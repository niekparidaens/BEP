
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


# Paths / I/O
PROJECT_ROOT = Path(
    os.environ.get("PROJECT_ROOT", "/tudelft.net/staff-umbrella/Xeniumenhancer")
).resolve()
ANN_DIR = Path(os.environ.get("ANN_DIR", PROJECT_ROOT / "AnnData")).resolve()
SAVE_DIR = Path(
    os.environ.get("OUTPUT_DIR", PROJECT_ROOT / "outputs" / "transformer_token_zinb")
).resolve()
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Default to backed mode on the cluster so X stays on disk during metadata prep
READ_MODE = os.environ.get("READ_MODE", "r") or "r"


# Split definitions
train = {
    "Bone marrow": ["TENX133", "TENX134"],
    "Bowel": ["TENX139"],
    "Brain": ["TENX138"],
    "Breast": [
        "TENX191", "TENX192", "TENX193",
        "TENX194", "TENX195", "TENX196",
        "NCBI783", "TENX95", "TENX98", "TENX99",
    ],
    "Femur bone": ["TENX132"],
    "Heart": ["TENX119"],
    "Kidney": ["TENX105", "TENX106"],
    "Liver": ["TENX120", "TENX121"],
    "Lung": [
        "NCBI856", "NCBI857", "NCBI858", "NCBI860", "NCBI861",
        "NCBI864", "NCBI865", "NCBI866", "NCBI867", "NCBI870",
        "NCBI873", "NCBI875", "NCBI876", "NCBI879", "NCBI880",
        "NCBI885", "NCBI886",
    ],
    "Ovary": ["TENX142"],
    "Tonsil": ["TENX124", "TENX125"],
}

val = {
    "Lung": ["NCBI883", "NCBI884"],
    "Breast": ["TENX197", "TENX198", "TENX199"],
}

test_seen_tissue_in_distribution = {
    "Lung": ["NCBI881", "NCBI882"],
    "Breast": ["TENX200", "TENX201", "TENX202"],
}

test_seen_tissue_distribution_shift = {
    "Lung": ["NCBI859", "TENX118", "TENX141", "TENX190"],
    "Breast": ["NCBI784", "NCBI785"],
}

test_unseen_tissue = {
    "Colon": ["TENX147", "TENX148", "TENX149", "TENX111", "TENX114"],
    "Skin": ["TENX122", "TENX123", "TENX115", "TENX117"],
    "Pancreas": ["TENX116", "TENX126", "TENX140"],
}


# Basic preprocessing / corruption config
threshold = 40
genes_threshold = 5

UNKNOWN_GENE_TOKEN = "__UNKNOWN_GENE__"
UNKNOWN_TISSUE_TOKEN = "__UNKNOWN_TISSUE__"

# Exactly like the VAE setup: one deterministic corrupted version per p-value
p_non_overlap_values = [0.19, 0.21, 0.25, 0.31]
base_seed = 42



# Helpers

def _adata_path(sample_id: str) -> Path:
    return ANN_DIR / f"{sample_id}_xenium_cell_level.h5ad"


def _load_sample_backed(sample_id: str, read_mode: str = READ_MODE):
    h5ad_path = _adata_path(sample_id)
    if not h5ad_path.exists():
        raise FileNotFoundError(f"Missing h5ad for {sample_id}: {h5ad_path}")
    return sc.read_h5ad(h5ad_path, backed=read_mode)


def _close_backed_adata(ad_panel) -> None:
    try:
        if getattr(ad_panel, "file", None) is not None:
            ad_panel.file.close()
    except Exception:
        pass


def _records_from_split_dict(split_dict, split_name, split_group):
    records = []
    for tissue_name, sample_ids in split_dict.items():
        for sample_id in sample_ids:
            records.append(
                {
                    "sample_id": str(sample_id).strip().upper(),
                    "tissue_name": tissue_name,
                    "split_name": split_name,
                    "split_group": split_group,
                }
            )
    return records


def _compute_qc_chunked(adata_backed, chunk_size=4096):
    """
    Compute total_counts and n_genes_by_counts without loading the whole matrix.
    """
    n = adata_backed.n_obs
    total_counts = np.zeros(n, dtype=np.float64)
    n_genes_by_counts = np.zeros(n, dtype=np.int32)

    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        X_chunk = adata_backed.X[start:stop]

        if sp.issparse(X_chunk):
            X_chunk = X_chunk.tocsr()
            total_counts[start:stop] = np.asarray(X_chunk.sum(axis=1)).ravel()
            n_genes_by_counts[start:stop] = np.diff(X_chunk.indptr)
        else:
            X_chunk = np.asarray(X_chunk)
            total_counts[start:stop] = X_chunk.sum(axis=1)
            n_genes_by_counts[start:stop] = np.count_nonzero(X_chunk, axis=1)

    return total_counts, n_genes_by_counts


def _prepare_panel_metadata(sample_id: str, threshold: int, genes_threshold: int):
    """
    Keep only lightweight metadata in RAM.
    """
    ad_panel = _load_sample_backed(sample_id)

    try:
        if {"total_counts", "n_genes_by_counts"}.issubset(ad_panel.obs.columns):
            total_counts = ad_panel.obs["total_counts"].to_numpy(copy=True)
            n_genes_by_counts = ad_panel.obs["n_genes_by_counts"].to_numpy(copy=True)
        else:
            print(f"{sample_id}: QC columns missing, computing them chunk-wise from backed X.")
            total_counts, n_genes_by_counts = _compute_qc_chunked(ad_panel)

        cell_mask = (total_counts >= threshold) & (n_genes_by_counts > genes_threshold)
        cell_pos = np.flatnonzero(cell_mask).astype(np.int64, copy=False)

        gene_names_full = pd.Index(ad_panel.var_names).astype(str)
        keep_gene_mask = (
            ~gene_names_full.str.startswith("UnassignedCodeword", na=False)
            & ~gene_names_full.str.lower().str.startswith("antisense", na=False)
        )
        keep_gene_mask = np.asarray(keep_gene_mask, dtype=bool)
        gene_pos_all = np.flatnonzero(keep_gene_mask).astype(np.int64, copy=False)
        gene_names_filtered = gene_names_full[keep_gene_mask]

        return {
            "h5ad_path": str(_adata_path(sample_id)),
            "cell_pos": cell_pos,
            "gene_pos_all": gene_pos_all,
            "gene_names": gene_names_filtered,
            "n_obs_raw": int(ad_panel.n_obs),
            "n_obs_filtered": int(cell_pos.size),
        }
    finally:
        _close_backed_adata(ad_panel)


def build_or_load_panel_cache(sample_id, rec, cache_dir, chunk_size=2048):
    """
    Cache each panel/sample as a dense float32 matrix in its native filtered gene order.
    This avoids repeated random backed .h5ad access during training.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{sample_id}_clean.npy"

    if cache_path.exists():
        return cache_path

    ad_panel = _load_sample_backed(sample_id)
    try:
        cell_pos = rec["cell_pos"]
        gene_pos_all = rec["gene_pos_all"]
        n_cells = int(cell_pos.size)

        Y = np.empty((n_cells, int(gene_pos_all.size)), dtype=np.float32)
        write_pos = 0

        for start in tqdm(
            range(0, n_cells, chunk_size),
            desc=f"Caching {sample_id}",
            unit="chunk",
            leave=False,
        ):
            stop = min(start + chunk_size, n_cells)
            rows = cell_pos[start:stop]

            try:
                block = ad_panel[rows, gene_pos_all].X
                if sp.issparse(block):
                    block = block.toarray()
                else:
                    block = np.asarray(block)
            except Exception:
                block_rows = []
                for r in rows:
                    row = ad_panel.X[int(r), gene_pos_all]
                    if sp.issparse(row):
                        row = row.toarray().ravel()
                    else:
                        row = np.asarray(row).ravel()
                    block_rows.append(row)
                block = np.vstack(block_rows)

            block = np.clip(block.astype(np.float32, copy=False), 0.0, None)
            Y[write_pos:write_pos + block.shape[0]] = block
            write_pos += block.shape[0]

        np.save(cache_path, Y)
        print(f"Saved cache for {sample_id}: {cache_path} | shape={Y.shape}")
        return cache_path
    finally:
        _close_backed_adata(ad_panel)


def build_gene_vocab(panel_data, train_sample_ids):
    genes = pd.Index([UNKNOWN_GENE_TOKEN])
    for sid in train_sample_ids:
        genes = genes.union(pd.Index(panel_data[sid]["gene_names"]).astype(str))
    genes = genes.sort_values()
    gene2id = {g: i for i, g in enumerate(genes)}
    id2gene = np.array(genes, dtype=object)
    return gene2id, id2gene


# Flatten split metadata

train_records = _records_from_split_dict(train, "train", "train")
val_records = _records_from_split_dict(val, "val", "val")
test_id_records = _records_from_split_dict(
    test_seen_tissue_in_distribution,
    "test",
    "seen_tissue_in_distribution",
)
test_shift_records = _records_from_split_dict(
    test_seen_tissue_distribution_shift,
    "test",
    "seen_tissue_distribution_shift",
)
test_unseen_records = _records_from_split_dict(
    test_unseen_tissue,
    "test",
    "unseen_tissue",
)

all_records = (
    train_records
    + val_records
    + test_id_records
    + test_shift_records
    + test_unseen_records
)

sample_to_tissue = {r["sample_id"]: r["tissue_name"] for r in all_records}
sample_to_group = {r["sample_id"]: r["split_group"] for r in all_records}
sample_to_split = {r["sample_id"]: r["split_name"] for r in all_records}

TRAIN_SAMPLE_IDS = [r["sample_id"] for r in train_records]
VAL_SAMPLE_IDS = [r["sample_id"] for r in val_records]
TEST_SAMPLE_IDS = [
    r["sample_id"] for r in (test_id_records + test_shift_records + test_unseen_records)
]
REQUESTED_IDS = sorted(set(TRAIN_SAMPLE_IDS + VAL_SAMPLE_IDS + TEST_SAMPLE_IDS))



# Load only requested sample IDs

available_ids = sorted(
    p.name.replace("_xenium_cell_level.h5ad", "")
    for p in ANN_DIR.glob("*_xenium_cell_level.h5ad")
)
available_set = set(available_ids)

missing_requested = [sid for sid in REQUESTED_IDS if sid not in available_set]
if missing_requested:
    print(f"Requested IDs missing from disk ({len(missing_requested)}): {missing_requested}")

panel_data = {}
load_failed = []

for sample_id in REQUESTED_IDS:
    if sample_id not in available_set:
        continue
    try:
        rec = _prepare_panel_metadata(
            sample_id=sample_id,
            threshold=threshold,
            genes_threshold=genes_threshold,
        )
        rec["tissue_name"] = sample_to_tissue[sample_id]
        rec["split_group"] = sample_to_group[sample_id]
        rec["split_name"] = sample_to_split[sample_id]
        panel_data[sample_id] = rec
    except Exception as e:
        load_failed.append((sample_id, str(e)))

TRAIN_SAMPLE_IDS = [sid for sid in TRAIN_SAMPLE_IDS if sid in panel_data]
VAL_SAMPLE_IDS = [sid for sid in VAL_SAMPLE_IDS if sid in panel_data]
TEST_SAMPLE_IDS = [sid for sid in TEST_SAMPLE_IDS if sid in panel_data]

if len(TRAIN_SAMPLE_IDS) == 0:
    raise ValueError("No training samples are available after loading.")
if len(VAL_SAMPLE_IDS) == 0:
    raise ValueError("No validation samples are available after loading.")
if len(TEST_SAMPLE_IDS) == 0:
    raise ValueError("No test samples are available after loading.")

print(f"Loaded requested panels: {len(panel_data)}")
print(f"Read mode: backed ({READ_MODE})")

if load_failed:
    print(f"Failed to load {len(load_failed)} sample(s):")
    for sid, msg in load_failed[:10]:
        print(f"  - {sid}: {msg}")

for sid in sorted(panel_data.keys()):
    rec = panel_data[sid]
    n_before = rec["n_obs_raw"]
    n_after = rec["n_obs_filtered"]
    n_removed = n_before - n_after
    pct_removed = (100.0 * n_removed / n_before) if n_before else 0.0
    print(
        f"{sid}: split={rec['split_name']} | group={rec['split_group']} | tissue={rec['tissue_name']} | "
        f"before={n_before}, after={n_after}, removed={n_removed} ({pct_removed:.1f}%), "
        f"kept_genes={len(rec['gene_names'])}"
    )


# Build cached clean matrices (all requested panels)

CACHE_DIR = SAVE_DIR / "panel_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

for sid in tqdm(sorted(panel_data.keys()), desc="Ensuring panel caches", unit="panel"):
    cache_path = build_or_load_panel_cache(sid, panel_data[sid], CACHE_DIR)
    panel_data[sid]["cache_path"] = str(cache_path)



# Vocab / token metadata from train only

gene2id, id2gene = build_gene_vocab(panel_data, TRAIN_SAMPLE_IDS)

shared_train_genes = set(map(str, panel_data[TRAIN_SAMPLE_IDS[0]]["gene_names"]))
for sid in TRAIN_SAMPLE_IDS[1:]:
    shared_train_genes &= set(map(str, panel_data[sid]["gene_names"]))

train_tissue_names = sorted({sample_to_tissue[sid] for sid in TRAIN_SAMPLE_IDS})
tissue_names = train_tissue_names + [UNKNOWN_TISSUE_TOKEN]
tissue2id = {t: i for i, t in enumerate(tissue_names)}

print(f"Train samples: {len(TRAIN_SAMPLE_IDS)}")
print(f"Val samples: {len(VAL_SAMPLE_IDS)}")
print(f"Test samples: {len(TEST_SAMPLE_IDS)}")
print(f"Vocabulary size (including unknown): {len(gene2id)}")
print(f"Genes shared across training samples: {len(shared_train_genes)}")
print(f"Tissues (includes unknown): {tissue2id}")
print(f"p_non_overlap_values: {p_non_overlap_values}")


# Token datasets

class GeneTokenDataset(Dataset):
    """
    One item = one cell from one cached panel/sample.

    When apply_corruption=True:
        dataset length = n_cells * len(p_non_overlap_values)
        and each version index corresponds to exactly one p-value,
        just like the VAE training / validation / test setup.

    When apply_corruption=False:
        dataset length = n_cells
        and x == y (clean input / clean target).
    """

    def __init__(
        self,
        panel_data,
        sample_ids,
        gene2id,
        shared_genes,
        sample_to_tissue,
        tissue2id,
        sample_to_group,
        p_non_overlap_values=None,
        base_seed=0,
        apply_corruption=True,
        split_seed_offset=0,
    ):
        self.panel_data = panel_data
        self.sample_ids = list(sample_ids)
        self.gene2id = dict(gene2id)
        self.shared_genes = set(map(str, shared_genes))
        self.sample_to_tissue = dict(sample_to_tissue)
        self.tissue2id = dict(tissue2id)
        self.sample_to_group = dict(sample_to_group)
        self.base_seed = int(base_seed)
        self.apply_corruption = bool(apply_corruption)
        self.split_seed_offset = int(split_seed_offset)

        self.unknown_gene_id = int(self.gene2id[UNKNOWN_GENE_TOKEN])
        self.unknown_tissue_id = int(self.tissue2id[UNKNOWN_TISSUE_TOKEN])

        if self.apply_corruption:
            self.p_values = np.atleast_1d(np.asarray(p_non_overlap_values, dtype=np.float64))
            if self.p_values.size < 1:
                raise ValueError("p_non_overlap_values must contain at least one value.")
            if np.any((self.p_values < 0.0) | (self.p_values > 1.0)):
                raise ValueError("All p_non_overlap_values must be between 0 and 1.")
            self.n_versions = int(self.p_values.size)
        else:
            self.p_values = np.array([], dtype=np.float64)
            self.n_versions = 1

        self.sample_defs = []
        self.sample_sizes = []
        self._mmap_cache = {}

        for sid in self.sample_ids:
            rec = self.panel_data[sid]

            tissue_name = self.sample_to_tissue.get(sid, UNKNOWN_TISSUE_TOKEN)
            tissue_id = int(self.tissue2id.get(tissue_name, self.unknown_tissue_id))

            gene_names = pd.Index(rec["gene_names"]).astype(str).to_numpy()
            gene_ids = np.array(
                [self.gene2id.get(g, self.unknown_gene_id) for g in gene_names],
                dtype=np.int64,
            )
            shared_mask = np.array(
                [(g in self.shared_genes) and (g != UNKNOWN_GENE_TOKEN) for g in gene_names],
                dtype=bool,
            )

            self.sample_defs.append(
                {
                    "sample_id": sid,
                    "cache_path": rec["cache_path"],
                    "n_cells": int(rec["n_obs_filtered"]),
                    "gene_ids": gene_ids,
                    "shared_mask": shared_mask,
                    "tissue_id": tissue_id,
                    "tissue_name": tissue_name,
                    "group_name": self.sample_to_group.get(sid, "unknown_group"),
                }
            )
            self.sample_sizes.append(int(rec["n_obs_filtered"]))

        self.sample_sizes = np.asarray(self.sample_sizes, dtype=np.int64)
        self.cum_sizes = np.cumsum(self.sample_sizes)
        self.total_cells = int(self.cum_sizes[-1]) if self.cum_sizes.size else 0

    def __len__(self):
        return self.total_cells * self.n_versions

    def _locate(self, global_cell_idx):
        if global_cell_idx < 0 or global_cell_idx >= self.total_cells:
            raise IndexError("Cell index out of range.")
        sample_idx = int(np.searchsorted(self.cum_sizes, global_cell_idx, side="right"))
        prev_cum = 0 if sample_idx == 0 else int(self.cum_sizes[sample_idx - 1])
        within_sample_idx = int(global_cell_idx - prev_cum)
        return sample_idx, within_sample_idx

    def _get_clean_matrix(self, cache_path):
        arr = self._mmap_cache.get(cache_path)
        if arr is None:
            arr = np.load(cache_path, mmap_mode="r", allow_pickle=False)
            self._mmap_cache[cache_path] = arr
        return arr

    def __getitem__(self, idx):
        version_idx = idx // self.total_cells
        within_idx = idx % self.total_cells

        sample_idx, row_idx = self._locate(within_idx)
        sample_def = self.sample_defs[sample_idx]

        Y_full = self._get_clean_matrix(sample_def["cache_path"])
        y = np.asarray(Y_full[row_idx], dtype=np.float32).copy()
        x = y.copy()

        if self.apply_corruption:
            p_non_overlap = float(self.p_values[version_idx])
            nz = x > 0
            if np.any(nz):
                counts = np.rint(x[nz]).astype(np.int64, copy=False)
                counts = np.clip(counts, 0, None)
                rng = np.random.default_rng(
                    self.base_seed
                    + self.split_seed_offset
                    + version_idx * 1_000_003
                    + within_idx
                )
                x[nz] = rng.binomial(counts, p_non_overlap).astype(np.float32, copy=False)
        else:
            p_non_overlap = np.nan

        return {
            "gene_ids": torch.tensor(sample_def["gene_ids"], dtype=torch.long),
            "x_vals": torch.tensor(x, dtype=torch.float32),
            "y_vals": torch.tensor(y, dtype=torch.float32),
            "shared_mask": torch.tensor(sample_def["shared_mask"], dtype=torch.bool),
            "tissue_id": torch.tensor(sample_def["tissue_id"], dtype=torch.long),
            "sample_id": sample_def["sample_id"],
            "tissue_name": sample_def["tissue_name"],
            "split_group": sample_def["group_name"],
            "p_non_overlap": float(p_non_overlap),
        }


def collate_gene_tokens(batch):
    B = len(batch)
    lengths = [len(item["gene_ids"]) for item in batch]
    Lmax = max(lengths)

    gene_ids = torch.zeros(B, Lmax, dtype=torch.long)
    x_vals = torch.zeros(B, Lmax, dtype=torch.float32)
    y_vals = torch.zeros(B, Lmax, dtype=torch.float32)
    attn_mask = torch.zeros(B, Lmax, dtype=torch.bool)
    shared_mask = torch.zeros(B, Lmax, dtype=torch.bool)
    tissue_ids = torch.zeros(B, dtype=torch.long)

    sample_ids = []
    tissue_names = []
    split_groups = []
    p_non_overlap = []

    for i, item in enumerate(batch):
        L = len(item["gene_ids"])
        gene_ids[i, :L] = item["gene_ids"]
        x_vals[i, :L] = item["x_vals"]
        y_vals[i, :L] = item["y_vals"]
        attn_mask[i, :L] = True
        shared_mask[i, :L] = item["shared_mask"]
        tissue_ids[i] = item["tissue_id"]

        sample_ids.append(item["sample_id"])
        tissue_names.append(item["tissue_name"])
        split_groups.append(item["split_group"])
        p_non_overlap.append(item["p_non_overlap"])

    return {
        "gene_ids": gene_ids,
        "x_vals": x_vals,
        "y_vals": y_vals,
        "attn_mask": attn_mask,
        "shared_mask": shared_mask,
        "tissue_id": tissue_ids,
        "sample_id": sample_ids,
        "tissue_name": tissue_names,
        "split_group": split_groups,
        "p_non_overlap": torch.tensor(p_non_overlap, dtype=torch.float32),
    }


train_dataset = GeneTokenDataset(
    panel_data=panel_data,
    sample_ids=TRAIN_SAMPLE_IDS,
    gene2id=gene2id,
    shared_genes=shared_train_genes,
    sample_to_tissue=sample_to_tissue,
    tissue2id=tissue2id,
    sample_to_group=sample_to_group,
    p_non_overlap_values=p_non_overlap_values,
    base_seed=base_seed,
    apply_corruption=True,
    split_seed_offset=0,
)

# Like the VAE: validation and test also use deterministic corrupted inputs
val_dataset = GeneTokenDataset(
    panel_data=panel_data,
    sample_ids=VAL_SAMPLE_IDS,
    gene2id=gene2id,
    shared_genes=shared_train_genes,
    sample_to_tissue=sample_to_tissue,
    tissue2id=tissue2id,
    sample_to_group=sample_to_group,
    p_non_overlap_values=p_non_overlap_values,
    base_seed=base_seed,
    apply_corruption=True,
    split_seed_offset=10_000_000,
)

test_dataset = GeneTokenDataset(
    panel_data=panel_data,
    sample_ids=TEST_SAMPLE_IDS,
    gene2id=gene2id,
    shared_genes=shared_train_genes,
    sample_to_tissue=sample_to_tissue,
    tissue2id=tissue2id,
    sample_to_group=sample_to_group,
    p_non_overlap_values=p_non_overlap_values,
    base_seed=base_seed,
    apply_corruption=True,
    split_seed_offset=20_000_000,
)

print(
    f"Token dataset cells | train={train_dataset.total_cells} | "
    f"val={val_dataset.total_cells} | test={test_dataset.total_cells}"
)
print(
    f"Token dataset examples | train={len(train_dataset)} | "
    f"val={len(val_dataset)} | test={len(test_dataset)}"
)


# Model

class GeneTokenAutoencoder(nn.Module):
    """
    Tissue-conditioned token autoencoder.
    No panel embeddings are used anywhere.
    """

    def __init__(
        self,
        n_genes_vocab,
        n_tissues,
        d_model=128,
        nhead=4,
        num_layers=3,
        latent_dim=32,
        dropout=0.1,
        theta_init=10.0,
    ):
        super().__init__()

        self.gene_emb = nn.Embedding(n_genes_vocab, d_model)
        self.tissue_emb = nn.Embedding(n_tissues, d_model)

        self.value_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.input_norm = nn.LayerNorm(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.post_encoder_norm = nn.LayerNorm(d_model)

        self.latent_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, latent_dim),
        )
        self.z_proj = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.decoder_trunk = nn.Sequential(
            nn.Linear(3 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
        )
        self.mu_head = nn.Linear(d_model, 1)
        self.pi_head = nn.Linear(d_model, 1)

        theta_unconstrained_init = float(np.log(np.expm1(theta_init)))
        self.log_theta_gene = nn.Embedding(n_genes_vocab, 1)
        nn.init.constant_(self.log_theta_gene.weight, theta_unconstrained_init)

    def encode(self, gene_ids, x_vals, attn_mask, tissue_id):
        g = self.gene_emb(gene_ids)
        x = self.value_mlp(x_vals.unsqueeze(-1))
        t = self.tissue_emb(tissue_id).unsqueeze(1)

        h = self.input_norm(g + x + t)
        pad_mask = ~attn_mask
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        h = self.post_encoder_norm(h)

        mask_f = attn_mask.unsqueeze(-1).float()
        pooled = (h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
        z = self.latent_proj(pooled)
        return z

    def decode_params(self, z, gene_ids, tissue_id):
        B, L = gene_ids.shape
        zg = self.z_proj(z).unsqueeze(1).expand(B, L, -1)
        g = self.gene_emb(gene_ids)
        t = self.tissue_emb(tissue_id).unsqueeze(1).expand(B, L, -1)

        dec_in = torch.cat([zg, g, t], dim=-1)
        h = self.decoder_trunk(dec_in)

        mu_logit = self.mu_head(h).squeeze(-1)
        pi_logit = self.pi_head(h).squeeze(-1)
        theta_unconstrained = self.log_theta_gene(gene_ids).squeeze(-1)

        return mu_logit, pi_logit, theta_unconstrained

    def forward_with_params(self, gene_ids, x_vals, attn_mask, tissue_id):
        z = self.encode(gene_ids, x_vals, attn_mask, tissue_id)
        mu_logit, pi_logit, theta_unconstrained = self.decode_params(z, gene_ids, tissue_id)

        mu = F.softplus(mu_logit)
        pi = torch.sigmoid(pi_logit)
        recon_counts = (1.0 - pi) * mu

        return recon_counts, z, mu_logit, pi_logit, theta_unconstrained

    def forward(self, gene_ids, x_vals, attn_mask, tissue_id):
        recon_counts, z, _, _, _ = self.forward_with_params(
            gene_ids=gene_ids,
            x_vals=x_vals,
            attn_mask=attn_mask,
            tissue_id=tissue_id,
        )
        return recon_counts, z


def token_zinb_nll_matrix(mu_logit, pi_logit, theta_unconstrained, target_counts, eps=1e-8):
    target_counts = target_counts.clamp_min(0.0).float()
    mu_logit = mu_logit.float()
    pi_logit = pi_logit.float()
    theta_unconstrained = theta_unconstrained.float()

    mu = F.softplus(mu_logit).clamp_min(eps)
    theta = F.softplus(theta_unconstrained).clamp_min(eps)

    log_pi = -F.softplus(-pi_logit)
    log_1m_pi = -F.softplus(pi_logit)

    log_theta = torch.log(theta)
    log_mu = torch.log(mu)
    log_theta_mu = torch.log(theta + mu)

    nb_log_prob = (
        torch.lgamma(target_counts + theta)
        - torch.lgamma(theta)
        - torch.lgamma(target_counts + 1.0)
        + theta * (log_theta - log_theta_mu)
        + target_counts * (log_mu - log_theta_mu)
    )

    nb_zero_log_prob = theta * (log_theta - log_theta_mu)

    zero_log_prob = torch.logaddexp(log_pi, log_1m_pi + nb_zero_log_prob)
    nonzero_log_prob = log_1m_pi + nb_log_prob

    return -torch.where(target_counts < eps, zero_log_prob, nonzero_log_prob)


def token_zinb_loss(mu_logit, pi_logit, theta_unconstrained, target_counts, attn_mask, eps=1e-8):
    zinb_nll = token_zinb_nll_matrix(
        mu_logit=mu_logit,
        pi_logit=pi_logit,
        theta_unconstrained=theta_unconstrained,
        target_counts=target_counts,
        eps=eps,
    )
    valid = zinb_nll[attn_mask]
    if valid.numel() == 0:
        return torch.tensor(0.0, device=zinb_nll.device)
    return valid.mean()


# Device / loaders

torch.set_float32_matmul_precision("high")
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
if use_cuda:
    torch.backends.cudnn.benchmark = True
print(f"Token model device: {device}")

batch_size_cuda = 512
batch_size_cpu = 128
batch_size = batch_size_cuda if use_cuda else batch_size_cpu

loader_kwargs = {
    "batch_size": batch_size,
    "collate_fn": collate_gene_tokens,
    "num_workers": 4 if use_cuda else 0,
    "pin_memory": use_cuda,
}
if loader_kwargs["num_workers"] > 0:
    loader_kwargs["persistent_workers"] = True
    loader_kwargs["prefetch_factor"] = 2

train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

print(f"train/val/test batches: {len(train_loader)} / {len(val_loader)} / {len(test_loader)}")



# Training / evaluation helpers

def run_epoch_token_ae(
    model,
    loader,
    optimizer,
    scaler,
    device,
    train=True,
    epoch_label="",
):
    model.train() if train else model.eval()

    loss_sum = 0.0
    n_batches = 0

    stage = "train" if train else "val"

    iter_loader = tqdm(
        loader,
        desc=f"{epoch_label} [{stage}]",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    )

    for batch in iter_loader:
        gene_ids = batch["gene_ids"].to(device, non_blocking=use_cuda)
        x_vals = batch["x_vals"].to(device, non_blocking=use_cuda)
        y_vals = batch["y_vals"].to(device, non_blocking=use_cuda)
        attn_mask = batch["attn_mask"].to(device, non_blocking=use_cuda)
        tissue_id = batch["tissue_id"].to(device, non_blocking=use_cuda)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=use_cuda):
                _, _, mu_logit, pi_logit, theta_unconstrained = model.forward_with_params(
                    gene_ids=gene_ids,
                    x_vals=x_vals,
                    attn_mask=attn_mask,
                    tissue_id=tissue_id,
                )

                loss = token_zinb_loss(
                    mu_logit=mu_logit,
                    pi_logit=pi_logit,
                    theta_unconstrained=theta_unconstrained,
                    target_counts=y_vals,
                    attn_mask=attn_mask,
                )

            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()

        loss_sum += float(loss.item())
        n_batches += 1
        iter_loader.set_postfix({"zinb_nll": f"{(loss_sum / n_batches):.5f}"})

    if n_batches == 0:
        return np.inf

    return loss_sum / n_batches


def collect_eval_rows(model, loader, device):
    model.eval()
    rows = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Token AE eval", unit="batch", leave=False):
            gene_ids = batch["gene_ids"].to(device, non_blocking=use_cuda)
            x_vals = batch["x_vals"].to(device, non_blocking=use_cuda)
            y_vals = batch["y_vals"].to(device, non_blocking=use_cuda)
            attn_mask = batch["attn_mask"].to(device, non_blocking=use_cuda)
            shared_mask = batch["shared_mask"].to(device, non_blocking=use_cuda)
            tissue_id = batch["tissue_id"].to(device, non_blocking=use_cuda)

            recon_counts, _, mu_logit, pi_logit, theta_unconstrained = model.forward_with_params(
                gene_ids=gene_ids,
                x_vals=x_vals,
                attn_mask=attn_mask,
                tissue_id=tissue_id,
            )

            diff2 = (recon_counts - y_vals) ** 2
            zinb_tok = token_zinb_nll_matrix(
                mu_logit=mu_logit,
                pi_logit=pi_logit,
                theta_unconstrained=theta_unconstrained,
                target_counts=y_vals,
            )

            p_vals = batch["p_non_overlap"].cpu().numpy()

            for i in range(recon_counts.shape[0]):
                valid = attn_mask[i]
                shared = valid & shared_mask[i]
                specific = valid & (~shared_mask[i])

                rows.append(
                    {
                        "sample_id": batch["sample_id"][i],
                        "tissue_name": batch["tissue_name"][i],
                        "split_group": batch["split_group"][i],
                        "p_non_overlap": float(p_vals[i]),
                        "mse_all": float(diff2[i][valid].mean().item()) if valid.any() else np.nan,
                        "mse_shared": float(diff2[i][shared].mean().item()) if shared.any() else np.nan,
                        "mse_specific": float(diff2[i][specific].mean().item()) if specific.any() else np.nan,
                        "zinb_nll_all": float(zinb_tok[i][valid].mean().item()) if valid.any() else np.nan,
                        "n_tokens": int(valid.sum().item()),
                        "n_shared_tokens": int(shared.sum().item()),
                        "n_specific_tokens": int(specific.sum().item()),
                    }
                )

    return pd.DataFrame(rows)



# Train

epochs = 30
learning_rate = 5e-4
early_stop_patience = 3
min_epochs_before_early_stop = 25

model = GeneTokenAutoencoder(
    n_genes_vocab=len(gene2id),
    n_tissues=len(tissue2id),
    d_model=128,
    nhead=4,
    num_layers=3,
    latent_dim=32,
    dropout=0.1,
    theta_init=10.0,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

hist_train = []
hist_val = []
best_val = np.inf
best_state = None
best_epoch = 0
stale = 0

print(
    f"Training config | epochs={epochs}, lr={learning_rate}, d_model=128, "
    f"layers=3, nhead=4, latent_dim=32, recon_loss=ZINB(raw counts)"
)

overall_t0 = time.time()
epoch_bar = tqdm(range(1, epochs + 1), desc="Token AE epochs", unit="epoch")

for epoch in epoch_bar:
    epoch_t0 = time.time()

    tr = run_epoch_token_ae(
        model=model,
        loader=train_loader,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        train=True,
        epoch_label=f"Epoch {epoch:02d}/{epochs}",
    )
    va = run_epoch_token_ae(
        model=model,
        loader=val_loader,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        train=False,
        epoch_label=f"Epoch {epoch:02d}/{epochs}",
    )

    hist_train.append(tr)
    hist_val.append(va)

    improved = va < (best_val - 1e-8)
    if improved:
        best_val = va
        best_epoch = epoch
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        stale = 0
    else:
        stale += 1

    if epoch % 5 == 0 or epoch == 1 or improved:
        print(f"Epoch {epoch:02d}/{epochs} | train_zinb={tr:.6f} | val_zinb={va:.6f}")

    epoch_sec = time.time() - epoch_t0
    elapsed_sec = time.time() - overall_t0
    epoch_bar.set_postfix(
        {
            "train": f"{tr:.4f}",
            "val": f"{va:.4f}",
            "best": best_epoch,
            "epoch_s": f"{epoch_sec:.1f}",
            "elapsed_m": f"{elapsed_sec/60:.1f}",
        }
    )

    if epoch >= min_epochs_before_early_stop and stale >= early_stop_patience:
        print(f"Early stopping at epoch {epoch}; best validation loss at epoch {best_epoch}.")
        break

epoch_bar.close()

if best_state is None:
    raise RuntimeError("No checkpoint saved for token AE.")

model.load_state_dict(best_state)
print(f"Loaded best token AE checkpoint from epoch {best_epoch} (val={best_val:.6f})")



# Save checkpoint / curves / eval
ckpt_path = SAVE_DIR / "transformer_token_zinb_best.pt"
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "best_epoch": int(best_epoch),
        "best_val_zinb_nll": float(best_val),
        "gene_vocab_size": int(len(gene2id)),
        "gene2id": gene2id,
        "tissue2id": tissue2id,
        "id2gene": id2gene.tolist(),
        "unknown_gene_token": UNKNOWN_GENE_TOKEN,
        "unknown_tissue_token": UNKNOWN_TISSUE_TOKEN,
        "train_sample_ids": TRAIN_SAMPLE_IDS,
        "val_sample_ids": VAL_SAMPLE_IDS,
        "test_sample_ids": TEST_SAMPLE_IDS,
        "p_non_overlap_values": list(map(float, p_non_overlap_values)),
        "threshold": int(threshold),
        "genes_threshold": int(genes_threshold),
    },
    ckpt_path,
)
print(f"Saved token AE checkpoint to: {ckpt_path}")

ep = np.arange(1, len(hist_train) + 1)
plt.figure(figsize=(7, 4))
plt.plot(ep, hist_train, label="Train", linewidth=2)
plt.plot(ep, hist_val, label="Validation", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("ZINB NLL")
plt.title("Transformer token AE learning curves (ZINB)")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
curve_path = SAVE_DIR / "transformer_token_zinb_learning_curves.png"
plt.savefig(curve_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved learning curves to: {curve_path}")

val_eval = collect_eval_rows(model, val_loader, device)
test_eval = collect_eval_rows(model, test_loader, device)

val_eval_path = SAVE_DIR / "transformer_token_zinb_val_metrics.csv"
test_eval_path = SAVE_DIR / "transformer_token_zinb_test_metrics.csv"
val_eval.to_csv(val_eval_path, index=False)
test_eval.to_csv(test_eval_path, index=False)

print(f"Saved validation metrics to: {val_eval_path}")
print(f"Saved test metrics to: {test_eval_path}")

print("Validation summary:")
print(
    val_eval.groupby(["split_group", "p_non_overlap"])[
        ["mse_all", "mse_shared", "mse_specific", "zinb_nll_all"]
    ].mean(numeric_only=True)
)

print("Test summary:")
print(
    test_eval.groupby(["split_group", "p_non_overlap"])[
        ["mse_all", "mse_shared", "mse_specific", "zinb_nll_all"]
    ].mean(numeric_only=True)
)
