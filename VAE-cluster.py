import json
import os
import sys
import time
from pathlib import Path

import anndata as ad
import annsel as an
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import zarr
from IPython.display import display
from scipy.sparse import csr_matrix, issparse
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset
from tqdm.auto import tqdm

matplotlib.use("Agg")


# Paths / I/O

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/tudelft.net/staff-umbrella/Xeniumenhancer")).resolve()
ANN_DIR = Path(os.environ.get("ANN_DIR", PROJECT_ROOT / "AnnData")).resolve()
SAVE_DIR = Path(os.environ.get("OUTPUT_DIR", PROJECT_ROOT / "outputs")).resolve()
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Default to backed mode on the cluster so X stays on disk
READ_MODE = os.environ.get("READ_MODE", "r") or "r"


def _adata_path(sample_id: str):
    return ANN_DIR / f"{sample_id}_xenium_cell_level.h5ad"


def _load_sample_backed(sample_id: str, read_mode=READ_MODE):
    h5ad_path = _adata_path(sample_id)
    if not h5ad_path.exists():
        raise FileNotFoundError(f"Missing h5ad for {sample_id}: {h5ad_path}")
    return sc.read_h5ad(h5ad_path, backed=read_mode)

# Fixed split definitions / corruption setup
threshold = 40
genes_threshold = 5
p_non_overlap_values = [0.15, 0.19, 0.23, 0.27, 0.31, 0.37]
base_seed = 42

split_samples_train = [
    "NCBI856", "NCBI857", "NCBI858", "NCBI860", "NCBI861", "NCBI864",
    "NCBI865", "NCBI866", "NCBI867", "NCBI870", "NCBI873", "NCBI875", "NCBI876",
]
split_samples_val = ["NCBI879", "NCBI880", "NCBI881"]
split_samples_test_a = ["NCBI882", "NCBI883", "NCBI884"]
split_samples_test_b = [
    "NCBI887", "NCBI888", "TENX189",
    "NCBI885", "NCBI886",
    "NCBI859",
    "TENX118",
    "TENX141",
    "TENX190",
]
split_samples_test_c = [
    "TENX191", "TENX192", "TENX193", "TENX194", "TENX195", "TENX196", "TENX197", "TENX198",
    "NCBI783", "TENX94", "TENX95", "TENX98", "TENX99",
    "TENX147", "TENX148", "TENX149", "TENX111", "TENX114",
    "TENX116", "TENX126", "TENX140",
    "TENX122", "TENX123", "TENX115", "TENX117", "TENX158",
]
split_samples_test = split_samples_test_a + split_samples_test_b + split_samples_test_c


def _select_available_ids(sample_ids, available_like):
    available_set = set(available_like)
    normalized = [str(s).strip().upper() for s in sample_ids]
    found = [sid for sid in normalized if sid in available_set]
    missing = [sid for sid in normalized if sid not in available_set]
    return found, missing



# Backed QC / metadata prep

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
    Keep the matrix backed on disk and store only:
      - filtered row indices
      - filtered gene indices
      - filtered gene names
    """
    ad_panel = _load_sample_backed(sample_id)

    if {"total_counts", "n_genes_by_counts"}.issubset(ad_panel.obs.columns):
        total_counts = ad_panel.obs["total_counts"].to_numpy(copy=True)
        n_genes_by_counts = ad_panel.obs["n_genes_by_counts"].to_numpy(copy=True)
    else:
        print(f"{sample_id}: QC columns missing, computing them chunk-wise from backed X.")
        total_counts, n_genes_by_counts = _compute_qc_chunked(ad_panel)

    cell_mask = (
        (total_counts >= threshold)
        & (n_genes_by_counts > genes_threshold)
    )
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
        "adata": ad_panel,                   # backed AnnData handle
        "cell_pos": cell_pos,               # kept rows in original X
        "gene_pos_all": gene_pos_all,       # kept cols in original X
        "gene_names": gene_names_filtered,  # names after gene filtering
        "n_obs_raw": int(ad_panel.n_obs),
        "n_obs_filtered": int(cell_pos.size),
    }



# Load only requested samples
available_ids = sorted(
    p.name.replace("_xenium_cell_level.h5ad", "")
    for p in ANN_DIR.glob("*_xenium_cell_level.h5ad")
)
available_set = set(available_ids)

train_panel_ids, missing_train = _select_available_ids(split_samples_train, available_set)
val_panel_ids, missing_val = _select_available_ids(split_samples_val, available_set)
test_panel_ids, missing_test = _select_available_ids(split_samples_test, available_set)
test_a_panel_ids, _ = _select_available_ids(split_samples_test_a, available_set)
test_b_panel_ids, _ = _select_available_ids(split_samples_test_b, available_set)
test_c_panel_ids, _ = _select_available_ids(split_samples_test_c, available_set)

requested_ids = sorted(set(train_panel_ids + val_panel_ids + test_panel_ids))

panel_data = {}
load_failed = []

for sample_id in requested_ids:
    try:
        panel_data[sample_id] = _prepare_panel_metadata(
            sample_id=sample_id,
            threshold=threshold,
            genes_threshold=genes_threshold,
        )
    except Exception as e:
        load_failed.append((sample_id, str(e)))

# Drop any samples that failed to open from the split lists
train_panel_ids = [sid for sid in train_panel_ids if sid in panel_data]
val_panel_ids = [sid for sid in val_panel_ids if sid in panel_data]
test_panel_ids = [sid for sid in test_panel_ids if sid in panel_data]
test_a_panel_ids = [sid for sid in test_a_panel_ids if sid in panel_data]
test_b_panel_ids = [sid for sid in test_b_panel_ids if sid in panel_data]
test_c_panel_ids = [sid for sid in test_c_panel_ids if sid in panel_data]

if len(train_panel_ids) == 0:
    raise ValueError("No train panels from split_samples_train are present in `panel_data`.")
if len(val_panel_ids) == 0:
    raise ValueError("No validation panels from split_samples_val are present in `panel_data`.")
if len(test_panel_ids) == 0:
    raise ValueError("No test panels from split_samples_test are present in `panel_data`.")

print(f"Loaded requested panels: {len(panel_data)}")
print(f"Read mode: backed ({READ_MODE})")

if missing_train or missing_val or missing_test:
    print("Requested sample IDs missing from disk:")
    if missing_train:
        print(f"  train missing: {missing_train}")
    if missing_val:
        print(f"  val missing: {missing_val}")
    if missing_test:
        print(f"  test missing: {missing_test}")

if load_failed:
    print(f"Failed to load {len(load_failed)} sample(s):")
    for sid, msg in load_failed[:10]:
        print(f"  - {sid}: {msg}")

for sid in requested_ids:
    if sid not in panel_data:
        continue
    rec = panel_data[sid]
    n_before = rec["n_obs_raw"]
    n_after = rec["n_obs_filtered"]
    n_removed = n_before - n_after
    pct_removed = (100.0 * n_removed / n_before) if n_before else 0.0
    print(
        f"{sid}: before={n_before}, after={n_after}, "
        f"removed={n_removed} ({pct_removed:.1f}%), kept_genes={len(rec['gene_names'])}"
    )



def _to_dense_float32(X):
    if sp.issparse(X):
        return X.toarray().astype(np.float32)
    return np.asarray(X, dtype=np.float32)


class PanelRowAccessor:
    """Access rows from multiple backed AnnData panels in one shared gene order."""

    def __init__(self, panel_data, panel_ids, common_genes):
        self.panel_defs = []
        self.panel_sizes = []
        self.common_genes = pd.Index(common_genes).astype(str)

        for sid in panel_ids:
            rec = panel_data[sid]

            # Map this panel's filtered genes onto the shared train gene space
            idx = rec["gene_names"].get_indexer(self.common_genes)
            present_mask = idx >= 0

            # Positions in the panel's filtered gene list
            filtered_pos = idx[present_mask].astype(np.int64, copy=False)

            # Convert to original X column positions
            panel_pos = rec["gene_pos_all"][filtered_pos]

            # Output positions in the common gene vector
            common_pos = np.flatnonzero(present_mask).astype(np.int64, copy=False)

            if panel_pos.size == 0:
                raise ValueError(f"Panel {sid} has zero overlap with the training gene space.")

            order = np.argsort(panel_pos)
            panel_pos = panel_pos[order]
            common_pos = common_pos[order]

            self.panel_defs.append((
                rec["adata"],       # backed AnnData
                rec["cell_pos"],    # kept rows in original X
                panel_pos,          # original X cols for common genes
                common_pos,         # output positions in shared vector
            ))
            self.panel_sizes.append(int(rec["cell_pos"].size))

        self.panel_sizes = np.asarray(self.panel_sizes, dtype=np.int64)
        self.cum_sizes = np.cumsum(self.panel_sizes)
        self.total_cells = int(self.cum_sizes[-1]) if self.cum_sizes.size else 0
        self.n_genes = int(len(self.common_genes))

    def _locate(self, global_cell_idx):
        if global_cell_idx < 0 or global_cell_idx >= self.total_cells:
            raise IndexError("Cell index out of range.")
        panel_idx = int(np.searchsorted(self.cum_sizes, global_cell_idx, side="right"))
        prev_cum = 0 if panel_idx == 0 else int(self.cum_sizes[panel_idx - 1])
        within_panel_idx = int(global_cell_idx - prev_cum)
        return panel_idx, within_panel_idx

    def get_row_dense(self, global_cell_idx):
        panel_idx, row_idx = self._locate(global_cell_idx)
        ad_panel, cell_pos, panel_pos, common_pos = self.panel_defs[panel_idx]

        orig_row_idx = int(cell_pos[row_idx])
        row = ad_panel.X[orig_row_idx, panel_pos]

        if sp.issparse(row):
            row_vals = row.toarray().ravel().astype(np.float32, copy=False)
        else:
            row_vals = np.asarray(row, dtype=np.float32).ravel()

        y = np.zeros(self.n_genes, dtype=np.float32)
        y[common_pos] = row_vals
        return np.clip(y, 0.0, None)


def load_or_build_dense_split(
    panel_data,
    panel_ids,
    common_genes,
    cache_path,
    dtype=np.float32,
    chunk_size=2048,
):
    """
    Build one dense matrix for a split in the shared gene space and cache it to disk.
    Reuses the cache on later runs.
    """
    cache_path = Path(cache_path)

    if cache_path.exists():
        arr = np.load(cache_path, allow_pickle=False)
        print(f"Loaded cached split: {cache_path} | shape={arr.shape} | dtype={arr.dtype}")
        return arr

    accessor = PanelRowAccessor(panel_data, panel_ids, common_genes)
    Y = np.empty((accessor.total_cells, accessor.n_genes), dtype=dtype)

    write_pos = 0
    panel_pbar = tqdm(panel_ids, desc=f"Building {cache_path.stem}", unit="panel")

    for sid in panel_pbar:
        rec = panel_data[sid]

        idx = rec["gene_names"].get_indexer(accessor.common_genes)
        present_mask = idx >= 0

        filtered_pos = idx[present_mask].astype(np.int64, copy=False)
        panel_pos = rec["gene_pos_all"][filtered_pos]
        common_pos = np.flatnonzero(present_mask).astype(np.int64, copy=False)

        if panel_pos.size == 0:
            raise ValueError(f"Panel {sid} has zero overlap with the training gene space.")

        order = np.argsort(panel_pos)
        panel_pos = panel_pos[order]
        common_pos = common_pos[order]

        cell_pos = rec["cell_pos"]
        ad_panel = rec["adata"]
        n_rows = int(cell_pos.size)

        for start in range(0, n_rows, chunk_size):
            stop = min(start + chunk_size, n_rows)
            rows = cell_pos[start:stop]

            # Fast path: try block loading from backed AnnData
            try:
                block = ad_panel[rows, panel_pos].X
                if sp.issparse(block):
                    block = block.toarray()
                else:
                    block = np.asarray(block)
            except Exception:
                # Fallback: row-by-row if backed fancy indexing fails
                block_rows = []
                for r in rows:
                    row = ad_panel.X[int(r), panel_pos]
                    if sp.issparse(row):
                        row = row.toarray().ravel()
                    else:
                        row = np.asarray(row).ravel()
                    block_rows.append(row)
                block = np.vstack(block_rows)

            block = block.astype(dtype, copy=False)

            Y_block = np.zeros((block.shape[0], accessor.n_genes), dtype=dtype)
            Y_block[:, common_pos] = block

            Y[write_pos:write_pos + block.shape[0]] = Y_block
            write_pos += block.shape[0]

    np.save(cache_path, Y)
    print(f"Saved cached split: {cache_path} | shape={Y.shape} | dtype={Y.dtype}")
    return Y



class MultiVersionTrainDataset(Dataset):
    def __init__(self, clean_matrix, p_non_overlap_values, base_seed=0):
        self.Y = np.ascontiguousarray(clean_matrix, dtype=np.float32)
        self.n_train, self.n_genes = self.Y.shape
        self.base_seed = int(base_seed)

        self.p_values = np.atleast_1d(np.asarray(p_non_overlap_values, dtype=np.float64))
        if self.p_values.size < 1:
            raise ValueError("p_non_overlap_values must contain at least one value.")
        if np.any((self.p_values < 0.0) | (self.p_values > 1.0)):
            raise ValueError("All p_non_overlap_values must be between 0 and 1.")

        self.n_versions = int(self.p_values.size)
        self.p_versions = []
        for p_non_overlap in self.p_values:
            p_rep = np.full(self.n_genes, float(p_non_overlap), dtype=np.float32)
            self.p_versions.append(p_rep)

    def __len__(self):
        return self.n_versions * self.n_train

    def __getitem__(self, idx):
        version_idx = idx // self.n_train
        within_idx = idx % self.n_train

        y = self.Y[within_idx].copy()
        x = y.copy()

        nz = x > 0
        if np.any(nz):
            counts = np.rint(x[nz]).astype(np.int64, copy=False)
            counts = np.clip(counts, 0, None)
            rng = np.random.default_rng(self.base_seed + version_idx * 1_000_003 + within_idx)
            p_entry = self.p_versions[version_idx][nz]
            x[nz] = rng.binomial(counts, p_entry).astype(np.float32, copy=False)

        return torch.from_numpy(x), torch.from_numpy(y)


class CleanEvalDataset(Dataset):
    """Validation dataset with clean inputs and clean targets (no corruption)."""

    def __init__(self, clean_matrix):
        self.Y = np.ascontiguousarray(clean_matrix, dtype=np.float32)
        self.n_cells, self.n_genes = self.Y.shape

    def __len__(self):
        return self.n_cells

    def __getitem__(self, idx):
        y = self.Y[idx].copy()
        x = y.copy()
        return torch.from_numpy(x), torch.from_numpy(y)


class CleanEvalDatasetLazy(Dataset):
    """Lazy test dataset with clean inputs and clean targets (no corruption)."""

    def __init__(self, panel_data, panel_ids, common_genes):
        self.rows = PanelRowAccessor(panel_data, panel_ids, common_genes)
        self.n_cells = self.rows.total_cells

    def __len__(self):
        return self.n_cells

    def __getitem__(self, idx):
        y = self.rows.get_row_dense(idx)
        x = y.copy()
        return torch.from_numpy(x), torch.from_numpy(y)


# Shared gene space from train panels only
common_genes = panel_data[train_panel_ids[0]]["gene_names"]
for sid in train_panel_ids[1:]:
    common_genes = common_genes.intersection(panel_data[sid]["gene_names"], sort=False)

if len(common_genes) == 0:
    raise ValueError("No shared genes across selected training panels.")

gene_names = common_genes.to_numpy()
n_genes = len(gene_names)


# Build / cache dense train + val matrices

CACHE_DIR = SAVE_DIR / "split_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

train_cache = CACHE_DIR / f"train_{len(train_panel_ids)}panels_{n_genes}genes.npy"
val_cache = CACHE_DIR / f"val_{len(val_panel_ids)}panels_{n_genes}genes.npy"

Y_train = load_or_build_dense_split(
    panel_data=panel_data,
    panel_ids=train_panel_ids,
    common_genes=common_genes,
    cache_path=train_cache,
    dtype=np.float32,
    chunk_size=2048,
)

Y_val = load_or_build_dense_split(
    panel_data=panel_data,
    panel_ids=val_panel_ids,
    common_genes=common_genes,
    cache_path=val_cache,
    dtype=np.float32,
    chunk_size=2048,
)

# Compatibility placeholders
X_inputs_model = [None] * len(p_non_overlap_values)
X_inputs = X_inputs_model
X_tgt = None
Y_test = None



train_dataset = MultiVersionTrainDataset(
    clean_matrix=Y_train,
    p_non_overlap_values=p_non_overlap_values,
    base_seed=base_seed,
)

val_dataset = CleanEvalDataset(
    clean_matrix=Y_val,
)

# Keep test lazy because it is very large
test_dataset = CleanEvalDatasetLazy(
    panel_data=panel_data,
    panel_ids=test_panel_ids,
    common_genes=common_genes,
)

train_idx = np.arange(train_dataset.n_train, dtype=np.int64)
val_idx = np.arange(len(val_dataset), dtype=np.int64)
test_idx = np.arange(len(test_dataset), dtype=np.int64)

test_a_cells = int(sum(panel_data[sid]["n_obs_filtered"] for sid in test_a_panel_ids))
test_b_cells = int(sum(panel_data[sid]["n_obs_filtered"] for sid in test_b_panel_ids))
test_c_cells = int(sum(panel_data[sid]["n_obs_filtered"] for sid in test_c_panel_ids))

n_cells = len(train_idx)
input_name = f"panel-wise cached train/val corruptions ({len(train_panel_ids)} train panels x {len(p_non_overlap_values)} p-values)"
split_mode = "fixed panel/sample split (panel-wise)"

print(f"Using input source: {input_name}")
print(f"Shared genes (train-defined): {n_genes}")
print(f"Train/Val/Test panels: {len(train_panel_ids)} / {len(val_panel_ids)} / {len(test_panel_ids)}")
print(f"Train/Val/Test cells: {train_dataset.n_train} / {len(val_dataset)} / {len(test_dataset)}")
print(f"Test A/B/C cells: {test_a_cells} / {test_b_cells} / {test_c_cells}")
print("Validation/Test inputs: clean (no corruption)")
print("Training data mode: raw counts (train/val cached dense, test lazy)")
print(f"Training examples: {len(train_dataset)}")
print(f"Validation examples: {len(val_dataset)} | Test examples: {len(test_dataset)}")


# Hyperparameters
epochs = 40
beta = 1e-3
latent_dim = 16
hidden_dim = 256
learning_rate = 1e-3
early_stop_patience = 5
theta_init = 10.0
pi_init = 0.1
batch_size_cuda = 1024
batch_size_cpu = 256

# Hyperparameters

epochs = 30
beta = 1e-3
latent_dim = 16
hidden_dim = 256
learning_rate = 1e-3
early_stop_patience = 3
theta_init = 10.0
pi_init = 0.1

# lower if processed gets killed immediately
batch_size_cuda = 512
batch_size_cpu = 128

torch.set_float32_matmul_precision("high")
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
if use_cuda:
    torch.backends.cudnn.benchmark = True
print(f"Running on device: {device}")

batch_size = batch_size_cuda if use_cuda else batch_size_cpu
loader_kwargs = {
    "num_workers": 0,
    "pin_memory": use_cuda,
}

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    **loader_kwargs,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    **loader_kwargs,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    **loader_kwargs,
)

class VAE(nn.Module):
    # Set up the encoder and decoder layers once
    def __init__(self, input_dim, latent_dim=16, hidden_dim=256):
        super().__init__()
        self.enc_fc1 = nn.Linear(input_dim, hidden_dim)
        self.enc_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

        self.dec_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.dec_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, input_dim)

    # Pass through two encoder layers to get mu/logvar
    def encode(self, x):
        h = F.relu(self.enc_fc1(x))
        h = F.relu(self.enc_fc2(h))
        return self.mu(h), self.logvar(h)

    # Sample a latent vector using the reparameterization trick
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # Map latent z back to gene-space logits
    def decode(self, z):
        h = F.relu(self.dec_fc1(z))
        h = F.relu(self.dec_fc2(h))
        return self.out(h)

    # Full pass: encode, sample, then decode
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


# Here we combine ZINB reconstruction loss with KL regularization
def vae_loss(
    recon,          # Decoder output from the VAE (not yet the counts)
    target,         # Target raw counts
    mu,             # Encoders latent mean vectors 
    logvar,         # Encoders latent log-variance vectors
    beta=1e-3,      # The KL term weight 
    loss_mask=None, # Optional mask, so we only compute the loss over certain genes 
    theta_param=None, # Negative bionomial dispersion parameter 
    zi_logits=None, # Zero-inflation probabilities for each gene (logits)
    eps=1e-8,       # help with stability in log/exp calculations
 ):
    # We use the raw counts here (this works better for ZINB)
    target_counts = target.clamp_min(0.0)
    mu_counts = F.softplus(recon) + eps

    # If not provided we say its 10 for everything (usual default in scVI)
    if theta_param is None:
        theta = torch.full((recon.shape[1],), 10.0, device=recon.device, dtype=recon.dtype)
    else:
        # If it is provided we make sure its positive and its a learned parameter
        theta = F.softplus(theta_param) + eps

    theta = theta.unsqueeze(0).expand_as(mu_counts)

    # We precompute this already since it is used multiple times
    log_theta_mu = torch.log(theta + mu_counts + eps)

    # Just the log probability, so if the model does well this is higher, if it does poor its lower. 
    nb_log_prob = (
        torch.lgamma(target_counts + theta)
        - torch.lgamma(theta)
        - torch.lgamma(target_counts + 1.0)
        + theta * (torch.log(theta + eps) - log_theta_mu)
        + target_counts * (torch.log(mu_counts + eps) - log_theta_mu)
    )

    # If its given we use ZINB instead of NB (which we do)
    if zi_logits is not None:
        pi = torch.sigmoid(zi_logits).unsqueeze(0).expand_as(mu_counts)
        zero_mask = target_counts < eps

        # NB probability mass at zero
        nb_log_prob_zero = theta * (torch.log(theta + eps) - log_theta_mu)

        # For zeros: log[ pi + (1-pi)*NB(0) ]
        zero_log_prob = torch.logaddexp(
            torch.log(pi + eps),
            torch.log1p(-pi + eps) + nb_log_prob_zero,
        )
        # For non-zeros: log[(1-pi)*NB(x)]
        nonzero_log_prob = torch.log1p(-pi + eps) + nb_log_prob

        recon_nll = -torch.where(zero_mask, zero_log_prob, nonzero_log_prob)
    else:
        recon_nll = -nb_log_prob

    # Compute the loss based on the genes we want from the mask (if we use it)
    if loss_mask is not None:
        recon_loss = recon_nll[:, loss_mask].mean()
    else:
        recon_loss = recon_nll.mean()

    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, recon_loss, kl

model = VAE(input_dim=n_genes, latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)

# Learn per-gene NB dispersion (theta) and per-gene zero-inflation logits (pi)
theta_unconstrained_init = float(np.log(np.expm1(theta_init)))
log_theta = nn.Parameter(
    torch.full((n_genes,), theta_unconstrained_init, device=device, dtype=torch.float32)
)

pi_logit_init = float(np.log(pi_init / (1.0 - pi_init)))
logit_pi = nn.Parameter(
    torch.full((n_genes,), pi_logit_init, device=device, dtype=torch.float32)
)

optimizer = torch.optim.Adam(list(model.parameters()) + [log_theta, logit_pi], lr=learning_rate)
scaler = torch.cuda.amp.GradScaler(enabled=use_cuda)

loss_mask_np = np.ones(n_genes, dtype=bool)
loss_mask_t = torch.from_numpy(loss_mask_np).to(device)

print(
    f"Training config | epochs={epochs}, beta={beta}, latent_dim={latent_dim}, "
    f"hidden_dim={hidden_dim}, lr={learning_rate}, recon_loss=ZINB(raw counts)"
 )

train_total_history = []
train_recon_history = []
train_kl_history = []
val_total_history = []
val_recon_history = []
val_kl_history = []

best_val_loss = np.inf
best_epoch = 0
best_state = None
epochs_without_improve = 0

overall_t0 = time.time()
epoch_pbar = tqdm(range(1, epochs + 1), desc="Epochs", unit="epoch")

for epoch in epoch_pbar:
    epoch_t0 = time.time()
    model.train()
    train_loss_sum = 0.0
    train_recon_sum = 0.0
    train_kl_sum = 0.0
    n_train_batches = 0

    train_iter = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{epochs} [train]", unit="batch", leave=False)
    for xb, yb in train_iter:
        xb = xb.to(device, non_blocking=use_cuda)
        yb = yb.to(device, non_blocking=use_cuda)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_cuda):
            recon, mu, logvar = model(xb)
            loss, recon_loss, kl = vae_loss(
                recon,
                yb,
                mu,
                logvar,
                beta=beta,
                loss_mask=loss_mask_t,
                theta_param=log_theta,
                zi_logits=logit_pi,
            )

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + [log_theta, logit_pi], max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        train_loss_sum += loss.item()
        train_recon_sum += recon_loss.item()
        train_kl_sum += kl.item()
        n_train_batches += 1

        if n_train_batches % 50 == 0:
            train_iter.set_postfix({
                "loss": f"{(train_loss_sum / n_train_batches):.4f}",
                "recon": f"{(train_recon_sum / n_train_batches):.4f}",
            })

    epoch_train_total = train_loss_sum / n_train_batches
    epoch_train_recon = train_recon_sum / n_train_batches
    epoch_train_kl = train_kl_sum / n_train_batches
    train_total_history.append(epoch_train_total)
    train_recon_history.append(epoch_train_recon)
    train_kl_history.append(epoch_train_kl)

    model.eval()
    val_loss_sum = 0.0
    val_recon_sum = 0.0
    val_kl_sum = 0.0
    n_val_batches = 0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device, non_blocking=use_cuda)
            yb = yb.to(device, non_blocking=use_cuda)
            with torch.cuda.amp.autocast(enabled=use_cuda):
                recon, mu, logvar = model(xb)
                val_loss, val_recon, val_kl = vae_loss(
                    recon,
                    yb,
                    mu,
                    logvar,
                    beta=beta,
                    loss_mask=loss_mask_t,
                    theta_param=log_theta,
                    zi_logits=logit_pi,
                )

            val_loss_sum += val_loss.item()
            val_recon_sum += val_recon.item()
            val_kl_sum += val_kl.item()
            n_val_batches += 1

    epoch_val_total = val_loss_sum / n_val_batches
    epoch_val_recon = val_recon_sum / n_val_batches
    epoch_val_kl = val_kl_sum / n_val_batches
    val_total_history.append(epoch_val_total)
    val_recon_history.append(epoch_val_recon)
    val_kl_history.append(epoch_val_kl)

    improved = epoch_val_total < (best_val_loss - 1e-8)
    if improved:
        best_val_loss = epoch_val_total
        best_epoch = epoch
        best_state = {
            "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "log_theta": log_theta.detach().cpu().clone(),
            "logit_pi": logit_pi.detach().cpu().clone(),
        }
        epochs_without_improve = 0
    else:
        epochs_without_improve += 1

    if epoch % 5 == 0 or epoch == 1 or improved:
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_total={epoch_train_total:.6f} | val_total={epoch_val_total:.6f} | "
            f"train_recon={epoch_train_recon:.6f} | val_recon={epoch_val_recon:.6f}"
        )

    epoch_sec = time.time() - epoch_t0
    elapsed_sec = time.time() - overall_t0
    epoch_pbar.set_postfix({
        "train": f"{epoch_train_total:.4f}",
        "val": f"{epoch_val_total:.4f}",
        "epoch_s": f"{epoch_sec:.1f}",
        "elapsed_m": f"{elapsed_sec/60:.1f}",
    })

    if epochs_without_improve >= early_stop_patience:
        print(f"Early stopping at epoch {epoch}; best validation loss at epoch {best_epoch}.")
        break

epoch_pbar.close()

if best_state is None:
    raise RuntimeError("No validation checkpoint was saved.")

model.load_state_dict(best_state["model"])
with torch.no_grad():
    log_theta.copy_(best_state["log_theta"].to(device))
    logit_pi.copy_(best_state["logit_pi"].to(device))

theta_learned = F.softplus(log_theta).detach().cpu().numpy()
pi_learned = torch.sigmoid(logit_pi).detach().cpu().numpy()
print(f"Loaded best checkpoint from epoch {best_epoch} (val_total={best_val_loss:.6f})")
print(f"Learned ZINB theta (median): {np.median(theta_learned):.4f}")
print(f"Learned ZINB pi (median): {np.median(pi_learned):.4f}")


# Training / validation loss curves (total + recon + KL)
epochs_ran = min(
    len(train_total_history), len(val_total_history),
    len(train_recon_history), len(val_recon_history),
    len(train_kl_history), len(val_kl_history),
)
epoch_axis = np.arange(1, epochs_ran + 1)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

axes[0].plot(epoch_axis, train_total_history[:epochs_ran], label="Train", linewidth=2)
axes[0].plot(epoch_axis, val_total_history[:epochs_ran], label="Validation", linewidth=2)
axes[0].set_title("Total loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].grid(alpha=0.25)
axes[0].legend()

axes[1].plot(epoch_axis, train_recon_history[:epochs_ran], label="Train", linewidth=2)
axes[1].plot(epoch_axis, val_recon_history[:epochs_ran], label="Validation", linewidth=2)
axes[1].set_title("Reconstruction loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].grid(alpha=0.25)
axes[1].legend()

axes[2].plot(epoch_axis, train_kl_history[:epochs_ran], label="Train", linewidth=2)
axes[2].plot(epoch_axis, val_kl_history[:epochs_ran], label="Validation", linewidth=2)
axes[2].set_title("KL loss")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Loss")
axes[2].grid(alpha=0.25)
axes[2].legend()

fig.suptitle("VAE training vs validation curves", y=1.02)
plt.tight_layout()

save_dir = SAVE_DIR
save_dir.mkdir(parents=True, exist_ok=True)
loss_curve_path = save_dir / "VAE_training_validation_curves.png"
fig.savefig(loss_curve_path, dpi=300, bbox_inches="tight")
print(f"Saved loss curves to: {loss_curve_path}")

plt.show()


# Save trained ZINB VAE weights
save_dir = SAVE_DIR
save_dir.mkdir(parents=True, exist_ok=True)
weights_path = save_dir / "VAE_ZINB_weights"

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "log_theta": log_theta.detach().cpu(),
        "logit_pi": logit_pi.detach().cpu(),
        "gene_names": gene_names.tolist(),
        "best_epoch": best_epoch if "best_epoch" in globals() else None,
        "best_val_loss": float(best_val_loss) if "best_val_loss" in globals() else None,
    },
    weights_path,
)
print(f"Saved ZINB VAE weights to: {weights_path}")