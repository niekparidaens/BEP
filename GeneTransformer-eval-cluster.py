import os
import time
from contextlib import nullcontext
from pathlib import Path

import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

matplotlib.use("Agg")


# Paths / I/O
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/tudelft.net/staff-umbrella/Xeniumenhancer")).resolve()
ANN_DIR = Path(os.environ.get("ANN_DIR", PROJECT_ROOT / "AnnData")).resolve()
SAVE_DIR = Path(os.environ.get("OUTPUT_DIR", PROJECT_ROOT / "outputs" / "gene_transformer" / "eval")).resolve()
SAVE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_WEIGHTS_PATH = PROJECT_ROOT / "outputs" / "gene_transformer" / "train" / "tokenized_ae_best.pt"
WEIGHTS_PATH = Path(os.environ.get("WEIGHTS_PATH", str(DEFAULT_WEIGHTS_PATH))).resolve()
READ_MODE = os.environ.get("READ_MODE", "r") or "r"

MODEL_D_MODEL = int(os.environ.get("MODEL_D_MODEL", 128))
MODEL_NHEAD = int(os.environ.get("MODEL_NHEAD", 4))
MODEL_NUM_LAYERS = int(os.environ.get("MODEL_NUM_LAYERS", 3))
MODEL_LATENT_DIM = int(os.environ.get("MODEL_LATENT_DIM", 32))
MODEL_DROPOUT = float(os.environ.get("MODEL_DROPOUT", 0.1))
MODEL_THETA_INIT = float(os.environ.get("MODEL_THETA_INIT", 10.0))

# -----------------------------------------------------------------------------
# Evaluation options
# -----------------------------------------------------------------------------
QUICK_MODE = False
QUICK_MAX_CELLS = None
QUICK_P_VALUES = [0.19, 0.25, 0.31]
FULL_P_VALUES = [0.19, 0.25, 0.31]

IO_CHUNK_SIZE = 2048
PAIR_INFER_BATCH_SIZE = 512
EXACT_CORRUPTION = True

RUN_GLOBAL_TEST_EVAL = True
RUN_PAIR_ANALYSIS = True
RUN_SINGLE_PANEL_GENE_DISTRIBUTIONS = True
RUN_SPLIT_PANEL_RECON_ANALYSIS = True

PAIR_5K_ID = "TENX189"
PAIR_V1_ID = "TENX190"

# Representative panels for histogram / spatial diagnostics.
SPLIT_PANEL_FOR_HIST = {
    "A": "NCBI882",
    "B": "TENX190",
    "C": "TENX116",
}
SPLIT_PANEL_FOR_RECON_ANALYSIS = {
    "A": "NCBI882",
    "B": "TENX190",
    "C": "TENX116",
}

SINGLE_PANEL_GENE_DIST_P = None
SINGLE_PANEL_TOP_HIGH = 4
SINGLE_PANEL_TOP_LOW = 4
RECON_PANEL_TOP_HIGH = 4
RECON_PANEL_TOP_LOW = 4
PAIR_TOP_HIGH_GENES = 4
PAIR_TOP_LOW_GENES = 4
ZINB_HIST_SAMPLE_SEED = 314159

threshold = 40
genes_threshold = 5
base_seed = 42

UNKNOWN_GENE_TOKEN = "__UNKNOWN_GENE__"
UNKNOWN_TISSUE_TOKEN = "__UNKNOWN_TISSUE__"


# Split definitions (same structure as transformer training)
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


def _safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _select_available_ids(sample_ids, available_like):
    available_set = set(available_like)
    normalized = [str(s).strip().upper() for s in sample_ids]
    found = [sid for sid in normalized if sid in available_set]
    missing = [sid for sid in normalized if sid not in available_set]
    return found, missing


def _to_dense_float32(X):
    if sp.issparse(X):
        return X.toarray().astype(np.float32, copy=False)
    return np.asarray(X, dtype=np.float32)


def autocast_context(use_cuda: bool):
    if use_cuda:
        return torch.amp.autocast(device_type="cuda", enabled=True)
    return nullcontext()


def _stable_string_seed(text: str) -> int:
    text = str(text)
    return int(sum((i + 1) * ord(ch) for i, ch in enumerate(text)))


def _clean_path_part(part):
    part = str(part).strip()
    if not part:
        return ""
    for old, new in [
        (" ", "_"),
        ("/", "_"),
        ("\\", "_"),
        ("->", "_to_"),
        (":", "_"),
        ("(", ""),
        (")", ""),
        (",", ""),
        ("=", "_"),
    ]:
        part = part.replace(old, new)
    while "__" in part:
        part = part.replace("__", "_")
    return part.strip("_")


def format_p_tag(p_val):
    return f"{float(p_val):.3f}".replace(".", "p")


def build_output_path(save_dir, category, name, ext, split=None, sample_id=None, p_val=None, extra_parts=None):
    parts = [category]
    if split is not None:
        parts.append(f"split_{split}")
    if sample_id is not None:
        parts.append(f"sample_{sample_id}")
    if p_val is not None:
        parts.append(f"p_{format_p_tag(p_val)}")
    if extra_parts:
        parts.extend(extra_parts)
    parts.append(name)

    stem = "__".join(filter(None, (_clean_path_part(p) for p in parts)))
    ext = ext.lstrip(".")
    return save_dir / f"{stem}.{ext}"


def flatten_simple_split_dict(split_dict, split_name, split_group):
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
            "sample_id": sample_id,
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
        n_genes = int(gene_pos_all.size)

        Y = np.empty((n_cells, n_genes), dtype=np.float32)
        write_pos = 0

        chunk_pbar = tqdm(
            range(0, n_cells, chunk_size),
            desc=f"Caching {sample_id}",
            unit="chunk",
            leave=False,
        )

        for start in chunk_pbar:
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


def _panel_global_bases(panel_data, panel_ids):
    bases = {}
    offset = 0
    for sid in panel_ids:
        if sid not in panel_data:
            continue
        bases[sid] = offset
        offset += int(panel_data[sid]["n_obs_filtered"])
    return bases


def _ensure_spatial_coords(adata_obj):
    if "spatial" in adata_obj.obsm:
        return

    candidate_pairs = [
        ("nucleus_centroid_x", "nucleus_centroid_y"),
        ("x_centroid", "y_centroid"),
        ("x", "y"),
        ("center_x", "center_y"),
        ("centroid_x", "centroid_y"),
        ("global_x", "global_y"),
    ]

    pair = next(
        ((x, y) for x, y in candidate_pairs if x in adata_obj.obs.columns and y in adata_obj.obs.columns),
        None,
    )
    if pair is None:
        raise KeyError(
            "No spatial coordinates found. Expected adata.obsm['spatial'] or one of: "
            f"{candidate_pairs}."
        )
    adata_obj.obsm["spatial"] = adata_obj.obs[[pair[0], pair[1]]].to_numpy()


def _find_spatial_xy(adata_obj):
    if "spatial" in adata_obj.obsm and adata_obj.obsm["spatial"].shape[1] >= 2:
        xy = np.asarray(adata_obj.obsm["spatial"])
        return xy[:, 0], xy[:, 1], "obsm['spatial']"

    candidate_pairs = [
        ("x_centroid", "y_centroid"),
        ("x", "y"),
        ("X", "Y"),
        ("center_x", "center_y"),
        ("global_x", "global_y"),
    ]
    for x_key, y_key in candidate_pairs:
        if x_key in adata_obj.obs.columns and y_key in adata_obj.obs.columns:
            return (
                adata_obj.obs[x_key].to_numpy(copy=True),
                adata_obj.obs[y_key].to_numpy(copy=True),
                f"obs[{x_key}, {y_key}]",
            )
    return None, None, None


def materialize_panel_native(panel_data, sample_id):
    rec = panel_data[sample_id]
    X = np.asarray(np.load(rec["cache_path"], allow_pickle=False), dtype=np.float32)

    ad_src = _load_sample_backed(sample_id)
    try:
        kept_rows = rec["cell_pos"]
        obs = ad_src.obs.iloc[kept_rows].copy()
        obs_names = pd.Index(ad_src.obs_names.astype(str))[kept_rows].copy()
        var = pd.DataFrame(index=pd.Index(rec["gene_names"]).astype(str))

        adata_out = ad.AnnData(X=X, obs=obs, var=var)
        adata_out.obs_names = obs_names

        for key in ad_src.obsm.keys():
            try:
                adata_out.obsm[key] = np.asarray(ad_src.obsm[key])[kept_rows].copy()
            except Exception:
                pass

        _ensure_spatial_coords(adata_out)
        return adata_out
    finally:
        _close_backed_adata(ad_src)


def compute_valid_overlap_gene_mask(gene_names, train_gene_name_set):
    gene_names = np.asarray(gene_names, dtype=object)
    return np.array([str(g) in train_gene_name_set for g in gene_names], dtype=bool)


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
class GeneTokenAutoencoder(nn.Module):
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
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.latent_proj = nn.Linear(d_model, latent_dim)
        self.z_proj = nn.Linear(latent_dim, d_model)

        self.decoder_trunk = nn.Sequential(
            nn.Linear(3 * d_model, 2 * d_model),
            nn.GELU(),
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

        h = g + x + t
        pad_mask = ~attn_mask
        h = self.encoder(h, src_key_padding_mask=pad_mask)

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
        recon_counts, z, _, _, _ = self.forward_with_params(gene_ids, x_vals, attn_mask, tissue_id)
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


# -----------------------------------------------------------------------------
# Corruption / inference helpers
# -----------------------------------------------------------------------------
def corrupt_batch_deterministic(x_clean, global_idx_np, version_idx, p_val, base_seed, exact=True):
    x_clean = np.asarray(x_clean, dtype=np.float32)
    global_idx_np = np.asarray(global_idx_np, dtype=np.int64)

    if exact:
        x_corrupt = x_clean.copy()
        for i, cell_idx in enumerate(global_idx_np):
            counts = np.rint(np.clip(x_corrupt[i], 0.0, None)).astype(np.int64, copy=False)
            rng = np.random.default_rng(int(base_seed + version_idx * 1_000_003 + int(cell_idx)))
            x_corrupt[i] = rng.binomial(counts, float(p_val)).astype(np.float32, copy=False)
        return x_corrupt

    counts = np.rint(np.clip(x_clean, 0.0, None)).astype(np.int64, copy=False)
    seed = int(base_seed + version_idx * 1_000_003 + int(global_idx_np[0]) + 13 * int(global_idx_np.size))
    rng = np.random.default_rng(seed)
    return rng.binomial(counts, float(p_val)).astype(np.float32, copy=False)


def sample_zinb_counts_np(mu_nb, theta, pi, rng, eps=1e-8):
    mu_nb = np.asarray(mu_nb, dtype=np.float32)
    theta = np.asarray(theta, dtype=np.float32)
    pi = np.asarray(pi, dtype=np.float32)

    mu_nb = np.clip(mu_nb, 0.0, None)
    theta = np.clip(theta, eps, None)
    pi = np.clip(pi, 0.0, 1.0)

    if theta.ndim == 1:
        gamma_shape = theta[None, :]
        gamma_scale = mu_nb / theta[None, :]
    else:
        gamma_shape = theta
        gamma_scale = mu_nb / theta

    lam = rng.gamma(shape=gamma_shape, scale=gamma_scale).astype(np.float32, copy=False)
    nb_sample = rng.poisson(lam).astype(np.float32, copy=False)

    zero_mask = rng.random(mu_nb.shape) < pi
    nb_sample[zero_mask] = 0.0
    return nb_sample


def run_token_model_batched(model, X_in, gene_ids_np, tissue_id_int, device, batch_size=512):
    model.eval()
    use_cuda = device.type == "cuda"

    expected_blocks = []
    mu_blocks = []
    pi_blocks = []
    theta_vec = None

    gene_ids_np = np.asarray(gene_ids_np, dtype=np.int64)
    L = gene_ids_np.shape[0]

    with torch.inference_mode():
        for start in tqdm(range(0, X_in.shape[0], batch_size), desc="Token model inference", unit="batch"):
            stop = min(start + batch_size, X_in.shape[0])
            batch_n = stop - start

            xb = torch.from_numpy(np.asarray(X_in[start:stop], dtype=np.float32)).to(device, non_blocking=use_cuda)
            gene_ids_b = torch.from_numpy(np.broadcast_to(gene_ids_np, (batch_n, L)).copy()).to(device, non_blocking=use_cuda)
            attn_mask_b = torch.ones((batch_n, L), dtype=torch.bool, device=device)
            tissue_id_b = torch.full((batch_n,), int(tissue_id_int), dtype=torch.long, device=device)

            with autocast_context(use_cuda):
                _, _, mu_logit, pi_logit, theta_unconstrained = model.forward_with_params(
                    gene_ids=gene_ids_b,
                    x_vals=xb,
                    attn_mask=attn_mask_b,
                    tissue_id=tissue_id_b,
                )

            mu_nb = F.softplus(mu_logit).float().cpu().numpy().astype(np.float32, copy=False)
            pi = torch.sigmoid(pi_logit).float().cpu().numpy().astype(np.float32, copy=False)
            if theta_vec is None:
                theta_vec = F.softplus(theta_unconstrained[0]).float().cpu().numpy().astype(np.float32, copy=False)

            expected_blocks.append((1.0 - pi) * mu_nb)
            mu_blocks.append(mu_nb)
            pi_blocks.append(pi)

    return np.vstack(expected_blocks), np.vstack(mu_blocks), np.vstack(pi_blocks), theta_vec


def expected_detection_rate_from_token_params(mu_nb, theta_vec, pi_mat, eps=1e-8):
    mu_nb = np.asarray(mu_nb, dtype=np.float32)
    theta_vec = np.asarray(theta_vec, dtype=np.float32)
    pi_mat = np.asarray(pi_mat, dtype=np.float32)

    nb_p0 = np.power(
        theta_vec[None, :] / (theta_vec[None, :] + np.clip(mu_nb, 0.0, None) + eps),
        theta_vec[None, :],
    )
    p0 = pi_mat + (1.0 - pi_mat) * nb_p0
    return (1.0 - p0).mean(axis=0)


# -----------------------------------------------------------------------------
# Metric helpers
# -----------------------------------------------------------------------------
def init_metric_acc():
    return {
        "sse_counts": 0.0,
        "sse_log1p": 0.0,
        "sum_true_counts": 0.0,
        "sumsq_true_counts": 0.0,
        "sum_true_log1p": 0.0,
        "sumsq_true_log1p": 0.0,
        "n_elements": 0,
        "n_cells": 0,
    }


def update_metric_acc(acc, pred, true, gene_mask=None):
    pred64 = np.asarray(pred, dtype=np.float64)
    true64 = np.asarray(true, dtype=np.float64)
    pred64, true64 = np.broadcast_arrays(pred64, true64)

    if gene_mask is not None:
        gene_mask = np.asarray(gene_mask, dtype=bool)
        if gene_mask.ndim != 1:
            raise ValueError("gene_mask must be a 1D boolean mask over genes.")
        pred64 = pred64[..., gene_mask]
        true64 = true64[..., gene_mask]

    if true64.size == 0:
        return

    pred64 = np.clip(pred64, 0.0, None)
    true64 = np.clip(true64, 0.0, None)

    pred_log = np.log1p(pred64)
    true_log = np.log1p(true64)

    acc["sse_counts"] += np.square(pred64 - true64).sum()
    acc["sse_log1p"] += np.square(pred_log - true_log).sum()
    acc["sum_true_counts"] += true64.sum()
    acc["sumsq_true_counts"] += np.square(true64).sum()
    acc["sum_true_log1p"] += true_log.sum()
    acc["sumsq_true_log1p"] += np.square(true_log).sum()
    acc["n_elements"] += true64.size
    acc["n_cells"] += true64.shape[0]


def finalize_metric_acc(acc):
    if acc["n_elements"] == 0:
        return {
            "mse_counts": np.nan,
            "rmse_counts": np.nan,
            "r2_counts": np.nan,
            "mse_log1p": np.nan,
            "rmse_log1p": np.nan,
            "r2_log1p": np.nan,
            "n_cells": 0,
        }

    n = float(acc["n_elements"])
    mse_counts = float(acc["sse_counts"] / n)
    mse_log1p = float(acc["sse_log1p"] / n)

    sst_counts = float(acc["sumsq_true_counts"] - (acc["sum_true_counts"] ** 2) / n)
    sst_log1p = float(acc["sumsq_true_log1p"] - (acc["sum_true_log1p"] ** 2) / n)

    r2_counts = np.nan if sst_counts <= 0 else float(1.0 - acc["sse_counts"] / sst_counts)
    r2_log1p = np.nan if sst_log1p <= 0 else float(1.0 - acc["sse_log1p"] / sst_log1p)

    return {
        "mse_counts": mse_counts,
        "rmse_counts": float(np.sqrt(mse_counts)),
        "r2_counts": r2_counts,
        "mse_log1p": mse_log1p,
        "rmse_log1p": float(np.sqrt(mse_log1p)),
        "r2_log1p": r2_log1p,
        "n_cells": int(acc["n_cells"]),
    }


def build_scoreboard_from_acc_dict(acc_dict):
    rows = []
    for method_name, acc in acc_dict.items():
        out = finalize_metric_acc(acc)
        rows.append(
            {
                "method": method_name,
                "rmse_counts": out["rmse_counts"],
                "r2_counts": out["r2_counts"],
                "rmse_log1p": out["rmse_log1p"],
                "r2_log1p": out["r2_log1p"],
                "mse_counts": out["mse_counts"],
                "mse_log1p": out["mse_log1p"],
                "n_cells": out["n_cells"],
            }
        )

    return pd.DataFrame(rows).sort_values(["rmse_counts", "rmse_log1p"], ascending=[True, True]).reset_index(drop=True)


def rmse_and_r2(pred, true):
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)

    rmse = float(np.sqrt(np.mean(np.square(pred - true))))
    ss_res = float(np.square(pred - true).sum())
    ss_tot = float(np.square(true - true.mean()).sum())
    r2 = np.nan if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)
    return rmse, r2


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------
def save_bar_plot(df, category_col, value_col, title, ylabel, save_path, rotate=True):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(df[category_col], df[value_col])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    if rotate:
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_exact_scatter_with_fit(
    x,
    y,
    title,
    x_label,
    y_label,
    color="#E68613",
    figsize=(7, 6),
    point_size=18,
    alpha=0.65,
    text_loc=(0.02, 0.98),
    line_pad=0.02,
    ax=None,
):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if x.size < 2:
        raise ValueError("Need at least 2 valid points to fit a line.")

    slope, intercept = np.polyfit(x, y, 1)
    pearson_r = float(np.corrcoef(x, y)[0, 1])
    line_max = float(max(x.max(), y.max(), 1e-8)) * (1.0 + line_pad)
    line_x = np.array([0.0, line_max], dtype=np.float64)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True

    ax.scatter(x, y, s=point_size, alpha=alpha, color=color)
    ax.plot(line_x, line_x, "k--", linewidth=1.2, alpha=0.8, label="y = x")
    ax.plot(line_x, slope * line_x + intercept, color=color, linewidth=1.6, alpha=0.95, label="fit")

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    ax.text(
        text_loc[0],
        text_loc[1],
        f"n={x.size}\nPearson r={pearson_r:.3f}\nfit: y={slope:.3f}x+{intercept:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    ax.set_xlim(-0.01 * line_max, line_max)
    ax.set_ylim(-0.01 * line_max, line_max)

    if created_fig:
        plt.tight_layout()
        plt.show()

    return {
        "n_points": int(x.size),
        "pearson_r": pearson_r,
        "slope": float(slope),
        "intercept": float(intercept),
    }


def save_reconstruction_scatter_suite(
    mean_input,
    mean_recon,
    mean_target,
    det_input,
    det_recon,
    det_target,
    title_prefix,
    input_label,
    recon_label,
    target_label,
    save_path_linear,
    save_path_log,
):
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    plot_exact_scatter_with_fit(
        x=mean_target,
        y=mean_input,
        title=f"Gene mean: {input_label} vs {target_label}",
        x_label=f"{target_label} gene mean",
        y_label=f"{input_label} gene mean",
        color="#1f77b4",
        ax=axes[0, 0],
    )
    plot_exact_scatter_with_fit(
        x=mean_target,
        y=mean_recon,
        title=f"Gene mean: {recon_label} vs {target_label}",
        x_label=f"{target_label} gene mean",
        y_label=f"{recon_label} gene mean",
        color="#E68613",
        ax=axes[0, 1],
    )
    plot_exact_scatter_with_fit(
        x=det_target,
        y=det_input,
        title=f"Detection: {input_label} vs {target_label}",
        x_label=f"{target_label} detection rate",
        y_label=f"{input_label} detection rate",
        color="#1f77b4",
        ax=axes[1, 0],
    )
    plot_exact_scatter_with_fit(
        x=det_target,
        y=det_recon,
        title=f"Detection: {recon_label} vs {target_label}",
        x_label=f"{target_label} detection rate",
        y_label=f"{recon_label} detection rate",
        color="#E68613",
        ax=axes[1, 1],
    )

    fig.suptitle(title_prefix, y=1.002)
    plt.tight_layout()
    fig.savefig(save_path_linear, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))
    plot_exact_scatter_with_fit(
        x=np.log1p(mean_target),
        y=np.log1p(mean_input),
        title=f"log gene mean: {input_label} vs {target_label}",
        x_label=f"log1p({target_label} gene mean)",
        y_label=f"log1p({input_label} gene mean)",
        color="#1f77b4",
        ax=axes[0],
    )
    plot_exact_scatter_with_fit(
        x=np.log1p(mean_target),
        y=np.log1p(mean_recon),
        title=f"log gene mean: {recon_label} vs {target_label}",
        x_label=f"log1p({target_label} gene mean)",
        y_label=f"log1p({recon_label} gene mean)",
        color="#E68613",
        ax=axes[1],
    )

    fig.suptitle(f"{title_prefix} | log-scale gene means", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path_log, dpi=300, bbox_inches="tight")
    plt.close(fig)


def select_high_low_genes(gene_names, mean_target, n_high=4, n_low=4):
    gene_names = np.asarray(gene_names, dtype=object)
    mean_target = np.asarray(mean_target, dtype=np.float64)

    high_order = np.argsort(mean_target)[::-1]
    high_genes = [gene_names[i] for i in high_order[: min(n_high, len(high_order))]]

    positive_idx = np.flatnonzero(mean_target > 0)
    if positive_idx.size > 0:
        low_order = positive_idx[np.argsort(mean_target[positive_idx])]
        low_genes = [gene_names[i] for i in low_order[: min(n_low, len(low_order))]]
    else:
        low_genes = []

    selected = []
    for g in high_genes + low_genes:
        if g not in selected:
            selected.append(g)
    return selected


def plot_selected_gene_histograms(
    selected_genes,
    gene_to_idx,
    X_input,
    X_recon,
    X_target,
    save_path,
    label_input="Input",
    label_recon="Reconstruction",
    label_target="Target",
):
    n_rows = len(selected_genes)
    if n_rows == 0:
        return

    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 3.2 * n_rows), squeeze=False)

    for i, gene in enumerate(selected_genes):
        g_idx = gene_to_idx[gene]

        x_in = np.clip(X_input[:, g_idx], 0.0, None)
        x_rec = np.clip(X_recon[:, g_idx], 0.0, None)
        x_tar = np.clip(X_target[:, g_idx], 0.0, None)

        ax_left = axes[i, 0]
        ax_left.hist(np.log1p(x_in), bins=30, density=True, alpha=0.40, label=f"{label_input} (all)")
        ax_left.hist(np.log1p(x_rec), bins=30, density=True, alpha=0.40, label=f"{label_recon} (all)")
        ax_left.hist(np.log1p(x_tar), bins=30, density=True, alpha=0.40, label=f"{label_target} (all)")
        ax_left.set_title(f"{gene} | all cells")
        ax_left.set_xlabel("log1p(count)")
        ax_left.set_ylabel("Density")
        ax_left.legend(frameon=True)

        ax_right = axes[i, 1]
        pos_in = x_in[x_in > 0]
        pos_rec = x_rec[x_rec > 0]
        pos_tar = x_tar[x_tar > 0]

        if pos_in.size > 0:
            ax_right.hist(np.log1p(pos_in), bins=30, density=True, alpha=0.40, label=f"{label_input} (>0)")
        if pos_rec.size > 0:
            ax_right.hist(np.log1p(pos_rec), bins=30, density=True, alpha=0.40, label=f"{label_recon} (>0)")
        if pos_tar.size > 0:
            ax_right.hist(np.log1p(pos_tar), bins=30, density=True, alpha=0.40, label=f"{label_target} (>0)")

        ax_right.set_title(f"{gene} | positive cells only")
        ax_right.set_xlabel("log1p(count)")
        ax_right.set_ylabel("Density")
        ax_right.legend(frameon=True)

    fig.suptitle("Selected gene distributions", y=1.002)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_selected_gene_spatial_triplets(
    selected_genes,
    gene_to_idx,
    adata_input,
    X_recon,
    adata_target,
    save_path,
    label_input="Input",
    label_recon="Reconstruction",
    label_target="Target",
    X_input=None,
    X_target=None,
):
    x_in, y_in, _ = _find_spatial_xy(adata_input)
    x_tar, y_tar, _ = _find_spatial_xy(adata_target)

    if x_in is None or x_tar is None:
        print("Skipping spatial gene plots: no spatial coordinates found.")
        return

    n_rows = len(selected_genes)
    if n_rows == 0:
        return

    if X_input is not None:
        X_input = np.asarray(X_input, dtype=np.float32)
    X_recon = np.asarray(X_recon, dtype=np.float32)
    if X_target is not None:
        X_target = np.asarray(X_target, dtype=np.float32)

    fig = plt.figure(figsize=(16, 4.0 * n_rows))
    gs = fig.add_gridspec(n_rows, 4, width_ratios=[1, 1, 1, 0.06], wspace=0.18, hspace=0.30)

    for i, gene in enumerate(selected_genes):
        g_idx = gene_to_idx[gene]

        if X_input is None:
            expr_input = np.log1p(np.clip(_to_dense_float32(adata_input.X[:, g_idx]).ravel(), 0.0, None))
        else:
            expr_input = np.log1p(np.clip(X_input[:, g_idx], 0.0, None))

        expr_recon = np.log1p(np.clip(X_recon[:, g_idx], 0.0, None))

        if X_target is None:
            expr_target = np.log1p(np.clip(_to_dense_float32(adata_target.X[:, g_idx]).ravel(), 0.0, None))
        else:
            expr_target = np.log1p(np.clip(X_target[:, g_idx], 0.0, None))

        vmax = float(
            max(
                np.percentile(expr_input, 99) if expr_input.size else 0.0,
                np.percentile(expr_recon, 99) if expr_recon.size else 0.0,
                np.percentile(expr_target, 99) if expr_target.size else 0.0,
                1e-6,
            )
        )

        ax0 = fig.add_subplot(gs[i, 0])
        ax1 = fig.add_subplot(gs[i, 1])
        ax2 = fig.add_subplot(gs[i, 2])
        cax = fig.add_subplot(gs[i, 3])

        ax0.scatter(x_in, y_in, c=expr_input, s=4, cmap="viridis", vmin=0.0, vmax=vmax)
        ax0.set_title(f"{gene} | {label_input}")
        ax0.set_xlabel("x")
        ax0.set_ylabel("y")
        ax0.invert_yaxis()
        ax0.set_aspect("equal", adjustable="box")

        ax1.scatter(x_in, y_in, c=expr_recon, s=4, cmap="viridis", vmin=0.0, vmax=vmax)
        ax1.set_title(f"{gene} | {label_recon}")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.invert_yaxis()
        ax1.set_aspect("equal", adjustable="box")

        sc2 = ax2.scatter(x_tar, y_tar, c=expr_target, s=4, cmap="viridis", vmin=0.0, vmax=vmax)
        ax2.set_title(f"{gene} | {label_target}")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.invert_yaxis()
        ax2.set_aspect("equal", adjustable="box")

        cbar = fig.colorbar(sc2, cax=cax)
        cbar.set_label("log1p(count)")

    fig.suptitle("Selected spatial gene patterns", y=0.995)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Train baseline in token vocabulary
# -----------------------------------------------------------------------------
def compute_train_gene_mean_by_vocab(panel_data, train_sample_ids, gene2id):
    vocab_size = int(len(gene2id))
    gene_sum = np.zeros(vocab_size, dtype=np.float64)
    total_cells = 0

    unknown_gene_id = int(gene2id[UNKNOWN_GENE_TOKEN])

    for sid in tqdm(train_sample_ids, desc="Computing train gene mean baseline", unit="panel"):
        rec = panel_data[sid]
        X = np.asarray(np.load(rec["cache_path"], mmap_mode="r", allow_pickle=False), dtype=np.float32)
        total_cells += int(X.shape[0])

        gene_ids = np.array([gene2id.get(str(g), unknown_gene_id) for g in rec["gene_names"]], dtype=np.int64)
        gene_sum_by_local = X.astype(np.float64, copy=False).sum(axis=0)
        np.add.at(gene_sum, gene_ids, gene_sum_by_local)

    if total_cells == 0:
        raise RuntimeError("No training cells found while computing train gene mean baseline.")

    return (gene_sum / float(total_cells)).astype(np.float32)


# -----------------------------------------------------------------------------
# Global evaluation
# -----------------------------------------------------------------------------
def run_global_test_evaluation(
    panel_data,
    train_sample_ids,
    test_sample_ids,
    test_a_sample_ids,
    test_b_sample_ids,
    test_c_sample_ids,
    panel_to_tissue,
    gene2id,
    tissue2id,
    train_gene_name_set,
    model,
    device,
    save_dir,
):
    eval_p_values = np.asarray(QUICK_P_VALUES if QUICK_MODE else FULL_P_VALUES, dtype=np.float64)
    if np.any((eval_p_values < 0.0) | (eval_p_values > 1.0)):
        raise ValueError("All evaluation p values must be between 0 and 1.")

    panel_bases = _panel_global_bases(panel_data, test_sample_ids)
    total_test_cells = int(sum(panel_data[sid]["n_obs_filtered"] for sid in test_sample_ids))

    selected_global_idx = None
    if QUICK_MODE and QUICK_MAX_CELLS is not None and QUICK_MAX_CELLS < total_test_cells:
        rng_sub = np.random.default_rng(base_seed + 2026)
        selected_global_idx = np.sort(rng_sub.choice(total_test_cells, size=int(QUICK_MAX_CELLS), replace=False))

    split_name_by_sid = {sid: "A" for sid in test_a_sample_ids}
    split_name_by_sid.update({sid: "B" for sid in test_b_sample_ids})
    split_name_by_sid.update({sid: "C" for sid in test_c_sample_ids})

    train_gene_mean_by_vocab = compute_train_gene_mean_by_vocab(panel_data, train_sample_ids, gene2id)
    print("Computed train gene-mean baseline in transformer vocabulary.")

    METHOD_MODEL = "Transformer (ZINB expected decode)"
    METHOD_ID = "Identity baseline (input passthrough)"
    METHOD_MEAN = "Gene-mean baseline (train)"
    METHODS = [METHOD_MODEL, METHOD_ID, METHOD_MEAN]

    overall_acc = {m: init_metric_acc() for m in METHODS}
    per_p_acc = {(int(v), m): init_metric_acc() for v in range(len(eval_p_values)) for m in METHODS}
    per_split_acc = {
        (int(v), split_name, m): init_metric_acc()
        for v in range(len(eval_p_values))
        for split_name in ["A", "B", "C"]
        for m in METHODS
    }
    per_panel_acc = {
        (int(v), split_name_by_sid.get(sid), sid, m): init_metric_acc()
        for v in range(len(eval_p_values))
        for sid in test_sample_ids
        for m in METHODS
    }
    per_panel_valid_overlap_acc = {
        (int(v), split_name_by_sid.get(sid), sid, m): init_metric_acc()
        for v in range(len(eval_p_values))
        for sid in test_sample_ids
        if split_name_by_sid.get(sid) in {"B", "C"}
        for m in METHODS
    }

    loss_sum_weighted = 0.0
    loss_weight = 0

    unknown_gene_id = int(gene2id[UNKNOWN_GENE_TOKEN])
    unknown_tissue_id = int(tissue2id[UNKNOWN_TISSUE_TOKEN])
    use_cuda = device.type == "cuda"

    print(f"Eval mode: {'QUICK' if QUICK_MODE else 'FULL'}")
    print(f"Cells used per p: {total_test_cells if selected_global_idx is None else int(selected_global_idx.size)} / {total_test_cells}")
    print(f"p values used: {eval_p_values.tolist()}")
    print(f"Exact corruption seeding: {EXACT_CORRUPTION}")

    for version_idx, p_val in enumerate(eval_p_values):
        processed_cells = 0

        for sample_id in tqdm(test_sample_ids, desc=f"Evaluating p={float(p_val):.3f}", unit="panel"):
            rec = panel_data[sample_id]
            X_clean = np.asarray(np.load(rec["cache_path"], mmap_mode="r", allow_pickle=False), dtype=np.float32)
            n_cells = int(X_clean.shape[0])
            if n_cells == 0:
                continue

            panel_base = panel_bases[sample_id]
            local_idx = np.arange(n_cells, dtype=np.int64)
            global_idx = panel_base + local_idx

            if selected_global_idx is not None:
                lo = np.searchsorted(selected_global_idx, panel_base, side="left")
                hi = np.searchsorted(selected_global_idx, panel_base + n_cells, side="left")
                chosen_local = selected_global_idx[lo:hi] - panel_base
                if chosen_local.size == 0:
                    continue
            else:
                chosen_local = local_idx

            gene_names = np.asarray(rec["gene_names"], dtype=object)
            gene_ids = np.array([gene2id.get(str(g), unknown_gene_id) for g in gene_names], dtype=np.int64)
            valid_overlap_mask = compute_valid_overlap_gene_mask(gene_names, train_gene_name_set)
            tissue_name = panel_to_tissue.get(sample_id, UNKNOWN_TISSUE_TOKEN)
            tissue_id_int = int(tissue2id.get(tissue_name, unknown_tissue_id))
            split_name = split_name_by_sid.get(sample_id)

            mean_baseline_vec = train_gene_mean_by_vocab[gene_ids][None, :]

            for start in range(0, chosen_local.size, IO_CHUNK_SIZE):
                stop = min(start + IO_CHUNK_SIZE, chosen_local.size)
                batch_local = chosen_local[start:stop]
                yb_np = np.asarray(X_clean[batch_local], dtype=np.float32)
                batch_global_idx = global_idx[batch_local]

                xb_np = corrupt_batch_deterministic(
                    x_clean=yb_np,
                    global_idx_np=batch_global_idx,
                    version_idx=version_idx,
                    p_val=float(p_val),
                    base_seed=base_seed,
                    exact=EXACT_CORRUPTION,
                )

                batch_n, L = xb_np.shape
                xb = torch.from_numpy(xb_np).to(device, non_blocking=use_cuda)
                yb = torch.from_numpy(yb_np).to(device, non_blocking=use_cuda)
                gene_ids_b = torch.from_numpy(np.broadcast_to(gene_ids, (batch_n, L)).copy()).to(device, non_blocking=use_cuda)
                attn_mask_b = torch.ones((batch_n, L), dtype=torch.bool, device=device)
                tissue_id_b = torch.full((batch_n,), tissue_id_int, dtype=torch.long, device=device)

                with torch.inference_mode():
                    with autocast_context(use_cuda):
                        recon_expected_t, _, mu_logit, pi_logit, theta_unconstrained = model.forward_with_params(
                            gene_ids=gene_ids_b,
                            x_vals=xb,
                            attn_mask=attn_mask_b,
                            tissue_id=tissue_id_b,
                        )
                        t_loss = token_zinb_loss(
                            mu_logit=mu_logit,
                            pi_logit=pi_logit,
                            theta_unconstrained=theta_unconstrained,
                            target_counts=yb,
                            attn_mask=attn_mask_b,
                        )

                recon_expected = recon_expected_t.float().cpu().numpy().astype(np.float32, copy=False)

                processed_cells += batch_n
                loss_sum_weighted += float(t_loss.item()) * float(batch_n * L)
                loss_weight += int(batch_n * L)

                update_metric_acc(overall_acc[METHOD_MODEL], recon_expected, yb_np)
                update_metric_acc(overall_acc[METHOD_ID], xb_np, yb_np)
                update_metric_acc(overall_acc[METHOD_MEAN], mean_baseline_vec, yb_np)

                update_metric_acc(per_p_acc[(version_idx, METHOD_MODEL)], recon_expected, yb_np)
                update_metric_acc(per_p_acc[(version_idx, METHOD_ID)], xb_np, yb_np)
                update_metric_acc(per_p_acc[(version_idx, METHOD_MEAN)], mean_baseline_vec, yb_np)

                if split_name in {"A", "B", "C"}:
                    update_metric_acc(per_split_acc[(version_idx, split_name, METHOD_MODEL)], recon_expected, yb_np)
                    update_metric_acc(per_split_acc[(version_idx, split_name, METHOD_ID)], xb_np, yb_np)
                    update_metric_acc(per_split_acc[(version_idx, split_name, METHOD_MEAN)], mean_baseline_vec, yb_np)

                    update_metric_acc(per_panel_acc[(version_idx, split_name, sample_id, METHOD_MODEL)], recon_expected, yb_np)
                    update_metric_acc(per_panel_acc[(version_idx, split_name, sample_id, METHOD_ID)], xb_np, yb_np)
                    update_metric_acc(per_panel_acc[(version_idx, split_name, sample_id, METHOD_MEAN)], mean_baseline_vec, yb_np)

                    if split_name in {"B", "C"} and np.any(valid_overlap_mask):
                        update_metric_acc(
                            per_panel_valid_overlap_acc[(version_idx, split_name, sample_id, METHOD_MODEL)],
                            recon_expected,
                            yb_np,
                            gene_mask=valid_overlap_mask,
                        )
                        update_metric_acc(
                            per_panel_valid_overlap_acc[(version_idx, split_name, sample_id, METHOD_ID)],
                            xb_np,
                            yb_np,
                            gene_mask=valid_overlap_mask,
                        )
                        update_metric_acc(
                            per_panel_valid_overlap_acc[(version_idx, split_name, sample_id, METHOD_MEAN)],
                            mean_baseline_vec,
                            yb_np,
                            gene_mask=valid_overlap_mask,
                        )

        print(f"Finished p={float(p_val):.3f} | processed_cells={processed_cells}")

    if loss_weight == 0:
        raise RuntimeError("No evaluation batches were produced.")

    overall_model = finalize_metric_acc(overall_acc[METHOD_MODEL])
    summary_table = pd.DataFrame(
        [
            {"metric": "test_zinb_nll", "value": loss_sum_weighted / float(loss_weight)},
            {"metric": "test_RMSE_counts", "value": overall_model["rmse_counts"]},
            {"metric": "test_R2_counts", "value": overall_model["r2_counts"]},
            {"metric": "test_RMSE_log1p", "value": overall_model["rmse_log1p"]},
            {"metric": "test_R2_log1p", "value": overall_model["r2_log1p"]},
        ]
    )
    print("\nGlobal test summary:")
    print(summary_table.to_string(index=False))
    summary_table.to_csv(build_output_path(save_dir, "global", "summary", "csv"), index=False)

    scoreboard_overall = build_scoreboard_from_acc_dict(overall_acc)
    print("\nOverall scoreboard:")
    print(scoreboard_overall.to_string(index=False))
    scoreboard_overall.to_csv(build_output_path(save_dir, "global", "scoreboard_overall", "csv"), index=False)

    save_bar_plot(
        scoreboard_overall,
        category_col="method",
        value_col="rmse_counts",
        title="Overall test-set comparison",
        ylabel="RMSE on counts (lower is better)",
        save_path=build_output_path(save_dir, "global", "scoreboard_overall_rmse_counts", "png"),
    )
    save_bar_plot(
        scoreboard_overall,
        category_col="method",
        value_col="r2_counts",
        title="Overall test-set comparison",
        ylabel="R² on counts (higher is better)",
        save_path=build_output_path(save_dir, "global", "scoreboard_overall_r2_counts", "png"),
    )

    per_p_rows = []
    per_p_scoreboards = []
    for version_idx, p_val in enumerate(eval_p_values):
        sb = build_scoreboard_from_acc_dict(
            {
                METHOD_MODEL: per_p_acc[(version_idx, METHOD_MODEL)],
                METHOD_ID: per_p_acc[(version_idx, METHOD_ID)],
                METHOD_MEAN: per_p_acc[(version_idx, METHOD_MEAN)],
            }
        ).copy()
        sb.insert(0, "p_non_overlap", float(p_val))
        per_p_scoreboards.append(sb)

        model_row = finalize_metric_acc(per_p_acc[(version_idx, METHOD_MODEL)])
        per_p_rows.append(
            {
                "p_non_overlap": float(p_val),
                "n_cells": model_row["n_cells"],
                "rmse_counts_model": model_row["rmse_counts"],
                "r2_counts_model": model_row["r2_counts"],
                "rmse_log1p_model": model_row["rmse_log1p"],
                "r2_log1p_model": model_row["r2_log1p"],
            }
        )

    per_p_summary = pd.DataFrame(per_p_rows)
    scoreboard_by_p = pd.concat(per_p_scoreboards, ignore_index=True)
    print("\nPer-p summary:")
    print(per_p_summary.to_string(index=False))
    per_p_summary.to_csv(build_output_path(save_dir, "global", "per_p_summary", "csv"), index=False)

    print("\nPer-p scoreboard:")
    print(scoreboard_by_p.to_string(index=False))
    scoreboard_by_p.to_csv(build_output_path(save_dir, "global", "scoreboard_by_p", "csv"), index=False)

    split_rows = []
    for version_idx, p_val in enumerate(eval_p_values):
        for split_name in ["A", "B", "C"]:
            for method_name in METHODS:
                out = finalize_metric_acc(per_split_acc[(version_idx, split_name, method_name)])
                if out["n_cells"] == 0:
                    continue
                split_rows.append(
                    {
                        "p_non_overlap": float(p_val),
                        "split": split_name,
                        "method": method_name,
                        "n_cells": out["n_cells"],
                        "rmse_counts": out["rmse_counts"],
                        "r2_counts": out["r2_counts"],
                        "rmse_log1p": out["rmse_log1p"],
                        "r2_log1p": out["r2_log1p"],
                        "mse_counts": out["mse_counts"],
                        "mse_log1p": out["mse_log1p"],
                    }
                )

    scoreboard_by_split = pd.DataFrame(split_rows).sort_values(
        ["p_non_overlap", "split", "rmse_counts"], ascending=[True, True, True]
    ).reset_index(drop=True)
    print("\nPer-split scoreboard:")
    print(scoreboard_by_split.to_string(index=False))
    scoreboard_by_split.to_csv(build_output_path(save_dir, "split", "scoreboard_by_split", "csv"), index=False)

    for split_name in ["A", "B", "C"]:
        df_split = scoreboard_by_split[scoreboard_by_split["split"] == split_name].copy()
        if df_split.empty:
            continue
        best_p = float(df_split["p_non_overlap"].iloc[0])
        df_best = df_split[df_split["p_non_overlap"] == best_p].copy()

        save_bar_plot(
            df_best,
            category_col="method",
            value_col="rmse_counts",
            title=f"Split {split_name} comparison (p={best_p:.2f})",
            ylabel="RMSE on counts",
            save_path=build_output_path(save_dir, "split", "method_comparison_rmse_counts", "png", split=split_name, p_val=best_p),
        )
        save_bar_plot(
            df_best,
            category_col="method",
            value_col="r2_counts",
            title=f"Split {split_name} comparison (p={best_p:.2f})",
            ylabel="R² on counts",
            save_path=build_output_path(save_dir, "split", "method_comparison_r2_counts", "png", split=split_name, p_val=best_p),
        )

    panel_rows = []
    for version_idx, p_val in enumerate(eval_p_values):
        for split_name in ["A", "B", "C"]:
            split_panel_ids = [sid for sid in test_sample_ids if split_name_by_sid.get(sid) == split_name]
            for sample_id in split_panel_ids:
                for method_name in METHODS:
                    out = finalize_metric_acc(per_panel_acc[(version_idx, split_name, sample_id, method_name)])
                    if out["n_cells"] == 0:
                        continue
                    panel_rows.append(
                        {
                            "p_non_overlap": float(p_val),
                            "split": split_name,
                            "sample_id": sample_id,
                            "method": method_name,
                            "n_cells": out["n_cells"],
                            "rmse_counts": out["rmse_counts"],
                            "r2_counts": out["r2_counts"],
                            "rmse_log1p": out["rmse_log1p"],
                            "r2_log1p": out["r2_log1p"],
                            "mse_counts": out["mse_counts"],
                            "mse_log1p": out["mse_log1p"],
                        }
                    )

    scoreboard_by_panel = pd.DataFrame(panel_rows).sort_values(
        ["p_non_overlap", "split", "sample_id", "rmse_counts"], ascending=[True, True, True, True]
    ).reset_index(drop=True)
    print("\nPer-panel scoreboard:")
    print(scoreboard_by_panel.to_string(index=False))
    scoreboard_by_panel.to_csv(build_output_path(save_dir, "panel", "scoreboard_by_panel", "csv"), index=False)

    panel_summary_model = scoreboard_by_panel[scoreboard_by_panel["method"] == METHOD_MODEL].copy()
    panel_summary_model.to_csv(build_output_path(save_dir, "panel", "summary_model", "csv"), index=False)

    for split_name in ["A", "B", "C"]:
        df_panel = panel_summary_model[panel_summary_model["split"] == split_name].copy()
        if df_panel.empty:
            continue
        best_p = float(df_panel["p_non_overlap"].iloc[0])
        df_best = df_panel[df_panel["p_non_overlap"] == best_p].sort_values("rmse_counts").copy()

        save_bar_plot(
            df_best,
            category_col="sample_id",
            value_col="rmse_counts",
            title=f"Split {split_name} | Transformer per-panel RMSE (p={best_p:.2f})",
            ylabel="RMSE on counts",
            save_path=build_output_path(save_dir, "panel", "transformer_per_panel_rmse_counts", "png", split=split_name, p_val=best_p),
        )
        save_bar_plot(
            df_best,
            category_col="sample_id",
            value_col="r2_counts",
            title=f"Split {split_name} | Transformer per-panel R² (p={best_p:.2f})",
            ylabel="R² on counts",
            save_path=build_output_path(save_dir, "panel", "transformer_per_panel_r2_counts", "png", split=split_name, p_val=best_p),
        )

    panel_valid_overlap_rows = []
    for version_idx, p_val in enumerate(eval_p_values):
        for split_name in ["B", "C"]:
            split_panel_ids = [sid for sid in test_sample_ids if split_name_by_sid.get(sid) == split_name]
            for sample_id in split_panel_ids:
                n_valid_genes = int(compute_valid_overlap_gene_mask(panel_data[sample_id]["gene_names"], train_gene_name_set).sum())
                for method_name in METHODS:
                    out = finalize_metric_acc(per_panel_valid_overlap_acc[(version_idx, split_name, sample_id, method_name)])
                    if out["n_cells"] == 0:
                        continue
                    panel_valid_overlap_rows.append(
                        {
                            "p_non_overlap": float(p_val),
                            "split": split_name,
                            "sample_id": sample_id,
                            "method": method_name,
                            "n_valid_overlap_genes": n_valid_genes,
                            "n_cells": out["n_cells"],
                            "rmse_counts": out["rmse_counts"],
                            "r2_counts": out["r2_counts"],
                            "rmse_log1p": out["rmse_log1p"],
                            "r2_log1p": out["r2_log1p"],
                            "mse_counts": out["mse_counts"],
                            "mse_log1p": out["mse_log1p"],
                        }
                    )

    if panel_valid_overlap_rows:
        scoreboard_by_panel_valid_overlap = pd.DataFrame(panel_valid_overlap_rows).sort_values(
            ["p_non_overlap", "split", "sample_id", "rmse_counts"], ascending=[True, True, True, True]
        ).reset_index(drop=True)
        print("\nPer-panel scoreboard on valid train-overlap genes (B/C):")
        print(scoreboard_by_panel_valid_overlap.to_string(index=False))
        scoreboard_by_panel_valid_overlap.to_csv(
            build_output_path(save_dir, "panel", "scoreboard_by_panel_valid_train_overlap", "csv"),
            index=False,
        )

        panel_valid_overlap_model = scoreboard_by_panel_valid_overlap[
            scoreboard_by_panel_valid_overlap["method"] == METHOD_MODEL
        ].copy()
        panel_valid_overlap_model.to_csv(
            build_output_path(save_dir, "panel", "summary_model_valid_train_overlap", "csv"),
            index=False,
        )

        for split_name in ["B", "C"]:
            df_panel = panel_valid_overlap_model[panel_valid_overlap_model["split"] == split_name].copy()
            if df_panel.empty:
                continue

            best_p = float(df_panel["p_non_overlap"].iloc[0])
            df_best = df_panel[df_panel["p_non_overlap"] == best_p].sort_values("rmse_counts").copy()

            save_bar_plot(
                df_best,
                category_col="sample_id",
                value_col="rmse_counts",
                title=f"Split {split_name} | Transformer RMSE on valid train-overlap genes (p={best_p:.2f})",
                ylabel="RMSE on counts",
                save_path=build_output_path(
                    save_dir,
                    "panel",
                    "transformer_per_panel_valid_train_overlap_rmse_counts",
                    "png",
                    split=split_name,
                    p_val=best_p,
                ),
            )
            save_bar_plot(
                df_best,
                category_col="sample_id",
                value_col="r2_counts",
                title=f"Split {split_name} | Transformer R² on valid train-overlap genes (p={best_p:.2f})",
                ylabel="R² on counts",
                save_path=build_output_path(
                    save_dir,
                    "panel",
                    "transformer_per_panel_valid_train_overlap_r2_counts",
                    "png",
                    split=split_name,
                    p_val=best_p,
                ),
            )


# -----------------------------------------------------------------------------
# Diagnostic analyses
# -----------------------------------------------------------------------------
def run_single_panel_gene_distribution_analysis(
    panel_data,
    test_sample_ids,
    sample_id,
    split_name,
    panel_to_tissue,
    gene2id,
    tissue2id,
    model,
    device,
    save_dir,
    version_idx,
    p_val,
):
    if sample_id not in panel_data:
        print(f"Skipping split {split_name}: sample {sample_id} not present in panel_data.")
        return

    print(f"\nRunning single-panel gene distribution analysis for split {split_name}, sample {sample_id}, p={p_val:.3f}")

    rec = panel_data[sample_id]
    X_clean = np.asarray(np.load(rec["cache_path"], allow_pickle=False), dtype=np.float32)
    if X_clean.shape[0] == 0:
        print(f"Skipping split {split_name}: sample {sample_id} has no cells.")
        return

    panel_bases = _panel_global_bases(panel_data, test_sample_ids)
    panel_base = panel_bases[sample_id]
    global_idx_np = panel_base + np.arange(X_clean.shape[0], dtype=np.int64)

    gene_names = np.asarray(rec["gene_names"], dtype=object)
    mean_target = X_clean.mean(axis=0)
    selected_genes = select_high_low_genes(
        gene_names=gene_names,
        mean_target=mean_target,
        n_high=SINGLE_PANEL_TOP_HIGH,
        n_low=SINGLE_PANEL_TOP_LOW,
    )
    if len(selected_genes) == 0:
        print(f"Skipping split {split_name}: no genes selected for {sample_id}.")
        return

    selected_idx = pd.Index(gene_names).get_indexer(selected_genes)
    gene_to_idx_local = {g: i for i, g in enumerate(selected_genes)}
    print(f"Selected genes for split {split_name}, sample {sample_id}: {selected_genes}")

    unknown_gene_id = int(gene2id[UNKNOWN_GENE_TOKEN])
    unknown_tissue_id = int(tissue2id[UNKNOWN_TISSUE_TOKEN])
    gene_ids = np.array([gene2id.get(str(g), unknown_gene_id) for g in gene_names], dtype=np.int64)
    tissue_name = panel_to_tissue.get(sample_id, UNKNOWN_TISSUE_TOKEN)
    tissue_id_int = int(tissue2id.get(tissue_name, unknown_tissue_id))

    X_corrupt = corrupt_batch_deterministic(
        x_clean=X_clean,
        global_idx_np=global_idx_np,
        version_idx=version_idx,
        p_val=float(p_val),
        base_seed=base_seed,
        exact=EXACT_CORRUPTION,
    )

    X_recon_expected, mu_nb, pi_mat, theta_vec = run_token_model_batched(
        model=model,
        X_in=X_corrupt,
        gene_ids_np=gene_ids,
        tissue_id_int=tissue_id_int,
        device=device,
        batch_size=PAIR_INFER_BATCH_SIZE,
    )

    hist_rng = np.random.default_rng(
        int(ZINB_HIST_SAMPLE_SEED + version_idx * 100_003 + _stable_string_seed(sample_id))
    )
    X_recon_sample = sample_zinb_counts_np(mu_nb=mu_nb, theta=theta_vec, pi=pi_mat, rng=hist_rng)

    plot_selected_gene_histograms(
        selected_genes=selected_genes,
        gene_to_idx=gene_to_idx_local,
        X_input=X_corrupt[:, selected_idx],
        X_recon=X_recon_sample[:, selected_idx],
        X_target=X_clean[:, selected_idx],
        save_path=build_output_path(
            save_dir,
            "single_panel",
            "gene_histograms",
            "png",
            split=split_name,
            sample_id=sample_id,
            p_val=p_val,
        ),
        label_input=f"{sample_id} corrupted",
        label_recon=f"{sample_id} through transformer (1x ZINB sample)",
        label_target=f"{sample_id} clean",
    )

    selected_gene_summary = pd.DataFrame(
        {
            "gene": selected_genes,
            "split": split_name,
            "sample_id": sample_id,
            "p_non_overlap": float(p_val),
            "mean_corrupted": X_corrupt[:, selected_idx].mean(axis=0),
            "mean_recon": X_recon_expected[:, selected_idx].mean(axis=0),
            "mean_clean": X_clean[:, selected_idx].mean(axis=0),
            "detect_corrupted": (X_corrupt[:, selected_idx] > 0).mean(axis=0),
            "detect_recon": (X_recon_expected[:, selected_idx] > 0).mean(axis=0),
            "detect_clean": (X_clean[:, selected_idx] > 0).mean(axis=0),
        }
    )
    selected_gene_summary.to_csv(
        build_output_path(
            save_dir,
            "single_panel",
            "selected_gene_summary",
            "csv",
            split=split_name,
            sample_id=sample_id,
            p_val=p_val,
        ),
        index=False,
    )


def run_split_panel_reconstruction_analysis(
    panel_data,
    test_sample_ids,
    sample_id,
    split_name,
    panel_to_tissue,
    gene2id,
    tissue2id,
    train_gene_name_set,
    model,
    device,
    save_dir,
    version_idx,
    p_val,
):
    if sample_id not in panel_data:
        print(f"Skipping reconstruction analysis for split {split_name}: sample {sample_id} not present in panel_data.")
        return

    panel_bases = _panel_global_bases(panel_data, test_sample_ids)
    if sample_id not in panel_bases:
        print(f"Skipping reconstruction analysis for split {split_name}: sample {sample_id} not found in full test ordering.")
        return

    print(f"\nRunning reconstruction analysis for split {split_name}, sample {sample_id}, p={p_val:.3f}")

    rec = panel_data[sample_id]
    gene_names_full = np.asarray(rec["gene_names"], dtype=object)
    valid_gene_mask = compute_valid_overlap_gene_mask(gene_names_full, train_gene_name_set)
    valid_gene_idx = np.flatnonzero(valid_gene_mask)
    if valid_gene_idx.size == 0:
        print(f"Skipping reconstruction analysis for split {split_name}: no valid train-overlap genes for {sample_id}.")
        return

    print(f"Using {valid_gene_idx.size} valid overlap genes for split {split_name}, sample {sample_id}.")
    adata_clean = materialize_panel_native(panel_data, sample_id)
    X_clean_full = np.clip(_to_dense_float32(adata_clean.X), 0.0, None)
    if X_clean_full.shape[0] == 0:
        print(f"Skipping reconstruction analysis for split {split_name}: sample {sample_id} has no cells.")
        return

    panel_base = panel_bases[sample_id]
    global_idx_np = panel_base + np.arange(X_clean_full.shape[0], dtype=np.int64)

    X_corrupt_full = corrupt_batch_deterministic(
        x_clean=X_clean_full,
        global_idx_np=global_idx_np,
        version_idx=version_idx,
        p_val=float(p_val),
        base_seed=base_seed,
        exact=EXACT_CORRUPTION,
    )

    unknown_gene_id = int(gene2id[UNKNOWN_GENE_TOKEN])
    unknown_tissue_id = int(tissue2id[UNKNOWN_TISSUE_TOKEN])
    gene_ids_full = np.array([gene2id.get(str(g), unknown_gene_id) for g in gene_names_full], dtype=np.int64)
    tissue_name = panel_to_tissue.get(sample_id, UNKNOWN_TISSUE_TOKEN)
    tissue_id_int = int(tissue2id.get(tissue_name, unknown_tissue_id))

    X_recon_expected_full, mu_nb_full, pi_mat_full, theta_vec_full = run_token_model_batched(
        model=model,
        X_in=X_corrupt_full,
        gene_ids_np=gene_ids_full,
        tissue_id_int=tissue_id_int,
        device=device,
        batch_size=PAIR_INFER_BATCH_SIZE,
    )

    gene_names_valid = gene_names_full[valid_gene_idx]
    X_clean = X_clean_full[:, valid_gene_idx]
    X_corrupt = X_corrupt_full[:, valid_gene_idx]
    X_recon_expected = X_recon_expected_full[:, valid_gene_idx]
    mu_nb = mu_nb_full[:, valid_gene_idx]
    pi_mat = pi_mat_full[:, valid_gene_idx]
    theta_vec = theta_vec_full[valid_gene_idx]

    mean_input = X_corrupt.mean(axis=0)
    mean_recon = X_recon_expected.mean(axis=0)
    mean_target = X_clean.mean(axis=0)

    det_input = (X_corrupt > 0).mean(axis=0)
    det_target = (X_clean > 0).mean(axis=0)
    det_recon = expected_detection_rate_from_token_params(mu_nb, theta_vec, pi_mat)

    rmse_mean_input, r2_mean_input = rmse_and_r2(mean_input, mean_target)
    rmse_mean_recon, r2_mean_recon = rmse_and_r2(mean_recon, mean_target)
    rmse_det_input, r2_det_input = rmse_and_r2(det_input, det_target)
    rmse_det_recon, r2_det_recon = rmse_and_r2(det_recon, det_target)

    recon_summary = pd.DataFrame(
        [
            {"comparison": "corrupted_vs_clean_gene_mean", "rmse": rmse_mean_input, "r2": r2_mean_input},
            {"comparison": "recon_vs_clean_gene_mean", "rmse": rmse_mean_recon, "r2": r2_mean_recon},
            {"comparison": "corrupted_vs_clean_detection", "rmse": rmse_det_input, "r2": r2_det_input},
            {"comparison": "recon_vs_clean_detection", "rmse": rmse_det_recon, "r2": r2_det_recon},
        ]
    )
    print("\nSplit-panel reconstruction summary:")
    print(recon_summary.to_string(index=False))
    recon_summary.to_csv(
        build_output_path(
            save_dir,
            "split_recon",
            "summary_valid_train_overlap",
            "csv",
            split=split_name,
            sample_id=sample_id,
            p_val=p_val,
        ),
        index=False,
    )

    gene_summary = pd.DataFrame(
        {
            "gene": pd.Index(gene_names_valid).astype(str),
            "mean_corrupted": mean_input,
            "mean_recon": mean_recon,
            "mean_clean": mean_target,
            "detect_corrupted": det_input,
            "detect_recon": det_recon,
            "detect_clean": det_target,
            "abs_err_mean_corrupted_vs_clean": np.abs(mean_input - mean_target),
            "abs_err_mean_recon_vs_clean": np.abs(mean_recon - mean_target),
        }
    ).sort_values("mean_clean", ascending=False).reset_index(drop=True)
    gene_summary.to_csv(
        build_output_path(
            save_dir,
            "split_recon",
            "gene_summary_valid_train_overlap",
            "csv",
            split=split_name,
            sample_id=sample_id,
            p_val=p_val,
        ),
        index=False,
    )

    save_reconstruction_scatter_suite(
        mean_input=mean_input,
        mean_recon=mean_recon,
        mean_target=mean_target,
        det_input=det_input,
        det_recon=det_recon,
        det_target=det_target,
        title_prefix=f"Split {split_name}: {sample_id} corrupted -> transformer -> clean | valid train-overlap genes",
        input_label=f"{sample_id} corrupted",
        recon_label="Reconstruction",
        target_label=f"{sample_id} clean",
        save_path_linear=build_output_path(
            save_dir,
            "split_recon",
            "scatter_valid_train_overlap",
            "png",
            split=split_name,
            sample_id=sample_id,
            p_val=p_val,
        ),
        save_path_log=build_output_path(
            save_dir,
            "split_recon",
            "scatter_log_gene_mean_valid_train_overlap",
            "png",
            split=split_name,
            sample_id=sample_id,
            p_val=p_val,
        ),
    )

    selected_genes = select_high_low_genes(
        gene_names=np.asarray(gene_names_valid, dtype=object),
        mean_target=mean_target,
        n_high=RECON_PANEL_TOP_HIGH,
        n_low=RECON_PANEL_TOP_LOW,
    )
    print("\nSelected genes for split-panel spatial plots:")
    print(selected_genes)

    selected_gene_df = gene_summary[gene_summary["gene"].isin(selected_genes)].copy()
    selected_gene_df.to_csv(
        build_output_path(
            save_dir,
            "split_recon",
            "selected_genes_valid_train_overlap",
            "csv",
            split=split_name,
            sample_id=sample_id,
            p_val=p_val,
        ),
        index=False,
    )

    gene_to_idx_valid = {g: i for i, g in enumerate(pd.Index(gene_names_valid).astype(str))}
    plot_selected_gene_spatial_triplets(
        selected_genes=selected_genes,
        gene_to_idx=gene_to_idx_valid,
        adata_input=adata_clean[:, gene_names_valid].copy(),
        X_input=X_corrupt,
        X_recon=X_recon_expected,
        adata_target=adata_clean[:, gene_names_valid].copy(),
        X_target=X_clean,
        save_path=build_output_path(
            save_dir,
            "split_recon",
            "gene_spatial_valid_train_overlap",
            "png",
            split=split_name,
            sample_id=sample_id,
            p_val=p_val,
        ),
        label_input=f"{sample_id} corrupted",
        label_recon=f"{sample_id} through transformer",
        label_target=f"{sample_id} clean",
    )


def run_pair_analysis(
    panel_data,
    panel_to_tissue,
    gene2id,
    tissue2id,
    train_gene_name_set,
    model,
    device,
    save_dir,
):
    if PAIR_5K_ID not in panel_data:
        raise KeyError(f"{PAIR_5K_ID} not found in panel_data.")
    if PAIR_V1_ID not in panel_data:
        raise KeyError(f"{PAIR_V1_ID} not found in panel_data.")

    print(f"\nRunning pair analysis for {PAIR_5K_ID} -> {PAIR_V1_ID}")
    adata_5k = materialize_panel_native(panel_data, PAIR_5K_ID)
    adata_v1 = materialize_panel_native(panel_data, PAIR_V1_ID)

    present_5k = set(pd.Index(panel_data[PAIR_5K_ID]["gene_names"]).astype(str))
    present_v1 = set(pd.Index(panel_data[PAIR_V1_ID]["gene_names"]).astype(str))
    common_pair_genes = pd.Index(sorted(present_5k & present_v1 & set(train_gene_name_set))).astype(str)
    if len(common_pair_genes) == 0:
        raise ValueError("No overlapping trained genes found between pair panels.")

    idx_5k = pd.Index(panel_data[PAIR_5K_ID]["gene_names"]).astype(str).get_indexer(common_pair_genes)
    idx_v1 = pd.Index(panel_data[PAIR_V1_ID]["gene_names"]).astype(str).get_indexer(common_pair_genes)

    X5k_in = np.clip(_to_dense_float32(adata_5k.X[:, idx_5k]), 0.0, None)
    Xv1 = np.clip(_to_dense_float32(adata_v1.X[:, idx_v1]), 0.0, None)

    unknown_gene_id = int(gene2id[UNKNOWN_GENE_TOKEN])
    unknown_tissue_id = int(tissue2id[UNKNOWN_TISSUE_TOKEN])
    gene_ids_pair = np.array([gene2id.get(str(g), unknown_gene_id) for g in common_pair_genes], dtype=np.int64)
    tissue_name_5k = panel_to_tissue.get(PAIR_5K_ID, UNKNOWN_TISSUE_TOKEN)
    tissue_id_5k = int(tissue2id.get(tissue_name_5k, unknown_tissue_id))

    X5k_recon_expected, mu_nb_pair, pi_pair, theta_pair = run_token_model_batched(
        model=model,
        X_in=X5k_in,
        gene_ids_np=gene_ids_pair,
        tissue_id_int=tissue_id_5k,
        device=device,
        batch_size=PAIR_INFER_BATCH_SIZE,
    )

    X5k_recon_sampled = sample_zinb_counts_np(
        mu_nb=mu_nb_pair,
        theta=theta_pair,
        pi=pi_pair,
        rng=np.random.default_rng(
            int(ZINB_HIST_SAMPLE_SEED + _stable_string_seed(PAIR_5K_ID) + 1009 * _stable_string_seed(PAIR_V1_ID))
        ),
    )

    mean_input = X5k_in.mean(axis=0)
    mean_recon = X5k_recon_expected.mean(axis=0)
    mean_target = Xv1.mean(axis=0)

    det_input = (X5k_in > 0).mean(axis=0)
    det_target = (Xv1 > 0).mean(axis=0)
    det_recon = expected_detection_rate_from_token_params(mu_nb_pair, theta_pair, pi_pair)

    rmse_mean_input, r2_mean_input = rmse_and_r2(mean_input, mean_target)
    rmse_mean_recon, r2_mean_recon = rmse_and_r2(mean_recon, mean_target)
    rmse_det_input, r2_det_input = rmse_and_r2(det_input, det_target)
    rmse_det_recon, r2_det_recon = rmse_and_r2(det_recon, det_target)

    pair_summary = pd.DataFrame(
        [
            {"comparison": "input_vs_target_gene_mean", "rmse": rmse_mean_input, "r2": r2_mean_input},
            {"comparison": "recon_vs_target_gene_mean", "rmse": rmse_mean_recon, "r2": r2_mean_recon},
            {"comparison": "input_vs_target_detection", "rmse": rmse_det_input, "r2": r2_det_input},
            {"comparison": "recon_vs_target_detection", "rmse": rmse_det_recon, "r2": r2_det_recon},
        ]
    )
    print("\nPair summary:")
    print(pair_summary.to_string(index=False))
    pair_summary.to_csv(
        build_output_path(
            save_dir,
            "pair",
            "summary",
            "csv",
            extra_parts=[f"source_{PAIR_5K_ID}", f"target_{PAIR_V1_ID}"],
        ),
        index=False,
    )

    gene_summary = pd.DataFrame(
        {
            "gene": common_pair_genes.astype(str),
            "mean_input": mean_input,
            "mean_recon": mean_recon,
            "mean_target": mean_target,
            "detect_input": det_input,
            "detect_recon": det_recon,
            "detect_target": det_target,
            "abs_err_mean_input_vs_target": np.abs(mean_input - mean_target),
            "abs_err_mean_recon_vs_target": np.abs(mean_recon - mean_target),
        }
    ).sort_values("mean_target", ascending=False).reset_index(drop=True)
    gene_summary.to_csv(
        build_output_path(
            save_dir,
            "pair",
            "gene_summary",
            "csv",
            extra_parts=[f"source_{PAIR_5K_ID}", f"target_{PAIR_V1_ID}"],
        ),
        index=False,
    )

    save_reconstruction_scatter_suite(
        mean_input=mean_input,
        mean_recon=mean_recon,
        mean_target=mean_target,
        det_input=det_input,
        det_recon=det_recon,
        det_target=det_target,
        title_prefix=f"Pair analysis: {PAIR_5K_ID} -> transformer -> {PAIR_V1_ID}",
        input_label=f"{PAIR_5K_ID} input",
        recon_label="Reconstruction",
        target_label=f"{PAIR_V1_ID} target",
        save_path_linear=build_output_path(
            save_dir,
            "pair",
            "scatter",
            "png",
            extra_parts=[f"source_{PAIR_5K_ID}", f"target_{PAIR_V1_ID}"],
        ),
        save_path_log=build_output_path(
            save_dir,
            "pair",
            "scatter_log_gene_mean",
            "png",
            extra_parts=[f"source_{PAIR_5K_ID}", f"target_{PAIR_V1_ID}"],
        ),
    )

    selected_genes = select_high_low_genes(
        gene_names=np.asarray(common_pair_genes, dtype=object),
        mean_target=mean_target,
        n_high=PAIR_TOP_HIGH_GENES,
        n_low=PAIR_TOP_LOW_GENES,
    )
    selected_gene_df = gene_summary[gene_summary["gene"].isin(selected_genes)].copy()
    selected_gene_df.to_csv(
        build_output_path(
            save_dir,
            "pair",
            "selected_genes",
            "csv",
            extra_parts=[f"source_{PAIR_5K_ID}", f"target_{PAIR_V1_ID}"],
        ),
        index=False,
    )

    print("\nSelected genes for detailed plots:")
    print(selected_genes)

    gene_to_idx_pair = {g: i for i, g in enumerate(common_pair_genes.astype(str))}
    plot_selected_gene_histograms(
        selected_genes=selected_genes,
        gene_to_idx=gene_to_idx_pair,
        X_input=X5k_in,
        X_recon=X5k_recon_sampled,
        X_target=Xv1,
        save_path=build_output_path(
            save_dir,
            "pair",
            "gene_histograms",
            "png",
            extra_parts=[f"source_{PAIR_5K_ID}", f"target_{PAIR_V1_ID}"],
        ),
        label_input=f"{PAIR_5K_ID} input",
        label_recon=f"{PAIR_5K_ID} through transformer (1x ZINB sample)",
        label_target=f"{PAIR_V1_ID} target",
    )

    adata_5k_pair = adata_5k[:, common_pair_genes].copy()
    adata_v1_pair = adata_v1[:, common_pair_genes].copy()
    plot_selected_gene_spatial_triplets(
        selected_genes=selected_genes,
        gene_to_idx=gene_to_idx_pair,
        adata_input=adata_5k_pair,
        X_input=X5k_in,
        X_recon=X5k_recon_expected,
        adata_target=adata_v1_pair,
        X_target=Xv1,
        save_path=build_output_path(
            save_dir,
            "pair",
            "gene_spatial",
            "png",
            extra_parts=[f"source_{PAIR_5K_ID}", f"target_{PAIR_V1_ID}"],
        ),
        label_input=f"{PAIR_5K_ID} input",
        label_recon=f"{PAIR_5K_ID} through transformer",
        label_target=f"{PAIR_V1_ID} target",
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    t0 = time.time()

    torch.set_float32_matmul_precision("high")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    print(f"Running on device: {device}")
    print(f"Checkpoint path: {WEIGHTS_PATH}")
    print(f"Output directory: {SAVE_DIR}")

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {WEIGHTS_PATH}")

    ckpt = _safe_torch_load(WEIGHTS_PATH, map_location=device)
    if "model_state_dict" not in ckpt:
        raise KeyError("Checkpoint is missing `model_state_dict`.")
    if "gene2id" not in ckpt or "tissue2id" not in ckpt:
        raise KeyError("Checkpoint is missing `gene2id` and/or `tissue2id`.")

    gene2id = dict(ckpt["gene2id"])
    tissue2id = dict(ckpt["tissue2id"])
    if UNKNOWN_GENE_TOKEN not in gene2id:
        raise KeyError(f"Checkpoint gene2id is missing {UNKNOWN_GENE_TOKEN}.")
    if UNKNOWN_TISSUE_TOKEN not in tissue2id:
        raise KeyError(f"Checkpoint tissue2id is missing {UNKNOWN_TISSUE_TOKEN}.")

    model = GeneTokenAutoencoder(
        n_genes_vocab=int(len(gene2id)),
        n_tissues=int(len(tissue2id)),
        d_model=MODEL_D_MODEL,
        nhead=MODEL_NHEAD,
        num_layers=MODEL_NUM_LAYERS,
        latent_dim=MODEL_LATENT_DIM,
        dropout=MODEL_DROPOUT,
        theta_init=MODEL_THETA_INIT,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    print(
        f"Loaded checkpoint | gene_vocab={len(gene2id)}, tissues={len(tissue2id)}, "
        f"d_model={MODEL_D_MODEL}, nhead={MODEL_NHEAD}, num_layers={MODEL_NUM_LAYERS}, latent={MODEL_LATENT_DIM}"
    )
    if ckpt.get("best_epoch") is not None:
        print(f"Checkpoint best_epoch: {ckpt['best_epoch']}")
    if ckpt.get("best_val_zinb_nll") is not None:
        print(f"Checkpoint best_val_zinb_nll: {float(ckpt['best_val_zinb_nll']):.6f}")

    train_records = flatten_simple_split_dict(train, "train", "train")
    val_records = flatten_simple_split_dict(val, "val", "val")
    test_a_records = flatten_simple_split_dict(test_seen_tissue_in_distribution, "test", "A")
    test_b_records = flatten_simple_split_dict(test_seen_tissue_distribution_shift, "test", "B")
    test_c_records = flatten_simple_split_dict(test_unseen_tissue, "test", "C")
    all_records = train_records + val_records + test_a_records + test_b_records + test_c_records

    panel_to_tissue = {r["sample_id"]: r["tissue_name"] for r in all_records}
    panel_to_group = {r["sample_id"]: r["split_group"] for r in all_records}
    panel_to_split = {r["sample_id"]: r["split_name"] for r in all_records}

    train_sample_ids = [r["sample_id"] for r in train_records]
    val_sample_ids = [r["sample_id"] for r in val_records]
    test_a_sample_ids = [r["sample_id"] for r in test_a_records]
    test_b_sample_ids = [r["sample_id"] for r in test_b_records]
    test_c_sample_ids = [r["sample_id"] for r in test_c_records]
    test_sample_ids = test_a_sample_ids + test_b_sample_ids + test_c_sample_ids

    available_ids = sorted(
        p.name.replace("_xenium_cell_level.h5ad", "")
        for p in ANN_DIR.glob("*_xenium_cell_level.h5ad")
    )
    available_set = set(available_ids)

    requested_ids = sorted(set(train_sample_ids + val_sample_ids + test_sample_ids + [PAIR_5K_ID, PAIR_V1_ID]))
    missing_requested = [sid for sid in requested_ids if sid not in available_set]
    if missing_requested:
        print(f"Requested IDs missing from disk ({len(missing_requested)}): {missing_requested}")

    panel_data = {}
    load_failed = []
    for sample_id in requested_ids:
        if sample_id not in available_set:
            continue
        try:
            rec = _prepare_panel_metadata(sample_id=sample_id, threshold=threshold, genes_threshold=genes_threshold)
            rec["tissue_name"] = panel_to_tissue.get(sample_id, UNKNOWN_TISSUE_TOKEN)
            rec["split_group"] = panel_to_group.get(sample_id, "other")
            rec["split_name"] = panel_to_split.get(sample_id, "other")
            panel_data[sample_id] = rec
        except Exception as e:
            load_failed.append((sample_id, str(e)))

    train_sample_ids = [sid for sid in train_sample_ids if sid in panel_data]
    val_sample_ids = [sid for sid in val_sample_ids if sid in panel_data]
    test_a_sample_ids = [sid for sid in test_a_sample_ids if sid in panel_data]
    test_b_sample_ids = [sid for sid in test_b_sample_ids if sid in panel_data]
    test_c_sample_ids = [sid for sid in test_c_sample_ids if sid in panel_data]
    test_sample_ids = test_a_sample_ids + test_b_sample_ids + test_c_sample_ids

    if len(train_sample_ids) == 0:
        raise ValueError("No training samples are available after loading.")
    if len(test_sample_ids) == 0:
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

    cache_dir = SAVE_DIR / "panel_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for sid in tqdm(sorted(panel_data.keys()), desc="Ensuring panel caches", unit="panel"):
        cache_path = build_or_load_panel_cache(sid, panel_data[sid], cache_dir)
        panel_data[sid]["cache_path"] = str(cache_path)

    train_gene_name_set = set(gene2id.keys()) - {UNKNOWN_GENE_TOKEN}
    print(f"Genes in transformer train vocabulary (excluding unknown): {len(train_gene_name_set)}")
    print(f"Tissues in transformer checkpoint: {sorted(tissue2id.keys())}")

    if RUN_GLOBAL_TEST_EVAL:
        run_global_test_evaluation(
            panel_data=panel_data,
            train_sample_ids=train_sample_ids,
            test_sample_ids=test_sample_ids,
            test_a_sample_ids=test_a_sample_ids,
            test_b_sample_ids=test_b_sample_ids,
            test_c_sample_ids=test_c_sample_ids,
            panel_to_tissue=panel_to_tissue,
            gene2id=gene2id,
            tissue2id=tissue2id,
            train_gene_name_set=train_gene_name_set,
            model=model,
            device=device,
            save_dir=SAVE_DIR,
        )

    if RUN_SINGLE_PANEL_GENE_DISTRIBUTIONS or RUN_SPLIT_PANEL_RECON_ANALYSIS:
        eval_p_values = np.asarray(QUICK_P_VALUES if QUICK_MODE else FULL_P_VALUES, dtype=np.float64)
        if np.any((eval_p_values < 0.0) | (eval_p_values > 1.0)):
            raise ValueError("All evaluation p values must be between 0 and 1.")

        if SINGLE_PANEL_GENE_DIST_P is None:
            single_panel_version_idx = 0
        else:
            matches = np.where(np.isclose(eval_p_values, float(SINGLE_PANEL_GENE_DIST_P)))[0]
            if matches.size == 0:
                raise ValueError(
                    f"SINGLE_PANEL_GENE_DIST_P={SINGLE_PANEL_GENE_DIST_P} not found in {eval_p_values.tolist()}"
                )
            single_panel_version_idx = int(matches[0])
        single_panel_p = float(eval_p_values[single_panel_version_idx])

    if RUN_SINGLE_PANEL_GENE_DISTRIBUTIONS:
        for split_name in ["A", "B", "C"]:
            sample_id = SPLIT_PANEL_FOR_HIST.get(split_name)
            if sample_id is None:
                continue
            run_single_panel_gene_distribution_analysis(
                panel_data=panel_data,
                test_sample_ids=test_sample_ids,
                sample_id=sample_id,
                split_name=split_name,
                panel_to_tissue=panel_to_tissue,
                gene2id=gene2id,
                tissue2id=tissue2id,
                model=model,
                device=device,
                save_dir=SAVE_DIR,
                version_idx=single_panel_version_idx,
                p_val=single_panel_p,
            )

    if RUN_SPLIT_PANEL_RECON_ANALYSIS:
        for split_name in ["A", "B", "C"]:
            sample_id = SPLIT_PANEL_FOR_RECON_ANALYSIS.get(split_name)
            if sample_id is None:
                continue
            run_split_panel_reconstruction_analysis(
                panel_data=panel_data,
                test_sample_ids=test_sample_ids,
                sample_id=sample_id,
                split_name=split_name,
                panel_to_tissue=panel_to_tissue,
                gene2id=gene2id,
                tissue2id=tissue2id,
                train_gene_name_set=train_gene_name_set,
                model=model,
                device=device,
                save_dir=SAVE_DIR,
                version_idx=single_panel_version_idx,
                p_val=single_panel_p,
            )

    if RUN_PAIR_ANALYSIS:
        run_pair_analysis(
            panel_data=panel_data,
            panel_to_tissue=panel_to_tissue,
            gene2id=gene2id,
            tissue2id=tissue2id,
            train_gene_name_set=train_gene_name_set,
            model=model,
            device=device,
            save_dir=SAVE_DIR,
        )

    elapsed_min = (time.time() - t0) / 60.0
    print(f"\nDone. Total elapsed time: {elapsed_min:.2f} minutes")
    print(f"Outputs written to: {SAVE_DIR}")


if __name__ == "__main__":
    main()
