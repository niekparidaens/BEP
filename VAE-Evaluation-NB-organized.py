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
SAVE_DIR = Path(os.environ.get("OUTPUT_DIR", PROJECT_ROOT / "outputs" / "vae_eval_nb")).resolve()
SAVE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_WEIGHTS_PATH = PROJECT_ROOT / "outputs" / "VAE_NB" / "VAE_NB_weights-NB-loss"
WEIGHTS_PATH = Path(os.environ.get("WEIGHTS_PATH", str(DEFAULT_WEIGHTS_PATH))).resolve()

# Keep backed mode on the cluster so we do not pull giant matrices into RAM by accident.
READ_MODE = os.environ.get("READ_MODE", "r") or "r"

# Evaluation options
QUICK_MODE = False
QUICK_MAX_CELLS = None          # e.g. 20000 for a smoke test
QUICK_P_VALUES = [0.19]
FULL_P_VALUES = [0.19]

IO_CHUNK_SIZE = 4096
PAIR_INFER_BATCH_SIZE = 1024

# If True, corruption is cell-wise deterministic and matches your old behaviour.
# If False, it uses a faster batch-style RNG that is still reproducible per batch,
# but not bit-for-bit identical to the older exact version.
EXACT_CORRUPTION = True

RUN_GLOBAL_TEST_EVAL = True
RUN_PAIR_ANALYSIS = True

PAIR_5K_ID = "TENX189"
PAIR_V1_ID = "TENX190"

# How many genes to visualize from the pair analysis.
# We take the highest-mean genes and the lowest positive-mean genes.
PAIR_TOP_HIGH_GENES = 4
PAIR_TOP_LOW_GENES = 4

threshold = 40
genes_threshold = 5
base_seed = 42
beta = 1e-3

split_samples_train = [
    "NCBI856", "NCBI857", "NCBI858", "NCBI860", "NCBI861", "NCBI864",
    "NCBI865", "NCBI866", "NCBI867", "NCBI870", "NCBI873", "NCBI875", "NCBI876",
]
split_samples_val = ["NCBI879", "NCBI880", "NCBI881"]
split_samples_test_a = ["NCBI882", "NCBI883", "NCBI884"]
split_samples_test_b = [
    "NCBI887", "NCBI888", "TENX189",
    # "NCBI885", "NCBI886",
    # "NCBI859",
    # "TENX118",
    # "TENX141",
    # "TENX190",
]
split_samples_test_c = [
    "TENX191", "TENX192", "TENX193", "TENX194",
    "TENX195", "TENX196", "TENX197", "TENX198",
]
split_samples_test = split_samples_test_a + split_samples_test_b + split_samples_test_c


# Small helpers
def _adata_path(sample_id: str):
    return ANN_DIR / f"{sample_id}_xenium_cell_level.h5ad"


def _load_sample_backed(sample_id: str, read_mode=READ_MODE):
    h5ad_path = _adata_path(sample_id)
    if not h5ad_path.exists():
        raise FileNotFoundError(f"Missing h5ad for {sample_id}: {h5ad_path}")
    return sc.read_h5ad(h5ad_path, backed=read_mode)


def _select_available_ids(sample_ids, available_like):
    available_set = set(available_like)
    normalized = [str(s).strip().upper() for s in sample_ids]
    found = [sid for sid in normalized if sid in available_set]
    missing = [sid for sid in normalized if sid not in available_set]
    return found, missing


def _safe_torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _to_dense_float32(X):
    if sp.issparse(X):
        return X.toarray().astype(np.float32, copy=False)
    return np.asarray(X, dtype=np.float32)


def autocast_context(use_cuda: bool):
    if use_cuda:
        return torch.amp.autocast(device_type="cuda", enabled=True)
    return nullcontext()


def _compute_qc_chunked(adata_backed, chunk_size=4096):
    """
    Compute total counts and detected genes without reading the whole matrix at once.
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
    Keep the panel backed on disk and only store the metadata needed later:
    - which rows survive QC
    - which genes survive filtering
    - the gene names after filtering
    """
    ad_panel = _load_sample_backed(sample_id)

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
        "adata": ad_panel,
        "cell_pos": cell_pos,
        "gene_pos_all": gene_pos_all,
        "gene_names": gene_names_filtered,
        "n_obs_raw": int(ad_panel.n_obs),
        "n_obs_filtered": int(cell_pos.size),
    }


# Model
class VAE(nn.Module):
    # Same model definition as in training, so checkpoint loading stays clean.
    def __init__(self, input_dim, latent_dim=16, hidden_dim=256):
        super().__init__()
        self.enc_fc1 = nn.Linear(input_dim, hidden_dim)
        self.enc_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

        self.dec_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.dec_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.enc_fc1(x))
        h = F.relu(self.enc_fc2(h))
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.dec_fc1(z))
        h = F.relu(self.dec_fc2(h))
        return self.out(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


def vae_loss(
    recon,
    target,
    mu,
    logvar,
    beta=1e-3,
    loss_mask=None,
    theta_param=None,
    eps=1e-8,
):
    """
    Same reconstruction objective as in training:
    NB reconstruction loss + KL penalty.
    """
    target_counts = target.clamp_min(0.0)
    mu_counts = F.softplus(recon) + eps

    if theta_param is None:
        theta = torch.full((recon.shape[1],), 10.0, device=recon.device, dtype=recon.dtype)
    else:
        theta = F.softplus(theta_param) + eps

    theta = theta.unsqueeze(0).expand_as(mu_counts)
    log_theta_mu = torch.log(theta + mu_counts + eps)

    nb_log_prob = (
        torch.lgamma(target_counts + theta)
        - torch.lgamma(theta)
        - torch.lgamma(target_counts + 1.0)
        + theta * (torch.log(theta + eps) - log_theta_mu)
        + target_counts * (torch.log(mu_counts + eps) - log_theta_mu)
    )

    recon_nll = -nb_log_prob

    if loss_mask is not None:
        recon_loss = recon_nll[:, loss_mask].mean()
    else:
        recon_loss = recon_nll.mean()

    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, recon_loss, kl


# Shared gene-space access
class PanelRowAccessor:
    """
    This mirrors the training-side helper style:
    it keeps track of how each backed panel maps into one shared gene order.
    """

    def __init__(self, panel_data, panel_ids, common_genes):
        self.panel_defs = []
        self.panel_sizes = []
        self.common_genes = pd.Index(common_genes).astype(str)

        for sid in panel_ids:
            rec = panel_data[sid]

            idx = rec["gene_names"].get_indexer(self.common_genes)
            present_mask = idx >= 0

            filtered_pos = idx[present_mask].astype(np.int64, copy=False)
            panel_pos = rec["gene_pos_all"][filtered_pos]
            common_pos = np.flatnonzero(present_mask).astype(np.int64, copy=False)

            if panel_pos.size == 0:
                raise ValueError(f"Panel {sid} has zero overlap with the shared gene space.")

            order = np.argsort(panel_pos)
            panel_pos = panel_pos[order]
            common_pos = common_pos[order]

            self.panel_defs.append({
                "sample_id": sid,
                "adata": rec["adata"],
                "cell_pos": rec["cell_pos"],
                "panel_pos": panel_pos,
                "common_pos": common_pos,
            })
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
        panel_def = self.panel_defs[panel_idx]

        orig_row_idx = int(panel_def["cell_pos"][row_idx])
        row = panel_def["adata"].X[orig_row_idx, panel_def["panel_pos"]]

        if sp.issparse(row):
            row_vals = row.toarray().ravel().astype(np.float32, copy=False)
        else:
            row_vals = np.asarray(row, dtype=np.float32).ravel()

        y = np.zeros(self.n_genes, dtype=np.float32)
        y[panel_def["common_pos"]] = row_vals
        return np.clip(y, 0.0, None)

    def iter_blocks(self, chunk_size=2048, selected_global_idx=None, split_name_by_sid=None):
        """
        Yield dense blocks in the shared gene order.
        This way we can stream evaluation without loading the full test split.
        """
        selected_global_idx = None if selected_global_idx is None else np.asarray(selected_global_idx, dtype=np.int64)

        global_offset = 0
        for panel_def in self.panel_defs:
            sid = panel_def["sample_id"]
            ad_panel = panel_def["adata"]
            cell_pos = panel_def["cell_pos"]
            panel_pos = panel_def["panel_pos"]
            common_pos = panel_def["common_pos"]

            n_rows = int(cell_pos.size)

            if selected_global_idx is None:
                local_blocks = [
                    np.arange(start, min(start + chunk_size, n_rows), dtype=np.int64)
                    for start in range(0, n_rows, chunk_size)
                ]
            else:
                lo = np.searchsorted(selected_global_idx, global_offset, side="left")
                hi = np.searchsorted(selected_global_idx, global_offset + n_rows, side="left")
                local_selected = selected_global_idx[lo:hi] - global_offset

                local_blocks = []
                for start in range(0, local_selected.size, chunk_size):
                    stop = min(start + chunk_size, local_selected.size)
                    local_blocks.append(local_selected[start:stop])

            for local_idx in local_blocks:
                if local_idx.size == 0:
                    continue

                rows = cell_pos[local_idx]

                try:
                    block = ad_panel[rows, panel_pos].X
                    block = _to_dense_float32(block)
                except Exception:
                    block_rows = []
                    for r in rows:
                        row = ad_panel.X[int(r), panel_pos]
                        block_rows.append(_to_dense_float32(row).ravel())
                    block = np.vstack(block_rows).astype(np.float32, copy=False)

                Y_block = np.zeros((block.shape[0], self.n_genes), dtype=np.float32)
                Y_block[:, common_pos] = block
                Y_block = np.clip(Y_block, 0.0, None)

                yield {
                    "sample_id": sid,
                    "split": None if split_name_by_sid is None else split_name_by_sid.get(sid),
                    "global_idx": global_offset + local_idx,
                    "Y": Y_block,
                }

            global_offset += n_rows


def compute_gene_mean_streaming(panel_data, panel_ids, common_genes, chunk_size):
    """
    Train-set gene mean baseline.
    We compute it in streaming mode so the code still works on large panels.
    """
    accessor = PanelRowAccessor(panel_data, panel_ids, common_genes)

    gene_sum = np.zeros(accessor.n_genes, dtype=np.float64)
    n_cells = 0

    for payload in tqdm(
        accessor.iter_blocks(chunk_size=chunk_size),
        desc="Computing train gene mean",
        unit="chunk",
    ):
        Y = payload["Y"].astype(np.float64, copy=False)
        gene_sum += Y.sum(axis=0)
        n_cells += Y.shape[0]

    if n_cells == 0:
        raise RuntimeError("No train cells found while computing gene mean.")

    return (gene_sum / n_cells).astype(np.float32)


def materialize_panel_in_gene_space(panel_data, sample_id, common_genes, chunk_size=4096):
    accessor = PanelRowAccessor(panel_data, [sample_id], common_genes)
    rec = panel_data[sample_id]

    X = np.empty((rec["cell_pos"].size, accessor.n_genes), dtype=np.float32)
    write_pos = 0

    for payload in tqdm(
        accessor.iter_blocks(chunk_size=chunk_size),
        desc=f"Materializing {sample_id}",
        unit="chunk",
    ):
        block = payload["Y"]
        n_block = block.shape[0]
        X[write_pos:write_pos + n_block] = block
        write_pos += n_block

    ad_src = rec["adata"]
    kept_rows = rec["cell_pos"]

    obs = ad_src.obs.iloc[kept_rows].copy()
    obs_names = pd.Index(ad_src.obs_names.astype(str))[kept_rows].copy()
    var = pd.DataFrame(index=pd.Index(common_genes).astype(str))

    adata_out = ad.AnnData(X=X, obs=obs, var=var)
    adata_out.obs_names = obs_names

    for key in ad_src.obsm.keys():
        try:
            adata_out.obsm[key] = np.asarray(ad_src.obsm[key])[kept_rows].copy()
        except Exception:
            pass

    _ensure_spatial_coords(adata_out)

    return adata_out


# Corruption
def corrupt_batch_deterministic(x_clean, global_idx_np, version_idx, p_val, base_seed, exact=True):
    """
    Apply the synthetic dropout / downsampling corruption.

    We keep this explicit because it is part of the experimental setup:
    evaluation should use the same corruption logic every time.
    """
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


# Metrics
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


def update_metric_acc(acc, pred, true):
    """
    We explicitly broadcast pred and true first.

    This matters for the gene-mean baseline:
    pred may come in as shape (1, n_genes) while true is (batch, n_genes).
    The arithmetic broadcasts fine, but metric counting needs the same broadcast too.
    """
    pred64 = np.asarray(pred, dtype=np.float64)
    true64 = np.asarray(true, dtype=np.float64)
    pred64, true64 = np.broadcast_arrays(pred64, true64)

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
        rows.append({
            "method": method_name,
            "rmse_counts": out["rmse_counts"],
            "r2_counts": out["r2_counts"],
            "rmse_log1p": out["rmse_log1p"],
            "r2_log1p": out["r2_log1p"],
            "mse_counts": out["mse_counts"],
            "mse_log1p": out["mse_log1p"],
            "n_cells": out["n_cells"],
        })

    return (
        pd.DataFrame(rows)
        .sort_values(["rmse_counts", "rmse_log1p"], ascending=[True, True])
        .reset_index(drop=True)
    )


def rmse_and_r2(pred, true):
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)

    rmse = float(np.sqrt(np.mean(np.square(pred - true))))
    ss_res = float(np.square(pred - true).sum())
    ss_tot = float(np.square(true - true.mean()).sum())
    r2 = np.nan if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)
    return rmse, r2


# Plot helpers
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
    line_max = float(max(x.max(), y.max())) * (1.0 + line_pad)

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


def _find_spatial_xy(adata_obj):
    """
    Try a few common coordinate conventions used in Xenium / AnnData objects.
    """
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


def _ensure_spatial_coords(adata):
    if "spatial" in adata.obsm:
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
        ((x, y) for x, y in candidate_pairs if x in adata.obs.columns and y in adata.obs.columns),
        None,
    )

    if pair is None:
        raise KeyError(
            "No spatial coordinates found. Expected adata.obsm['spatial'] or one of: "
            f"{candidate_pairs}."
        )

    adata.obsm["spatial"] = adata.obs[[pair[0], pair[1]]].to_numpy()


def nb_expected_counts_batched(model, X_in, device, batch_size=1024):
    """
    For the NB model, the decoder output is transformed into the NB mean by softplus.
    There is no zero-inflation factor.
    """
    model.eval()
    out_blocks = []
    use_cuda = device.type == "cuda"

    with torch.inference_mode():
        for start in tqdm(range(0, X_in.shape[0], batch_size), desc="VAE inference", unit="batch"):
            stop = min(start + batch_size, X_in.shape[0])
            xb = torch.from_numpy(X_in[start:stop]).to(device, non_blocking=use_cuda)

            with autocast_context(use_cuda):
                raw_recon, _, _ = model(xb)

            mu_nb = F.softplus(raw_recon).float().cpu().numpy().astype(np.float32)
            out_blocks.append(mu_nb)

    return np.vstack(out_blocks)


def select_high_low_genes(gene_names, mean_target, n_high=4, n_low=4):
    gene_names = np.asarray(gene_names, dtype=object)
    mean_target = np.asarray(mean_target, dtype=np.float64)

    high_order = np.argsort(mean_target)[::-1]
    high_genes = [gene_names[i] for i in high_order[:min(n_high, len(high_order))]]

    positive_idx = np.flatnonzero(mean_target > 0)
    if positive_idx.size > 0:
        low_order = positive_idx[np.argsort(mean_target[positive_idx])]
        low_genes = [gene_names[i] for i in low_order[:min(n_low, len(low_order))]]
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
):
    x_in, y_in, in_source = _find_spatial_xy(adata_input)
    x_tar, y_tar, tar_source = _find_spatial_xy(adata_target)

    if x_in is None or x_tar is None:
        print("Skipping spatial gene plots: no spatial coordinates found.")
        return

    n_rows = len(selected_genes)
    if n_rows == 0:
        return

    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4.2 * n_rows), squeeze=False)

    for i, gene in enumerate(selected_genes):
        g_idx = gene_to_idx[gene]

        expr_input = np.log1p(np.clip(_to_dense_float32(adata_input.X[:, g_idx]).ravel(), 0.0, None))
        expr_recon = np.log1p(np.clip(X_recon[:, g_idx], 0.0, None))
        expr_target = np.log1p(np.clip(_to_dense_float32(adata_target.X[:, g_idx]).ravel(), 0.0, None))

        vmax = float(max(
            np.percentile(expr_input, 99) if expr_input.size else 0.0,
            np.percentile(expr_recon, 99) if expr_recon.size else 0.0,
            np.percentile(expr_target, 99) if expr_target.size else 0.0,
            1e-6,
        ))

        ax0, ax1, ax2 = axes[i]

        sc0 = ax0.scatter(x_in, y_in, c=expr_input, s=4, cmap="viridis", vmin=0.0, vmax=vmax)
        ax0.set_title(f"{gene} | {label_input}")
        ax0.set_xlabel(in_source)
        ax0.set_ylabel("y")
        ax0.invert_yaxis()
        ax0.set_aspect("equal", adjustable="box")

        sc1 = ax1.scatter(x_in, y_in, c=expr_recon, s=4, cmap="viridis", vmin=0.0, vmax=vmax)
        ax1.set_title(f"{gene} | {label_recon}")
        ax1.set_xlabel(in_source)
        ax1.set_ylabel("y")
        ax1.invert_yaxis()
        ax1.set_aspect("equal", adjustable="box")

        sc2 = ax2.scatter(x_tar, y_tar, c=expr_target, s=4, cmap="viridis", vmin=0.0, vmax=vmax)
        ax2.set_title(f"{gene} | {label_target}")
        ax2.set_xlabel(tar_source)
        ax2.set_ylabel("y")
        ax2.invert_yaxis()
        ax2.set_aspect("equal", adjustable="box")

        cbar = fig.colorbar(sc2, ax=[ax0, ax1, ax2], fraction=0.02, pad=0.01)
        cbar.set_label("log1p(count)")

    fig.suptitle("Selected spatial gene patterns", y=1.002)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_global_test_evaluation(
    panel_data,
    train_panel_ids,
    test_panel_ids,
    test_a_panel_ids,
    test_b_panel_ids,
    test_c_panel_ids,
    gene_names,
    model,
    log_theta,
    device,
    save_dir,
):
    eval_p_values = np.asarray(
        QUICK_P_VALUES if QUICK_MODE else FULL_P_VALUES,
        dtype=np.float64,
    )
    if np.any((eval_p_values < 0.0) | (eval_p_values > 1.0)):
        raise ValueError("All evaluation p values must be between 0 and 1.")

    test_accessor = PanelRowAccessor(panel_data, test_panel_ids, gene_names)
    base_n_cells = test_accessor.total_cells

    selected_cell_idx = None
    if QUICK_MODE and QUICK_MAX_CELLS is not None and QUICK_MAX_CELLS < base_n_cells:
        rng_sub = np.random.default_rng(base_seed + 2026)
        selected_cell_idx = np.sort(
            rng_sub.choice(base_n_cells, size=int(QUICK_MAX_CELLS), replace=False)
        )

    n_eval_cells = base_n_cells if selected_cell_idx is None else int(selected_cell_idx.size)

    split_name_by_sid = {sid: "A" for sid in test_a_panel_ids}
    split_name_by_sid.update({sid: "B" for sid in test_b_panel_ids})
    split_name_by_sid.update({sid: "C" for sid in test_c_panel_ids})

    print(f"Eval mode: {'QUICK' if QUICK_MODE else 'FULL'}")
    print(f"Cells used per p: {n_eval_cells} / {base_n_cells}")
    print(f"p values used: {eval_p_values.tolist()}")
    print(f"Exact corruption seeding: {EXACT_CORRUPTION}")

    gene_mean_train = compute_gene_mean_streaming(
        panel_data=panel_data,
        panel_ids=train_panel_ids,
        common_genes=gene_names,
        chunk_size=IO_CHUNK_SIZE,
    )
    print("Computed gene-mean baseline from train panels.")

    METHOD_VAE = "VAE (NB expected mean)"
    METHOD_ID = "Identity baseline (input passthrough)"
    METHOD_MEAN = "Gene-mean baseline (train)"
    METHODS = [METHOD_VAE, METHOD_ID, METHOD_MEAN]

    overall_acc = {m: init_metric_acc() for m in METHODS}
    per_p_acc = {(int(v), m): init_metric_acc() for v in range(len(eval_p_values)) for m in METHODS}
    per_split_acc = {
        (int(v), split_name, m): init_metric_acc()
        for v in range(len(eval_p_values))
        for split_name in ["A", "B", "C"]
        for m in METHODS
    }

    loss_sum_weighted = 0.0
    recon_sum_weighted = 0.0
    kl_sum_weighted = 0.0
    loss_weight = 0

    use_cuda = device.type == "cuda"

    for version_idx, p_val in enumerate(eval_p_values):
        processed_cells = 0

        eval_iter = tqdm(
            test_accessor.iter_blocks(
                chunk_size=IO_CHUNK_SIZE,
                selected_global_idx=selected_cell_idx,
                split_name_by_sid=split_name_by_sid,
            ),
            desc=f"Evaluating p={float(p_val):.3f}",
            unit="chunk",
        )

        for payload in eval_iter:
            global_idx_np = payload["global_idx"]
            yb_np = payload["Y"]
            split_name = payload["split"]

            xb_np = corrupt_batch_deterministic(
                x_clean=yb_np,
                global_idx_np=global_idx_np,
                version_idx=version_idx,
                p_val=float(p_val),
                base_seed=base_seed,
                exact=EXACT_CORRUPTION,
            )

            xb = torch.from_numpy(xb_np).to(device, non_blocking=use_cuda)
            yb = torch.from_numpy(yb_np).to(device, non_blocking=use_cuda)

            with torch.inference_mode():
                with autocast_context(use_cuda):
                    recon, mu, logvar = model(xb)
                    t_loss, t_recon, t_kl = vae_loss(
                        recon,
                        yb,
                        mu,
                        logvar,
                        beta=beta,
                        loss_mask=None,
                        theta_param=log_theta,
                    )
                recon_expected = F.softplus(recon).float().cpu().numpy().astype(np.float32)

            batch_n = int(yb_np.shape[0])
            processed_cells += batch_n
            eval_iter.set_postfix({"cells": processed_cells})

            loss_sum_weighted += float(t_loss.item()) * batch_n
            recon_sum_weighted += float(t_recon.item()) * batch_n
            kl_sum_weighted += float(t_kl.item()) * batch_n
            loss_weight += batch_n

            update_metric_acc(overall_acc[METHOD_VAE], recon_expected, yb_np)
            update_metric_acc(overall_acc[METHOD_ID], xb_np, yb_np)
            update_metric_acc(overall_acc[METHOD_MEAN], gene_mean_train[None, :], yb_np)

            update_metric_acc(per_p_acc[(version_idx, METHOD_VAE)], recon_expected, yb_np)
            update_metric_acc(per_p_acc[(version_idx, METHOD_ID)], xb_np, yb_np)
            update_metric_acc(per_p_acc[(version_idx, METHOD_MEAN)], gene_mean_train[None, :], yb_np)

            if split_name in {"A", "B", "C"}:
                update_metric_acc(per_split_acc[(version_idx, split_name, METHOD_VAE)], recon_expected, yb_np)
                update_metric_acc(per_split_acc[(version_idx, split_name, METHOD_ID)], xb_np, yb_np)
                update_metric_acc(per_split_acc[(version_idx, split_name, METHOD_MEAN)], gene_mean_train[None, :], yb_np)

    if loss_weight == 0:
        raise RuntimeError("No evaluation batches were produced.")

    overall_vae = finalize_metric_acc(overall_acc[METHOD_VAE])
    summary_table = pd.DataFrame([
        {"metric": "test_total_loss_nb", "value": loss_sum_weighted / loss_weight},
        {"metric": "test_recon_loss_nb", "value": recon_sum_weighted / loss_weight},
        {"metric": "test_kl", "value": kl_sum_weighted / loss_weight},
        {"metric": "test_RMSE_counts", "value": overall_vae["rmse_counts"]},
        {"metric": "test_R2_counts", "value": overall_vae["r2_counts"]},
        {"metric": "test_RMSE_log1p", "value": overall_vae["rmse_log1p"]},
        {"metric": "test_R2_log1p", "value": overall_vae["r2_log1p"]},
    ])
    print("\nGlobal test summary:")
    print(summary_table.to_string(index=False))
    summary_table.to_csv(save_dir / "eval_summary.csv", index=False)

    scoreboard_overall = build_scoreboard_from_acc_dict(overall_acc)
    print("\nOverall scoreboard:")
    print(scoreboard_overall.to_string(index=False))
    scoreboard_overall.to_csv(save_dir / "eval_scoreboard_overall.csv", index=False)

    save_bar_plot(
        scoreboard_overall,
        category_col="method",
        value_col="rmse_counts",
        title="Overall test-set comparison",
        ylabel="RMSE on counts (lower is better)",
        save_path=save_dir / "eval_scoreboard_overall_rmse_counts.png",
    )
    save_bar_plot(
        scoreboard_overall,
        category_col="method",
        value_col="r2_counts",
        title="Overall test-set comparison",
        ylabel="R² on counts (higher is better)",
        save_path=save_dir / "eval_scoreboard_overall_r2_counts.png",
    )

    per_p_rows = []
    per_p_scoreboards = []

    for version_idx, p_val in enumerate(eval_p_values):
        sb = build_scoreboard_from_acc_dict({
            METHOD_VAE: per_p_acc[(version_idx, METHOD_VAE)],
            METHOD_ID: per_p_acc[(version_idx, METHOD_ID)],
            METHOD_MEAN: per_p_acc[(version_idx, METHOD_MEAN)],
        }).copy()
        sb.insert(0, "p_non_overlap", float(p_val))
        per_p_scoreboards.append(sb)

        vae_row = finalize_metric_acc(per_p_acc[(version_idx, METHOD_VAE)])
        per_p_rows.append({
            "p_non_overlap": float(p_val),
            "n_cells": vae_row["n_cells"],
            "rmse_counts_vae": vae_row["rmse_counts"],
            "r2_counts_vae": vae_row["r2_counts"],
            "rmse_log1p_vae": vae_row["rmse_log1p"],
            "r2_log1p_vae": vae_row["r2_log1p"],
        })

    per_p_summary = pd.DataFrame(per_p_rows)
    scoreboard_by_p = pd.concat(per_p_scoreboards, ignore_index=True)

    print("\nPer-p summary:")
    print(per_p_summary.to_string(index=False))
    per_p_summary.to_csv(save_dir / "eval_per_p_summary.csv", index=False)

    print("\nPer-p scoreboard:")
    print(scoreboard_by_p.to_string(index=False))
    scoreboard_by_p.to_csv(save_dir / "eval_scoreboard_by_p.csv", index=False)

    split_rows = []
    for version_idx, p_val in enumerate(eval_p_values):
        for split_name in ["A", "B", "C"]:
            for method_name in METHODS:
                out = finalize_metric_acc(per_split_acc[(version_idx, split_name, method_name)])
                if out["n_cells"] == 0:
                    continue
                split_rows.append({
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
                })

    scoreboard_by_split = (
        pd.DataFrame(split_rows)
        .sort_values(["p_non_overlap", "split", "rmse_counts"], ascending=[True, True, True])
        .reset_index(drop=True)
    )
    print("\nPer-split scoreboard:")
    print(scoreboard_by_split.to_string(index=False))
    scoreboard_by_split.to_csv(save_dir / "eval_scoreboard_by_split.csv", index=False)

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
            save_path=save_dir / f"eval_split_{split_name}_rmse_counts.png",
        )
        save_bar_plot(
            df_best,
            category_col="method",
            value_col="r2_counts",
            title=f"Split {split_name} comparison (p={best_p:.2f})",
            ylabel="R² on counts",
            save_path=save_dir / f"eval_split_{split_name}_r2_counts.png",
        )


def run_pair_analysis(
    panel_data,
    gene_names,
    model,
    log_theta,
    device,
    save_dir,
):
    if PAIR_5K_ID not in panel_data:
        raise KeyError(f"{PAIR_5K_ID} not found in panel_data.")
    if PAIR_V1_ID not in panel_data:
        raise KeyError(f"{PAIR_V1_ID} not found in panel_data.")

    print(f"\nRunning pair analysis for {PAIR_5K_ID} -> {PAIR_V1_ID}")

    adata_5k = materialize_panel_in_gene_space(
        panel_data=panel_data,
        sample_id=PAIR_5K_ID,
        common_genes=gene_names,
        chunk_size=IO_CHUNK_SIZE,
    )
    adata_v1 = materialize_panel_in_gene_space(
        panel_data=panel_data,
        sample_id=PAIR_V1_ID,
        common_genes=gene_names,
        chunk_size=IO_CHUNK_SIZE,
    )

    print(f"{PAIR_5K_ID} cells x genes: {adata_5k.shape}")
    print(f"{PAIR_V1_ID} cells x genes: {adata_v1.shape}")

    present_5k = set(panel_data[PAIR_5K_ID]["gene_names"].astype(str))
    present_v1 = set(panel_data[PAIR_V1_ID]["gene_names"].astype(str))
    common_pair_genes = pd.Index(gene_names).astype(str)
    common_pair_genes = common_pair_genes.intersection(pd.Index(sorted(present_5k)), sort=False)
    common_pair_genes = common_pair_genes.intersection(pd.Index(sorted(present_v1)), sort=False)

    if len(common_pair_genes) == 0:
        raise ValueError("No overlapping genes found between pair panels and model gene space.")

    idx_pair = pd.Index(gene_names).get_indexer(common_pair_genes)

    X5k_in = np.clip(_to_dense_float32(adata_5k.X), 0.0, None)
    Xv1 = np.clip(_to_dense_float32(adata_v1.X), 0.0, None)

    X5k_recon = nb_expected_counts_batched(
        model=model,
        X_in=X5k_in,
        device=device,
        batch_size=PAIR_INFER_BATCH_SIZE,
    )

    X5k_in_pair = X5k_in[:, idx_pair]
    X5k_recon_pair = X5k_recon[:, idx_pair]
    Xv1_pair = Xv1[:, idx_pair]

    theta_vec = F.softplus(log_theta.detach()).cpu().numpy().astype(np.float32)
    theta_pair = theta_vec[idx_pair]

    mean_input = X5k_in_pair.mean(axis=0)
    mean_recon = X5k_recon_pair.mean(axis=0)
    mean_target = Xv1_pair.mean(axis=0)

    det_input = (X5k_in_pair > 0).mean(axis=0)
    det_target = (Xv1_pair > 0).mean(axis=0)

    nb_p0_pair = np.power(
        theta_pair[None, :] / (theta_pair[None, :] + np.clip(X5k_recon_pair, 0.0, None) + 1e-8),
        theta_pair[None, :],
    )
    det_recon = (1.0 - nb_p0_pair).mean(axis=0)

    rmse_mean_input, r2_mean_input = rmse_and_r2(mean_input, mean_target)
    rmse_mean_recon, r2_mean_recon = rmse_and_r2(mean_recon, mean_target)
    rmse_det_input, r2_det_input = rmse_and_r2(det_input, det_target)
    rmse_det_recon, r2_det_recon = rmse_and_r2(det_recon, det_target)

    pair_summary = pd.DataFrame([
        {"comparison": "input_vs_target_gene_mean", "rmse": rmse_mean_input, "r2": r2_mean_input},
        {"comparison": "recon_vs_target_gene_mean", "rmse": rmse_mean_recon, "r2": r2_mean_recon},
        {"comparison": "input_vs_target_detection", "rmse": rmse_det_input, "r2": r2_det_input},
        {"comparison": "recon_vs_target_detection", "rmse": rmse_det_recon, "r2": r2_det_recon},
    ])
    print("\nPair summary:")
    print(pair_summary.to_string(index=False))
    pair_summary.to_csv(save_dir / f"pair_summary_{PAIR_5K_ID}_to_{PAIR_V1_ID}.csv", index=False)

    gene_summary = pd.DataFrame({
        "gene": common_pair_genes.astype(str),
        "mean_input": mean_input,
        "mean_recon": mean_recon,
        "mean_target": mean_target,
        "detect_input": det_input,
        "detect_recon": det_recon,
        "detect_target": det_target,
        "abs_err_mean_input_vs_target": np.abs(mean_input - mean_target),
        "abs_err_mean_recon_vs_target": np.abs(mean_recon - mean_target),
    }).sort_values("mean_target", ascending=False).reset_index(drop=True)
    gene_summary.to_csv(save_dir / f"pair_gene_summary_{PAIR_5K_ID}_to_{PAIR_V1_ID}.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    plot_exact_scatter_with_fit(
        x=mean_target,
        y=mean_input,
        title=f"Gene mean: {PAIR_5K_ID} input vs {PAIR_V1_ID}",
        x_label=f"{PAIR_V1_ID} gene mean",
        y_label=f"{PAIR_5K_ID} input gene mean",
        color="#1f77b4",
        ax=axes[0, 0],
    )
    plot_exact_scatter_with_fit(
        x=mean_target,
        y=mean_recon,
        title=f"Gene mean: reconstruction vs {PAIR_V1_ID}",
        x_label=f"{PAIR_V1_ID} gene mean",
        y_label="Reconstructed gene mean",
        color="#E68613",
        ax=axes[0, 1],
    )
    plot_exact_scatter_with_fit(
        x=det_target,
        y=det_input,
        title=f"Detection: {PAIR_5K_ID} input vs {PAIR_V1_ID}",
        x_label=f"{PAIR_V1_ID} detection rate",
        y_label=f"{PAIR_5K_ID} input detection rate",
        color="#1f77b4",
        ax=axes[1, 0],
    )
    plot_exact_scatter_with_fit(
        x=det_target,
        y=det_recon,
        title=f"Detection: reconstruction vs {PAIR_V1_ID}",
        x_label=f"{PAIR_V1_ID} detection rate",
        y_label="Reconstructed detection rate",
        color="#E68613",
        ax=axes[1, 1],
    )

    fig.suptitle(f"Pair analysis: {PAIR_5K_ID} -> NB-VAE -> {PAIR_V1_ID}", y=1.002)
    plt.tight_layout()
    fig.savefig(save_dir / f"pair_scatter_{PAIR_5K_ID}_to_{PAIR_V1_ID}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    selected_genes = select_high_low_genes(
        gene_names=np.asarray(common_pair_genes, dtype=object),
        mean_target=mean_target,
        n_high=PAIR_TOP_HIGH_GENES,
        n_low=PAIR_TOP_LOW_GENES,
    )
    selected_gene_df = gene_summary[gene_summary["gene"].isin(selected_genes)].copy()
    selected_gene_df.to_csv(
        save_dir / f"pair_selected_genes_{PAIR_5K_ID}_to_{PAIR_V1_ID}.csv",
        index=False,
    )

    print("\nSelected genes for detailed plots:")
    print(selected_genes)

    gene_to_idx_pair = {g: i for i, g in enumerate(common_pair_genes.astype(str))}

    plot_selected_gene_histograms(
        selected_genes=selected_genes,
        gene_to_idx=gene_to_idx_pair,
        X_input=X5k_in_pair,
        X_recon=X5k_recon_pair,
        X_target=Xv1_pair,
        save_path=save_dir / f"pair_gene_histograms_{PAIR_5K_ID}_to_{PAIR_V1_ID}.png",
        label_input=f"{PAIR_5K_ID} input",
        label_recon=f"{PAIR_5K_ID} through NB-VAE",
        label_target=f"{PAIR_V1_ID} target",
    )

    adata_5k_pair = adata_5k[:, common_pair_genes].copy()
    adata_v1_pair = adata_v1[:, common_pair_genes].copy()

    plot_selected_gene_spatial_triplets(
        selected_genes=selected_genes,
        gene_to_idx=gene_to_idx_pair,
        adata_input=adata_5k_pair,
        X_recon=X5k_recon_pair,
        adata_target=adata_v1_pair,
        save_path=save_dir / f"pair_gene_spatial_{PAIR_5K_ID}_to_{PAIR_V1_ID}.png",
        label_input=f"{PAIR_5K_ID} input",
        label_recon=f"{PAIR_5K_ID} through NB-VAE",
        label_target=f"{PAIR_V1_ID} target",
    )


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

    state = ckpt["model_state_dict"]
    if "enc_fc1.weight" not in state or "mu.weight" not in state or "out.weight" not in state:
        raise KeyError("Checkpoint model_state_dict is missing expected VAE layer weights.")

    hidden_dim_ckpt = int(state["enc_fc1.weight"].shape[0])
    latent_dim_ckpt = int(state["mu.weight"].shape[0])
    input_dim_ckpt = int(state["out.weight"].shape[0])

    if "gene_names" not in ckpt:
        raise KeyError("Checkpoint is missing `gene_names`; needed to fix the evaluation gene order.")
    gene_names = np.asarray(ckpt["gene_names"], dtype=object)

    if gene_names.size != input_dim_ckpt:
        raise ValueError(
            f"Checkpoint mismatch: output dim={input_dim_ckpt}, gene_names={gene_names.size}"
        )

    model = VAE(
        input_dim=input_dim_ckpt,
        latent_dim=latent_dim_ckpt,
        hidden_dim=hidden_dim_ckpt,
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    if "log_theta" in ckpt:
        init_theta = ckpt["log_theta"].to(device=device, dtype=torch.float32)
    else:
        init_theta = torch.zeros(input_dim_ckpt, device=device, dtype=torch.float32)

    log_theta = nn.Parameter(init_theta.clone().detach())

    print(
        f"Loaded checkpoint | input={input_dim_ckpt}, hidden={hidden_dim_ckpt}, latent={latent_dim_ckpt}"
    )
    if ckpt.get("best_epoch") is not None:
        print(f"Checkpoint best_epoch: {ckpt['best_epoch']}")
    if ckpt.get("best_val_loss") is not None:
        print(f"Checkpoint best_val_loss: {float(ckpt['best_val_loss']):.6f}")

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

    requested_ids = sorted(set(train_panel_ids + val_panel_ids + test_panel_ids + [PAIR_5K_ID, PAIR_V1_ID]))

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

    train_panel_ids = [sid for sid in train_panel_ids if sid in panel_data]
    val_panel_ids = [sid for sid in val_panel_ids if sid in panel_data]
    test_panel_ids = [sid for sid in test_panel_ids if sid in panel_data]
    test_a_panel_ids = [sid for sid in test_a_panel_ids if sid in panel_data]
    test_b_panel_ids = [sid for sid in test_b_panel_ids if sid in panel_data]
    test_c_panel_ids = [sid for sid in test_c_panel_ids if sid in panel_data]

    if len(train_panel_ids) == 0:
        raise ValueError("No train panels are present in `panel_data`.")
    if len(test_panel_ids) == 0:
        raise ValueError("No test panels are present in `panel_data`.")

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

    print(f"Loaded requested panels: {len(panel_data)}")
    print(f"Read mode: backed ({READ_MODE})")
    print(f"Model gene space from checkpoint: {len(gene_names)}")

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

    if RUN_GLOBAL_TEST_EVAL:
        run_global_test_evaluation(
            panel_data=panel_data,
            train_panel_ids=train_panel_ids,
            test_panel_ids=test_panel_ids,
            test_a_panel_ids=test_a_panel_ids,
            test_b_panel_ids=test_b_panel_ids,
            test_c_panel_ids=test_c_panel_ids,
            gene_names=gene_names,
            model=model,
            log_theta=log_theta,
            device=device,
            save_dir=SAVE_DIR,
        )

    if RUN_PAIR_ANALYSIS:
        run_pair_analysis(
            panel_data=panel_data,
            gene_names=gene_names,
            model=model,
            log_theta=log_theta,
            device=device,
            save_dir=SAVE_DIR,
        )

    elapsed_min = (time.time() - t0) / 60.0
    print(f"\nDone. Total elapsed time: {elapsed_min:.2f} minutes")
    print(f"Outputs written to: {SAVE_DIR}")


if __name__ == "__main__":
    main()