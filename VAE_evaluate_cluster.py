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
SAVE_DIR = Path(os.environ.get("OUTPUT_DIR", PROJECT_ROOT / "outputs" / "test_run")).resolve()
SAVE_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS_PATH = Path(
    os.environ.get(
        "WEIGHTS_PATH",
        str(SAVE_DIR / "VAE_ZINB_weights"),
    )
).resolve()

READ_MODE = os.environ.get("READ_MODE", "r") or "r"


# Evaluation options
QUICK_MODE = False
QUICK_MAX_CELLS = None           # e.g. 20000 for a quick smoke test
QUICK_P_VALUES = [0.19]
FULL_P_VALUES = [0.19]

EVAL_BATCH_SIZE = 1024           # kept for compatibility / pair inference sizes
IO_CHUNK_SIZE = 4096             # backed h5ad read chunk size
PAIR_INFER_BATCH_SIZE = 1024

# Keep exact old corruption seeding, or switch off for more speed.
EXACT_CORRUPTION = True

RUN_GLOBAL_TEST_EVAL = True
RUN_STRICT_PAIR_ANALYSIS = True
PAIR_5K_ID = "TENX189"
PAIR_V1_ID = "TENX190"
PAIR_TOP_GENES = 10


# Fixed split definitions / filtering setup
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
split_samples_test_a = ["NCBI882", "NCBI883", "NCBI884"]
split_samples_test_b = [
    "NCBI887", "NCBI888", "TENX189", # panel
    # "NCBI885", "NCBI886", # panel
    #"NCBI859", # panel
    # "TENX118", # panel
    # "TENX141", # panel
    # "TENX190", # panel
]
split_samples_test_c = [
    "TENX191", "TENX192", "TENX193", "TENX194",
    "TENX195", "TENX196", "TENX197", "TENX198"
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


def init_err_acc():
    return {
        "sq_counts": 0.0,
        "sq_log1p": 0.0,
        "n_elements": 0,
        "n_cells": 0,
    }


def update_err_acc(acc, pred, true):
    pred64 = np.asarray(pred, dtype=np.float64)
    true64 = np.asarray(true, dtype=np.float64)

    acc["sq_counts"] += np.square(pred64 - true64).sum()
    acc["sq_log1p"] += np.square(np.log1p(pred64) - np.log1p(true64)).sum()
    acc["n_elements"] += pred64.size
    acc["n_cells"] += pred64.shape[0]


def finalize_err_acc(acc):
    if acc["n_elements"] == 0:
        return {"mse_counts": np.nan, "mse_log1p": np.nan, "n_cells": 0}
    return {
        "mse_counts": float(acc["sq_counts"] / acc["n_elements"]),
        "mse_log1p": float(acc["sq_log1p"] / acc["n_elements"]),
        "n_cells": int(acc["n_cells"]),
    }


def build_scoreboard_from_acc_dict(acc_dict):
    rows = []
    for method_name, acc in acc_dict.items():
        out = finalize_err_acc(acc)
        rows.append({
            "method": method_name,
            "mse_counts": out["mse_counts"],
            "mse_log1p": out["mse_log1p"],
            "n_cells": out["n_cells"],
        })
    return pd.DataFrame(rows).sort_values("mse_counts", ascending=True).reset_index(drop=True)


# Model
class VAE(nn.Module):
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
    zi_logits=None,
    eps=1e-8,
):
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

    if zi_logits is not None:
        pi = torch.sigmoid(zi_logits).unsqueeze(0).expand_as(mu_counts)
        zero_mask = target_counts < eps

        nb_log_prob_zero = theta * (torch.log(theta + eps) - log_theta_mu)

        zero_log_prob = torch.logaddexp(
            torch.log(pi + eps),
            torch.log1p(-pi + eps) + nb_log_prob_zero,
        )
        nonzero_log_prob = torch.log1p(-pi + eps) + nb_log_prob

        recon_nll = -torch.where(zero_mask, zero_log_prob, nonzero_log_prob)
    else:
        recon_nll = -nb_log_prob

    if loss_mask is not None:
        recon_loss = recon_nll[:, loss_mask].mean()
    else:
        recon_loss = recon_nll.mean()

    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, recon_loss, kl


# Gene-space mapping + block loading
def attach_common_gene_mappings(panel_data, common_genes):
    common_genes = pd.Index(common_genes).astype(str)

    for sid, rec in panel_data.items():
        idx = rec["gene_names"].get_indexer(common_genes)
        present_mask = idx >= 0

        filtered_pos = idx[present_mask].astype(np.int64, copy=False)
        panel_pos = rec["gene_pos_all"][filtered_pos]
        common_pos = np.flatnonzero(present_mask).astype(np.int64, copy=False)

        if panel_pos.size == 0:
            raise ValueError(f"Panel {sid} has zero overlap with the evaluation gene space.")

        order = np.argsort(panel_pos)
        rec["mapped_panel_pos"] = panel_pos[order]
        rec["mapped_output_pos"] = common_pos[order]
        rec["mapped_n_genes"] = int(common_genes.size)


def load_block_in_common_gene_space(rec, local_indices, n_genes):
    local_indices = np.asarray(local_indices, dtype=np.int64)
    if local_indices.size == 0:
        return np.empty((0, n_genes), dtype=np.float32)

    rows = rec["cell_pos"][local_indices]
    mapped_panel_pos = rec["mapped_panel_pos"]
    mapped_output_pos = rec["mapped_output_pos"]
    ad_panel = rec["adata"]

    try:
        block = ad_panel[rows, mapped_panel_pos].X
        block = _to_dense_float32(block)
    except Exception:
        block_rows = []
        for r in rows:
            row = ad_panel.X[int(r), mapped_panel_pos]
            block_rows.append(_to_dense_float32(row).ravel())
        block = np.vstack(block_rows).astype(np.float32, copy=False)

    Y_block = np.zeros((block.shape[0], n_genes), dtype=np.float32)
    Y_block[:, mapped_output_pos] = block
    return np.clip(Y_block, 0.0, None)


def iter_panel_blocks(panel_data, panel_ids, n_genes, chunk_size, selected_global_idx=None, split_name_by_sid=None):
    selected_global_idx = None if selected_global_idx is None else np.asarray(selected_global_idx, dtype=np.int64)
    global_offset = 0

    for sid in panel_ids:
        rec = panel_data[sid]
        n_rows = int(rec["cell_pos"].size)

        if selected_global_idx is None:
            for start in range(0, n_rows, chunk_size):
                stop = min(start + chunk_size, n_rows)
                local_idx = np.arange(start, stop, dtype=np.int64)
                Y_block = load_block_in_common_gene_space(rec, local_idx, n_genes)
                yield {
                    "sample_id": sid,
                    "split": None if split_name_by_sid is None else split_name_by_sid.get(sid),
                    "global_idx": global_offset + local_idx,
                    "Y": Y_block,
                }
        else:
            lo = np.searchsorted(selected_global_idx, global_offset, side="left")
            hi = np.searchsorted(selected_global_idx, global_offset + n_rows, side="left")
            if hi > lo:
                local_selected = selected_global_idx[lo:hi] - global_offset
                for start in range(0, local_selected.size, chunk_size):
                    stop = min(start + chunk_size, local_selected.size)
                    local_idx = local_selected[start:stop]
                    Y_block = load_block_in_common_gene_space(rec, local_idx, n_genes)
                    yield {
                        "sample_id": sid,
                        "split": None if split_name_by_sid is None else split_name_by_sid.get(sid),
                        "global_idx": global_offset + local_idx,
                        "Y": Y_block,
                    }

        global_offset += n_rows


def compute_gene_mean_streaming(panel_data, panel_ids, n_genes, chunk_size):
    gene_sum = np.zeros(n_genes, dtype=np.float64)
    n_cells = 0

    for payload in tqdm(
        iter_panel_blocks(panel_data, panel_ids, n_genes=n_genes, chunk_size=chunk_size),
        desc="Computing train gene mean",
        unit="chunk",
    ):
        Y = payload["Y"].astype(np.float64, copy=False)
        gene_sum += Y.sum(axis=0)
        n_cells += Y.shape[0]

    if n_cells == 0:
        raise RuntimeError("No train cells found while computing gene mean.")
    return (gene_sum / n_cells).astype(np.float32)


# Corruption
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


# Plotting
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
    n_genes = x.size
    if n_genes < 2:
        raise ValueError("Need at least 2 valid points to fit a line.")

    slope, intercept = np.polyfit(x, y, 1)
    pearson_r = float(np.corrcoef(x, y)[0, 1])

    line_max = float(max(x.max(), y.max())) * (1.0 + line_pad) if n_genes > 0 else 1.0
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
        text_loc[0], text_loc[1],
        f"genes={n_genes}\nPearson r={pearson_r:.3f}\nfit: y={slope:.3f}x+{intercept:.3f}",
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
        "n_genes": int(n_genes),
        "pearson_r": pearson_r,
        "slope": float(slope),
        "intercept": float(intercept),
        "line_max": line_max,
    }


# Pair analysis helpers
def materialize_filtered_panel(panel_data, sample_id, dtype=np.float32, chunk_size=4096):
    rec = panel_data[sample_id]
    rows = np.asarray(rec["cell_pos"], dtype=np.int64)
    cols = np.asarray(rec["gene_pos_all"], dtype=np.int64)
    ad_panel = rec["adata"]

    X = np.empty((rows.size, cols.size), dtype=dtype)

    for start in tqdm(range(0, rows.size, chunk_size), desc=f"Materializing {sample_id}", unit="chunk"):
        stop = min(start + chunk_size, rows.size)
        block_rows = rows[start:stop]

        try:
            block = ad_panel[block_rows, cols].X
            block = _to_dense_float32(block)
        except Exception:
            block_list = []
            for r in block_rows:
                row = ad_panel.X[int(r), cols]
                block_list.append(_to_dense_float32(row).ravel())
            block = np.vstack(block_list).astype(dtype, copy=False)

        X[start:stop] = np.clip(block, 0.0, None)

    obs = ad_panel.obs.iloc[rows].copy()
    obs_names = pd.Index(ad_panel.obs_names.astype(str))[rows].copy()
    var_names = pd.Index(rec["gene_names"]).astype(str)

    adata_out = ad.AnnData(
        X=X,
        obs=obs,
        var=pd.DataFrame(index=var_names),
    )
    adata_out.obs_names = obs_names
    return adata_out


def zinb_expected_counts_batched(model, X_in, logit_pi, device, batch_size=1024):
    model.eval()
    pi_vec = torch.sigmoid(logit_pi.detach()).cpu().numpy().astype(np.float32)

    out_blocks = []
    with torch.inference_mode():
        for start in tqdm(range(0, X_in.shape[0], batch_size), desc="ZINB inference", unit="batch"):
            stop = min(start + batch_size, X_in.shape[0])
            xin = torch.from_numpy(X_in[start:stop]).to(device, non_blocking=torch.cuda.is_available())
            with autocast_context(torch.cuda.is_available()):
                raw_recon, _, _ = model(xin)
            mu_nb = F.softplus(raw_recon).float().cpu().numpy().astype(np.float32)
            out_blocks.append(mu_nb * (1.0 - pi_vec[None, :]))

    return np.vstack(out_blocks), pi_vec


def run_pair_analysis(panel_data, gene_names, model, log_theta, logit_pi, device, save_dir):
    if PAIR_5K_ID not in panel_data:
        raise KeyError(f"{PAIR_5K_ID} not found in panel_data.")
    if PAIR_V1_ID not in panel_data:
        raise KeyError(f"{PAIR_V1_ID} not found in panel_data.")

    adata_5k = materialize_filtered_panel(panel_data, PAIR_5K_ID, chunk_size=IO_CHUNK_SIZE)
    adata_v1 = materialize_filtered_panel(panel_data, PAIR_V1_ID, chunk_size=IO_CHUNK_SIZE)

    print(f"{PAIR_5K_ID} cells x genes: {adata_5k.shape}")
    print(f"{PAIR_V1_ID} cells x genes: {adata_v1.shape}")

    genes_model = np.array(gene_names, dtype=object)
    common_5k_model = np.intersect1d(genes_model, adata_5k.var_names.values)
    common_v1_model = np.intersect1d(genes_model, adata_v1.var_names.values)

    print(f"Model genes: {len(genes_model)}")
    print(f"{PAIR_5K_ID} genes in model: {len(common_5k_model)}")
    print(f"{PAIR_V1_ID} genes in model: {len(common_v1_model)}")

    if len(common_5k_model) == 0:
        raise ValueError(f"No overlap between {PAIR_5K_ID} genes and model gene space.")
    if len(common_v1_model) == 0:
        raise ValueError(f"No overlap between {PAIR_V1_ID} genes and model gene space.")

    X5k_in = np.zeros((adata_5k.n_obs, len(genes_model)), dtype=np.float32)
    model_pos = pd.Index(genes_model).get_indexer(common_5k_model)
    g5k_pos = pd.Index(adata_5k.var_names).get_indexer(common_5k_model)
    X5k_in[:, model_pos] = _to_dense_float32(adata_5k.X[:, g5k_pos])
    X5k_in = np.clip(X5k_in, 0.0, None)

    X5k_recon_mu, pi_vec = zinb_expected_counts_batched(
        model=model,
        X_in=X5k_in,
        logit_pi=logit_pi,
        device=device,
        batch_size=PAIR_INFER_BATCH_SIZE,
    )

    strict_eval_genes = common_v1_model
    idx_model_eval = pd.Index(genes_model).get_indexer(strict_eval_genes)
    idx_v1_eval = pd.Index(adata_v1.var_names).get_indexer(strict_eval_genes)

    X_recon_eval = X5k_recon_mu[:, idx_model_eval]
    X_v1_eval = _to_dense_float32(adata_v1.X[:, idx_v1_eval])

    finite_recon = np.isfinite(X_recon_eval).all()
    finite_v1 = np.isfinite(X_v1_eval).all()
    if not (finite_recon and finite_v1):
        raise RuntimeError("Non-finite values detected in strict pair comparison arrays.")

    theta_vec = F.softplus(log_theta.detach()).cpu().numpy().astype(np.float32)
    theta_eval = theta_vec[idx_model_eval]
    pi_eval = pi_vec[idx_model_eval]

    mean_recon = X_recon_eval.mean(axis=0)
    mean_v1 = X_v1_eval.mean(axis=0)

    nb_p0 = np.power(theta_eval[None, :] / (theta_eval[None, :] + X_recon_eval + 1e-8), theta_eval[None, :])
    p0 = pi_eval[None, :] + (1.0 - pi_eval[None, :]) * nb_p0
    det_recon = (1.0 - p0).mean(axis=0)
    det_v1 = (X_v1_eval > 0).mean(axis=0)

    compare_zinb = pd.DataFrame([
        {"metric": "MSE_gene_mean_counts", "value": float(np.mean((mean_recon - mean_v1) ** 2))},
        {"metric": "MSE_gene_mean_log1p", "value": float(np.mean((np.log1p(mean_recon) - np.log1p(mean_v1)) ** 2))},
        {"metric": "MSE_gene_detection_rate_zinb_implied", "value": float(np.mean((det_recon - det_v1) ** 2))},
    ])
    print("\nStrict pair reconstruction summary:")
    print(compare_zinb.to_string(index=False))
    compare_zinb.to_csv(save_dir / f"strict_pair_summary_{PAIR_5K_ID}_to_{PAIR_V1_ID}.csv", index=False)

    common_triplet = pd.Index(adata_5k.var_names.astype(str))
    common_triplet = common_triplet.intersection(pd.Index(adata_v1.var_names.astype(str)))
    common_triplet = common_triplet.intersection(pd.Index(genes_model.astype(str)))

    a5 = adata_5k[:, common_triplet]
    av = adata_v1[:, common_triplet]

    idx_triplet_model = pd.Index(genes_model).get_indexer(common_triplet)
    xr_expected = X5k_recon_mu[:, idx_triplet_model]

    x5 = _to_dense_float32(a5.X)
    xv = _to_dense_float32(av.X)

    mean_5k = x5.mean(axis=0)
    mean_v1_triplet = xv.mean(axis=0)
    mean_recon_triplet = xr_expected.mean(axis=0)

    det_5k = (x5 > 0).mean(axis=0)
    det_v1_triplet = (xv > 0).mean(axis=0)

    theta_triplet = theta_vec[idx_triplet_model]
    pi_triplet = pi_vec[idx_triplet_model]
    mu_nb_triplet = xr_expected / (1.0 - pi_triplet[None, :] + 1e-8)
    nb_p0_triplet = np.power(
        theta_triplet[None, :] / (theta_triplet[None, :] + np.clip(mu_nb_triplet, 0.0, None) + 1e-8),
        theta_triplet[None, :],
    )
    p0_triplet = pi_triplet[None, :] + (1.0 - pi_triplet[None, :]) * nb_p0_triplet
    det_recon_triplet = (1.0 - p0_triplet).mean(axis=0)

    fig, axes = plt.subplots(3, 2, figsize=(14, 16))

    plot_exact_scatter_with_fit(
        x=mean_v1_triplet, y=mean_recon_triplet,
        title=f"Gene mean: reconstructed {PAIR_5K_ID} vs {PAIR_V1_ID}",
        x_label=f"{PAIR_V1_ID} gene mean",
        y_label=f"Reconstructed {PAIR_5K_ID} gene mean",
        color="#E68613",
        ax=axes[0, 0],
    )
    plot_exact_scatter_with_fit(
        x=det_v1_triplet, y=det_recon_triplet,
        title=f"Detection: reconstructed {PAIR_5K_ID} vs {PAIR_V1_ID}",
        x_label=f"{PAIR_V1_ID} detection rate",
        y_label=f"Reconstructed {PAIR_5K_ID} detection rate",
        color="#E68613",
        ax=axes[0, 1],
    )
    plot_exact_scatter_with_fit(
        x=mean_5k, y=mean_recon_triplet,
        title=f"Gene mean: reconstructed {PAIR_5K_ID} vs original {PAIR_5K_ID}",
        x_label=f"Original {PAIR_5K_ID} gene mean",
        y_label=f"Reconstructed {PAIR_5K_ID} gene mean",
        color="#2ca02c",
        ax=axes[1, 0],
    )
    plot_exact_scatter_with_fit(
        x=det_5k, y=det_recon_triplet,
        title=f"Detection: reconstructed {PAIR_5K_ID} vs original {PAIR_5K_ID}",
        x_label=f"Original {PAIR_5K_ID} detection rate",
        y_label=f"Reconstructed {PAIR_5K_ID} detection rate",
        color="#2ca02c",
        ax=axes[1, 1],
    )
    plot_exact_scatter_with_fit(
        x=mean_v1_triplet, y=mean_5k,
        title=f"Gene mean: {PAIR_V1_ID} vs original {PAIR_5K_ID}",
        x_label=f"{PAIR_V1_ID} gene mean",
        y_label=f"Original {PAIR_5K_ID} gene mean",
        color="#1f77b4",
        ax=axes[2, 0],
    )
    plot_exact_scatter_with_fit(
        x=det_v1_triplet, y=det_5k,
        title=f"Detection: {PAIR_V1_ID} vs original {PAIR_5K_ID}",
        x_label=f"{PAIR_V1_ID} detection rate",
        y_label=f"Original {PAIR_5K_ID} detection rate",
        color="#1f77b4",
        ax=axes[2, 1],
    )

    fig.suptitle(
        f"Shared-gene comparisons: {PAIR_5K_ID} -> VAE -> {PAIR_V1_ID} (n_genes={len(common_triplet)})",
        y=1.002,
    )
    plt.tight_layout()
    fig.savefig(save_dir / f"strict_pair_scatter_{PAIR_5K_ID}_to_{PAIR_V1_ID}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    gene_priority = np.argsort(mean_v1)[::-1]
    top_k = min(PAIR_TOP_GENES, len(strict_eval_genes))
    plot_genes = [strict_eval_genes[i] for i in gene_priority[:top_k]]

    eval_gene_pos = {g: i for i, g in enumerate(strict_eval_genes)}
    rng = np.random.default_rng(42)

    n_rows = len(plot_genes)
    fig, axs = plt.subplots(n_rows, 2, figsize=(12, 3.2 * n_rows), squeeze=False)
    summary_rows = []

    for i, gene in enumerate(plot_genes):
        g_idx = eval_gene_pos[gene]
        xv_all = X_v1_eval[:, g_idx]

        mu_g = np.clip(X_recon_eval[:, g_idx].astype(np.float32), 0.0, None)
        theta_g = float(theta_eval[g_idx])
        pi_g = float(pi_eval[g_idx])

        lam = rng.gamma(shape=theta_g, scale=mu_g / (theta_g + 1e-8))
        x_first_all = rng.poisson(lam).astype(np.float32)
        drop_mask = rng.random(x_first_all.shape[0]) < pi_g
        x_first_all[drop_mask] = 0.0

        det_model_implied = float(
            (
                1.0
                - (
                    pi_g
                    + (1.0 - pi_g)
                    * np.power(
                        theta_g / (theta_g + np.clip(X_recon_eval[:, g_idx], 0.0, None) + 1e-8),
                        theta_g,
                    )
                )
            ).mean()
        )

        x_first_pos = x_first_all[x_first_all > 0]
        xv_pos = xv_all[xv_all > 0]

        x_first_plot_all = np.log1p(np.clip(x_first_all, 0.0, None))
        xv_plot_all = np.log1p(np.clip(xv_all, 0.0, None))
        x_first_plot_pos = np.log1p(np.clip(x_first_pos, 0.0, None))
        xv_plot_pos = np.log1p(np.clip(xv_pos, 0.0, None))

        axL = axs[i, 0]
        axL.hist(x_first_plot_all, bins=30, density=True, alpha=0.4, label=f"{PAIR_5K_ID} through VAE (n={len(x_first_plot_all)})")
        axL.hist(xv_plot_all, bins=30, density=True, alpha=0.4, label=f"{PAIR_V1_ID} original (n={len(xv_plot_all)})")
        axL.set_title(f"{gene} | all cells")
        axL.legend(frameon=True)

        axR = axs[i, 1]
        if x_first_plot_pos.size > 0:
            axR.hist(x_first_plot_pos, bins=30, density=True, alpha=0.4, label=f"{PAIR_5K_ID} through VAE >0 (n={len(x_first_plot_pos)})")
        if xv_plot_pos.size > 0:
            axR.hist(xv_plot_pos, bins=30, density=True, alpha=0.4, label=f"{PAIR_V1_ID} original >0 (n={len(xv_plot_pos)})")
        axR.set_title(f"{gene} | >0 only")
        axR.legend(frameon=True)

        summary_rows.append(
            {
                "gene": gene,
                f"mean_{PAIR_5K_ID}_through_VAE": float(np.mean(x_first_all)),
                f"mean_{PAIR_V1_ID}_original": float(np.mean(xv_all)),
                f"detect_pct_{PAIR_5K_ID}_sampled": float(100.0 * np.mean(x_first_all > 0)),
                f"detect_pct_{PAIR_5K_ID}_model_implied": float(100.0 * det_model_implied),
                f"detect_pct_{PAIR_V1_ID}_original": float(100.0 * np.mean(xv_all > 0)),
            }
        )

    fig.suptitle(f"Per-gene distributions: {PAIR_5K_ID} through VAE vs {PAIR_V1_ID} original", y=1.002)
    plt.tight_layout()
    fig.savefig(save_dir / f"strict_pair_gene_hists_{PAIR_5K_ID}_to_{PAIR_V1_ID}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(save_dir / f"strict_pair_gene_summary_{PAIR_5K_ID}_to_{PAIR_V1_ID}.csv", index=False)
    print("\nPer-gene summary:")
    print(summary_df.to_string(index=False))


# Main
def main():
    t0 = time.time()

    torch.set_float32_matmul_precision("high")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    print(f"Running on device: {device}")
    print(f"Checkpoint path: {WEIGHTS_PATH}")

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
            f"Checkpoint mismatch: out dim={input_dim_ckpt}, gene_names={gene_names.size}"
        )

    model = VAE(
        input_dim=input_dim_ckpt,
        latent_dim=latent_dim_ckpt,
        hidden_dim=hidden_dim_ckpt,
    ).to(device)
    model.load_state_dict(state, strict=True)

    if "log_theta" in ckpt:
        init_theta = ckpt["log_theta"].to(device=device, dtype=torch.float32)
    else:
        init_theta = torch.zeros(input_dim_ckpt, device=device, dtype=torch.float32)
    if "logit_pi" in ckpt:
        init_pi = ckpt["logit_pi"].to(device=device, dtype=torch.float32)
    else:
        init_pi = torch.zeros(input_dim_ckpt, device=device, dtype=torch.float32)

    log_theta = nn.Parameter(init_theta.clone().detach())
    logit_pi = nn.Parameter(init_pi.clone().detach())

    model.eval()

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

    attach_common_gene_mappings(panel_data, gene_names)

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
        eval_p_values = np.asarray(
            QUICK_P_VALUES if QUICK_MODE else FULL_P_VALUES,
            dtype=np.float64,
        )
        if np.any((eval_p_values < 0.0) | (eval_p_values > 1.0)):
            raise ValueError("All evaluation p values must be between 0 and 1.")

        base_n_cells = int(sum(panel_data[sid]["n_obs_filtered"] for sid in test_panel_ids))
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
            n_genes=len(gene_names),
            chunk_size=IO_CHUNK_SIZE,
        )
        print("Computed gene-mean baseline from train panels.")

        METHOD_VAE = "VAE (ZINB expected decode)"
        METHOD_ID = "Identity baseline (input passthrough)"
        METHOD_MEAN = "Gene-mean baseline (train)"
        METHODS = [METHOD_VAE, METHOD_ID, METHOD_MEAN]

        overall_acc = {m: init_err_acc() for m in METHODS}
        per_p_acc = {(int(v), m): init_err_acc() for v in range(len(eval_p_values)) for m in METHODS}
        per_split_acc = {
            (int(v), split_name, m): init_err_acc()
            for v in range(len(eval_p_values))
            for split_name in ["A", "B", "C"]
            for m in METHODS
        }

        loss_sum_weighted = 0.0
        recon_sum_weighted = 0.0
        kl_sum_weighted = 0.0
        loss_weight = 0

        pi_vec = torch.sigmoid(logit_pi.detach()).cpu().numpy().astype(np.float32)

        for version_idx, p_val in enumerate(eval_p_values):
            processed_cells = 0
            eval_iter = tqdm(
                iter_panel_blocks(
                    panel_data=panel_data,
                    panel_ids=test_panel_ids,
                    n_genes=len(gene_names),
                    chunk_size=IO_CHUNK_SIZE,
                    selected_global_idx=selected_cell_idx,
                    split_name_by_sid=split_name_by_sid,
                ),
                total=None,
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
                            zi_logits=logit_pi,
                        )
                    recon_mu = F.softplus(recon).float().cpu().numpy().astype(np.float32)

                recon_expected = recon_mu * (1.0 - pi_vec[None, :])

                batch_n = int(yb_np.shape[0])
                processed_cells += batch_n
                eval_iter.set_postfix({"cells": processed_cells})

                loss_sum_weighted += float(t_loss.item()) * batch_n
                recon_sum_weighted += float(t_recon.item()) * batch_n
                kl_sum_weighted += float(t_kl.item()) * batch_n
                loss_weight += batch_n

                update_err_acc(overall_acc[METHOD_VAE], recon_expected, yb_np)
                update_err_acc(overall_acc[METHOD_ID], xb_np, yb_np)
                update_err_acc(overall_acc[METHOD_MEAN], gene_mean_train[None, :], yb_np)

                update_err_acc(per_p_acc[(version_idx, METHOD_VAE)], recon_expected, yb_np)
                update_err_acc(per_p_acc[(version_idx, METHOD_ID)], xb_np, yb_np)
                update_err_acc(per_p_acc[(version_idx, METHOD_MEAN)], gene_mean_train[None, :], yb_np)

                if split_name in {"A", "B", "C"}:
                    update_err_acc(per_split_acc[(version_idx, split_name, METHOD_VAE)], recon_expected, yb_np)
                    update_err_acc(per_split_acc[(version_idx, split_name, METHOD_ID)], xb_np, yb_np)
                    update_err_acc(per_split_acc[(version_idx, split_name, METHOD_MEAN)], gene_mean_train[None, :], yb_np)

        if loss_weight == 0:
            raise RuntimeError("No evaluation batches were produced.")

        overall_vae = finalize_err_acc(overall_acc[METHOD_VAE])
        summary_table_nb = pd.DataFrame([
            {"metric": "test_total_loss_zinb", "value": loss_sum_weighted / loss_weight},
            {"metric": "test_recon_loss_zinb", "value": recon_sum_weighted / loss_weight},
            {"metric": "test_kl", "value": kl_sum_weighted / loss_weight},
            {"metric": "test_MSE_counts", "value": overall_vae["mse_counts"]},
            {"metric": "test_MSE_log1p", "value": overall_vae["mse_log1p"]},
        ])
        print("\nZINB test summary:")
        print(summary_table_nb.to_string(index=False))
        summary_table_nb.to_csv(SAVE_DIR / "eval_summary.csv", index=False)

        scoreboard_nb = build_scoreboard_from_acc_dict(overall_acc)
        print("\nOverall baseline scoreboard (lower is better):")
        print(scoreboard_nb.to_string(index=False))
        scoreboard_nb.to_csv(SAVE_DIR / "eval_scoreboard_overall.csv", index=False)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(scoreboard_nb["method"], scoreboard_nb["mse_counts"])
        ax.set_title("Overall test-set comparison")
        ax.set_ylabel("MSE on counts (lower is better)")
        ax.grid(axis="y", alpha=0.25)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        fig.savefig(SAVE_DIR / "eval_scoreboard_overall.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        per_p_rows = []
        per_p_scoreboards = []

        for version_idx, p_val in enumerate(eval_p_values):
            row_vae = finalize_err_acc(per_p_acc[(version_idx, METHOD_VAE)])
            row_id = finalize_err_acc(per_p_acc[(version_idx, METHOD_ID)])

            per_p_rows.append({
                "p_non_overlap": float(p_val),
                "n_cells": row_vae["n_cells"],
                "mse_counts_vae": row_vae["mse_counts"],
                "mse_log1p_vae": row_vae["mse_log1p"],
                "mse_counts_identity": row_id["mse_counts"],
                "mse_log1p_identity": row_id["mse_log1p"],
            })

            sb_v = build_scoreboard_from_acc_dict({
                METHOD_VAE: per_p_acc[(version_idx, METHOD_VAE)],
                METHOD_ID: per_p_acc[(version_idx, METHOD_ID)],
                METHOD_MEAN: per_p_acc[(version_idx, METHOD_MEAN)],
            }).copy()
            sb_v.insert(0, "p_non_overlap", float(p_val))
            per_p_scoreboards.append(sb_v)

        per_p_summary = pd.DataFrame(per_p_rows)
        scoreboard_by_p = pd.concat(per_p_scoreboards, ignore_index=True)

        print("\nPer-p VAE vs identity summary:")
        print(per_p_summary.to_string(index=False))
        per_p_summary.to_csv(SAVE_DIR / "eval_per_p_summary.csv", index=False)

        print("\nPer-p scoreboard:")
        print(scoreboard_by_p.to_string(index=False))
        scoreboard_by_p.to_csv(SAVE_DIR / "eval_scoreboard_by_p.csv", index=False)

        split_rows = []
        for version_idx, p_val in enumerate(eval_p_values):
            for split_name in ["A", "B", "C"]:
                for method_name in METHODS:
                    out = finalize_err_acc(per_split_acc[(version_idx, split_name, method_name)])
                    if out["n_cells"] == 0:
                        continue
                    split_rows.append({
                        "p_non_overlap": float(p_val),
                        "split": split_name,
                        "n_cells": out["n_cells"],
                        "method": method_name,
                        "mse_counts": out["mse_counts"],
                        "mse_log1p": out["mse_log1p"],
                    })

        scoreboard_by_split = (
            pd.DataFrame(split_rows)
            .sort_values(["p_non_overlap", "split", "mse_counts"], ascending=[True, True, True])
            .reset_index(drop=True)
        )
        print("\nPer-split scoreboard:")
        print(scoreboard_by_split.to_string(index=False))
        scoreboard_by_split.to_csv(SAVE_DIR / "eval_scoreboard_by_split.csv", index=False)

    if RUN_STRICT_PAIR_ANALYSIS:
        run_pair_analysis(
            panel_data=panel_data,
            gene_names=gene_names,
            model=model,
            log_theta=log_theta,
            logit_pi=logit_pi,
            device=device,
            save_dir=SAVE_DIR,
        )

    elapsed_min = (time.time() - t0) / 60.0
    print(f"\nDone. Total elapsed time: {elapsed_min:.2f} minutes")
    print(f"Outputs written to: {SAVE_DIR}")


if __name__ == "__main__":
    main()
