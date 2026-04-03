import json
import os
import sys
import time
from pathlib import Path

import anndata as ad
import annsel as an
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


# Load all saved Xenium samples as AnnData only (.h5ad)

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", "/tudelft.net/staff-umbrella/Xeniumenhancer")).resolve()
ANN_DIR = Path(os.environ.get("ANN_DIR", PROJECT_ROOT / "AnnData")).resolve()
SAVE_DIR = Path(os.environ.get("OUTPUT_DIR", PROJECT_ROOT / "outputs")).resolve()
SAVE_DIR.mkdir(parents=True, exist_ok=True)

READ_MODE = os.environ.get("READ_MODE") or None


def _adata_path(sample_id: str):
    return ANN_DIR / f"{sample_id}_xenium_cell_level.h5ad"


def _load_sample(sample_id: str, read_mode=READ_MODE):
    h5ad_path = _adata_path(sample_id)
    if not h5ad_path.exists():
        raise FileNotFoundError(f"Missing h5ad for {sample_id}: {h5ad_path}")
    return sc.read_h5ad(h5ad_path, backed=read_mode)


# Discover available sample IDs from written h5ad files
available_ids = sorted(
    p.name.replace("_xenium_cell_level.h5ad", "")
    for p in ANN_DIR.glob("*_xenium_cell_level.h5ad")
)

ids_to_load = available_ids
adatas_raw = {}
load_failed = []

for sample_id in ids_to_load:
    try:
        adatas_raw[sample_id] = _load_sample(sample_id)
    except Exception as e:
        load_failed.append((sample_id, str(e)))

# Working copy
adatas = dict(adatas_raw)

print(f"Loaded samples: {len(adatas_raw)}")
print(f"Read mode: {'backed (r)' if READ_MODE == 'r' else 'in-memory'}")
for sample_id, ad in adatas_raw.items():
    print(f"  {sample_id}: {ad.shape}")

if load_failed:
    print(f"Failed to load {len(load_failed)} sample(s):")
    for sid, msg in load_failed[:10]:
        print(f"  - {sid}: {msg}")


threshold = 40
genes_threshold = 5

# Filter every loaded panel/sample
adatas_f = {}
for panel_name, ad in adatas_raw.items():
    # Start from an in-memory object so slicing/copy works for both backed and non-backed AnnData.
    ad_qc = ad.to_memory() if getattr(ad, "isbacked", False) else ad.copy()

    # Ensure required QC fields exist for each panel.
    if not {"total_counts", "n_genes_by_counts"}.issubset(ad_qc.obs.columns):
        sc.pp.calculate_qc_metrics(ad_qc, inplace=True, percent_top=(50, 100))

    panel_mask = (
        (ad_qc.obs["total_counts"] >= threshold)
        & (ad_qc.obs["n_genes_by_counts"] > genes_threshold)
    )

    ad_cells = ad_qc[panel_mask].copy()
    gene_names = pd.Index(ad_cells.var_names).astype(str)
    keep_genes = (
    ~gene_names.str.startswith("UnassignedCodeword", na=False)
    & ~gene_names.str.lower().str.startswith("antisense", na=False)
    )
    adatas_f[panel_name] = ad_cells[:, keep_genes].copy()

# Refresh working copies used later in the notebook
adatas = dict(adatas_f)

# Report filtering impact for all panels
for panel_name in sorted(adatas_raw.keys()):
    n_before = adatas_raw[panel_name].n_obs
    n_after = adatas_f[panel_name].n_obs
    n_removed = n_before - n_after
    pct_removed = (100.0 * n_removed / n_before) if n_before else 0.0
    print(
        f"{panel_name}: before={n_before}, after={n_after}, "
        f"removed={n_removed} ({pct_removed:.1f}%)"
    )


def make_multiple_v1_like_corruptions(
    adata_input,
    p_non_overlap_values,
    base_seed=0,
):
    """
    Create multiple corrupted versions of one AnnData object by sweeping p values.
    For each p in p_non_overlap_values:
    1) Start with a per-gene base vector filled with p.
    2) Binomially thin each nonzero count entry.
    """
    p_values = np.atleast_1d(np.asarray(p_non_overlap_values, dtype=np.float64))

    if np.any((p_values < 0.0) | (p_values > 1.0)):
        raise ValueError("All p_non_overlap_values must be between 0 and 1.")

    X = adata_input.X
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)

    # Check if the structure of adata is correct
    if X.data.size > 0:
        if np.any(X.data < 0):
            raise ValueError("adata_input.X contains negative values (expected raw counts).")
        if np.issubdtype(X.data.dtype, np.floating):
            frac = np.abs(X.data - np.rint(X.data))
            if np.nanmax(frac) > 1e-6:
                raise ValueError("adata_input.X looks non-integer (use raw counts)")

    # Round the data to integers
    data_int = np.rint(X.data).astype(np.int64, copy=False)

    adata_versions = []
    p_versions = []
    p_base_versions = []

    # for each p value, create a new corrupted version of the input adata
    for rep, p_non_overlap in enumerate(p_values):
        p_base = np.full(adata_input.n_vars, float(p_non_overlap), dtype=np.float64)
        p_rep = p_base.copy()

        # bionomial thinning by drawwing fro the distiribution for each nonzero entry
        rng = np.random.default_rng(base_seed + rep)
        p_entry = p_rep[X.indices]
        data_new = rng.binomial(data_int, p_entry).astype(np.int64, copy=False)

        # Keep the CSR structure consistent, and drop the zeros
        X_new = sp.csr_matrix(
            (data_new, X.indices.copy(), X.indptr.copy()),
            shape=X.shape,
        )
        X_new.eliminate_zeros()

        # Create the new Anndata object and calculate the QC metrics
        adata_rep = adata_input.copy()
        adata_rep.X = X_new
        sc.pp.calculate_qc_metrics(adata_rep, inplace=True, percent_top=None)

        adata_versions.append(adata_rep)
        p_versions.append(p_rep)
        p_base_versions.append(p_base)

    return adata_versions, p_versions, p_base_versions

p_non_overlap_values = [0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.29, 0.31, 0.33, 0.35, 0.37, 0.40]

# Basic conditional-style VAE reconstruction (GPU-optimized)

def _to_dense_float32(X):
    if sp.issparse(X):
        return X.toarray().astype(np.float32)
    return np.asarray(X, dtype=np.float32)


class PanelRowAccessor:
    """Accesses rows from multiple AnnData panels in one shared gene order.
       Used by training and eval datasets to have consistent gene indexing for different panels """

    def __init__(self, panel_data, panel_ids, common_genes):
        self.panel_defs = []
        self.panel_sizes = []
        self.common_genes = pd.Index(common_genes).astype(str)

        # For each panel, we have to determine how its genes map to the common gene space
        for sid in panel_ids:
            ad_panel = panel_data[sid]
            # Match the panel genes to the shared genes so keep track of the position of the common genes
            idx = pd.Index(ad_panel.var_names).astype(str).get_indexer(self.common_genes)
            present_mask = idx >= 0

            # Get local panel columns for present genes 
            panel_pos = idx[present_mask].astype(np.int64, copy=False)
            # Get global output positions for those same genes
            common_pos = np.where(present_mask)[0].astype(np.int64, copy=False)

            if panel_pos.size == 0:
                raise ValueError(f"Panel {sid} has zero overlap with the training gene space.")

            # store the mappings and sizes (for when retrieving rows)
            self.panel_defs.append((ad_panel, panel_pos, common_pos))
            self.panel_sizes.append(int(ad_panel.n_obs))

        # After processing all panels, we can compute cumulative sizes for fast row indexing
        self.panel_sizes = np.asarray(self.panel_sizes, dtype=np.int64)
        self.cum_sizes = np.cumsum(self.panel_sizes)
        self.total_cells = int(self.cum_sizes[-1]) if self.cum_sizes.size else 0
        self.n_genes = int(len(self.common_genes))

    # Given a global cell index, determine which panel it belongs to and the local row index within that panel
    def _locate(self, global_cell_idx):
        if global_cell_idx < 0 or global_cell_idx >= self.total_cells:
            raise IndexError("Cell index out of range.")
        panel_idx = int(np.searchsorted(self.cum_sizes, global_cell_idx, side="right"))
        prev_cum = 0 if panel_idx == 0 else int(self.cum_sizes[panel_idx - 1])
        within_panel_idx = int(global_cell_idx - prev_cum)
        return panel_idx, within_panel_idx

    # Retrieves the specified row as a dense vector in the common gene space, filling missing genes with zeros
    def get_row_dense(self, global_cell_idx):
        panel_idx, row_idx = self._locate(global_cell_idx)
        ad_panel, panel_pos, common_pos = self.panel_defs[panel_idx]

        row = ad_panel.X[row_idx, panel_pos]
        if sp.issparse(row):
            row_vals = row.toarray().ravel().astype(np.float32, copy=False)
        else:
            row_vals = np.asarray(row, dtype=np.float32).ravel()

        y = np.zeros(self.n_genes, dtype=np.float32)
        y[common_pos] = row_vals
        return np.clip(y, 0.0, None)


class MultiVersionTrainDataset(Dataset):
    def __init__(self, panel_data, panel_ids, common_genes, p_non_overlap_values, base_seed=0):
        self.rows = PanelRowAccessor(panel_data, panel_ids, common_genes)
        self.n_train = self.rows.total_cells
        self.n_genes = self.rows.n_genes
        self.base_seed = int(base_seed)

        self.p_values = np.atleast_1d(np.asarray(p_non_overlap_values, dtype=np.float64))
        if self.p_values.size < 1:
            raise ValueError("p_non_overlap_values must contain at least one value.")
        if np.any((self.p_values < 0.0) | (self.p_values > 1.0)):
            raise ValueError("All p_non_overlap_values must be between 0 and 1.")

        self.n_versions = int(self.p_values.size)
        self.p_versions = []
        for _, p_non_overlap in enumerate(self.p_values):
            p_rep = np.full(self.n_genes, float(p_non_overlap), dtype=np.float32)
            self.p_versions.append(p_rep)

    def __len__(self):
        return self.n_versions * self.n_train

    def __getitem__(self, idx):
        version_idx = idx // self.n_train
        within_idx = idx % self.n_train

        y = self.rows.get_row_dense(within_idx)
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
    """Validation/test dataset with clean inputs and clean targets (no corruption)."""

    def __init__(self, panel_data, panel_ids, common_genes):
        self.rows = PanelRowAccessor(panel_data, panel_ids, common_genes)
        self.n_cells = self.rows.total_cells

    def __len__(self):
        return self.n_cells

    def __getitem__(self, idx):
        y = self.rows.get_row_dense(idx)
        x = y.copy()
        return torch.from_numpy(x), torch.from_numpy(y)

# Build panel-wise denoising data lazily to avoid giant dense matrices in RAM

panel_data = adatas  # Filtered per-sample AnnData objects from earlier QC
p_non_overlap_values = [0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.29, 0.31, 0.33, 0.35, 0.37, 0.40]
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

def _select_available_ids(sample_ids, panel_dict):
    normalized = [str(s).strip().upper() for s in sample_ids]
    found = [sid for sid in normalized if sid in panel_dict]
    missing = [sid for sid in normalized if sid not in panel_dict]
    return found, missing

train_panel_ids, missing_train = _select_available_ids(split_samples_train, panel_data)
val_panel_ids, missing_val = _select_available_ids(split_samples_val, panel_data)
test_panel_ids, missing_test = _select_available_ids(split_samples_test, panel_data)
test_a_panel_ids, _ = _select_available_ids(split_samples_test_a, panel_data)
test_b_panel_ids, _ = _select_available_ids(split_samples_test_b, panel_data)
test_c_panel_ids, _ = _select_available_ids(split_samples_test_c, panel_data)

if len(train_panel_ids) == 0:
    raise ValueError("No train panels from split_samples_train are present in `adatas`.")
if len(val_panel_ids) == 0:
    raise ValueError("No validation panels from split_samples_val are present in `adatas`.")
if len(test_panel_ids) == 0:
    raise ValueError("No test panels from split_samples_test are present in `adatas`.")

if missing_train or missing_val or missing_test:
    print("Requested sample IDs missing from loaded panel_data:")
    if missing_train:
        print(f"  train missing: {missing_train}")
    if missing_val:
        print(f"  val missing: {missing_val}")
    if missing_test:
        print(f"  test missing: {missing_test}")

# Define model gene space from training panels only.
all_panel_ids = train_panel_ids
common_genes = pd.Index(panel_data[all_panel_ids[0]].var_names).astype(str)
for sid in all_panel_ids[1:]:
    common_genes = common_genes.intersection(pd.Index(panel_data[sid].var_names).astype(str))
if len(common_genes) == 0:
    raise ValueError("No shared genes across selected training panels.")

gene_names = common_genes.to_numpy()
n_genes = len(gene_names)

# Keep compatibility variables without allocating giant dense arrays.
X_inputs_model = [None] * len(p_non_overlap_values)
X_inputs = X_inputs_model
X_tgt = None
Y_val = None
Y_test = None

# Build datasets.
train_dataset = MultiVersionTrainDataset(
    panel_data=panel_data,
    panel_ids=train_panel_ids,
    common_genes=common_genes,
    p_non_overlap_values=p_non_overlap_values,
    base_seed=base_seed,
)

val_dataset = CleanEvalDataset(
    panel_data=panel_data,
    panel_ids=val_panel_ids,
    common_genes=common_genes,
)

test_dataset = CleanEvalDataset(
    panel_data=panel_data,
    panel_ids=test_panel_ids,
    common_genes=common_genes,
)

# Split indices are split-local arrays.
train_idx = np.arange(train_dataset.rows.total_cells, dtype=np.int64)
val_idx = np.arange(len(val_dataset), dtype=np.int64)
test_idx = np.arange(len(test_dataset), dtype=np.int64)

# Panel-wise test subset sizes (A/B/C)
test_a_cells = int(sum(panel_data[sid].n_obs for sid in test_a_panel_ids))
test_b_cells = int(sum(panel_data[sid].n_obs for sid in test_b_panel_ids))
test_c_cells = int(sum(panel_data[sid].n_obs for sid in test_c_panel_ids))

n_cells = len(train_idx)
input_name = f"panel-wise lazy corruptions ({len(train_panel_ids)} train panels x {len(p_non_overlap_values)} p-values)"
split_mode = "fixed panel/sample split (panel-wise)"

print(f"Using input source: {input_name}")
print(f"Shared genes (train-defined): {n_genes}")
print(f"Train/Val/Test panels: {len(train_panel_ids)} / {len(val_panel_ids)} / {len(test_panel_ids)}")
print(f"Train/Val/Test cells: {train_dataset.rows.total_cells} / {len(val_dataset)} / {len(test_dataset)}")
print(f"Test A/B/C cells: {test_a_cells} / {test_b_cells} / {test_c_cells}")
print("Validation/Test inputs: clean (no corruption)")
print("Training data mode: raw counts (lazy streaming)")
print(f"Training examples: {len(train_dataset)}")
print(f"Validation examples: {len(val_dataset)} | Test examples: {len(test_dataset)}")

# Tunable hyperparameters 
epochs = 60
beta = 1e-3
latent_dim = 16
hidden_dim = 256
learning_rate = 1e-3
early_stop_patience = 5
theta_init = 10.0
pi_init = 0.1
batch_size_cuda = 512
batch_size_cpu = 256


# Here we just do some device setup & let it run on the  CPU
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