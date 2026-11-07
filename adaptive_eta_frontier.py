"""
adaptive_eta_frontier.py

Adaptive cold-start eta-frontier experiment for the spectral latent graph
model. Produces a high-resolution AUC-vs-V frontier with structural-event
diagnostics across multiple ranks and seeds.

This script is SELF-CONTAINED: model, training loop, sampler, and data
loaders are inlined. It does not import from any other project file.

Setup:
  - Datasets:       --datasets GRQC ...
  - Rank caps:      r in {64, 128, 256, 512} (configurable)
  - Seeds:          {0..4} (3 for debug)
  - eta_start = -1.0 -> eta_end = +0.3
  - Adaptive step:  delta_eta in [1e-3, 0.1], init 0.05
  - Cold-start always, fixed-epoch training (default 2000)
  - K_restarts=3, NLL-based selection
  - V_upper = max(10, 0.03 * r), V_lower = 2.0
  - max_rejections_per_step = 5, max_total_etas = 50

Robustness layer (always on):
  - SVD parameterisation as DEFAULT (--no_svd to disable)
  - Bounded beta = beta_max * sigmoid(beta_raw)
  - Centered kernel
  - Per-restart bias warmup

Cross-seed sharing (NEW):
  - For each (dataset, rank), the FIRST seed in --seeds runs the full
    adaptive sweep (Phase A: search). Its accepted eta values become the
    shared eta-grid for that (dataset, rank).
  - Subsequent seeds run Phase B (replication): cold-start fits at exactly
    the captured eta-grid, with no rejection logic and no anchor pass.
  - Single-seed runs are unchanged.

Output structure (--output_dir):
  <output_dir>/
    <dataset_name>/
      frontier_points_<dataset>_r<rank>_seed<seed>.csv
      frontier_events_<dataset>_r<rank>_seed<seed>.csv
      frontier_full_<dataset>_r<rank>_seed<seed>.json
    frontier_all_points.csv
    frontier_all_events.csv
    frontier_all_summaries.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics


# ===========================================================================
# 1. SEEDING
# ===========================================================================
def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ===========================================================================
# 2. DATA LOADING
# ===========================================================================
def _loadtxt_int(path):
    arr = np.loadtxt(path, dtype=np.float64)
    if not np.allclose(arr, np.round(arr)):
        bad = np.where(~np.isclose(arr, np.round(arr)))[0][:5]
        raise ValueError(f"{path}: non-integer entries at rows {bad.tolist()}")
    return arr.astype(np.int64)


def load_split_from_files(train_i_path, train_j_path,
                          test_pos_i_path=None, test_pos_j_path=None,
                          test_neg_i_path=None, test_neg_j_path=None):
    out = {}
    out["train_i"] = _loadtxt_int(train_i_path)
    out["train_j"] = _loadtxt_int(train_j_path)
    if test_pos_i_path:
        out["test_pos_i"] = _loadtxt_int(test_pos_i_path)
        out["test_pos_j"] = _loadtxt_int(test_pos_j_path)
    if test_neg_i_path:
        out["test_neg_i"] = _loadtxt_int(test_neg_i_path)
        out["test_neg_j"] = _loadtxt_int(test_neg_j_path)
    return out


def num_nodes_from_split(split):
    arrs = [v for k, v in split.items() if k.endswith("_i") or k.endswith("_j")]
    return int(max(a.max() for a in arrs)) + 1


def resolve_dataset_dir(name, root="./datasets"):
    candidates = []
    for r in [root, "./dataset", "./datasets", "../datasets", "../dataset"]:
        for nm in [name, name.lower(), name.upper(),
                   name.capitalize(), name.replace("-", "_"), name.replace("_", "-")]:
            d = os.path.join(r, nm)
            if os.path.isdir(d):
                candidates.append(d)
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        searched = ", ".join([os.path.join(r, name) for r in [root, "./dataset", "./datasets"]])
        raise SystemExit(f"Dataset directory not found for '{name}'. Searched: {searched}")
    base = candidates[0]
    print(f"  Resolved --dataset {name!r} -> {base}")
    files = {
        "train_i": None, "train_j": None,
        "test_pos_i": None, "test_pos_j": None,
        "test_neg_i": None, "test_neg_j": None,
        "labels_file": None,
    }
    pairs = [
        ("train_i", ["sparse_i.txt", "sparse_i", "edges_train_i.txt", "train_i.txt"]),
        ("train_j", ["sparse_j.txt", "sparse_j", "edges_train_j.txt", "train_j.txt"]),
        ("test_pos_i", ["sparse_i_rem.txt", "sparse_i_rem", "test_pos_i.txt"]),
        ("test_pos_j", ["sparse_j_rem.txt", "sparse_j_rem", "test_pos_j.txt"]),
        ("test_neg_i", ["non_sparse_i.txt", "non_sparse_i", "test_neg_i.txt"]),
        ("test_neg_j", ["non_sparse_j.txt", "non_sparse_j", "test_neg_j.txt"]),
        ("labels_file", ["labels.txt", "labels", "label.txt", "y.txt"]),
    ]
    for key, names_to_try in pairs:
        for nm in names_to_try:
            p = os.path.join(base, nm)
            if os.path.isfile(p):
                files[key] = p
                break
    if not (files["train_i"] and files["train_j"]):
        raise SystemExit(
            f"sparse_i.txt + sparse_j.txt not found in {base}. "
            f"Files present: {sorted(os.listdir(base))[:20]}")
    found = {k: v for k, v in files.items() if v is not None}
    print(f"  Found in {base}: {list(found.keys())}")
    return files


# ===========================================================================
# 3. NEGATIVE SAMPLER (GPU-native; no CPU<->GPU per epoch)
# ===========================================================================
class GPUUniformNonEdgeSampler:
    """Negative sampler that runs entirely on GPU using torch ops.

    Construction:
      Builds canonical edge int64 keys (i*n + j with i<j) once and stores
      them as a unique sorted tensor on the target device.

    Sampling:
      Uses torch.randint on device to draw candidate pairs, canonicalises
      with elementwise min/max, drops self-loops, and rejects real edges
      via torch.isin against the edge-key tensor. No Python sets, no
      CPU<->GPU transfers per epoch.

    Note: does NOT dedupe within a batch. For sparse graphs the duplicate
    rate at typical batch sizes is well below 1% (birthday paradox), well
    below the variance from sampling itself.
    """
    def __init__(self, n: int, train_i: torch.Tensor, train_j: torch.Tensor,
                 device: str):
        self.n = int(n)
        self.device = device
        i = train_i.long()
        j = train_j.long()
        ii = torch.minimum(i, j)
        jj = torch.maximum(i, j)
        mask = ii != jj
        ii = ii[mask]
        jj = jj[mask]
        keys = ii * self.n + jj
        self.edge_keys = torch.unique(keys).to(device)

    def sample(self, m: int):
        target = int(m)
        out_i = torch.empty(0, dtype=torch.long, device=self.device)
        out_j = torch.empty(0, dtype=torch.long, device=self.device)
        while out_i.numel() < target:
            need = target - out_i.numel()
            batch = max(int(need * 1.5) + 16, 1024)
            i = torch.randint(0, self.n, (batch,), device=self.device, dtype=torch.long)
            j = torch.randint(0, self.n, (batch,), device=self.device, dtype=torch.long)
            ii = torch.minimum(i, j)
            jj = torch.maximum(i, j)
            mask = ii != jj
            ii = ii[mask]
            jj = jj[mask]
            keys = ii * self.n + jj
            is_edge = torch.isin(keys, self.edge_keys)
            keep = ~is_edge
            out_i = torch.cat([out_i, ii[keep]])
            out_j = torch.cat([out_j, jj[keep]])
        return out_i[:target], out_j[:target]


# ===========================================================================
# 4. MODELS
# ===========================================================================
def init_beta_raw_for_bounded_beta(init_beta, beta_max):
    p = min(max(init_beta / beta_max, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


class _VKLSMBase(nn.Module):
    """Bounded beta + centered kernel + warmup-freeze interface."""
    def __init__(self, n, rank, beta_max=5.0, init_beta=1.0, device="cpu"):
        super().__init__()
        self.n = n; self.rank = rank; self.beta_max = float(beta_max)
        beta_raw0 = init_beta_raw_for_bounded_beta(init_beta=init_beta, beta_max=self.beta_max)
        self.beta_raw = nn.Parameter(torch.tensor(beta_raw0, dtype=torch.float32, device=device))

    def beta(self):
        return self.beta_max * torch.sigmoid(self.beta_raw) + 1e-8

    def L_centered(self) -> torch.Tensor: raise NotImplementedError
    def factor_param_names(self) -> List[str]: raise NotImplementedError

    def trace_LL(self):
        Lc = self.L_centered()
        return Lc.pow(2).sum().clamp_min(1e-12)

    def K_entries(self, i, j):
        Lc = self.L_centered()
        return self.n * (Lc[i] * Lc[j]).sum(-1) / self.trace_LL()

    def freeze_factor(self):
        for name in self.factor_param_names():
            getattr(self, name).requires_grad_(False)

    def unfreeze_factor(self):
        for name in self.factor_param_names():
            getattr(self, name).requires_grad_(True)

    def spectrum_squared(self, Lc: torch.Tensor) -> torch.Tensor:
        """Return eigenvalues of Lc^T Lc (any order; entropy is order-invariant).
        Default uses eigvalsh; SVD subclasses override with sigma**2."""
        G = Lc.T @ Lc
        return torch.clamp(torch.linalg.eigvalsh(G), min=0.0)


# --- SVD parameterisation (DEFAULT) ---
class VKLSMGlobalBias_SVD(_VKLSMBase):
    def __init__(self, n, rank, init_scale=0.01, beta_max=5.0, init_beta=1.0, device="cpu"):
        super().__init__(n, rank, beta_max=beta_max, init_beta=init_beta, device=device)
        self.U_raw = nn.Parameter(init_scale * torch.randn(n, rank, device=device))
        sigma_init = 1.0 / torch.sqrt(torch.arange(1, rank + 1, dtype=torch.float32, device=device))
        sigma_raw_init = torch.log(torch.expm1(sigma_init.clamp_min(1e-6)))
        self.sigma_raw = nn.Parameter(sigma_raw_init)
        self.alpha = nn.Parameter(torch.tensor(-3.0, device=device))

    def sigma(self):
        return F.softplus(self.sigma_raw) + 1e-8

    def U(self):
        Uc = self.U_raw - self.U_raw.mean(0, keepdim=True)
        Q, _ = torch.linalg.qr(Uc, mode="reduced")
        return Q

    def L_centered(self):
        return self.U() * self.sigma().unsqueeze(0)

    def factor_param_names(self):
        return ["U_raw", "sigma_raw"]

    def spectrum_squared(self, Lc):
        # SVD: Lc = Q diag(sigma), Q^T Q = I, so eigvals(Lc^T Lc) = sigma^2.
        return self.sigma().pow(2)

    @property
    def L(self):
        return self.L_centered()

    def logits_for_pairs(self, i, j):
        return self.alpha + self.beta() * self.K_entries(i, j)


class VKLSMNodeBias_SVD(_VKLSMBase):
    def __init__(self, n, rank, init_scale=0.01, beta_max=5.0, init_beta=1.0, device="cpu"):
        super().__init__(n, rank, beta_max=beta_max, init_beta=init_beta, device=device)
        self.U_raw = nn.Parameter(init_scale * torch.randn(n, rank, device=device))
        sigma_init = 1.0 / torch.sqrt(torch.arange(1, rank + 1, dtype=torch.float32, device=device))
        sigma_raw_init = torch.log(torch.expm1(sigma_init.clamp_min(1e-6)))
        self.sigma_raw = nn.Parameter(sigma_raw_init)
        self.node_bias = nn.Parameter(torch.zeros(n, device=device))

    def sigma(self):
        return F.softplus(self.sigma_raw) + 1e-8

    def U(self):
        Uc = self.U_raw - self.U_raw.mean(0, keepdim=True)
        Q, _ = torch.linalg.qr(Uc, mode="reduced")
        return Q

    def L_centered(self):
        return self.U() * self.sigma().unsqueeze(0)

    def factor_param_names(self):
        return ["U_raw", "sigma_raw"]

    def spectrum_squared(self, Lc):
        # SVD: Lc = Q diag(sigma), Q^T Q = I, so eigvals(Lc^T Lc) = sigma^2.
        return self.sigma().pow(2)

    @property
    def L(self):
        return self.L_centered()

    def logits_for_pairs(self, i, j):
        return self.node_bias[i] + self.node_bias[j] + self.beta() * self.K_entries(i, j)


# --- Free-L parameterisation (--no_svd fallback) ---
class VKLSMGlobalBias_FreeL(_VKLSMBase):
    def __init__(self, n, rank, init_scale=0.01, beta_max=5.0, init_beta=1.0, device="cpu"):
        super().__init__(n, rank, beta_max=beta_max, init_beta=init_beta, device=device)
        self.L = nn.Parameter(init_scale * torch.randn(n, rank, device=device))
        self.alpha = nn.Parameter(torch.tensor(-3.0, device=device))

    def L_centered(self):
        return self.L - self.L.mean(0, keepdim=True)

    def factor_param_names(self):
        return ["L"]

    def logits_for_pairs(self, i, j):
        return self.alpha + self.beta() * self.K_entries(i, j)


class VKLSMNodeBias_FreeL(_VKLSMBase):
    def __init__(self, n, rank, init_scale=0.01, beta_max=5.0, init_beta=1.0, device="cpu"):
        super().__init__(n, rank, beta_max=beta_max, init_beta=init_beta, device=device)
        self.L = nn.Parameter(init_scale * torch.randn(n, rank, device=device))
        self.node_bias = nn.Parameter(torch.zeros(n, device=device))

    def L_centered(self):
        return self.L - self.L.mean(0, keepdim=True)

    def factor_param_names(self):
        return ["L"]

    def logits_for_pairs(self, i, j):
        return self.node_bias[i] + self.node_bias[j] + self.beta() * self.K_entries(i, j)


def make_model(bias_model, n, rank, device, use_svd, beta_max, init_beta,
               init_scale=0.01):
    if use_svd:
        if bias_model == "global":
            return VKLSMGlobalBias_SVD(n, rank, init_scale=init_scale,
                                        beta_max=beta_max, init_beta=init_beta,
                                        device=device).to(device)
        if bias_model == "node":
            return VKLSMNodeBias_SVD(n, rank, init_scale=init_scale,
                                      beta_max=beta_max, init_beta=init_beta,
                                      device=device).to(device)
    else:
        if bias_model == "global":
            return VKLSMGlobalBias_FreeL(n, rank, init_scale=init_scale,
                                          beta_max=beta_max, init_beta=init_beta,
                                          device=device).to(device)
        if bias_model == "node":
            return VKLSMNodeBias_FreeL(n, rank, init_scale=init_scale,
                                        beta_max=beta_max, init_beta=init_beta,
                                        device=device).to(device)
    raise ValueError(f"bias_model must be 'global' or 'node', got {bias_model!r}")


# ===========================================================================
# 5. TRAINING
# ===========================================================================
def train_loss_components(m, ti, tj, sampler, eta,
                           lambda_theta=1e-4, lambda_bias=1e-4, neg_per_edge=5):
    """Single-step loss with cached Lc.

    Computes L_centered() ONCE per step and reuses it for positive logits,
    negative logits, and the spectrum/entropy term. For SVD models the
    spectrum is read off as sigma**2 (no eigensolve)."""
    ni, nj = sampler.sample(neg_per_edge * ti.numel())

    # Compute Lc once per step.
    Lc = m.L_centered()
    trace_LL = Lc.pow(2).sum().clamp_min(1e-12)
    n = m.n

    # Kernel entries reuse the cached Lc.
    pos_kernel = n * (Lc[ti] * Lc[tj]).sum(-1) / trace_LL
    neg_kernel = n * (Lc[ni] * Lc[nj]).sum(-1) / trace_LL

    beta = m.beta()
    if hasattr(m, "alpha"):
        pl = m.alpha + beta * pos_kernel
        nl = m.alpha + beta * neg_kernel
    else:
        pl = m.node_bias[ti] + m.node_bias[tj] + beta * pos_kernel
        nl = m.node_bias[ni] + m.node_bias[nj] + beta * neg_kernel

    nll = F.softplus(-pl).mean() + F.softplus(nl).mean()

    reg = torch.tensor(0.0, device=ti.device)
    if hasattr(m, "alpha"):
        reg = reg + 0.5 * lambda_theta * m.alpha.pow(2)
    if hasattr(m, "beta_raw"):
        reg = reg + 0.5 * lambda_theta * beta.pow(2)
    if hasattr(m, "node_bias"):
        reg = reg + 0.5 * lambda_bias * m.node_bias.pow(2).mean()

    # Spectrum: sigma^2 for SVD, eigvalsh(Lc^T Lc) for free-L.
    evals = m.spectrum_squared(Lc)
    p = evals / evals.sum().clamp_min(1e-12)
    log_v = -(p * torch.log(p.clamp_min(1e-12))).sum()
    return nll, log_v, reg


def train_model(m, ti, tj, n, epochs=200, lr=1e-2, eta=0.0,
                lambda_theta=1e-4, lambda_bias=1e-4, bias_warmup_epochs=0):
    """Fixed-epoch cold-start training. No early stopping."""
    sampler = GPUUniformNonEdgeSampler(n, ti, tj, device=ti.device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)

    do_warmup = bias_warmup_epochs > 0
    if do_warmup:
        m.freeze_factor()

    for epoch in range(1, epochs + 1):
        if do_warmup and epoch == bias_warmup_epochs + 1:
            m.unfreeze_factor()
        m.train()
        opt.zero_grad(set_to_none=True)
        nll, log_v, reg = train_loss_components(
            m, ti, tj, sampler, eta=eta,
            lambda_theta=lambda_theta, lambda_bias=lambda_bias,
        )
        loss = nll - eta * log_v + reg
        loss.backward()
        opt.step()
    return m


# ===========================================================================
# 6. EVALUATION + DIAGNOSTICS
# ===========================================================================
@torch.no_grad()
def evaluate_train_nll(m, ti, tj, sampler):
    m.eval()
    pl = m.logits_for_pairs(ti, tj)
    ni, nj = sampler.sample(ti.numel())
    nl = m.logits_for_pairs(ni, nj)
    nll_p = F.softplus(-pl).mean()
    nll_n = F.softplus(nl).mean()
    return float((nll_p + nll_n).cpu())


@torch.no_grad()
def evaluate_test_loglik(m, tpi, tpj, tni, tnj):
    m.eval()
    pl = m.logits_for_pairs(tpi, tpj)
    nl = m.logits_for_pairs(tni, tnj)
    ll_pos = -F.softplus(-pl)
    ll_neg = -F.softplus(nl)
    n_pos = pl.numel(); n_neg = nl.numel()
    ll_total = (ll_pos.sum() + ll_neg.sum()) / max(n_pos + n_neg, 1)
    return float(ll_total.cpu())


@torch.no_grad()
def eval_auc_and_ap(model, tpi, tpj, tni, tnj):
    model.eval()
    pl = model.logits_for_pairs(tpi, tpj)
    nl = model.logits_for_pairs(tni, tnj)
    logits = torch.cat([nl, pl])
    y = torch.cat([torch.zeros(nl.numel()), torch.ones(pl.numel())]).cpu().numpy()
    s = torch.sigmoid(logits).cpu().numpy()
    auc = float(metrics.roc_auc_score(y, s))
    ap = float(metrics.average_precision_score(y, s))
    return auc, ap


@torch.no_grad()
def diagnostics(m, eps=1e-12):
    """Diagnostics on the centered factor.

    Uses m.spectrum_squared(L) which dispatches:
      - SVD subclasses: returns sigma**2 directly (exact equivalence to
        eigvalsh(L^T L) since L = Q diag(sigma) with Q^T Q = I_r).
      - Free-L subclasses: returns eigvalsh(L^T L).
    The eigenvector path (Z = L Q, hard/soft assignments) was unused
    downstream and has been removed.
    """
    L = m.L_centered().detach()
    evals = m.spectrum_squared(L).detach()
    evals = torch.clamp(evals, min=0.0)
    evals, _ = torch.sort(evals, descending=True)
    p = (evals / evals.sum().clamp_min(eps)).cpu().numpy()
    log_v = -(p * np.log(np.maximum(p, eps))).sum()
    V = float(np.exp(log_v))
    evals_np = evals.cpu().numpy()
    lam_max = evals_np.max() if evals_np.max() > 0 else 1e-12
    k_tau_01 = int((evals_np >= 0.01 * lam_max).sum())
    k_tau_001 = int((evals_np >= 0.001 * lam_max).sum())
    cum = np.cumsum(p)
    k90 = int(np.searchsorted(cum, 0.9) + 1)
    k99 = int(np.searchsorted(cum, 0.99) + 1)
    return {
        "V": V, "log_V": float(log_v),
        "k_tau_01": k_tau_01, "k_tau_001": k_tau_001,
        "k90": k90, "k99": k99,
        "evals_sorted": evals_np.tolist(),
        "p_sorted": p.tolist(),
    }


def extract_bias_summary(model, bias_model: str) -> Dict:
    out = {
        "alpha": float("nan"),
        "node_bias_mean": float("nan"),
        "node_bias_std": float("nan"),
        "node_bias_min": float("nan"),
        "node_bias_max": float("nan"),
    }
    if bias_model == "global" and hasattr(model, "alpha"):
        out["alpha"] = float(model.alpha.detach().cpu().item())
    if bias_model == "node" and hasattr(model, "node_bias"):
        nb = model.node_bias.detach().cpu().numpy()
        out["node_bias_mean"] = float(np.mean(nb))
        out["node_bias_std"] = float(np.std(nb))
        out["node_bias_min"] = float(np.min(nb))
        out["node_bias_max"] = float(np.max(nb))
    return out


# ===========================================================================
# 7. ONE-ETA COLD-START FIT (K restarts, NLL selection)
# ===========================================================================
def prepare_tensors(split: Dict[str, np.ndarray], device: str):
    train_i = torch.tensor(split["train_i"], dtype=torch.long, device=device)
    train_j = torch.tensor(split["train_j"], dtype=torch.long, device=device)
    have_test = "test_pos_i" in split and "test_neg_i" in split
    if have_test:
        tpi = torch.tensor(split["test_pos_i"], dtype=torch.long, device=device)
        tpj = torch.tensor(split["test_pos_j"], dtype=torch.long, device=device)
        tni = torch.tensor(split["test_neg_i"], dtype=torch.long, device=device)
        tnj = torch.tensor(split["test_neg_j"], dtype=torch.long, device=device)
    else:
        tpi = tpj = tni = tnj = None
    return train_i, train_j, tpi, tpj, tni, tnj, have_test


def fit_one_eta_cold_full(
    bias_model, n, rank,
    train_i, train_j, tpi, tpj, tni, tnj,
    eta, K_restarts, epochs, lambda_theta, lambda_bias,
    base_seed, kk_index, device, have_test,
    eval_sampler,
    use_svd, beta_max, init_beta, bias_warmup_epochs,
):
    per_restart = []
    for r in range(K_restarts):
        seed_r = base_seed * 1000 + kk_index * 100 + r
        torch.manual_seed(seed_r)
        np.random.seed(seed_r)
        m = make_model(bias_model, n, rank, device,
                        use_svd=use_svd, beta_max=beta_max, init_beta=init_beta)
        m = train_model(m, train_i, train_j, n,
                        epochs=epochs, eta=eta,
                        lambda_theta=lambda_theta, lambda_bias=lambda_bias,
                        bias_warmup_epochs=bias_warmup_epochs)
        train_nll = evaluate_train_nll(m, train_i, train_j, eval_sampler)
        if have_test:
            test_loglik = evaluate_test_loglik(m, tpi, tpj, tni, tnj)
            auc, ap = eval_auc_and_ap(m, tpi, tpj, tni, tnj)
        else:
            test_loglik = float("nan")
            auc, ap = float("nan"), float("nan")
        diag = diagnostics(m)
        bias_sum = extract_bias_summary(m, bias_model)
        beta_value = float(m.beta().detach().cpu().item())
        per_restart.append({
            "restart": r,
            "train_nll": train_nll,
            "test_auc": auc,
            "test_ap": ap,
            "test_loglik": test_loglik,
            "V": diag["V"], "log_V": diag["log_V"],
            "k_tau_01": diag["k_tau_01"], "k_tau_001": diag["k_tau_001"],
            "k90": diag["k90"], "k99": diag["k99"],
            "evals_sorted": diag["evals_sorted"],
            "p_sorted": diag["p_sorted"],
            "beta": beta_value,
            **bias_sum,
        })
    per_restart.sort(key=lambda x: x["train_nll"])
    return per_restart[0]


# ===========================================================================
# 8. TOP-K SPECTRUM SUMMARIES
# ===========================================================================
def topk_eigval_summaries(evals_sorted: List[float],
                           p_sorted: List[float]) -> Dict[str, float]:
    e = np.array(evals_sorted, dtype=float)
    p = np.array(p_sorted, dtype=float)
    out = {}
    out["eval_top1"] = float(e[0]) if len(e) >= 1 else float("nan")
    out["eval_top2"] = float(e[1]) if len(e) >= 2 else float("nan")
    out["eval_top3"] = float(e[2]) if len(e) >= 3 else float("nan")
    out["eval_top5_sum"] = float(e[:5].sum()) if len(e) >= 1 else float("nan")
    out["eval_top10_sum"] = float(e[:10].sum()) if len(e) >= 1 else float("nan")
    out["p_top1"] = float(p[0]) if len(p) >= 1 else float("nan")
    out["p_top2"] = float(p[1]) if len(p) >= 2 else float("nan")
    out["p_top3"] = float(p[2]) if len(p) >= 3 else float("nan")
    out["p_top5_sum"] = float(p[:5].sum()) if len(p) >= 1 else float("nan")
    out["p_top10_sum"] = float(p[:10].sum()) if len(p) >= 1 else float("nan")
    return out


# ===========================================================================
# 9. NEAR-DEGENERACY EVENT CHECK (proxy via small simplex-eigenvalue gaps)
# ===========================================================================
def near_degeneracy_event_check(p_a, p_b, near_gap_tol=5e-3) -> bool:
    a = np.sort(np.array(p_a, dtype=float))[::-1]
    b = np.sort(np.array(p_b, dtype=float))[::-1]
    k = min(8, len(a), len(b))
    if k < 2:
        return False
    a_diff = np.abs(a[:k - 1] - a[1:k])
    b_diff = np.abs(b[:k - 1] - b[1:k])
    return bool(np.any(a_diff < near_gap_tol) or np.any(b_diff < near_gap_tol))


# ===========================================================================
# 10. POINT ROW BUILDER
# ===========================================================================
def build_point_row(
    dataset_name: str, seed: int, rank: int, step_index: int,
    eta_value: float, best: Dict, bias_model: str,
    delta_eta_used: float, V_prev: Optional[float],
    wall_t: float, rejection_count: int,
    is_initial: bool, is_forced_event: bool,
    V_lower_used: float, V_upper_used: float,
    delta_eta_init: float, delta_eta_min: float, delta_eta_max: float,
    is_anchor_eta: bool = False,
    phase: str = "search",
    eta_grid_source_seed: Optional[int] = None,
):
    V = float(best["V"])
    delta_V_signed = float("nan") if V_prev is None else float(V - V_prev)
    delta_V_abs = float("nan") if V_prev is None else float(abs(V - V_prev))
    delta_V_norm = float("nan") if V_prev is None else float(abs(V - V_prev) / max(rank, 1))
    spec_summary = topk_eigval_summaries(best["evals_sorted"], best["p_sorted"])
    pt = {
        "dataset": dataset_name, "seed": seed, "rank_cap": rank,
        "step_index": step_index,
        "eta": float(eta_value),
        "V": V,
        "log_V": float(best["log_V"]),
        "AUC": float(best["test_auc"]),
        "AP": float(best["test_ap"]),
        "train_NLL": float(best["train_nll"]),
        "test_loglik": float(best["test_loglik"]),
        "k90": int(best["k90"]),
        "k99": int(best["k99"]),
        "k_tau_01": int(best["k_tau_01"]),
        "k_tau_001": int(best["k_tau_001"]),
        "eigenvalues_sorted": json.dumps(best["evals_sorted"]),
        "p_sorted": json.dumps(best["p_sorted"]),
        **spec_summary,
        "alpha": best.get("alpha", float("nan")),
        "beta": float(best.get("beta", float("nan"))),
        "node_bias_mean": best.get("node_bias_mean", float("nan")),
        "node_bias_std": best.get("node_bias_std", float("nan")),
        "node_bias_min": best.get("node_bias_min", float("nan")),
        "node_bias_max": best.get("node_bias_max", float("nan")),
        "delta_eta_used": float(delta_eta_used),
        "delta_V_abs": delta_V_abs,
        "delta_V_signed": delta_V_signed,
        "delta_V_norm": delta_V_norm,
        "wall_time_s": float(wall_t),
        "rejection_count_before_accept": int(rejection_count),
        "is_initial": bool(is_initial),
        "is_forced_event": bool(is_forced_event),
        "is_anchor_eta": bool(is_anchor_eta),
        "V_lower_used": float(V_lower_used),
        "V_upper_used": float(V_upper_used),
        "delta_eta_init": float(delta_eta_init),
        "delta_eta_min": float(delta_eta_min),
        "delta_eta_max": float(delta_eta_max),
        "phase": str(phase),
        "eta_grid_source_seed": (-1 if eta_grid_source_seed is None
                                  else int(eta_grid_source_seed)),
    }
    return pt


# ===========================================================================
# 11a. ADAPTIVE COLD-START FRONTIER SWEEP (Phase A: search)
# ===========================================================================
def run_adaptive_frontier_one(
    dataset_name: str, split: Dict, rank: int, seed: int,
    eta_start: float, eta_end: float,
    delta_eta_init: float, delta_eta_min: float, delta_eta_max: float,
    V_lower: float, V_upper: float,
    max_rejections_per_step: int, max_total_etas: int,
    K_restarts: int, epochs: int,
    bias_model: str, lambda_theta: float, lambda_bias: float,
    use_svd: bool, beta_max: float, init_beta: float, bias_warmup_epochs: int,
    near_gap_tol: float, device: str,
) -> Tuple[List[Dict], List[Dict], Dict]:
    n = num_nodes_from_split(split)
    train_i, train_j, tpi, tpj, tni, tnj, have_test = prepare_tensors(split, device)
    eval_sampler = GPUUniformNonEdgeSampler(n, train_i, train_j, device=device)

    # Anchor etas: values that MUST be in the final saved set if they fall
    # inside [eta_start, eta_end]. We do NOT clamp during the adaptive sweep
    # (that interacts badly with rejection logic and distorts ΔV statistics).
    # Instead, after the sweep finishes we check which anchors were missed
    # and append one extra cold-start fit per missing anchor. Cold-start
    # makes ordering irrelevant.
    anchor_etas = [a for a in [0.0] if eta_start <= a <= eta_end]

    points: List[Dict] = []
    events: List[Dict] = []
    fit_idx = 0
    n_fits_total = 0
    n_fits_rejected = 0
    n_forced_events = 0
    n_anchor_appends = 0
    t_run = time.time()

    # Initial point
    print(f"\n  [r={rank} s={seed}] starting at eta={eta_start:+.4f}")
    t0 = time.time()
    best = fit_one_eta_cold_full(
        bias_model, n, rank,
        train_i, train_j, tpi, tpj, tni, tnj,
        eta_start, K_restarts, epochs, lambda_theta, lambda_bias,
        seed, fit_idx, device, have_test, eval_sampler,
        use_svd=use_svd, beta_max=beta_max, init_beta=init_beta,
        bias_warmup_epochs=bias_warmup_epochs,
    )
    fit_idx += 1
    n_fits_total += 1
    wall_t = time.time() - t0

    pt0 = build_point_row(
        dataset_name=dataset_name, seed=seed, rank=rank, step_index=0,
        eta_value=eta_start, best=best, bias_model=bias_model,
        delta_eta_used=0.0, V_prev=None,
        wall_t=wall_t, rejection_count=0,
        is_initial=True, is_forced_event=False,
        V_lower_used=V_lower, V_upper_used=V_upper,
        delta_eta_init=delta_eta_init,
        delta_eta_min=delta_eta_min, delta_eta_max=delta_eta_max,
        is_anchor_eta=abs(eta_start) < 1e-12,
        phase="search", eta_grid_source_seed=None,
    )
    points.append(pt0)
    print(f"  [INIT  ] {dataset_name} s={seed} r={rank}  "
          f"eta={pt0['eta']:+.4f}  V={pt0['V']:7.2f}  "
          f"AUC={pt0['AUC']:.4f}  AP={pt0['AP']:.4f}  "
          f"k_tau01={pt0['k_tau_01']:3d}  beta={pt0['beta']:.3f}  "
          f"({wall_t:.1f}s)")

    eta_curr = eta_start
    V_curr = pt0["V"]
    delta_eta = float(delta_eta_init)
    prev_pt = pt0

    while eta_curr < eta_end - 1e-9 and len(points) < max_total_etas:
        rejections = 0
        accepted = False
        last_attempt = None

        while rejections <= max_rejections_per_step and not accepted:
            eta_try = min(eta_curr + delta_eta, eta_end)
            t0 = time.time()
            best = fit_one_eta_cold_full(
                bias_model, n, rank,
                train_i, train_j, tpi, tpj, tni, tnj,
                eta_try, K_restarts, epochs, lambda_theta, lambda_bias,
                seed, fit_idx, device, have_test, eval_sampler,
                use_svd=use_svd, beta_max=beta_max, init_beta=init_beta,
                bias_warmup_epochs=bias_warmup_epochs,
            )
            fit_idx += 1
            n_fits_total += 1
            wall_t = time.time() - t0
            V_try = float(best["V"])
            delta_V = abs(V_try - V_curr)
            at_floor = delta_eta <= delta_eta_min + 1e-12
            last_attempt = (eta_try, best, wall_t)

            if delta_V > V_upper and not at_floor:
                rejections += 1
                n_fits_rejected += 1
                old_de = delta_eta
                delta_eta = max(delta_eta / 2.0, delta_eta_min)
                print(f"  [REJECT] {dataset_name} s={seed} r={rank}  "
                      f"eta={eta_try:+.4f}  V={V_try:7.2f}  "
                      f"ΔV={delta_V:6.2f}  threshold(V_upper)={V_upper:.2f}  "
                      f"old Δη={old_de:.5f} -> new Δη={delta_eta:.5f}  "
                      f"({wall_t:.1f}s)")
                continue

            forced_event = (delta_V > V_upper and at_floor)
            if forced_event:
                n_forced_events += 1
            old_de = delta_eta
            if delta_V < V_lower:
                delta_eta = min(delta_eta * 2.0, delta_eta_max)
                tag = "GROW"
            elif delta_V > V_upper:
                tag = "EVENT"
            else:
                tag = "KEEP"

            pt = build_point_row(
                dataset_name=dataset_name, seed=seed, rank=rank,
                step_index=len(points),
                eta_value=eta_try, best=best, bias_model=bias_model,
                delta_eta_used=eta_try - eta_curr, V_prev=V_curr,
                wall_t=wall_t, rejection_count=rejections,
                is_initial=False, is_forced_event=forced_event,
                V_lower_used=V_lower, V_upper_used=V_upper,
                delta_eta_init=delta_eta_init,
                delta_eta_min=delta_eta_min, delta_eta_max=delta_eta_max,
                is_anchor_eta=False,
                phase="search", eta_grid_source_seed=None,
            )
            points.append(pt)

            d_AUC = pt["AUC"] - prev_pt["AUC"] if (
                not math.isnan(pt["AUC"]) and not math.isnan(prev_pt["AUC"])
            ) else float("nan")
            d_AP = pt["AP"] - prev_pt["AP"] if (
                not math.isnan(pt["AP"]) and not math.isnan(prev_pt["AP"])
            ) else float("nan")
            rank_transition = pt["k_tau_01"] != prev_pt["k_tau_01"]
            near_deg = near_degeneracy_event_check(
                json.loads(prev_pt["p_sorted"]), json.loads(pt["p_sorted"]),
                near_gap_tol=near_gap_tol,
            )
            ev = {
                "dataset": dataset_name, "seed": seed, "rank_cap": rank,
                "from_step": len(points) - 2, "to_step": len(points) - 1,
                "from_eta": float(prev_pt["eta"]), "to_eta": float(pt["eta"]),
                "midpoint_eta": float(0.5 * (prev_pt["eta"] + pt["eta"])),
                "from_V": float(prev_pt["V"]), "to_V": float(pt["V"]),
                "midpoint_V": float(0.5 * (prev_pt["V"] + pt["V"])),
                "delta_V_abs": float(abs(pt["V"] - prev_pt["V"])),
                "delta_V_signed": float(pt["V"] - prev_pt["V"]),
                "delta_V_norm": float(abs(pt["V"] - prev_pt["V"]) / max(rank, 1)),
                "delta_AUC": float(d_AUC),
                "delta_AP": float(d_AP),
                "from_k_tau_01": int(prev_pt["k_tau_01"]),
                "to_k_tau_01": int(pt["k_tau_01"]),
                "rank_transition": bool(rank_transition),
                "near_degeneracy_event": bool(near_deg),
                "is_forced_event": bool(forced_event),
                "any_structural_event": bool(rank_transition or near_deg or forced_event),
                "phase": "search",
            }
            events.append(ev)

            print(f"  [ACCEPT] {dataset_name} s={seed} r={rank}  "
                  f"eta={pt['eta']:+.4f}  V={V_try:7.2f}  ΔV={delta_V:6.2f}  "
                  f"AUC={pt['AUC']:.4f}  AP={pt['AP']:.4f}  "
                  f"k_tau01={pt['k_tau_01']:3d}  beta={pt['beta']:.3f}  "
                  f"Δη: {old_de:.5f}->{delta_eta:.5f} [{tag}"
                  f"{' / forced' if forced_event else ''}]  "
                  f"rejections={rejections}  ({wall_t:.1f}s)")

            eta_curr = pt["eta"]
            V_curr = V_try
            prev_pt = pt
            accepted = True

        if not accepted:
            print(f"  [WARN  ] {dataset_name} s={seed} r={rank} max rejections; "
                  f"force-accept last attempt.")
            if last_attempt is not None:
                eta_try, best, wall_t = last_attempt
                V_try = float(best["V"])
                pt = build_point_row(
                    dataset_name=dataset_name, seed=seed, rank=rank,
                    step_index=len(points),
                    eta_value=eta_try, best=best, bias_model=bias_model,
                    delta_eta_used=eta_try - eta_curr, V_prev=V_curr,
                    wall_t=wall_t, rejection_count=rejections,
                    is_initial=False, is_forced_event=True,
                    V_lower_used=V_lower, V_upper_used=V_upper,
                    delta_eta_init=delta_eta_init,
                    delta_eta_min=delta_eta_min, delta_eta_max=delta_eta_max,
                    is_anchor_eta=False,
                    phase="search", eta_grid_source_seed=None,
                )
                points.append(pt)
                eta_curr = pt["eta"]
                V_curr = V_try
                prev_pt = pt
                n_forced_events += 1

    # ----- Post-sweep anchor pass -----
    # Cold-start makes ordering irrelevant, so we add any missing anchor
    # eta values as supplemental fits AFTER the adaptive sweep completes.
    # This guarantees anchors are present without distorting ΔV statistics
    # or interacting with rejection logic. No event row is generated for
    # an anchor append (events describe the adaptive trajectory only).
    for anchor in anchor_etas:
        already_have = any(abs(p["eta"] - anchor) < 1e-12 for p in points)
        if already_have:
            continue
        print(f"\n  [ANCHOR] {dataset_name} s={seed} r={rank}  "
              f"appending missing anchor eta={anchor:+.4f}")
        t0 = time.time()
        best_anchor = fit_one_eta_cold_full(
            bias_model, n, rank,
            train_i, train_j, tpi, tpj, tni, tnj,
            anchor, K_restarts, epochs, lambda_theta, lambda_bias,
            seed, fit_idx, device, have_test, eval_sampler,
            use_svd=use_svd, beta_max=beta_max, init_beta=init_beta,
            bias_warmup_epochs=bias_warmup_epochs,
        )
        fit_idx += 1
        n_fits_total += 1
        wall_anchor = time.time() - t0
        # V_prev=None so delta_V_* are NaN: the anchor is not an adaptive
        # transition. delta_eta_used is also undefined for the anchor.
        pt_anchor = build_point_row(
            dataset_name=dataset_name, seed=seed, rank=rank,
            step_index=len(points),
            eta_value=anchor, best=best_anchor, bias_model=bias_model,
            delta_eta_used=float("nan"), V_prev=None,
            wall_t=wall_anchor, rejection_count=0,
            is_initial=False, is_forced_event=False,
            V_lower_used=V_lower, V_upper_used=V_upper,
            delta_eta_init=delta_eta_init,
            delta_eta_min=delta_eta_min, delta_eta_max=delta_eta_max,
            is_anchor_eta=True,
            phase="search", eta_grid_source_seed=None,
        )
        points.append(pt_anchor)
        n_anchor_appends += 1
        print(f"  [ANCHOR] {dataset_name} s={seed} r={rank}  "
              f"eta={pt_anchor['eta']:+.4f}  V={pt_anchor['V']:7.2f}  "
              f"AUC={pt_anchor['AUC']:.4f}  AP={pt_anchor['AP']:.4f}  "
              f"k_tau01={pt_anchor['k_tau_01']:3d}  beta={pt_anchor['beta']:.3f}  "
              f"({wall_anchor:.1f}s)")

    # ----- Summary -----
    summary = _build_summary(
        dataset_name=dataset_name, seed=seed, rank=rank,
        points=points, events=events,
        n_anchor_appends=n_anchor_appends,
        n_fits_rejected=n_fits_rejected,
        n_forced_events=n_forced_events,
        n_fits_total=n_fits_total,
        t_run_start=t_run,
        eta_start=eta_start, eta_end=eta_end,
        phase="search", eta_grid_source_seed=None,
    )

    # Hard guarantee: if zero lies in the sweep interval (inclusive), eta=0
    # MUST appear in the saved points. The anchor pass should ensure this;
    # if it does not, that is a logic error and we fail loudly.
    if eta_start <= 0.0 <= eta_end and not summary["has_eta_zero"]:
        raise RuntimeError(
            f"{dataset_name} r={rank} seed={seed}: sweep crossed zero "
            f"(eta in [{eta_start}, {eta_end}]) but eta=0 was not saved. "
            f"Anchor logic failed."
        )

    _print_summary(summary)
    return points, events, summary


# ===========================================================================
# 11b. FIXED-GRID REPLICATION (Phase B)
# ===========================================================================
def run_replication_one(
    dataset_name: str, split: Dict, rank: int, seed: int,
    eta_grid: List[float], eta_grid_source_seed: int,
    K_restarts: int, epochs: int,
    bias_model: str, lambda_theta: float, lambda_bias: float,
    use_svd: bool, beta_max: float, init_beta: float, bias_warmup_epochs: int,
    near_gap_tol: float, device: str,
    # Carried through for output-row schema parity with search-phase rows.
    V_lower: float, V_upper: float,
    delta_eta_init: float, delta_eta_min: float, delta_eta_max: float,
) -> Tuple[List[Dict], List[Dict], Dict]:
    """Cold-start fits at a fixed eta grid produced by Phase A on another seed.

    Differences vs run_adaptive_frontier_one:
      - No rejection logic, no delta_eta adaptation: the grid is fixed.
      - No anchor pass: eta=0 is in the grid by construction whenever the
        Phase A sweep produced it. (If somehow not, we trust the grid.)
      - Events are still recorded across consecutive grid points (in eta-
        sorted order), giving per-seed structural-event diagnostics.
    """
    n = num_nodes_from_split(split)
    train_i, train_j, tpi, tpj, tni, tnj, have_test = prepare_tensors(split, device)
    eval_sampler = GPUUniformNonEdgeSampler(n, train_i, train_j, device=device)

    # Fix iteration order: ascending eta. This matches the natural axis for
    # ΔV diagnostics across consecutive points.
    eta_grid_sorted = sorted(set(float(e) for e in eta_grid))

    points: List[Dict] = []
    events: List[Dict] = []
    fit_idx = 0
    n_fits_total = 0
    t_run = time.time()

    print(f"\n  [r={rank} s={seed}] REPLICATION at fixed eta-grid "
          f"({len(eta_grid_sorted)} points; from seed={eta_grid_source_seed})")

    prev_pt: Optional[Dict] = None

    for k, eta_value in enumerate(eta_grid_sorted):
        t0 = time.time()
        best = fit_one_eta_cold_full(
            bias_model, n, rank,
            train_i, train_j, tpi, tpj, tni, tnj,
            eta_value, K_restarts, epochs, lambda_theta, lambda_bias,
            seed, fit_idx, device, have_test, eval_sampler,
            use_svd=use_svd, beta_max=beta_max, init_beta=init_beta,
            bias_warmup_epochs=bias_warmup_epochs,
        )
        fit_idx += 1
        n_fits_total += 1
        wall_t = time.time() - t0

        V_prev = float(prev_pt["V"]) if prev_pt is not None else None
        pt = build_point_row(
            dataset_name=dataset_name, seed=seed, rank=rank,
            step_index=k,
            eta_value=eta_value, best=best, bias_model=bias_model,
            delta_eta_used=(eta_value - prev_pt["eta"]) if prev_pt is not None else 0.0,
            V_prev=V_prev,
            wall_t=wall_t, rejection_count=0,
            is_initial=(k == 0), is_forced_event=False,
            V_lower_used=V_lower, V_upper_used=V_upper,
            delta_eta_init=delta_eta_init,
            delta_eta_min=delta_eta_min, delta_eta_max=delta_eta_max,
            is_anchor_eta=(abs(eta_value) < 1e-12),
            phase="replication", eta_grid_source_seed=eta_grid_source_seed,
        )
        points.append(pt)

        if prev_pt is not None:
            d_AUC = pt["AUC"] - prev_pt["AUC"] if (
                not math.isnan(pt["AUC"]) and not math.isnan(prev_pt["AUC"])
            ) else float("nan")
            d_AP = pt["AP"] - prev_pt["AP"] if (
                not math.isnan(pt["AP"]) and not math.isnan(prev_pt["AP"])
            ) else float("nan")
            rank_transition = pt["k_tau_01"] != prev_pt["k_tau_01"]
            near_deg = near_degeneracy_event_check(
                json.loads(prev_pt["p_sorted"]), json.loads(pt["p_sorted"]),
                near_gap_tol=near_gap_tol,
            )
            ev = {
                "dataset": dataset_name, "seed": seed, "rank_cap": rank,
                "from_step": k - 1, "to_step": k,
                "from_eta": float(prev_pt["eta"]), "to_eta": float(pt["eta"]),
                "midpoint_eta": float(0.5 * (prev_pt["eta"] + pt["eta"])),
                "from_V": float(prev_pt["V"]), "to_V": float(pt["V"]),
                "midpoint_V": float(0.5 * (prev_pt["V"] + pt["V"])),
                "delta_V_abs": float(abs(pt["V"] - prev_pt["V"])),
                "delta_V_signed": float(pt["V"] - prev_pt["V"]),
                "delta_V_norm": float(abs(pt["V"] - prev_pt["V"]) / max(rank, 1)),
                "delta_AUC": float(d_AUC),
                "delta_AP": float(d_AP),
                "from_k_tau_01": int(prev_pt["k_tau_01"]),
                "to_k_tau_01": int(pt["k_tau_01"]),
                "rank_transition": bool(rank_transition),
                "near_degeneracy_event": bool(near_deg),
                "is_forced_event": False,
                "any_structural_event": bool(rank_transition or near_deg),
                "phase": "replication",
            }
            events.append(ev)

        print(f"  [REPL  ] {dataset_name} s={seed} r={rank}  "
              f"eta={pt['eta']:+.4f}  V={pt['V']:7.2f}  "
              f"AUC={pt['AUC']:.4f}  AP={pt['AP']:.4f}  "
              f"k_tau01={pt['k_tau_01']:3d}  beta={pt['beta']:.3f}  "
              f"({wall_t:.1f}s)")

        prev_pt = pt

    summary = _build_summary(
        dataset_name=dataset_name, seed=seed, rank=rank,
        points=points, events=events,
        n_anchor_appends=0,
        n_fits_rejected=0,
        n_forced_events=0,
        n_fits_total=n_fits_total,
        t_run_start=t_run,
        eta_start=min(eta_grid_sorted), eta_end=max(eta_grid_sorted),
        phase="replication", eta_grid_source_seed=eta_grid_source_seed,
    )
    _print_summary(summary)
    return points, events, summary


# ===========================================================================
# 11c. SHARED SUMMARY HELPERS
# ===========================================================================
def _build_summary(dataset_name, seed, rank, points, events,
                   n_anchor_appends, n_fits_rejected, n_forced_events,
                   n_fits_total, t_run_start, eta_start, eta_end,
                   phase, eta_grid_source_seed):
    Vs = np.array([p["V"] for p in points], dtype=float)
    AUCs = np.array([p["AUC"] for p in points], dtype=float)
    APs = np.array([p["AP"] for p in points], dtype=float)

    if np.any(~np.isnan(AUCs)):
        auc_peak_idx = int(np.nanargmax(AUCs))
        AUC_peak = float(AUCs[auc_peak_idx])
        V_at_AUC_peak = float(Vs[auc_peak_idx])
        eta_at_AUC_peak = float(points[auc_peak_idx]["eta"])
    else:
        AUC_peak, V_at_AUC_peak, eta_at_AUC_peak = float("nan"), float("nan"), float("nan")

    if np.any(~np.isnan(APs)):
        ap_peak_idx = int(np.nanargmax(APs))
        AP_peak = float(APs[ap_peak_idx])
        V_at_AP_peak = float(Vs[ap_peak_idx])
    else:
        AP_peak, V_at_AP_peak = float("nan"), float("nan")

    if events:
        largest_idx = int(np.argmax([abs(e["delta_V_abs"]) for e in events]))
        largest_delta_V = float(events[largest_idx]["delta_V_abs"])
        largest_delta_V_aligns_with_event = bool(events[largest_idx]["any_structural_event"])
    else:
        largest_delta_V = float("nan")
        largest_delta_V_aligns_with_event = None

    has_eta_zero = bool(any(abs(p["eta"]) < 1e-12 for p in points))

    return {
        "dataset": dataset_name, "seed": seed, "rank_cap": rank,
        # n_accepted_points kept for backward compatibility; counts adaptive-
        # trajectory points only (excludes anchor appends). For replication
        # phase there are no anchor appends, so this equals the grid size.
        "n_accepted_points": int(len(points) - n_anchor_appends),
        "n_points_total": int(len(points)),
        "n_adaptive_points": int(len(points) - n_anchor_appends),
        "n_anchor_appends": int(n_anchor_appends),
        "n_rejected_fits": int(n_fits_rejected),
        "n_forced_events": int(n_forced_events),
        "total_fits": int(n_fits_total),
        "V_min": float(Vs.min()) if len(Vs) else float("nan"),
        "V_max": float(Vs.max()) if len(Vs) else float("nan"),
        "AUC_peak": AUC_peak,
        "V_at_AUC_peak": V_at_AUC_peak,
        "eta_at_AUC_peak": eta_at_AUC_peak,
        "AP_peak": AP_peak,
        "V_at_AP_peak": V_at_AP_peak,
        "largest_delta_V": largest_delta_V,
        "largest_delta_V_aligns_with_event": largest_delta_V_aligns_with_event,
        "has_eta_zero": has_eta_zero,
        "total_wall_seconds": float(time.time() - t_run_start),
        "phase": str(phase),
        "eta_grid_source_seed": (-1 if eta_grid_source_seed is None
                                  else int(eta_grid_source_seed)),
        "eta_start": float(eta_start),
        "eta_end": float(eta_end),
    }


def _print_summary(summary):
    print(f"\n  === SUMMARY {summary['dataset']} r={summary['rank_cap']} "
          f"seed={summary['seed']} [{summary['phase']}] ===")
    if summary["phase"] == "replication":
        print(f"    grid source seed: {summary['eta_grid_source_seed']}")
    print(f"    adaptive points: {summary['n_adaptive_points']}  "
          f"anchor appends: {summary['n_anchor_appends']}  "
          f"total points: {summary['n_points_total']}")
    print(f"    rejected fits:   {summary['n_rejected_fits']}  "
          f"forced events: {summary['n_forced_events']}  "
          f"total fits: {summary['total_fits']}")
    print(f"    V range:         [{summary['V_min']:.2f}, {summary['V_max']:.2f}]")
    print(f"    AUC peak:        {summary['AUC_peak']:.4f}  at "
          f"V={summary['V_at_AUC_peak']:.2f}, eta={summary['eta_at_AUC_peak']:+.4f}")
    print(f"    AP peak:         {summary['AP_peak']:.4f}  at "
          f"V={summary['V_at_AP_peak']:.2f}")
    if summary['largest_delta_V_aligns_with_event'] is not None:
        print(f"    largest |ΔV|:    {summary['largest_delta_V']:.2f}  "
              f"aligns with structural event: {summary['largest_delta_V_aligns_with_event']}")
    print(f"    has eta=0:       {summary['has_eta_zero']}")
    print(f"    wall time:       {summary['total_wall_seconds']:.1f}s")


# ===========================================================================
# 12. CSV SCHEMAS + WRITERS
# ===========================================================================
POINT_COLS = [
    "dataset", "seed", "rank_cap", "step_index",
    "eta", "V", "log_V",
    "AUC", "AP", "train_NLL", "test_loglik",
    "k90", "k99", "k_tau_01", "k_tau_001",
    "eigenvalues_sorted", "p_sorted",
    "eval_top1", "eval_top2", "eval_top3", "eval_top5_sum", "eval_top10_sum",
    "p_top1", "p_top2", "p_top3", "p_top5_sum", "p_top10_sum",
    "alpha", "beta",
    "node_bias_mean", "node_bias_std", "node_bias_min", "node_bias_max",
    "delta_eta_used", "delta_V_abs", "delta_V_signed", "delta_V_norm",
    "wall_time_s", "rejection_count_before_accept",
    "is_initial", "is_forced_event", "is_anchor_eta",
    "V_lower_used", "V_upper_used",
    "delta_eta_init", "delta_eta_min", "delta_eta_max",
    "phase", "eta_grid_source_seed",
]

EVENT_COLS = [
    "dataset", "seed", "rank_cap", "from_step", "to_step",
    "from_eta", "to_eta", "midpoint_eta",
    "from_V", "to_V", "midpoint_V",
    "delta_V_abs", "delta_V_signed", "delta_V_norm",
    "delta_AUC", "delta_AP",
    "from_k_tau_01", "to_k_tau_01",
    "rank_transition", "near_degeneracy_event",
    "is_forced_event", "any_structural_event",
    "phase",
]


def write_csv(path: str, rows: List[Dict], cols: List[str]):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def append_csv(path: str, rows: List[Dict], cols: List[str], header: bool):
    if header:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in cols})
    else:
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            for r in rows:
                w.writerow({c: r.get(c, "") for c in cols})


# ===========================================================================
# 13. CLI
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", nargs="+", required=True)
    p.add_argument("--datasets_root", type=str, default="./datasets")
    p.add_argument("--ranks", type=int, nargs="+", default=[64, 128, 256, 512])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--output_dir", type=str, default="./adaptive_eta_out")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--eta_start", type=float, default=-0.25)
    p.add_argument("--eta_end", type=float, default=0.25)
    p.add_argument("--delta_eta_init", type=float, default=0.05)
    p.add_argument("--delta_eta_min", type=float, default=1e-3)
    p.add_argument("--delta_eta_max", type=float, default=0.1)
    p.add_argument("--V_lower", type=float, default=2.0)
    p.add_argument("--V_upper_floor", type=float, default=5.0)
    p.add_argument("--V_upper_alpha", type=float, default=0.03)
    p.add_argument("--max_rejections_per_step", type=int, default=5)
    p.add_argument("--max_total_etas", type=int, default=120)
    p.add_argument("--near_gap_tol", type=float, default=5e-3)
    p.add_argument("--K_restarts", type=int, default=1)
    p.add_argument("--epochs", type=int, default=6000,
                   help="Fixed epoch budget per fit. Default 6000.")
    p.add_argument("--bias_model", type=str, default="node",
                   choices=["global", "node"])
    p.add_argument("--lambda_theta", type=float, default=1e-4)
    p.add_argument("--lambda_bias", type=float, default=1e-4)
    p.add_argument("--no_svd", action="store_true")
    p.add_argument("--beta_max", type=float, default=5.0)
    p.add_argument("--init_beta", type=float, default=1.0)
    p.add_argument("--bias_warmup_epochs", type=int, default=200)
    p.add_argument("--no_share_eta_grid", action="store_true",
                   help="Disable cross-seed eta-grid sharing. If set, every "
                        "seed runs the full adaptive sweep independently "
                        "(legacy behaviour).")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    use_svd = not args.no_svd
    share_eta_grid = not args.no_share_eta_grid

    print("=" * 78)
    print("ADAPTIVE-ETA FRONTIER EXPERIMENT")
    print("=" * 78)
    print(f"datasets:        {args.datasets}")
    print(f"ranks:           {args.ranks}")
    print(f"seeds:           {args.seeds}")
    print(f"eta range:       [{args.eta_start}, {args.eta_end}]")
    print(f"delta_eta:       init={args.delta_eta_init}, "
          f"range=[{args.delta_eta_min}, {args.delta_eta_max}]")
    print(f"V thresholds:    V_lower={args.V_lower}, "
          f"V_upper=max({args.V_upper_floor}, {args.V_upper_alpha}*r)")
    print(f"max_rejections:  {args.max_rejections_per_step}")
    print(f"max_total_etas:  {args.max_total_etas}")
    print(f"K_restarts:      {args.K_restarts}")
    print(f"epochs:          {args.epochs}  (fixed; no early stopping)")
    print(f"SVD:             {use_svd}")
    print(f"bias_model:      {args.bias_model}")
    print(f"share_eta_grid:  {share_eta_grid}  "
          f"(first seed in --seeds searches; rest replicate)")
    print(f"output_dir:      {args.output_dir}")
    print("=" * 78)

    points_all_path = os.path.join(args.output_dir, "frontier_all_points.csv")
    events_all_path = os.path.join(args.output_dir, "frontier_all_events.csv")
    summaries_all_path = os.path.join(args.output_dir, "frontier_all_summaries.json")
    append_csv(points_all_path, [], POINT_COLS, header=True)
    append_csv(events_all_path, [], EVENT_COLS, header=True)
    all_summaries: List[Dict] = []

    overall_t0 = time.time()
    for dataset_name in args.datasets:
        dataset_dir = os.path.join(args.output_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        files = resolve_dataset_dir(dataset_name, root=args.datasets_root)
        split = load_split_from_files(
            files["train_i"], files["train_j"],
            files.get("test_pos_i"), files.get("test_pos_j"),
            files.get("test_neg_i"), files.get("test_neg_j"),
        )
        print(f"\n  Dataset {dataset_name}: n={num_nodes_from_split(split)}, "
              f"|E_train|={len(split['train_i'])}, "
              f"have_test={'test_pos_i' in split and 'test_neg_i' in split}")
        print(f"  per-dataset output dir: {dataset_dir}")

        for rank in args.ranks:
            V_upper = max(args.V_upper_floor, args.V_upper_alpha * rank)
            # Per (dataset, rank), the FIRST seed runs Phase A (adaptive
            # search) and produces the eta-grid. Subsequent seeds run
            # Phase B (replication at the captured grid).
            shared_eta_grid: Optional[List[float]] = None
            shared_eta_grid_source_seed: Optional[int] = None

            for seed_idx, seed in enumerate(args.seeds):
                set_seed(seed)
                use_replication = (
                    share_eta_grid
                    and seed_idx > 0
                    and shared_eta_grid is not None
                )

                if use_replication:
                    points, events, summary = run_replication_one(
                        dataset_name=dataset_name, split=split,
                        rank=rank, seed=seed,
                        eta_grid=shared_eta_grid,
                        eta_grid_source_seed=shared_eta_grid_source_seed,
                        K_restarts=args.K_restarts, epochs=args.epochs,
                        bias_model=args.bias_model,
                        lambda_theta=args.lambda_theta,
                        lambda_bias=args.lambda_bias,
                        use_svd=use_svd, beta_max=args.beta_max,
                        init_beta=args.init_beta,
                        bias_warmup_epochs=args.bias_warmup_epochs,
                        near_gap_tol=args.near_gap_tol, device=args.device,
                        V_lower=args.V_lower, V_upper=V_upper,
                        delta_eta_init=args.delta_eta_init,
                        delta_eta_min=args.delta_eta_min,
                        delta_eta_max=args.delta_eta_max,
                    )
                else:
                    points, events, summary = run_adaptive_frontier_one(
                        dataset_name=dataset_name, split=split,
                        rank=rank, seed=seed,
                        eta_start=args.eta_start, eta_end=args.eta_end,
                        delta_eta_init=args.delta_eta_init,
                        delta_eta_min=args.delta_eta_min,
                        delta_eta_max=args.delta_eta_max,
                        V_lower=args.V_lower, V_upper=V_upper,
                        max_rejections_per_step=args.max_rejections_per_step,
                        max_total_etas=args.max_total_etas,
                        K_restarts=args.K_restarts, epochs=args.epochs,
                        bias_model=args.bias_model,
                        lambda_theta=args.lambda_theta,
                        lambda_bias=args.lambda_bias,
                        use_svd=use_svd, beta_max=args.beta_max,
                        init_beta=args.init_beta,
                        bias_warmup_epochs=args.bias_warmup_epochs,
                        near_gap_tol=args.near_gap_tol, device=args.device,
                    )
                    # Capture the eta-grid for replication on subsequent seeds.
                    if share_eta_grid and shared_eta_grid is None:
                        shared_eta_grid = sorted(
                            set(float(p["eta"]) for p in points)
                        )
                        shared_eta_grid_source_seed = int(seed)

                tag = f"{dataset_name}_r{rank}_seed{seed}"
                p_path = os.path.join(dataset_dir, f"frontier_points_{tag}.csv")
                e_path = os.path.join(dataset_dir, f"frontier_events_{tag}.csv")
                j_path = os.path.join(dataset_dir, f"frontier_full_{tag}.json")
                write_csv(p_path, points, POINT_COLS)
                write_csv(e_path, events, EVENT_COLS)
                with open(j_path, "w") as f:
                    json.dump({
                        "config": {
                            "dataset": dataset_name, "rank": rank, "seed": seed,
                            "eta_start": args.eta_start, "eta_end": args.eta_end,
                            "delta_eta_init": args.delta_eta_init,
                            "delta_eta_min": args.delta_eta_min,
                            "delta_eta_max": args.delta_eta_max,
                            "V_lower": args.V_lower, "V_upper": V_upper,
                            "max_rejections_per_step": args.max_rejections_per_step,
                            "max_total_etas": args.max_total_etas,
                            "K_restarts": args.K_restarts,
                            "epochs": args.epochs,
                            "bias_model": args.bias_model,
                            "lambda_theta": args.lambda_theta,
                            "lambda_bias": args.lambda_bias,
                            "use_svd": use_svd,
                            "beta_max": args.beta_max,
                            "init_beta": args.init_beta,
                            "bias_warmup_epochs": args.bias_warmup_epochs,
                            "near_gap_tol": args.near_gap_tol,
                            "share_eta_grid": share_eta_grid,
                        },
                        "summary": summary,
                        "eta_grid_used": (sorted(set(float(p["eta"]) for p in points))
                                          if use_replication else None),
                        "points": points,
                        "events": events,
                    }, f, indent=2, default=float)
                print(f"\n  saved -> {p_path}")
                print(f"  saved -> {e_path}")
                print(f"  saved -> {j_path}")

                append_csv(points_all_path, points, POINT_COLS, header=False)
                append_csv(events_all_path, events, EVENT_COLS, header=False)
                all_summaries.append(summary)

    with open(summaries_all_path, "w") as f:
        json.dump(all_summaries, f, indent=2, default=float)

    print(f"\n{'='*78}")
    print(f"DONE. Total wall time: {time.time() - overall_t0:.1f}s")
    print(f"  -> {points_all_path}")
    print(f"  -> {events_all_path}")
    print(f"  -> {summaries_all_path}")
    print(f"{'='*78}")


if __name__ == "__main__":
    main()