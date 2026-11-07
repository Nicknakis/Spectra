"""
overparameterization_experiment.py

End-to-end overparameterization study for SPECTRA.

For each (dataset, V_target, rank_cap, seed) cell, this script runs the
two-phase contract:

  Phase 0 (anchor): full-epoch fit at eta=0, no early stop. Provides the
      "no-regularization" reference and the V(0)=V_anchor curve.
  Phase 1 (acquisition): bidirectional or directional-guarded bisection,
      with probe early-stop ON by default. Cheap probes; the search only
      consumes V.
  Phase 2 (retrain): full-epoch cold-start fit at eta_star, no early stop.
      Headline AUC/AP/V come from this model.

Outputs (per cell):
  <out>/<dataset>/cell_<dataset>_V<V*>_r<r>_seed<s>.json
    -- fully serializable; contains anchor metrics, acquisition trace,
       retrain metrics, calibration cost, and the eigenvalues of the
       retrained kernel for post-hoc analysis.
  <out>/<dataset>/L_<dataset>_V<V*>_r<r>_seed<s>.pt
    -- torch state_dict of the retrained model (for mode-connectivity).

After all cells finish, the script produces:
  <out>/figures/variance_<dataset>.pdf
    -- mean / std AUC vs r at each V* (the headline plot).
  <out>/figures/frontier_<dataset>.pdf
    -- AUC vs V_achieved at multiple r, overlay curves.
  <out>/figures/loss_curves_<dataset>.pdf
    -- mean +/- band of training loss per (V*, r), for the optimization-
       stability claim.
  <out>/figures/mode_connectivity_<dataset>.pdf
    -- post-hoc linear interpolation between seed pairs at each cell.
  <out>/tables/variance_<dataset>.csv
    -- the variance-comparison table with bootstrap CIs and Levene p-values.
  <out>/tables/calibration_cost_<dataset>.csv
    -- per-cell calibration cost statistics (probe epochs / full epochs).

Seed-count guidance (printed at script start as well):
  n=3:  preliminary only; not publishable for variance claims.
  n=5:  detects ~3x variance ratios. Acceptable as prelim with explicit label.
  n=10: detects ~2x ratios reliably; bootstrap CIs become tight. Recommended.
  n=20: gold standard.

Example usage:

  python3 overparameterization_experiment.py \\
      --datasets cora ca-grqc \\
      --datasets_root ./datasets \\
      --rank_caps 64 128 256 \\
      --V_targets 16 32 64 \\
      --seeds 0 1 2 3 4 5 6 7 8 9 \\
      --epochs 6000 --tau 0.05 \\
      --enable_probe_early_stop \\
      --output_dir ./op_out

Resume / skip-existing:

  Re-running with the same arguments will skip cells whose output JSON
  already exists. Use --force to overwrite.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Import the corrected engine. Keep this import path stable; if you rename
# the engine file, update here.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import visualize_eta_solution as engine  # noqa: E402


# ===========================================================================
# 1. CELL EXECUTION
# ===========================================================================

def _v_token(V: float) -> str:
    """Filename-safe token for V_target. Avoids '.' in the body of the
    filename (would collide with the extension dot)."""
    return f"{V:g}".replace(".", "p").replace("-", "m")


def _neg_seq_seed_for_cell(base: int, dataset: str, V_target: float,
                            rank: int) -> int:
    """Deterministic per-cell negative-sequence seed.

    Different (V*, r) cells get different sequences (so we don't accidentally
    bias the comparison by reusing the same negatives across capacity
    levels). All model seeds within the same (V*, r) cell share this.
    """
    # Mix dataset name + V* + r into a small integer. Use a stable
    # string hash via Python's built-in hash() with a fixed PYTHONHASHSEED?
    # Avoid that -- it isn't stable across processes. Use a manual mix.
    h = 0
    for c in dataset:
        h = (h * 131 + ord(c)) & 0x7fffffff
    h = (h + int(round(V_target * 1000))) & 0x7fffffff
    h = (h * 17 + int(rank)) & 0x7fffffff
    return (int(base) + h) & 0x7fffffff


def cell_output_path(out_dir: str, dataset: str, V_target: float,
                      rank: int, seed: int) -> str:
    return os.path.join(
        out_dir, dataset,
        f"cell_{dataset}_V{_v_token(V_target)}_r{rank}_seed{seed}.json")


def state_dict_path(out_dir: str, dataset: str, V_target: float,
                     rank: int, seed: int) -> str:
    """Path for the RETRAIN model's state_dict (Phase 2 output).
    Used for mode-connectivity in OP and rank-matched-with-control arms."""
    return os.path.join(
        out_dir, dataset,
        f"L_{dataset}_V{_v_token(V_target)}_r{rank}_seed{seed}.pt")


def anchor_state_dict_path(out_dir: str, dataset: str, V_target: float,
                            rank: int, seed: int) -> str:
    """Path for the ANCHOR model's state_dict (Phase 0 output).
    Used for mode-connectivity in the baseline arm at (V*, r=V*)."""
    return os.path.join(
        out_dir, dataset,
        f"L_anchor_{dataset}_V{_v_token(V_target)}_r{rank}_seed{seed}.pt")


def loss_curve_path(out_dir: str, dataset: str, V_target: float,
                     rank: int, seed: int) -> str:
    return os.path.join(
        out_dir, dataset,
        f"loss_{dataset}_V{_v_token(V_target)}_r{rank}_seed{seed}.npy")


def anchor_loss_curve_path(out_dir: str, dataset: str, V_target: float,
                            rank: int, seed: int) -> str:
    return os.path.join(
        out_dir, dataset,
        f"loss_anchor_{dataset}_V{_v_token(V_target)}_r{rank}_seed{seed}.npy")


def _train_with_loss_logging(model_factory, train_i, train_j, n,
                              epochs, lr, eta, lambda_theta, lambda_bias,
                              bias_warmup_epochs, log_every, log_prefix,
                              tpi, tpj, tni, tnj,
                              enable_probe_early_stop=False,
                              min_probe_epochs=500,
                              probe_V_stability_tol=0.02,
                              probe_stability_patience=3,
                              probe_stability_window=5,
                              train_neg_sampler_seed=None):
    """Wrapper around engine.train_model that also returns the training-loss
    trajectory (sampled every log_every epochs). Loss curves are essential
    for the optimization-stability figure but engine.train_model doesn't
    return them — so we re-run a thin loop here that mirrors its semantics
    exactly and collects loss every log_every epochs.

    SCOPE: This is used ONLY for Phase 0 (anchor at eta=0, full epochs)
    and Phase 2 (retrain at eta_star, full epochs) in run_one_cell. Both
    call sites pass enable_probe_early_stop=False. The probe phase
    (Phase 1) calls engine.train_model directly via fit_one_eta_keep_model
    and uses the engine's full early-stopping logic. So while this loop
    has an early-stop branch for completeness, it is not exercised in
    production and is intentionally simpler than the engine version
    (V-stability only, not V_and_NLL; requires window to be full before
    triggering, which the engine does not).

    NOTE: For Phase 0 / Phase 2 use the loop semantics match engine.train_model
    bit-for-bit (same RNG consumption order, same negative-sampling sequence
    given the same train_neg_sampler_seed, same Adam updates).

    `train_neg_sampler_seed` is forwarded to GPUUniformNonEdgeSampler so
    the training-time negative sequence can be made reproducible across
    runs (overparameterization variance experiment).
    """
    import torch
    import torch.nn.functional as F
    from collections import deque

    if enable_probe_early_stop and log_every <= 0:
        raise ValueError(
            "enable_probe_early_stop requires log_every > 0")

    m = model_factory()
    sampler = engine.GPUUniformNonEdgeSampler(
        n, train_i, train_j, device=train_i.device,
        generator_seed=train_neg_sampler_seed,
    )
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    do_warmup = bias_warmup_epochs > 0
    if do_warmup:
        m.freeze_factor()
    have_test = (tpi is not None and tni is not None)
    loss_history: List[Tuple[int, float]] = []   # (epoch, loss)
    V_history: List[Tuple[int, float]] = []      # (epoch, V)
    do_es = enable_probe_early_stop and log_every > 0
    V_window: deque = deque(maxlen=max(2, probe_stability_window))
    stable_consecutive = 0
    early_stop_epoch: Optional[int] = None

    for epoch in range(1, epochs + 1):
        if do_warmup and epoch == bias_warmup_epochs + 1:
            m.unfreeze_factor()
        m.train()
        opt.zero_grad(set_to_none=True)
        nll, log_v, reg = engine.train_loss_components(
            m, train_i, train_j, sampler, eta=eta,
            lambda_theta=lambda_theta, lambda_bias=lambda_bias,
        )
        loss = nll - eta * log_v + reg
        loss.backward()
        opt.step()

        if log_every > 0 and (epoch % log_every == 0 or epoch == epochs):
            V_now = engine.compute_V_quick(m)
            loss_now = float(loss.detach().item())
            loss_history.append((epoch, loss_now))
            V_history.append((epoch, V_now))

            if do_es:
                V_window.append(V_now)
                if epoch >= min_probe_epochs and len(V_window) >= probe_stability_window:
                    if engine._stability_check(V_window, probe_V_stability_tol):
                        stable_consecutive += 1
                    else:
                        stable_consecutive = 0
                    if stable_consecutive >= probe_stability_patience and epoch < epochs:
                        early_stop_epoch = epoch
                        break

    return m, loss_history, V_history, early_stop_epoch


def run_one_cell(dataset: str, datasets_root: str,
                 V_target: float, rank: int, seed: int,
                 args, out_dir: str, force: bool = False) -> Dict:
    """Run one (dataset, V*, r, seed) cell with the three-phase contract.

    Returns the cell-result dict and writes it (plus state_dict and loss
    curves) to disk.
    """
    cell_path = cell_output_path(out_dir, dataset, V_target, rank, seed)
    if (not force) and os.path.isfile(cell_path):
        with open(cell_path) as f:
            existing = json.load(f)
        existing["_skipped_existing"] = True
        return existing

    os.makedirs(os.path.dirname(cell_path), exist_ok=True)

    print("\n" + "=" * 78)
    print(f"CELL: dataset={dataset}  V*={V_target}  r={rank}  seed={seed}")
    print("=" * 78)

    engine.set_seed(seed)

    # --- Load data ---
    files = engine.resolve_dataset_dir(dataset, root=datasets_root)
    split = engine.load_split_from_files(
        files["train_i"], files["train_j"],
        files.get("test_pos_i"), files.get("test_pos_j"),
        files.get("test_neg_i"), files.get("test_neg_j"),
    )
    n = engine.num_nodes_from_split(split)
    ti, tj, tpi, tpj, tni, tnj, have_test = engine.prepare_tensors(
        split, args.device,
    )
    eval_sampler = engine.GPUUniformNonEdgeSampler(
        n, ti, tj, device=args.device)

    # ---- Cell-level negative-sequence seed (NEW) ----
    # When --shared_neg_sequence is on, all model seeds within this cell
    # see the same training-time negative sequence. Different cells get
    # different sequences so we don't bias capacity-comparisons by reusing
    # negatives across r values.
    if getattr(args, "shared_neg_sequence", False):
        cell_neg_seed: Optional[int] = _neg_seq_seed_for_cell(
            args.neg_sequence_seed_base, dataset, V_target, rank,
        )
        print(f"  [shared_neg_sequence] cell_neg_seed={cell_neg_seed}")
    else:
        cell_neg_seed = None

    if V_target > rank:
        result = {
            "dataset": dataset, "V_target": V_target,
            "rank_cap": rank, "seed": seed,
            "status": "impossible_target_V_gt_rank",
            "anchor": None, "acquisition": None, "retrain": None,
        }
        with open(cell_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[SKIP] V_target={V_target} > rank={rank}; recorded as impossible.")
        return result

    # =====================================================================
    # PHASE 0: ANCHOR at eta=0, full epochs, no early stop, with loss curve.
    # =====================================================================
    print("\n--- PHASE 0: anchor at eta=0 (full epochs) ---")
    t_anchor = time.time()

    def anchor_factory():
        return engine.make_model(
            args.bias_model, n, rank, args.device,
            use_svd=not args.no_svd, beta_max=args.beta_max,
            init_beta=args.init_beta,
        )

    m_anchor, anchor_loss_hist, anchor_V_hist, _ = _train_with_loss_logging(
        anchor_factory, ti, tj, n,
        epochs=args.epochs, lr=args.lr, eta=0.0,
        lambda_theta=args.lambda_theta, lambda_bias=args.lambda_bias,
        bias_warmup_epochs=args.bias_warmup_epochs,
        log_every=args.log_every, log_prefix="    [anchor] ",
        tpi=tpi, tpj=tpj, tni=tni, tnj=tnj,
        enable_probe_early_stop=False,
        train_neg_sampler_seed=cell_neg_seed,
    )
    # MC NLL (NEW): more reliable than single-draw.
    if args.n_mc_eval and args.n_mc_eval > 1:
        anchor_train_nll, anchor_train_nll_std, _ = engine.evaluate_train_nll_mc(
            m_anchor, ti, tj, n, n_mc=int(args.n_mc_eval),
            mc_generator_seed=int(args.mc_generator_seed),
        )
    else:
        anchor_train_nll = engine.evaluate_train_nll(
            m_anchor, ti, tj, eval_sampler)
        anchor_train_nll_std = float("nan")
    anchor_diag = engine.diagnostics(m_anchor)
    if have_test:
        anchor_loglik = engine.evaluate_test_loglik(m_anchor, tpi, tpj, tni, tnj)
        anchor_auc, anchor_ap = engine.eval_auc_and_ap(
            m_anchor, tpi, tpj, tni, tnj)
    else:
        anchor_loglik = float("nan")
        anchor_auc, anchor_ap = float("nan"), float("nan")
    anchor_result = {
        "wall_seconds": time.time() - t_anchor,
        "epochs": args.epochs,
        "V": float(anchor_diag["V"]),
        "AUC": float(anchor_auc), "AP": float(anchor_ap),
        "test_loglik": float(anchor_loglik),
        "train_NLL": float(anchor_train_nll),
        "train_NLL_mc_std": float(anchor_train_nll_std),
        "k_tau_01": int(anchor_diag["k_tau_01"]),
        "k_tau_001": int(anchor_diag["k_tau_001"]),
        "k90": int(anchor_diag["k90"]), "k99": int(anchor_diag["k99"]),
        "evals_top16": [float(x) for x in
                         anchor_diag["evals_sorted"][:16]],
    }
    print(f"  [anchor] V={anchor_result['V']:.2f}  "
          f"AUC={anchor_result['AUC']:.4f}  AP={anchor_result['AP']:.4f}  "
          f"wall={anchor_result['wall_seconds']:.1f}s")

    # Save anchor's loss curve (for the optimization-stability figure).
    np.save(anchor_loss_curve_path(out_dir, dataset, V_target, rank, seed),
            np.array(anchor_loss_hist, dtype=np.float64))
    # Save anchor's state_dict (for mode connectivity on the baseline arm).
    torch.save(m_anchor.state_dict(),
               anchor_state_dict_path(out_dir, dataset, V_target, rank, seed))

    # =====================================================================
    # PHASE 1: ACQUISITION via bisection. Probe early-stop optional.
    # =====================================================================
    print("\n--- PHASE 1: bisection acquisition ---")
    cache = engine.ProbeCache()
    # Seed the cache with the anchor we just trained — saves the bisection
    # one probe call. We construct the cache entry with the anchor's REAL
    # AUC/V (it was full-epoch, not early-stopped).
    anchor_cache_entry = {
        "model": m_anchor, "restart": 0,
        "train_nll": anchor_train_nll,
        "test_auc": anchor_auc, "test_ap": anchor_ap,
        "test_loglik": anchor_loglik,
        "V": anchor_diag["V"], "log_V": anchor_diag["log_V"],
        "k_tau_01": anchor_diag["k_tau_01"],
        "k_tau_001": anchor_diag["k_tau_001"],
        "k90": anchor_diag["k90"], "k99": anchor_diag["k99"],
        "evals_sorted": anchor_diag["evals_sorted"],
        "p_sorted": anchor_diag["p_sorted"],
        "beta": float(m_anchor.beta().detach().cpu().item()),
        "epoch_first_target_hit": None,
        "early_stop_epoch": None,
        "was_early_stopped": False,
        "selected_restart": 0,
        "alpha": float("nan"),
        "node_bias_mean": float("nan"), "node_bias_std": float("nan"),
        "node_bias_min": float("nan"), "node_bias_max": float("nan"),
    }
    bias_sum = engine.extract_bias_summary(m_anchor, args.bias_model)
    anchor_cache_entry.update(bias_sum)
    cache.put(0.0, anchor_cache_entry)

    summary = engine.run_target_v_search(
        dataset_name=dataset, split=split,
        rank=rank, seed=seed, V_target=float(V_target),
        cache=cache,
        eta_anchor=0.0,
        eta_initial_step=args.eta_initial_step,
        eta_expand_factor=args.eta_expand_factor,
        eta_min=args.eta_min, eta_max=args.eta_max,
        tau=args.tau,
        max_bracket_expansions=args.max_bracket_expansions,
        max_bisection_steps=args.max_bisection_steps,
        min_eta_interval=args.min_eta_interval,
        K_restarts=args.K_restarts, epochs=args.epochs,
        bias_model=args.bias_model,
        lambda_theta=args.lambda_theta, lambda_bias=args.lambda_bias,
        lr=args.lr,
        use_svd=not args.no_svd, beta_max=args.beta_max,
        init_beta=args.init_beta,
        bias_warmup_epochs=args.bias_warmup_epochs,
        device=args.device, log_every=args.log_every,
        search_mode=args.search_mode,
        monotone_guard_tol=args.monotone_guard_tol,
        warm_start_probes=False,
        enable_probe_early_stop=args.enable_probe_early_stop,
        min_probe_epochs=args.min_probe_epochs,
        probe_V_stability_tol=args.probe_V_stability_tol,
        probe_NLL_stability_tol=args.probe_NLL_stability_tol,
        probe_stability_patience=args.probe_stability_patience,
        probe_stability_window=args.probe_stability_window,
        probe_stop_metric=args.probe_stop_metric,
        train_neg_sampler_seed=cell_neg_seed,
        n_mc_eval=int(args.n_mc_eval),
        mc_generator_seed=int(args.mc_generator_seed),
    )
    eta_star = float(summary["eta_selected"])
    print(f"  [acquisition done] eta*={eta_star:+.6f}  "
          f"V_at_probe={summary['V_selected']:.2f}  "
          f"status={summary['status']}  "
          f"probes={summary['n_probes_total']}")

    # Tabulate probe costs for the calibration-cost paragraph.
    probe_costs = []
    for eta_key in cache.keys():
        probe = cache.get(eta_key)
        es = probe.get("early_stop_epoch")
        cost = es if es is not None else args.epochs
        probe_costs.append({
            "eta": float(eta_key),
            "V": float(probe["V"]),
            "early_stop_epoch": es,
            "cost_epochs": int(cost),
            "was_early_stopped": bool(probe.get("was_early_stopped", False)),
        })
    total_probe_epochs = sum(p["cost_epochs"] for p in probe_costs)

    acquisition_result = {
        "summary": summary,
        "probe_costs": probe_costs,
        "total_probe_epochs": int(total_probe_epochs),
        "calibration_cost_in_full_runs":
            float(total_probe_epochs / args.epochs),
    }

    if math.isnan(eta_star):
        # Impossible-target should have been caught above; defensive.
        result = {
            "dataset": dataset, "V_target": V_target,
            "rank_cap": rank, "seed": seed,
            "status": "acquisition_failed_nan_eta",
            "anchor": anchor_result,
            "acquisition": acquisition_result, "retrain": None,
        }
        with open(cell_path, "w") as f:
            json.dump(result, f, indent=2)
        return result

    # =====================================================================
    # PHASE 2: RETRAIN at eta_star, full epochs, no early stop, with loss.
    # =====================================================================
    print(f"\n--- PHASE 2: retrain at eta*={eta_star:+.6f} (full epochs) ---")
    t_retrain = time.time()
    engine.set_seed(seed)   # match anchor's seed regime; kk_index keeps RNGs distinct

    def retrain_factory():
        # Use a fresh seed_r distinct from any probe to avoid RNG coupling.
        seed_r = seed * 1000 + 9999 * 100 + 0
        torch.manual_seed(seed_r); np.random.seed(seed_r)
        return engine.make_model(
            args.bias_model, n, rank, args.device,
            use_svd=not args.no_svd, beta_max=args.beta_max,
            init_beta=args.init_beta,
        )

    m_retrain, retrain_loss_hist, retrain_V_hist, _ = _train_with_loss_logging(
        retrain_factory, ti, tj, n,
        epochs=args.epochs, lr=args.lr, eta=eta_star,
        lambda_theta=args.lambda_theta, lambda_bias=args.lambda_bias,
        bias_warmup_epochs=args.bias_warmup_epochs,
        log_every=args.log_every, log_prefix="    [retrain] ",
        tpi=tpi, tpj=tpj, tni=tni, tnj=tnj,
        enable_probe_early_stop=False,
        train_neg_sampler_seed=cell_neg_seed,
    )
    if args.n_mc_eval and args.n_mc_eval > 1:
        retrain_train_nll, retrain_train_nll_std, _ = engine.evaluate_train_nll_mc(
            m_retrain, ti, tj, n, n_mc=int(args.n_mc_eval),
            mc_generator_seed=int(args.mc_generator_seed),
        )
    else:
        retrain_train_nll = engine.evaluate_train_nll(
            m_retrain, ti, tj, eval_sampler)
        retrain_train_nll_std = float("nan")
    retrain_diag = engine.diagnostics(m_retrain)
    if have_test:
        retrain_loglik = engine.evaluate_test_loglik(m_retrain, tpi, tpj, tni, tnj)
        retrain_auc, retrain_ap = engine.eval_auc_and_ap(
            m_retrain, tpi, tpj, tni, tnj)
    else:
        retrain_loglik = float("nan")
        retrain_auc, retrain_ap = float("nan"), float("nan")
    retrain_result = {
        "wall_seconds": time.time() - t_retrain,
        "epochs": args.epochs,
        "eta_star": eta_star,
        "V": float(retrain_diag["V"]),
        "AUC": float(retrain_auc), "AP": float(retrain_ap),
        "test_loglik": float(retrain_loglik),
        "train_NLL": float(retrain_train_nll),
        "train_NLL_mc_std": float(retrain_train_nll_std),
        "k_tau_01": int(retrain_diag["k_tau_01"]),
        "k_tau_001": int(retrain_diag["k_tau_001"]),
        "k90": int(retrain_diag["k90"]), "k99": int(retrain_diag["k99"]),
        "evals_full": [float(x) for x in retrain_diag["evals_sorted"]],
        "abs_error_V": float(abs(retrain_diag["V"] - V_target)),
        "rel_error_V": float(abs(retrain_diag["V"] - V_target)
                             / max(V_target, 1e-12)),
    }
    print(f"  [retrain] V={retrain_result['V']:.2f}  "
          f"AUC={retrain_result['AUC']:.4f}  AP={retrain_result['AP']:.4f}  "
          f"rel_err_V={retrain_result['rel_error_V']:.4f}  "
          f"wall={retrain_result['wall_seconds']:.1f}s")

    # Save retrain's loss curve and state_dict for mode-connectivity etc.
    np.save(loss_curve_path(out_dir, dataset, V_target, rank, seed),
            np.array(retrain_loss_hist, dtype=np.float64))
    torch.save(m_retrain.state_dict(),
               state_dict_path(out_dir, dataset, V_target, rank, seed))

    # =====================================================================
    # WRITE CELL JSON
    # =====================================================================
    result = {
        "dataset": dataset, "V_target": V_target,
        "rank_cap": rank, "seed": seed,
        "status": "ok",
        "anchor": anchor_result,
        "acquisition": acquisition_result,
        "retrain": retrain_result,
        # "config" snapshot for reproducibility:
        "config": {
            "epochs": args.epochs, "lr": args.lr,
            "bias_model": args.bias_model,
            "K_restarts": args.K_restarts,
            "tau": args.tau, "search_mode": args.search_mode,
            "enable_probe_early_stop": args.enable_probe_early_stop,
            "min_probe_epochs": args.min_probe_epochs,
            "probe_V_stability_tol": args.probe_V_stability_tol,
            "probe_stability_patience": args.probe_stability_patience,
            "probe_stability_window": args.probe_stability_window,
            "shared_neg_sequence":
                bool(getattr(args, "shared_neg_sequence", False)),
            "neg_sequence_seed_base":
                int(getattr(args, "neg_sequence_seed_base", -1)),
            "cell_neg_seed":
                cell_neg_seed if cell_neg_seed is not None else None,
            "n_mc_eval": int(getattr(args, "n_mc_eval", 1)),
            "mc_generator_seed":
                int(getattr(args, "mc_generator_seed", 12345)),
        },
    }
    with open(cell_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[SAVED] {cell_path}")
    return result


# ===========================================================================
# 2. ANALYSIS — VARIANCE COMPARISON
# ===========================================================================

def _bootstrap_var_ratio(x: np.ndarray, y: np.ndarray,
                          n_boot: int = 10000, rng_seed: int = 0
                          ) -> Tuple[float, Tuple[float, float]]:
    """Bootstrap 95% CI for var(y)/var(x). Both arrays should have ddof=1
    sample variance. Returns (point estimate, (ci_low, ci_high))."""
    x = np.asarray(x); y = np.asarray(y)
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        return float("nan"), (float("nan"), float("nan"))
    rng = np.random.default_rng(rng_seed)
    point = float(np.var(y, ddof=1) / max(np.var(x, ddof=1), 1e-12))
    ratios = np.empty(n_boot)
    for i in range(n_boot):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        vx = np.var(xb, ddof=1)
        ratios[i] = (np.var(yb, ddof=1) / vx) if vx > 1e-12 else np.nan
    ratios = ratios[~np.isnan(ratios)]
    if len(ratios) < 100:
        return point, (float("nan"), float("nan"))
    return point, (float(np.percentile(ratios, 2.5)),
                    float(np.percentile(ratios, 97.5)))


def _levene_pvalue(*samples) -> float:
    """Brown-Forsythe (Levene with median). Returns p-value or NaN."""
    try:
        from scipy.stats import levene
    except ImportError:
        return float("nan")
    cleaned = [np.asarray(s)[~np.isnan(np.asarray(s))] for s in samples]
    cleaned = [s for s in cleaned if len(s) >= 2]
    if len(cleaned) < 2:
        return float("nan")
    try:
        return float(levene(*cleaned, center="median").pvalue)
    except Exception:
        return float("nan")


def collect_cells(out_dir: str, dataset: str) -> List[Dict]:
    """Load all cell_*.json files for one dataset."""
    ds_dir = os.path.join(out_dir, dataset)
    if not os.path.isdir(ds_dir):
        return []
    cells = []
    for fn in sorted(os.listdir(ds_dir)):
        if fn.startswith("cell_") and fn.endswith(".json"):
            with open(os.path.join(ds_dir, fn)) as f:
                cells.append(json.load(f))
    return cells


def variance_table(cells: List[Dict], V_targets: List[float],
                    rank_caps: List[int], baseline_r: int,
                    out_csv: str) -> List[Dict]:
    """Three-arm variance table.

    For each V*, three arms are produced:

      1. baseline (anchor at r=V*, eta=0):
            AUCs come from cell["anchor"]["AUC"] for cells with rank=V*.
            "no spectral control, rank cap matched to the desired V*."

      2. rank-matched-with-control (retrain at r=V*, eta selected):
            AUCs come from cell["retrain"]["AUC"] for cells with rank=V*.
            "spectral control on, but no overparameterization headroom."

      3. OP (retrain at r > V*, eta selected):
            AUCs come from cell["retrain"]["AUC"] for cells with rank > V*.
            "your method: overparameterized + spectral control."

    Variance ratios are computed against arm-1 (the baseline) for the
    same V*. The legacy `baseline_r` argument is ignored when arm-1 is
    available; falls back to per-r retrain AUCs only if no anchor is
    found at r=V*.
    """
    by_cell: Dict[Tuple[float, int], List[Dict]] = defaultdict(list)
    for c in cells:
        if c.get("status") != "ok": continue
        by_cell[(float(c["V_target"]), int(c["rank_cap"]))].append(c)

    rows: List[Dict] = []
    for V in V_targets:
        # ---- ARM 1: baseline (anchor at r=V*, eta=0) ----
        r_eq = int(round(V))
        anchor_cells = [c for c in by_cell.get((V, r_eq), [])
                         if c.get("anchor") is not None]
        baseline_aucs = np.array([
            float(c["anchor"]["AUC"]) for c in anchor_cells
            if not math.isnan(c["anchor"]["AUC"])
        ])
        baseline_Vs = np.array([
            float(c["anchor"]["V"]) for c in anchor_cells
            if not math.isnan(c["anchor"]["V"])
        ])
        # n_seeds counts USABLE AUC measurements (consistent with AUC_mean
        # / AUC_std). Anchor cells with NaN AUC (e.g. no test data) are
        # excluded from variance estimation.
        n_baseline = len(baseline_aucs)
        baseline_row = {
            "V_target": V, "arm": "baseline",
            "rank_cap": r_eq,
            "description": "anchor at r=V*, eta=0 (no spectral control)",
            "n_seeds": n_baseline,
            "AUC_mean": float(np.mean(baseline_aucs))
                if n_baseline >= 1 else float("nan"),
            "AUC_std": float(np.std(baseline_aucs, ddof=1))
                if n_baseline >= 2 else float("nan"),
            "V_mean": float(np.mean(baseline_Vs))
                if len(baseline_Vs) >= 1 else float("nan"),
            "V_std": float(np.std(baseline_Vs, ddof=1))
                if len(baseline_Vs) >= 2 else float("nan"),
            "var_ratio_vs_baseline": 1.0 if n_baseline >= 2 else float("nan"),
            "var_ratio_ci_low": float("nan"),
            "var_ratio_ci_high": float("nan"),
            "levene_p_vs_baseline": float("nan"),
            "is_baseline": True,
        }
        rows.append(baseline_row)

        # ---- ARM 2: rank-matched-with-control (retrain at r=V*) ----
        rmc_cells = [c for c in by_cell.get((V, r_eq), [])
                      if c.get("retrain") is not None]
        rmc_aucs = np.array([
            float(c["retrain"]["AUC"]) for c in rmc_cells
            if not math.isnan(c["retrain"]["AUC"])
        ])
        rmc_Vs = np.array([
            float(c["retrain"]["V"]) for c in rmc_cells
            if not math.isnan(c["retrain"]["V"])
        ])
        n_rmc = len(rmc_aucs)   # consistent with AUC stats
        if n_rmc >= 2 and len(baseline_aucs) >= 2:
            point, (lo, hi) = _bootstrap_var_ratio(baseline_aucs, rmc_aucs)
            p_lev = _levene_pvalue(baseline_aucs, rmc_aucs)
        else:
            point, lo, hi, p_lev = (float("nan"),) * 4
        rows.append({
            "V_target": V, "arm": "rank_matched_with_control",
            "rank_cap": r_eq,
            "description": "retrain at r=V*, eta selected by bisection",
            "n_seeds": n_rmc,
            "AUC_mean": float(np.mean(rmc_aucs))
                if n_rmc >= 1 else float("nan"),
            "AUC_std": float(np.std(rmc_aucs, ddof=1))
                if n_rmc >= 2 else float("nan"),
            "V_mean": float(np.mean(rmc_Vs))
                if len(rmc_Vs) >= 1 else float("nan"),
            "V_std": float(np.std(rmc_Vs, ddof=1))
                if len(rmc_Vs) >= 2 else float("nan"),
            "var_ratio_vs_baseline": point,
            "var_ratio_ci_low": lo,
            "var_ratio_ci_high": hi,
            "levene_p_vs_baseline": p_lev,
            "is_baseline": False,
        })

        # ---- ARM 3: OP (retrain at r > V*) ----
        for r in [rk for rk in rank_caps if rk > V]:
            op_cells = [c for c in by_cell.get((V, r), [])
                         if c.get("retrain") is not None]
            op_aucs = np.array([
                float(c["retrain"]["AUC"]) for c in op_cells
                if not math.isnan(c["retrain"]["AUC"])
            ])
            op_Vs = np.array([
                float(c["retrain"]["V"]) for c in op_cells
                if not math.isnan(c["retrain"]["V"])
            ])
            n_op = len(op_aucs)   # consistent with AUC stats
            if n_op >= 2 and len(baseline_aucs) >= 2:
                point, (lo, hi) = _bootstrap_var_ratio(baseline_aucs, op_aucs)
                p_lev = _levene_pvalue(baseline_aucs, op_aucs)
            else:
                point, lo, hi, p_lev = (float("nan"),) * 4
            rows.append({
                "V_target": V, "arm": f"OP_r{r}",
                "rank_cap": r,
                "description": f"retrain at r={r} > V*={V:g}, eta selected",
                "n_seeds": n_op,
                "AUC_mean": float(np.mean(op_aucs))
                    if n_op >= 1 else float("nan"),
                "AUC_std": float(np.std(op_aucs, ddof=1))
                    if n_op >= 2 else float("nan"),
                "V_mean": float(np.mean(op_Vs))
                    if len(op_Vs) >= 1 else float("nan"),
                "V_std": float(np.std(op_Vs, ddof=1))
                    if len(op_Vs) >= 2 else float("nan"),
                "var_ratio_vs_baseline": point,
                "var_ratio_ci_low": lo,
                "var_ratio_ci_high": hi,
                "levene_p_vs_baseline": p_lev,
                "is_baseline": False,
            })

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"\n[variance_table] -> {out_csv}")
    # Header
    print(f"\n  V*  | arm                          |    r |  n | "
          f"V_achieved (mean ± std)  | rel_err |  AUC mean  |  AUC std  | "
          f"var ratio vs baseline (95% CI) | Lev p")
    print("  " + "-" * 145)
    for r in rows:
        if r["is_baseline"]:
            if r["n_seeds"] >= 2:
                ratio_str = "1.000 (baseline)"
            else:
                ratio_str = "      — (no data)"
        elif not math.isnan(r["var_ratio_vs_baseline"]):
            ratio_str = (f"{r['var_ratio_vs_baseline']:.3f} "
                         f"[{r['var_ratio_ci_low']:.2f},"
                         f"{r['var_ratio_ci_high']:.2f}]")
        else:
            ratio_str = "      —"
        p_str = (f"{r['levene_p_vs_baseline']:.3f}"
                  if not math.isnan(r["levene_p_vs_baseline"]) else "—")
        # V_achieved ± std with rel_err flag.
        if not math.isnan(r["V_mean"]):
            if not math.isnan(r["V_std"]):
                v_str = f"{r['V_mean']:6.2f} ± {r['V_std']:5.2f}"
            else:
                v_str = f"{r['V_mean']:6.2f} ±   —  "
            rel_err = abs(r["V_mean"] - r["V_target"]) / max(r["V_target"], 1e-12)
            if rel_err <= 0.05:
                rel_flag = f"  {rel_err:.3f} "
            elif rel_err <= 0.15:
                rel_flag = f"  {rel_err:.3f}*"
            else:
                rel_flag = f"  {rel_err:.3f}!"
        else:
            v_str = "       — ±   —  "
            rel_flag = "    —  "
        auc_str = (f"{r['AUC_mean']:9.4f}"
                    if not math.isnan(r["AUC_mean"]) else "       —")
        std_str = (f"{r['AUC_std']:8.4f}"
                    if not math.isnan(r["AUC_std"]) else "      —")
        print(f"  {r['V_target']:>4g} | {r['arm']:<28s} | "
              f"{r['rank_cap']:>4d} | {r['n_seeds']:>2d} | "
              f"{v_str:>22s}  | {rel_flag} | {auc_str}  | {std_str} | "
              f"{ratio_str:>30} | {p_str:>5}")
    print("  Legend: rel_err marker — '*' = within 15% of V_target; "
          "'!' = >15% off (probe-vs-retrain mismatch).")
    print("  Tighter targeting can be obtained with --K_restarts 2-3 or "
          "--tau 0.02 at extra cost.")
    return rows


# ===========================================================================
# 3. FIGURES
# ===========================================================================

# Color palettes that are colorblind-safe and print well.
def _color_for_rank(r: int, all_ranks: List[int]) -> Tuple[float, ...]:
    cmap = plt.get_cmap("viridis")
    rs = sorted(all_ranks)
    if len(rs) == 1: return cmap(0.5)
    return cmap(rs.index(r) / (len(rs) - 1))


def _color_for_V(V: float, all_Vs: List[float]) -> Tuple[float, ...]:
    cmap = plt.get_cmap("plasma")
    vs = sorted(all_Vs)
    if len(vs) == 1: return cmap(0.5)
    return cmap(vs.index(V) / (len(vs) - 1))


def fig_variance(rows: List[Dict], V_targets: List[float],
                  rank_caps: List[int], save_path: str):
    """Three-arm variance figure.

    Two panels: (a) mean AUC vs r, (b) std AUC vs r. For each V*:
      - baseline arm (anchor at r=V*)         drawn at x=V*, marker='s'
      - rank_matched_with_control (retrain at r=V*)  drawn at x=V*, marker='o'
      - OP arms (retrain at r > V*)            drawn at x=r, marker='^'
    Colors encode V*; markers encode arm. A separate legend describes
    arm markers.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
    by_V: Dict[float, List[Dict]] = defaultdict(list)
    for r in rows:
        by_V[r["V_target"]].append(r)

    arm_marker = {
        "baseline": ("s", "anchor at r=V*, η=0 (baseline)"),
        "rank_matched_with_control": ("o", "retrain at r=V*, η selected"),
        # OP arms have arm string "OP_r{r}" -- handled separately.
    }

    for V in sorted(by_V):
        rs_all = by_V[V]
        c = _color_for_V(V, V_targets)
        # OP arms: line connecting points at r > V*.
        op_rs = sorted([r for r in rs_all if r["arm"].startswith("OP_")
                         and not math.isnan(r["AUC_mean"])],
                       key=lambda r: r["rank_cap"])
        if op_rs:
            xs = [r["rank_cap"] for r in op_rs]
            means = [r["AUC_mean"] for r in op_rs]
            stds = [r["AUC_std"] for r in op_rs]
            sems = [(s / math.sqrt(rr["n_seeds"])
                     if rr["n_seeds"] > 0 and not math.isnan(s)
                     else 0.0)
                    for rr, s in zip(op_rs, stds)]
            axes[0].errorbar(xs, means, yerr=sems,
                             fmt="^-", color=c, lw=2.0, ms=8,
                             capsize=4, mew=0.6,
                             markeredgecolor="black",
                             label=f"V*={V:g} OP arms (r>V*)")
            axes[1].plot(xs, stds, "^-", color=c, lw=2.0, ms=8,
                          markeredgecolor="black", mew=0.6)
        # Baseline arm: square at r=V*.
        bl = next((r for r in rs_all if r["arm"] == "baseline"
                    and not math.isnan(r["AUC_mean"])), None)
        if bl is not None:
            sem_b = (bl["AUC_std"] / math.sqrt(bl["n_seeds"])
                      if bl["n_seeds"] > 1 and not math.isnan(bl["AUC_std"])
                      else 0.0)
            axes[0].errorbar([bl["rank_cap"]], [bl["AUC_mean"]],
                              yerr=[sem_b],
                              fmt="s", color=c, ms=11, capsize=4,
                              markeredgecolor="black", mew=0.8,
                              label=f"V*={V:g} baseline (anchor)")
            if not math.isnan(bl["AUC_std"]):
                axes[1].plot([bl["rank_cap"]], [bl["AUC_std"]],
                              "s", color=c, ms=11,
                              markeredgecolor="black", mew=0.8)
        # Rank-matched-with-control arm: circle at r=V*.
        rmc = next((r for r in rs_all
                     if r["arm"] == "rank_matched_with_control"
                     and not math.isnan(r["AUC_mean"])), None)
        if rmc is not None:
            sem_r = (rmc["AUC_std"] / math.sqrt(rmc["n_seeds"])
                      if rmc["n_seeds"] > 1 and not math.isnan(rmc["AUC_std"])
                      else 0.0)
            axes[0].errorbar([rmc["rank_cap"]], [rmc["AUC_mean"]],
                              yerr=[sem_r],
                              fmt="o", color=c, ms=10, capsize=4,
                              markeredgecolor="black", mew=0.8,
                              label=f"V*={V:g} rank-matched + control")
            if not math.isnan(rmc["AUC_std"]):
                axes[1].plot([rmc["rank_cap"]], [rmc["AUC_std"]],
                              "o", color=c, ms=10,
                              markeredgecolor="black", mew=0.8)

    axes[0].set_xlabel("rank cap r", fontsize=15)
    axes[0].set_ylabel("mean test AUC ± SEM", fontsize=15)
    axes[0].set_xscale("log", base=2)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("(a) Mean AUC: three-arm comparison at each V*",
                       fontsize=12)
    axes[0].legend(fontsize=8, loc="best", ncol=1)

    axes[1].set_xlabel("rank cap r", fontsize=15)
    axes[1].set_ylabel("std AUC across seeds", fontsize=15)
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].set_title("(b) Variance of AUC: lower at higher r is the claim",
                       fontsize=12)

    # Add an arm-marker legend on panel (b).
    arm_handles = [
        Line2D([0], [0], marker="s", color="gray", linestyle="",
                markersize=11, markeredgecolor="black", mew=0.8,
                label="baseline (r=V*, η=0)"),
        Line2D([0], [0], marker="o", color="gray", linestyle="",
                markersize=10, markeredgecolor="black", mew=0.8,
                label="rank-matched + control (r=V*, η selected)"),
        Line2D([0], [0], marker="^", color="gray", linestyle="-",
                markersize=8, markeredgecolor="black", mew=0.6,
                label="OP (r > V*, η selected)  [our method]"),
    ]
    axes[1].legend(handles=arm_handles, fontsize=8, loc="best")

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[variance figure] -> {save_path}")


def fig_targeting_precision(rows: List[Dict], V_targets: List[float],
                             rank_caps: List[int], save_path: str):
    """Sanity-check figure: V_achieved (mean ± std across seeds) vs V_target,
    one column of dots per arm. The diagonal y=x line shows perfect
    targeting. Deviations from the diagonal reveal probe-vs-retrain V
    mismatch (expected at high overparameterization).

    This figure is intended for the paper supplement: it lets reviewers
    verify that the experimental claim 'matched V*' holds in practice.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    # Per arm: collect (V_target, V_mean, V_std) tuples.
    by_arm: Dict[str, List[Tuple[float, float, float, int]]] = defaultdict(list)
    for r in rows:
        if math.isnan(r["V_mean"]):
            continue
        # Bucket OP arms together for readability.
        if r["arm"].startswith("OP_"):
            arm_key = "OP (r > V*)"
        else:
            arm_key = r["arm"]
        v_std = r["V_std"] if not math.isnan(r["V_std"]) else 0.0
        by_arm[arm_key].append((r["V_target"], r["V_mean"], v_std,
                                  r["rank_cap"]))

    arm_styles = {
        "baseline":                     ("s", "tab:blue",
                                          "baseline (anchor at r=V*, η=0)"),
        "rank_matched_with_control":    ("o", "tab:orange",
                                          "rank-matched + control (r=V*, η selected)"),
        "OP (r > V*)":                  ("^", "tab:green",
                                          "OP (r > V*, η selected)"),
    }

    # Diagonal y=x line.
    if V_targets:
        v_lo = min(V_targets) * 0.5
        v_hi = max(V_targets) * 2.0
        ax.plot([v_lo, v_hi], [v_lo, v_hi], "--", color="gray", lw=1.0,
                label="y = x (perfect targeting)")
        # ±5%, ±15% bands.
        xs = np.linspace(v_lo, v_hi, 50)
        ax.fill_between(xs, xs * 0.95, xs * 1.05, color="gray", alpha=0.10,
                         label="±5% band")
        ax.fill_between(xs, xs * 0.85, xs * 1.15, color="gray", alpha=0.05,
                         label="±15% band")

    for arm_key, points in by_arm.items():
        marker, color, label = arm_styles.get(
            arm_key, ("d", "tab:gray", arm_key))
        # Within an arm, jitter x slightly so multiple ranks at same V*
        # don't overlap (only relevant for OP).
        for i, (V_t, V_mean, V_std, r) in enumerate(points):
            # No jitter — different ranks plot at same V_target by design.
            ax.errorbar([V_t], [V_mean], yerr=[V_std],
                        fmt=marker, color=color, ms=10, capsize=4,
                        markeredgecolor="black", mew=0.6,
                        alpha=0.85,
                        label=label if i == 0 else None)
            # Annotate rank if OP.
            if arm_key == "OP (r > V*)":
                ax.annotate(f"r={r}",
                             xy=(V_t, V_mean),
                             xytext=(7, 3), textcoords="offset points",
                             fontsize=8, color=color, alpha=0.8)

    ax.set_xlabel("V_target", fontsize=14)
    ax.set_ylabel("V_achieved (mean ± std across seeds)", fontsize=14)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title("Targeting precision: V_achieved vs V_target per arm",
                 fontsize=12)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[targeting figure] -> {save_path}")


def fig_frontier(cells: List[Dict], rank_caps: List[int],
                  save_path: str):
    """AUC vs achieved V scatter, one line per r, points = (V*, seed) cells.
    Demonstrates 'rank is a ceiling, V is the operative coordinate'.

    Uses RETRAIN AUC and RETRAIN V (the deployment-model values).
    The baseline anchor's behaviour is shown in the variance figure, not here.
    """
    by_r: Dict[int, List[Dict]] = defaultdict(list)
    for c in cells:
        if c.get("status") != "ok" or c.get("retrain") is None: continue
        # Defensive: drop cells with NaN retrain AUC or V (shouldn't
        # happen with the corrected code, but covers edge cases like
        # have_test=False).
        if (math.isnan(float(c["retrain"]["AUC"]))
                or math.isnan(float(c["retrain"]["V"]))):
            continue
        by_r[int(c["rank_cap"])].append(c)

    fig, ax = plt.subplots(figsize=(10, 6))
    for r in sorted(by_r):
        cs = by_r[r]
        if not cs:
            continue
        Vs = np.array([float(c["retrain"]["V"]) for c in cs])
        AUCs = np.array([float(c["retrain"]["AUC"]) for c in cs])
        # Aggregate by V_target for a clean curve.
        V_tgts = np.array([float(c["V_target"]) for c in cs])
        unique_V_tgts = sorted(set(V_tgts.tolist()))
        x_means, y_means, y_los, y_his = [], [], [], []
        for V_t in unique_V_tgts:
            mask = V_tgts == V_t
            if mask.sum() < 1: continue
            x_means.append(float(np.mean(Vs[mask])))
            y_means.append(float(np.mean(AUCs[mask])))
            y_los.append(float(np.min(AUCs[mask])))
            y_his.append(float(np.max(AUCs[mask])))
        if not x_means:
            continue
        c = _color_for_rank(r, rank_caps)
        # Sort by x for a clean line.
        order = np.argsort(x_means)
        x_means = np.array(x_means)[order]
        y_means = np.array(y_means)[order]
        y_los = np.array(y_los)[order]
        y_his = np.array(y_his)[order]
        ax.plot(x_means, y_means, "o-", color=c, lw=2.4, ms=8,
                label=f"r = {r}",
                markeredgecolor="black", markeredgewidth=0.4)
        ax.fill_between(x_means, y_los, y_his, color=c, alpha=0.18)
        # Per-cell scatter (faint).
        ax.scatter(Vs, AUCs, color=c, s=14, alpha=0.35, marker="o")

    ax.set_xlabel("achieved effective rank V (retrained model)", fontsize=15)
    ax.set_ylabel("test AUC", fontsize=15)
    ax.set_xscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=12, loc="lower right")
    ax.set_title("Capacity frontier: AUC organised by realised V, "
                 "not by rank cap", fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[frontier figure] -> {save_path}")


def fig_loss_curves(out_dir: str, dataset: str,
                     V_targets: List[float], rank_caps: List[int],
                     save_path: str):
    """For each V*, plot mean training loss vs epoch with band across seeds,
    one curve per r. Demonstrates optimization stability.

    Reads `loss_*.npy` files (the retrain loss curves). Each file is an
    (n_logged_steps, 2) array with columns [epoch, loss]. We align them
    by epoch (using a dict for robustness against float-precision issues)
    and report mean ± std across seeds.
    """
    n_V = len(V_targets)
    fig, axes = plt.subplots(1, n_V, figsize=(5.2 * n_V, 5.2),
                              squeeze=False)
    axes = axes[0]
    for i, V in enumerate(V_targets):
        ax = axes[i]
        for r in sorted(rank_caps):
            # Each seed's curve as a dict {epoch: loss}.
            seed_dicts: List[Dict[int, float]] = []
            for fn in os.listdir(os.path.join(out_dir, dataset)):
                if not fn.startswith("loss_"): continue
                if fn.startswith("loss_anchor_"): continue
                if f"_V{_v_token(V)}_" not in fn: continue
                if f"_r{r}_" not in fn: continue
                arr = np.load(os.path.join(out_dir, dataset, fn))
                if arr.size == 0: continue
                d = {int(round(arr[k, 0])): float(arr[k, 1])
                     for k in range(arr.shape[0])}
                seed_dicts.append(d)
            if not seed_dicts: continue
            # Common epoch grid: union of all epochs, sorted.
            all_epochs = sorted(set().union(*[set(d.keys())
                                                for d in seed_dicts]))
            losses_grid = np.full((len(seed_dicts), len(all_epochs)),
                                   np.nan)
            for j, d in enumerate(seed_dicts):
                for k, ep in enumerate(all_epochs):
                    if ep in d:
                        losses_grid[j, k] = d[ep]
            mean_loss = np.nanmean(losses_grid, axis=0)
            if losses_grid.shape[0] >= 2:
                std_loss = np.nanstd(losses_grid, axis=0, ddof=1)
            else:
                std_loss = np.zeros_like(mean_loss)
            c = _color_for_rank(r, rank_caps)
            ax.plot(all_epochs, mean_loss, "-", color=c, lw=2.0,
                    label=f"r = {r} (n={losses_grid.shape[0]})")
            ax.fill_between(all_epochs,
                             mean_loss - std_loss,
                             mean_loss + std_loss,
                             color=c, alpha=0.18)
        ax.set_xlabel("epoch", fontsize=14)
        ax.set_ylabel("training loss (retrain)", fontsize=14)
        ax.set_title(f"V* = {V:g}", fontsize=13)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=10, loc="upper right")
    fig.suptitle(f"Training loss curves at matched V* "
                 f"(mean ± std across seeds, dataset={dataset})",
                 fontsize=14)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[loss curves figure] -> {save_path}")


# ---------------------------------------------------------------------------
# Mode-connectivity post-hoc
# ---------------------------------------------------------------------------

def _eval_loss_at_state(state_dict, model_factory, ti, tj, sampler,
                          eta, lambda_theta, lambda_bias, n_samples=4):
    """Average training-style loss over n_samples negative draws."""
    import torch
    import torch.nn.functional as F
    m = model_factory()
    try:
        m.load_state_dict(state_dict, strict=True)
    except Exception:
        return float("nan")
    m.eval()
    losses = []
    with torch.no_grad():
        for _ in range(n_samples):
            nll, log_v, reg = engine.train_loss_components(
                m, ti, tj, sampler, eta=eta,
                lambda_theta=lambda_theta, lambda_bias=lambda_bias,
            )
            loss = nll - eta * log_v + reg
            losses.append(float(loss.item()))
    return float(np.mean(losses))


def fig_mode_connectivity(out_dir: str, dataset: str,
                           V_targets: List[float], rank_caps: List[int],
                           args, save_path: str,
                           n_pairs: int = 5, n_alpha: int = 11):
    """For each (V*, r) cell with >=2 seeds, linearly interpolate between
    pairs of trained state_dicts and compute loss along the path. The
    mean barrier height is reported as a scalar per cell.

    NOTE: linear interpolation in raw L parameter space is meaningful for
    SPECTRA because the trace-normalisation is a property of the kernel,
    not the parameters; the kernel is a smooth function of L. The plot
    shows the loss landscape AROUND the trained region, which is the
    object the benignity claim is about.
    """
    files = engine.resolve_dataset_dir(dataset, root=args.datasets_root)
    split = engine.load_split_from_files(
        files["train_i"], files["train_j"],
        files.get("test_pos_i"), files.get("test_pos_j"),
        files.get("test_neg_i"), files.get("test_neg_j"),
    )
    n = engine.num_nodes_from_split(split)
    ti, tj, tpi, tpj, tni, tnj, _ = engine.prepare_tensors(split, args.device)
    sampler = engine.GPUUniformNonEdgeSampler(n, ti, tj, args.device)

    n_V = len(V_targets)
    fig, axes = plt.subplots(1, n_V, figsize=(5.2 * n_V, 5.2),
                              squeeze=False)
    axes = axes[0]
    barrier_summary = []

    for i, V in enumerate(V_targets):
        ax = axes[i]
        for r in sorted(rank_caps):
            ds_dir = os.path.join(out_dir, dataset)
            if not os.path.isdir(ds_dir): continue
            sds = []
            for fn in sorted(os.listdir(ds_dir)):
                if not fn.startswith("L_"): continue
                if fn.startswith("L_anchor_"): continue   # only retrain models
                if f"_V{_v_token(V)}_" not in fn: continue
                if f"_r{r}_" not in fn: continue
                sds.append(os.path.join(ds_dir, fn))
            if len(sds) < 2: continue
            # Sample n_pairs pairs.
            rng = np.random.default_rng(0)
            pairs_idx = []
            for _ in range(n_pairs):
                a, b = rng.choice(len(sds), 2, replace=False)
                pairs_idx.append((int(a), int(b)))

            def factory():
                return engine.make_model(
                    args.bias_model, n, r, args.device,
                    use_svd=not args.no_svd, beta_max=args.beta_max,
                    init_beta=args.init_beta,
                )

            barrier_per_pair = []
            alphas = np.linspace(0, 1, n_alpha)
            curve_stack = np.full((len(pairs_idx), n_alpha), np.nan)
            for pidx, (a, b) in enumerate(pairs_idx):
                sd_a = torch.load(sds[a], map_location=args.device)
                sd_b = torch.load(sds[b], map_location=args.device)
                # All keys must align.
                if set(sd_a.keys()) != set(sd_b.keys()):
                    continue
                losses = []
                for alpha in alphas:
                    interp_sd = {
                        k: (1 - alpha) * sd_a[k].float() + alpha * sd_b[k].float()
                        for k in sd_a
                    }
                    L_at = _eval_loss_at_state(
                        interp_sd, factory, ti, tj, sampler,
                        eta=0.0,  # use eta=0 for a "geometry" view of the loss
                        lambda_theta=args.lambda_theta,
                        lambda_bias=args.lambda_bias,
                    )
                    losses.append(L_at)
                curve_stack[pidx, :] = losses
                endpoint_max = max(losses[0], losses[-1])
                barrier = max(losses) - endpoint_max
                barrier_per_pair.append(barrier)

            mean_curve = np.nanmean(curve_stack, axis=0)
            std_curve = np.nanstd(curve_stack, axis=0, ddof=1)
            c = _color_for_rank(r, rank_caps)
            ax.plot(alphas, mean_curve, "-", color=c, lw=2.0,
                    label=f"r = {r}")
            ax.fill_between(alphas, mean_curve - std_curve,
                             mean_curve + std_curve, color=c, alpha=0.18)
            barrier_summary.append({
                "V_target": V, "rank_cap": r,
                "mean_barrier": float(np.mean(barrier_per_pair))
                    if barrier_per_pair else float("nan"),
                "n_pairs": len(barrier_per_pair),
            })

        ax.set_xlabel("interpolation α", fontsize=14)
        ax.set_ylabel("loss along path", fontsize=14)
        ax.set_title(f"V* = {V:g}", fontsize=13)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=11)
    fig.suptitle(f"Mode connectivity: linear path between independently "
                 f"trained seeds (dataset={dataset})", fontsize=14)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"[mode connectivity figure] -> {save_path}")
    print(f"  barrier summary:")
    for b in barrier_summary:
        print(f"    V*={b['V_target']:>4g}  r={b['rank_cap']:>4d}  "
              f"mean barrier={b['mean_barrier']:.4f}  "
              f"(n_pairs={b['n_pairs']})")


# ===========================================================================
# 4. CALIBRATION COST TABLE
# ===========================================================================

def calibration_table(cells: List[Dict], out_csv: str) -> None:
    rows = []
    for c in cells:
        if c.get("status") != "ok": continue
        if c.get("acquisition") is None: continue
        acq = c["acquisition"]
        rows.append({
            "dataset": c["dataset"],
            "V_target": c["V_target"],
            "rank_cap": c["rank_cap"],
            "seed": c["seed"],
            "n_probes": acq["summary"]["n_probes_total"],
            "total_probe_epochs": acq["total_probe_epochs"],
            "calibration_cost_full_runs":
                acq["calibration_cost_in_full_runs"],
            "status": acq["summary"]["status"],
            "selected_was_early_stopped":
                acq["summary"].get("selected_was_early_stopped", False),
        })
    if not rows: return
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    costs = np.array([r["calibration_cost_full_runs"] for r in rows])
    print(f"\n[calibration_table] -> {out_csv}")
    print(f"  median calibration cost (full-run-equivalents): "
          f"{np.median(costs):.2f}")
    print(f"  IQR: [{np.percentile(costs, 25):.2f}, "
          f"{np.percentile(costs, 75):.2f}]")
    print(f"  n_cells: {len(rows)}")


# ===========================================================================
# 5. CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--datasets", type=str, nargs="+", required=True)
    p.add_argument("--datasets_root", type=str, default="./datasets")
    p.add_argument("--rank_caps", type=int, nargs="+", required=True)
    p.add_argument("--V_targets", type=float, nargs="+", required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--baseline_r", type=int, default=None,
                   help="Rank cap to use as variance-ratio denominator. "
                        "Defaults to the smallest in --rank_caps.")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_dir", type=str, default="./op_out")
    p.add_argument("--force", action="store_true")
    p.add_argument("--analyze_only", action="store_true",
                   help="Skip running cells; only build figures/tables "
                        "from existing JSONs.")

    # Training (passthrough to engine).
    p.add_argument("--epochs", type=int, default=6000)
    p.add_argument("--K_restarts", type=int, default=1)
    p.add_argument("--bias_model", type=str, default="node",
                   choices=["node", "global"])
    p.add_argument("--lambda_theta", type=float, default=1e-4)
    p.add_argument("--lambda_bias", type=float, default=1e-4)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--bias_warmup_epochs", type=int, default=200)
    p.add_argument("--beta_max", type=float, default=5.0)
    p.add_argument("--init_beta", type=float, default=1.0)
    p.add_argument("--no_svd", action="store_true")
    p.add_argument("--log_every", type=int, default=100)

    # Acquisition.
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--search_mode", type=str,
                   default="from_zero_directional_guarded",
                   choices=["from_zero_bidirectional",
                            "from_zero_directional_guarded"])
    p.add_argument("--eta_initial_step", type=float, default=0.1)
    p.add_argument("--eta_expand_factor", type=float, default=2.0)
    p.add_argument("--eta_min", type=float, default=-10.0)
    p.add_argument("--eta_max", type=float, default=10.0)
    p.add_argument("--max_bracket_expansions", type=int, default=10)
    p.add_argument("--max_bisection_steps", type=int, default=20)
    p.add_argument("--min_eta_interval", type=float, default=1e-3)
    p.add_argument("--monotone_guard_tol", type=float, default=0.02)

    # Probe early-stop.
    p.add_argument("--enable_probe_early_stop", action="store_true",
                   default=True)
    p.add_argument("--no_probe_early_stop",
                   dest="enable_probe_early_stop", action="store_false")
    p.add_argument("--min_probe_epochs", type=int, default=500)
    p.add_argument("--probe_V_stability_tol", type=float, default=0.02)
    p.add_argument("--probe_NLL_stability_tol", type=float, default=0.002)
    p.add_argument("--probe_stability_patience", type=int, default=3)
    p.add_argument("--probe_stability_window", type=int, default=5)
    p.add_argument("--probe_stop_metric", type=str, default="V",
                   choices=["V", "V_and_NLL", "none"])

    # Negative-sample variance control (NEW).
    p.add_argument("--shared_neg_sequence", action="store_true",
                   default=False,
                   help="If set, the training-time negative-sampling sequence "
                        "is fixed across model seeds within each (V*, r) "
                        "cell. This isolates initialisation variance from "
                        "negative-sampling variance. Recommended for the "
                        "headline variance comparison.")
    p.add_argument("--neg_sequence_seed_base", type=int, default=20260101,
                   help="Base seed used to derive the per-cell negative "
                        "sequence seed when --shared_neg_sequence is set. "
                        "Different (V*, r) cells get different sequences; "
                        "different model seeds within a cell share the "
                        "sequence.")

    # Monte Carlo evaluation (NEW).
    p.add_argument("--n_mc_eval", type=int, default=20,
                   help="Number of Monte Carlo draws for post-training "
                        "train_NLL evaluation. Higher = lower noise floor "
                        "in the reported train_NLL and in restart selection. "
                        "Default 20.")
    p.add_argument("--mc_generator_seed", type=int, default=12345,
                   help="Seed for the MC eval sampler (held fixed across "
                        "all cells). Default 12345.")

    # Three-arm structure (NEW).
    p.add_argument("--three_arm", action="store_true", default=True,
                   help="When set (default), every V_target in --V_targets "
                        "is automatically added as a rank cap if missing, "
                        "ensuring the rank-matched-with-control arm "
                        "(r=V*, eta selected) and the baseline anchor "
                        "(r=V*, eta=0) are both produced.")
    p.add_argument("--no_three_arm", dest="three_arm",
                   action="store_false")

    # Mode-connectivity post-hoc.
    p.add_argument("--mc_n_pairs", type=int, default=5)
    p.add_argument("--mc_n_alpha", type=int, default=11)
    p.add_argument("--skip_mode_connectivity", action="store_true")
    return p.parse_args()


def _print_seed_count_guidance(seeds: List[int]):
    n = len(seeds)
    print("\n" + "=" * 70)
    print(f"SEED COUNT: n = {n}")
    if n < 3:
        print("  WARNING: n < 3 means no useful variance estimate.")
    elif n < 5:
        print("  PRELIMINARY only. Variance comparisons will be unreliable.")
        print("  Use only for spot-checking or pilot runs.")
    elif n < 10:
        print("  Acceptable for prelim results (detects ~3x variance ratios).")
        print("  For paper, increase to 10+ if compute allows.")
    elif n < 20:
        print("  Recommended range: detects ~2x variance ratios with ~80% power.")
        print("  Bootstrap CIs become tight enough for confident reporting.")
    else:
        print("  Gold standard: detects subtle variance differences.")
    print("=" * 70)


def main():
    args = parse_args()
    _print_seed_count_guidance(args.seeds)

    # Three-arm structure: ensure r=V* exists in rank_caps for every
    # V* in V_targets. The baseline (r=V*, eta=0 anchor) and the rank-
    # matched-with-control arm (r=V*, retrain at eta selected) are both
    # produced from the (V*, r=V*) cell automatically.
    rank_caps_set = set(int(r) for r in args.rank_caps)
    if args.three_arm:
        for V in args.V_targets:
            r_eq = int(round(V))
            rank_caps_set.add(r_eq)
        if rank_caps_set != set(int(r) for r in args.rank_caps):
            added = sorted(rank_caps_set - set(int(r) for r in args.rank_caps))
            print(f"\n[three_arm] adding rank caps {added} so that "
                  f"r = V* exists for every V* in --V_targets")
        args.rank_caps = sorted(rank_caps_set)
    if args.baseline_r is None:
        args.baseline_r = min(args.rank_caps)
    print(f"\nrank_caps (after three_arm augmentation): {args.rank_caps}")
    print(f"V_targets: {args.V_targets}")
    print(f"baseline_r (legacy fallback): {args.baseline_r}")
    print(f"shared_neg_sequence: {args.shared_neg_sequence}  "
          f"n_mc_eval: {args.n_mc_eval}")

    if not args.analyze_only:
        # Build the cell list, skipping (V*, r) where r < V* (impossible).
        cells_to_run = []
        for dataset in args.datasets:
            for V_target in args.V_targets:
                for rank in args.rank_caps:
                    if rank < V_target:
                        # The cell is impossible (V_target > rank cap).
                        # We skip to avoid logging spam; an "impossible"
                        # marker is recorded only if the user explicitly
                        # asks for that combo. With three_arm, every
                        # (V*, r=V*) is in the grid, plus user-specified
                        # r > V*; r < V* combinations are dropped.
                        continue
                    for seed in args.seeds:
                        cells_to_run.append((dataset, V_target, rank, seed))
        n_total = len(cells_to_run)
        print(f"\nTotal cells to run: {n_total}")
        n_done = 0
        t_grid_start = time.time()
        for (dataset, V_target, rank, seed) in cells_to_run:
            n_done += 1
            print(f"\n>>> CELL {n_done}/{n_total}  "
                  f"(elapsed: {(time.time() - t_grid_start)/60:.1f} min) <<<")
            try:
                run_one_cell(
                    dataset, args.datasets_root,
                    V_target, rank, seed,
                    args, args.output_dir, force=args.force,
                )
            except Exception as e:
                print(f"[CELL ERROR] {dataset} V*={V_target} "
                      f"r={rank} seed={seed}: {e}")
                import traceback; traceback.print_exc()

    # Analysis phase.
    print("\n" + "#" * 78)
    print("ANALYSIS PHASE")
    print("#" * 78)
    for dataset in args.datasets:
        cells = collect_cells(args.output_dir, dataset)
        if not cells:
            print(f"[{dataset}] no cells found; skipping analysis.")
            continue
        print(f"\n[{dataset}] {len(cells)} cells")

        # Variance table + figure.
        rows = variance_table(
            cells, args.V_targets, args.rank_caps, args.baseline_r,
            os.path.join(args.output_dir, "tables",
                         f"variance_{dataset}.csv"),
        )
        fig_variance(rows, args.V_targets, args.rank_caps,
                      os.path.join(args.output_dir, "figures",
                                    f"variance_{dataset}.pdf"))

        # Targeting precision figure (sanity check: V_achieved vs V_target).
        fig_targeting_precision(rows, args.V_targets, args.rank_caps,
                      os.path.join(args.output_dir, "figures",
                                    f"targeting_{dataset}.pdf"))

        # Frontier figure.
        fig_frontier(cells, args.rank_caps,
                      os.path.join(args.output_dir, "figures",
                                    f"frontier_{dataset}.pdf"))

        # Loss-curves figure.
        fig_loss_curves(args.output_dir, dataset,
                         args.V_targets, args.rank_caps,
                         os.path.join(args.output_dir, "figures",
                                       f"loss_curves_{dataset}.pdf"))

        # Calibration cost table.
        calibration_table(
            cells,
            os.path.join(args.output_dir, "tables",
                         f"calibration_cost_{dataset}.csv"),
        )

        # Mode-connectivity figure.
        if not args.skip_mode_connectivity:
            try:
                fig_mode_connectivity(
                    args.output_dir, dataset,
                    args.V_targets, args.rank_caps,
                    args,
                    os.path.join(args.output_dir, "figures",
                                  f"mode_connectivity_{dataset}.pdf"),
                    n_pairs=args.mc_n_pairs, n_alpha=args.mc_n_alpha,
                )
            except Exception as e:
                print(f"[mode connectivity ERROR] {e}")
                import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()