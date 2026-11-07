"""
spectra_link_prediction.py

Run SPECTRA on the standard unsupervised link-prediction task and save
results in a CSV format matching the baseline summary CSVs, so the
table-generation script can pick up SPECTRA rows alongside baselines.

Three modes:
  --eta0              : matched-rank-cap, no spectral control (eta=0).
  --eta_value <eta>   : matched-rank-cap, fixed eta (no bisection).
  --V_target <V>      : eta-bisection to target V.

In all modes, SPECTRA uses rank cap r = d (matched parameter count to
the baseline at the same nominal dimension).

SCORING: log-odds  a_i + a_j + beta * K_ij(L)  (node-bias models) or
alpha + beta * K_ij  (global-bias models), produced by the engine's
canonical `model.logits_for_pairs(i, j)` method. These log-odds are
used DIRECTLY as the ranking score for test edges (no logistic
classifier).

AUC-ROC and AUC-PR via the user-provided link_prediction_scores:

    def link_prediction_scores(rates, target):
        precision, recall, _ = metrics.precision_recall_curve(target, rates)
        return (metrics.roc_auc_score(target, rates),
                metrics.auc(recall, precision))

Note: this uses metrics.auc(recall, precision), which differs slightly
from average_precision_score (the latter is what the engine's own
eval_auc_and_ap uses). The difference is typically below 1e-3 in
practice but is reported here for transparency.

Per-cell JSON: <out>/<dataset>/cell_<dataset>_d<d>_<mode>_seed<s>.json
Aggregate CSV: <out>/summary_<dataset>_spectra.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn import metrics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import visualize_eta_solution as engine  # noqa: E402


# =============================================================================
# SCORING (user-provided definition)
# =============================================================================

def link_prediction_scores(rates: np.ndarray,
                            target: np.ndarray) -> Tuple[float, float]:
    precision, recall, _ = metrics.precision_recall_curve(target, rates)
    auc_roc = metrics.roc_auc_score(target, rates)
    auc_pr  = metrics.auc(recall, precision)
    return float(auc_roc), float(auc_pr)


@torch.no_grad()
def compute_test_logodds(model: torch.nn.Module,
                          test_pos_i: torch.Tensor, test_pos_j: torch.Tensor,
                          test_neg_i: torch.Tensor, test_neg_j: torch.Tensor,
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (rates, target) on the concatenation of positive and
    negative test pairs. `rates` is the model log-odds for each pair;
    `target` is 1 for positives, 0 for negatives.

    Uses engine's `model.logits_for_pairs(i, j)`, which is implemented
    in every VKLSM variant (Global/Node bias, SVD/FreeL).
    """
    model.eval()
    pos_logits = model.logits_for_pairs(test_pos_i, test_pos_j)
    neg_logits = model.logits_for_pairs(test_neg_i, test_neg_j)
    rates = torch.cat([pos_logits, neg_logits]).cpu().numpy()
    target = np.concatenate([
        np.ones(len(test_pos_i), dtype=np.int32),
        np.zeros(len(test_neg_i), dtype=np.int32),
    ])
    return rates, target


# =============================================================================
# CELL EXECUTION
# =============================================================================

def _eta_token(eta: float) -> str:
    """Filesystem-safe token for an eta value, e.g. 0.05 -> 'p0p05',
    -0.1 -> 'm0p1'. Used in cell filenames for the eta_value mode."""
    s = f"{eta:g}"
    return s.replace("-", "m").replace(".", "p").replace("+", "")


def _mode_token(mode: str, V_target: Optional[float],
                 eta_value: Optional[float]) -> str:
    if mode == "eta0":
        return "eta0"
    if mode == "eta_value":
        return f"eta_{_eta_token(float(eta_value))}"
    return f"V{V_target:g}".replace(".", "p").replace("-", "m")


def _baseline_label(mode: str, V_target: Optional[float],
                     eta_value: Optional[float]) -> str:
    if mode == "eta0":
        return "SPECTRA-eta0"
    if mode == "eta_value":
        return f"SPECTRA-eta-{eta_value:g}"
    return f"SPECTRA-Vtarget-{V_target:g}"


def cell_path(out_dir: str, dataset: str, d: int, mode: str,
               V_target: Optional[float], eta_value: Optional[float],
               seed: int) -> str:
    return os.path.join(
        out_dir, dataset,
        f"cell_{dataset}_d{d}_{_mode_token(mode, V_target, eta_value)}"
        f"_seed{seed}.json")


def _train_at_fixed_eta(args, n: int, rank: int, ti, tj, eta: float, seed: int):
    """Common training loop for both eta=0 and eta_value modes.

    Cold-start, fixed-epoch training at the supplied eta. Returns the
    trained model and wall time."""
    t0 = time.time()
    engine.set_seed(seed)
    m = engine.make_model(
        args.bias_model, n, rank, args.device,
        use_svd=not args.no_svd, beta_max=args.beta_max,
        init_beta=args.init_beta,
    )
    sampler = engine.GPUUniformNonEdgeSampler(
        n, ti, tj, device=args.device, generator_seed=seed,
    )
    opt = torch.optim.Adam(m.parameters(), lr=args.lr)
    if args.bias_warmup_epochs > 0:
        m.freeze_factor()
    for epoch in range(1, args.epochs + 1):
        if args.bias_warmup_epochs > 0 and epoch == args.bias_warmup_epochs + 1:
            m.unfreeze_factor()
        m.train()
        opt.zero_grad(set_to_none=True)
        nll, log_v, reg = engine.train_loss_components(
            m, ti, tj, sampler, eta=eta,
            lambda_theta=args.lambda_theta,
            lambda_bias=args.lambda_bias,
        )
        # Loss: nll - eta * log_v + reg. When eta=0 the spectral term
        # contributes zero gradient; the same expression covers both
        # eta=0 and arbitrary eta_value modes.
        loss = nll - eta * log_v + reg
        loss.backward()
        opt.step()
    return m, time.time() - t0


def run_one_cell(dataset: str, datasets_root: str,
                 d: int, mode: str, V_target: Optional[float],
                 eta_value: Optional[float],
                 seed: int, args, out_dir: str,
                 force: bool = False) -> Dict:
    cp = cell_path(out_dir, dataset, d, mode, V_target, eta_value, seed)
    if (not force) and os.path.isfile(cp):
        with open(cp) as f:
            existing = json.load(f)
        existing["_skipped_existing"] = True
        return existing

    os.makedirs(os.path.dirname(cp), exist_ok=True)

    print("\n" + "=" * 78)
    print(f"CELL: dataset={dataset}  d={d}  mode={mode}"
          f"  V_target={V_target}  eta_value={eta_value}  seed={seed}")
    print("=" * 78)

    engine.set_seed(seed)

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
    if not have_test:
        raise RuntimeError(f"Dataset {dataset} has no test split.")

    rank = d  # matched-rank-cap protocol

    # ----- Mode 1: eta=0 -----
    if mode == "eta0":
        m, wall_seconds = _train_at_fixed_eta(args, n, rank, ti, tj, 0.0, seed)
        eta_star = 0.0

    # ----- Mode 2: fixed user-supplied eta -----
    elif mode == "eta_value":
        if eta_value is None:
            raise ValueError("eta_value mode requires --eta_value.")
        m, wall_seconds = _train_at_fixed_eta(
            args, n, rank, ti, tj, float(eta_value), seed,
        )
        eta_star = float(eta_value)

    # ----- Mode 3: V_target via eta-bisection -----
    elif mode == "V_target":
        if V_target is None:
            raise ValueError("V_target mode requires --V_target.")
        if V_target > rank:
            result = {
                "dataset": dataset, "d": d, "rank_cap": rank,
                "mode": mode, "V_target": V_target, "seed": seed,
                "status": "impossible_target_V_gt_rank",
            }
            with open(cp, "w") as f:
                json.dump(result, f, indent=2)
            return result

        cache = engine.ProbeCache()
        summary = engine.run_target_v_search(
            dataset_name=dataset, split=split,
            rank=rank, seed=seed, V_target=float(V_target),
            cache=cache, eta_anchor=0.0,
            eta_initial_step=args.eta_initial_step,
            eta_expand_factor=args.eta_expand_factor,
            eta_min=args.eta_min, eta_max=args.eta_max, tau=args.tau,
            max_bracket_expansions=args.max_bracket_expansions,
            max_bisection_steps=args.max_bisection_steps,
            min_eta_interval=args.min_eta_interval,
            K_restarts=args.K_restarts, epochs=args.epochs,
            bias_model=args.bias_model,
            lambda_theta=args.lambda_theta,
            lambda_bias=args.lambda_bias, lr=args.lr,
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
            train_neg_sampler_seed=seed,
            n_mc_eval=int(args.n_mc_eval),
            mc_generator_seed=int(args.mc_generator_seed),
        )
        eta_star = float(summary["eta_selected"])
        if math.isnan(eta_star):
            result = {
                "dataset": dataset, "d": d, "rank_cap": rank,
                "mode": mode, "V_target": V_target, "seed": seed,
                "status": "acquisition_failed_nan_eta",
            }
            with open(cp, "w") as f:
                json.dump(result, f, indent=2)
            return result

        # Phase 2: retrain at eta_star, full epochs, no early stop.
        m, wall_seconds = _train_at_fixed_eta(
            args, n, rank, ti, tj, eta_star, seed,
        )

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # ----- Score on test edges -----
    rates, target = compute_test_logodds(m, tpi, tpj, tni, tnj)
    auc_roc, auc_pr = link_prediction_scores(rates, target)

    diag = engine.diagnostics(m)
    achieved_V = float(diag["V"])

    result = {
        "dataset": dataset, "d": d, "rank_cap": rank,
        "mode": mode,
        "V_target": V_target if mode == "V_target" else None,
        "eta_value": eta_value if mode == "eta_value" else None,
        "seed": seed, "status": "ok",
        "achieved_V": achieved_V,
        "eta_star": float(eta_star),
        "best_ROC_AUC": auc_roc,
        "best_PR_AUC":  auc_pr,
        "scoring_method": "link_prediction_scores (auc(recall,precision))",
        "wall_seconds": wall_seconds,
        "config": {
            "epochs": args.epochs, "lr": args.lr,
            "bias_model": args.bias_model,
            "lambda_theta": args.lambda_theta,
            "lambda_bias": args.lambda_bias,
            "bias_warmup_epochs": args.bias_warmup_epochs,
        },
    }

    with open(cp, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  achieved_V={achieved_V:.2f}  eta*={eta_star:+.5f}  "
          f"AUC-ROC={auc_roc:.4f}  AUC-PR={auc_pr:.4f}  "
          f"wall={wall_seconds:.1f}s")
    print(f"[SAVED] {cp}")
    return result


# =============================================================================
# AGGREGATE -> CSV (baseline-format)
# =============================================================================

def aggregate_to_csv(out_dir: str, dataset: str,
                      mode: str, V_target: Optional[float],
                      eta_value: Optional[float],
                      d_list: List[int], seeds: List[int],
                      out_csv: str):
    label = _baseline_label(mode, V_target, eta_value)
    rows = []
    for d in d_list:
        cells = []
        for s in seeds:
            cp = cell_path(out_dir, dataset, d, mode, V_target,
                            eta_value, s)
            if not os.path.isfile(cp):
                continue
            with open(cp) as f:
                cell = json.load(f)
            if cell.get("status") != "ok":
                continue
            cells.append(cell)
        if not cells:
            continue
        Vs   = np.array([c["achieved_V"]   for c in cells])
        aucs = np.array([c["best_ROC_AUC"] for c in cells])
        aps  = np.array([c["best_PR_AUC"]  for c in cells])
        etas = np.array([c["eta_star"]     for c in cells])
        n = len(cells)
        rows.append({
            "dataset": dataset, "baseline": label,
            "dim": d, "n_seeds": n,
            "V_star_mean": float(np.mean(Vs)),
            "V_star_std": float(np.std(Vs, ddof=1)) if n > 1 else 0.0,
            "best_ROC_AUC_mean": float(np.mean(aucs)),
            "best_ROC_AUC_std": float(np.std(aucs, ddof=1)) if n > 1 else 0.0,
            "best_PR_AUC_mean": float(np.mean(aps)),
            "best_PR_AUC_std": float(np.std(aps, ddof=1)) if n > 1 else 0.0,
            "chosen_hyperparam_mode": "N/A", "chosen_op_mode": "N/A",
            "hyperparam_distribution": "", "op_distribution": "",
            "spectra_mode": mode,
            "V_target_input": V_target if V_target is not None else "",
            "eta_value_input": eta_value if eta_value is not None else "",
            "eta_star_mean": float(np.mean(etas)),
            "eta_star_std": float(np.std(etas, ddof=1)) if n > 1 else 0.0,
        })
    if not rows:
        print(f"[aggregate] no cells; nothing written.")
        return

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    file_exists = os.path.isfile(out_csv)
    fieldnames = list(rows[0].keys())
    write_header = (not file_exists)
    if file_exists:
        with open(out_csv, "r") as f:
            existing_header = f.readline().strip().split(",")
        if existing_header != fieldnames:
            print(f"[aggregate WARNING] schema mismatch; overwriting.")
            write_header = True; file_exists = False
    open_mode = "a" if file_exists else "w"
    with open(out_csv, open_mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[aggregate] wrote {len(rows)} rows to {out_csv}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--datasets_root", type=str, default="./datasets")
    p.add_argument("--d", type=int, nargs="+", required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--eta0", action="store_true",
                     help="Train with eta=0 (matched-rank-cap, no spectral "
                          "control).")
    grp.add_argument("--eta_value", type=float, default=None,
                     help="Train with a fixed user-supplied eta. No "
                          "bisection; rank cap r=d as in the other modes.")
    grp.add_argument("--V_target", type=float, default=None,
                     help="Train with eta-bisection to target effective "
                          "spectral dimension V_target.")
    p.add_argument("--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_dir", type=str, default="./spectra_lp_out")
    p.add_argument("--force", action="store_true")
    p.add_argument("--analyze_only", action="store_true")
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
    p.add_argument("--tau", type=float, default=0.02)
    p.add_argument("--search_mode", type=str,
                   default="from_zero_directional_guarded",
                   choices=["from_zero_bidirectional",
                            "from_zero_directional_guarded"])
    p.add_argument("--eta_initial_step", type=float, default=0.1)
    p.add_argument("--eta_expand_factor", type=float, default=2.0)
    p.add_argument("--eta_min", type=float, default=-10.0)
    p.add_argument("--eta_max", type=float, default=10.0)
    p.add_argument("--max_bracket_expansions", type=int, default=10)
    p.add_argument("--max_bisection_steps", type=int, default=30)
    p.add_argument("--min_eta_interval", type=float, default=1e-4)
    p.add_argument("--monotone_guard_tol", type=float, default=0.02)
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
    p.add_argument("--n_mc_eval", type=int, default=1)
    p.add_argument("--mc_generator_seed", type=int, default=12345)
    return p.parse_args()


def main():
    args = parse_args()
    # Resolve which mode the user picked.
    if args.eta0:
        mode, V_target, eta_value = "eta0", None, None
    elif args.eta_value is not None:
        mode, V_target, eta_value = "eta_value", None, float(args.eta_value)
    else:
        mode, V_target, eta_value = "V_target", float(args.V_target), None

    print("=" * 78)
    print(f"SPECTRA link-prediction runner")
    print(f"  dataset    : {args.dataset}")
    print(f"  d          : {args.d}")
    print(f"  seeds      : {args.seeds}")
    print(f"  mode       : {mode}")
    if mode == "V_target":
        print(f"  V_target   : {V_target}")
    if mode == "eta_value":
        print(f"  eta_value  : {eta_value:+g}")
    print(f"  output_dir : {args.output_dir}")
    print(f"  device     : {args.device}")
    print(f"  scoring    : link_prediction_scores  "
          f"(auc(recall,precision) for AUC-PR)")
    print("=" * 78)

    if not args.analyze_only:
        n_total = len(args.d) * len(args.seeds)
        n_done = 0
        t_start = time.time()
        for d in args.d:
            for seed in args.seeds:
                n_done += 1
                print(f"\n>>> CELL {n_done}/{n_total}  "
                      f"(elapsed {(time.time() - t_start)/60:.1f} min) <<<")
                try:
                    run_one_cell(args.dataset, args.datasets_root,
                        d, mode, V_target, eta_value, seed, args,
                        args.output_dir, force=args.force)
                except Exception as e:
                    print(f"[CELL ERROR] d={d} seed={seed}: {e}")
                    import traceback; traceback.print_exc()

    out_csv = os.path.join(
    args.output_dir,
    f"summary_{args.dataset}_d{'-'.join(str(x) for x in args.d)}_spectra.csv",
    )
    aggregate_to_csv(args.output_dir, args.dataset,
        mode, V_target, eta_value, args.d, args.seeds, out_csv)


if __name__ == "__main__":
    main()