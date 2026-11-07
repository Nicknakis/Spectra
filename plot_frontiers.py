"""
plot_frontier_multirank.py

Multi-rank-cap version of plot_frontier.py. Loads frontier sweeps at
several rank caps for a single dataset and produces two figures:

    1. Vveta:  d_spec vs eta, one curve per rank cap, with rank-cap
               saturation guides at y = r showing why each curve
               plateaus where it does.
    2. AUCvV:  Test AUC vs achieved d_spec, one curve per rank cap.
               This is the "collapse" plot: where the rank caps overlap
               in d_spec, the AUC curves should lie on top of each
               other -- the visual proof that performance is governed
               by realised capacity rather than nominal rank.

Visual language is inherited from plot_frontier.py and extended to
three rank caps via:
  * color  -> rank cap (cool to warm: r=64 blue, r=128 purple, r=256 red)
  * style  -> rank cap (dotted, dashed, solid) so coinciding curves
              read as coincidence rather than over-paint
  * width  -> smaller r drawn slightly thicker to remain visible under
              the largest-r curve where they overlap
  * band   -> per-rank +/-1 sigma at low alpha (0.13); three bands at
              full strength would mud the figure

Per-rank peaks are marked with diamond + errorbar (xerr/yerr from seed
spread). All three peaks are visible; their AUC and d_spec values appear
in the legend on dedicated lines so the rank-by-rank ranking is
explicit.

Expected file layout (same as plot_frontier.py):

    <root>/<dataset>/frontier_full_<dataset>_r<rank>_seed<seed>.json

CLI
---
    python3 plot_frontier_multirank.py \\
        --dataset inf-power \\
        --ranks 64 128 256 \\
        --seeds 0 1 2 \\
        --root ./adaptive_eta_out \\
        --output_prefix frontier_multirank_inf-power
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

# Silence the same cosmetic backend chatter as plot_frontier.py.
warnings.filterwarnings("ignore",
    message=r".*NOT subset; don't know how to subset; dropped.*")
warnings.filterwarnings("ignore",
    message=r"Glyph \d+ \(.*\) missing from current font.*")
warnings.filterwarnings("ignore",
    message=r".*This figure includes Axes that are not compatible "
            r"with tight_layout.*")
warnings.filterwarnings("ignore",
    message=r".*extra bytes in post\.stringData array.*")
warnings.filterwarnings("ignore",
    message=r".*'created' timestamp seems very low.*")
logging.getLogger("fontTools").setLevel(logging.ERROR)
logging.getLogger("fontTools.subset").setLevel(logging.ERROR)
logging.getLogger("fontTools.ttLib").setLevel(logging.ERROR)
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# ===========================================================================
# Style: inherited from plot_frontier.py, extended for multiple rank caps.
# ===========================================================================

# Colors: cool -> warm so rank ordering is implicit in the color ordering.
# These are saturation-matched to read clearly on white at print scale.
RANK_COLOR = {
    64:  "#1F6FB4",   # blue
    128: "#7B3294",   # purple
    256: "#C7261D",   # red
    512: "#E08214",   # orange (extra cap if needed)
}

# Line style by rank: smaller r dotted, mid dashed, largest solid.
# When curves coincide on the AUC-vs-dspec plot the reader sees all three
# styles at once -- this is what makes the collapse legible.
RANK_LINESTYLE = {
    64:  (0, (1.4, 1.6)),       # dotted
    128: (0, (5.5, 2.4)),       # dashed
    256: "-",                   # solid
    512: (0, (8, 2, 1.5, 2)),   # dash-dot
}

# Smaller r drawn a hair thicker so it remains visible under overpaint.
RANK_LINEWIDTH = {64: 2.7, 128: 2.5, 256: 2.3, 512: 2.1}

# Halo widths track line widths.
RANK_HALOWIDTH = {64: 5.4, 128: 5.0, 256: 4.6, 512: 4.4}

INK_TEXT = "#1a1a1a"
INK_AXIS = "#2b2b2b"
SOFT_GRAY = "#9aa0a6"
REF_GRAY = "#7d8590"

PANEL_FIGSIZE = (9.0, 5.6)
AUC_PANEL_FIGSIZE = (9.4, 5.6)


def _color(rank: int) -> str:
    return RANK_COLOR.get(int(rank), "#404040")

def _ls(rank: int):
    return RANK_LINESTYLE.get(int(rank), "-")

def _lw(rank: int) -> float:
    return RANK_LINEWIDTH.get(int(rank), 2.0)

def _halo(rank: int) -> float:
    return RANK_HALOWIDTH.get(int(rank), 4.4)


def _apply_style():
    plt.rcParams.update({
        "font.family":      "sans-serif",
        "font.sans-serif":  ["Helvetica", "Arial", "Liberation Sans",
                             "DejaVu Sans"],
        "font.size":         13.0,
        "axes.labelsize":    15.0,
        "axes.labelcolor":   INK_TEXT,
        "axes.titlepad":     10.0,
        "axes.labelpad":     6.0,
        "xtick.labelsize":   12.0,
        "ytick.labelsize":   12.0,
        "xtick.color":       INK_TEXT,
        "ytick.color":       INK_TEXT,
        "text.color":        INK_TEXT,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    1.0,
        "axes.edgecolor":    INK_AXIS,
        "axes.grid":         True,
        "axes.axisbelow":    True,
        "grid.alpha":        0.35,
        "grid.linewidth":    0.6,
        "grid.color":        "#d1d5db",
        "xtick.direction":   "out",
        "ytick.direction":   "out",
        "legend.frameon":    False,
        "axes.facecolor":    "white",
        "figure.facecolor":  "white",
        "savefig.facecolor": "white",
        "lines.solid_capstyle":  "round",
        "lines.solid_joinstyle": "round",
        "lines.dash_capstyle":   "round",
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype":     42,
        "ps.fonttype":      42,
        "savefig.bbox":     "tight",
        "savefig.pad_inches": 0.04,
        "figure.dpi":       110,
    })


def _polish_axes(ax, axis_color="#111827"):
    ax.set_axisbelow(True)
    ax.grid(True, which="major", color="#d1d5db", lw=0.8, alpha=0.55)
    ax.grid(True, which="minor", color="#e5e7eb", lw=0.45, alpha=0.30)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(axis_color)
        ax.spines[side].set_linewidth(1.15)
    ax.tick_params(axis="both", which="major", labelsize=18.5,
                   labelcolor=axis_color, color=axis_color,
                   length=5.5, width=1.1, direction="out")
    ax.tick_params(axis="both", which="minor", labelcolor=axis_color,
                   color=axis_color, length=3.2, width=0.8,
                   direction="out")
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("semibold")


# ===========================================================================
# Data loading.
# ===========================================================================

def _candidate_dirs(root: str, dataset: str) -> List[str]:
    out = [os.path.join(root, dataset), root]
    seen, uniq = set(), []
    for p in out:
        if os.path.isdir(p) and p not in seen:
            seen.add(p); uniq.append(p)
    return uniq


def _resolve_seed_json(root: str, dataset: str, rank: int, seed: int) -> str:
    name = f"frontier_full_{dataset}_r{rank}_seed{seed}.json"
    for d in _candidate_dirs(root, dataset):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"Could not find {name} under {root!r} for dataset={dataset!r}.")


def _load_seed_points(root: str, dataset: str, rank: int, seed: int
                      ) -> List[Dict]:
    p = _resolve_seed_json(root, dataset, rank, seed)
    with open(p, "r") as f:
        full = json.load(f)
    return sorted(full.get("points", []), key=lambda x: float(x["eta"]))


def _aggregate_one_rank(root: str, dataset: str, rank: int,
                         seeds: Sequence[int]
                         ) -> Dict[str, np.ndarray]:
    """Return per-eta mean, std, plus per-seed peak (V, AUC, eta) lists.

    The peak summary is computed per seed (so we can show xerr/yerr on
    the peak marker), then aggregated.
    """
    seed_pts = [_load_seed_points(root, dataset, rank, s) for s in seeds]
    eta_sets = [set(round(float(p["eta"]), 10) for p in pts) for pts in seed_pts]
    shared = sorted(set.intersection(*eta_sets)) if eta_sets else []
    if not shared:
        shared = sorted(set.union(*eta_sets)) if eta_sets else []
    if not shared:
        empty = np.array([])
        return dict(eta=empty, V=empty, V_std=empty,
                    AUC=empty, AUC_std=empty,
                    peak_V=np.nan, peak_V_std=np.nan,
                    peak_AUC=np.nan, peak_AUC_std=np.nan,
                    peak_eta=np.nan)

    Vs = np.full((len(seed_pts), len(shared)), np.nan)
    AUCs = np.full((len(seed_pts), len(shared)), np.nan)
    eta_to_idx = {e: i for i, e in enumerate(shared)}
    for s_idx, pts in enumerate(seed_pts):
        for p in pts:
            j = eta_to_idx.get(round(float(p["eta"]), 10))
            if j is None:
                continue
            Vs[s_idx, j] = float(p["V"])
            AUCs[s_idx, j] = float(p["AUC"])

    # Per-seed peaks (computed on each seed's own raw points to avoid
    # quantization to the shared grid).
    peak_V_per_seed, peak_AUC_per_seed, peak_eta_per_seed = [], [], []
    for pts in seed_pts:
        if not pts:
            continue
        V_arr = np.array([float(p["V"]) for p in pts])
        A_arr = np.array([float(p["AUC"]) for p in pts])
        e_arr = np.array([float(p["eta"]) for p in pts])
        if A_arr.size and not np.all(np.isnan(A_arr)):
            k = int(np.nanargmax(A_arr))
            peak_V_per_seed.append(float(V_arr[k]))
            peak_AUC_per_seed.append(float(A_arr[k]))
            peak_eta_per_seed.append(float(e_arr[k]))

    n = len(seed_pts)
    with np.errstate(all="ignore"):
        V_mean = np.nanmean(Vs, axis=0)
        AUC_mean = np.nanmean(AUCs, axis=0)
        if n > 1:
            V_std = np.nanstd(Vs, axis=0, ddof=1)
            AUC_std = np.nanstd(AUCs, axis=0, ddof=1)
        else:
            V_std = np.zeros_like(V_mean)
            AUC_std = np.zeros_like(AUC_mean)

    if peak_V_per_seed:
        pV = np.array(peak_V_per_seed)
        pA = np.array(peak_AUC_per_seed)
        pE = np.array(peak_eta_per_seed)
        peak_V = float(pV.mean())
        peak_AUC = float(pA.mean())
        peak_eta = float(pE.mean())
        if len(pV) > 1:
            peak_V_std = float(pV.std(ddof=1))
            peak_AUC_std = float(pA.std(ddof=1))
        else:
            peak_V_std = 0.0
            peak_AUC_std = 0.0
    else:
        peak_V = peak_AUC = peak_eta = np.nan
        peak_V_std = peak_AUC_std = np.nan

    return dict(
        eta=np.array(shared, dtype=float),
        V=V_mean, V_std=V_std,
        AUC=AUC_mean, AUC_std=AUC_std,
        peak_V=peak_V, peak_V_std=peak_V_std,
        peak_AUC=peak_AUC, peak_AUC_std=peak_AUC_std,
        peak_eta=peak_eta,
    )


# ===========================================================================
# Figure builders.
# ===========================================================================

def _plot_curve_with_halo(ax, x, y, *, color, ls, lw, halo_lw, zorder=4):
    """Plot a curve with a white halo underneath for figure-ground separation.

    The halo is drawn as a separate Line2D (not via path_effects) because
    path_effects.Stroke has trouble with non-solid line styles -- the
    dashed pattern propagates through the stroke and produces visible
    halo gaps. Two stacked Line2Ds give clean halos at any line style.
    """
    halo = ax.plot(x, y, color="white", lw=halo_lw, alpha=0.96,
                    zorder=zorder - 1, solid_capstyle="round",
                    solid_joinstyle="round")
    line, = ax.plot(x, y, color=color, lw=lw, ls=ls,
                     zorder=zorder, solid_capstyle="round",
                     dash_capstyle="round")
    return line


def _make_fig_Vveta(per_rank: Dict[int, Dict[str, np.ndarray]],
                     v_scale: str = "log",
                     eta_xlim: Tuple[float, float] = (-0.25, 0.25),
                     show_rank_caps: bool = True,
                     show_eta_zero: bool = True,
                     band_alpha: float = 0.13,
                     ) -> plt.Figure:
    """d_spec vs eta, one curve per rank cap.

    Rank-cap saturation guides (faint horizontal dashed line at y = r,
    colored by rank) explain why each curve plateaus where it does.
    """
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE, constrained_layout=True)
    axis_color = "#111827"
    ax.set_xlim(*eta_xlim)

    ranks_sorted = sorted(per_rank.keys())

    # Saturation guides drawn first so they sit behind the curves.
    if show_rank_caps:
        for rank in ranks_sorted:
            c = _color(rank)
            ax.axhline(rank, color=c, ls=(0, (5, 3)), lw=1.0,
                        alpha=0.45, zorder=0)

    if show_eta_zero:
        ax.axvline(0.0, color="#4b5563", ls=(0, (1.5, 2.2)),
                    lw=1.2, alpha=0.85, zorder=0)

    # Bands first, then curves on top.
    for rank in ranks_sorted:
        agg = per_rank[rank]
        if agg["eta"].size == 0:
            continue
        c = _color(rank)
        if np.any(agg["V_std"] > 0):
            band_lo = np.maximum(agg["V"] - agg["V_std"], 1e-3)
            band_hi = agg["V"] + agg["V_std"]
            ax.fill_between(agg["eta"], band_lo, band_hi,
                             color=c, alpha=band_alpha, linewidth=0,
                             zorder=2)

    for rank in ranks_sorted:
        agg = per_rank[rank]
        if agg["eta"].size == 0:
            continue
        _plot_curve_with_halo(
            ax, agg["eta"], agg["V"],
            color=_color(rank), ls=_ls(rank),
            lw=_lw(rank), halo_lw=_halo(rank),
            zorder=5,
        )

    if v_scale == "log":
        ax.set_yscale("log")

    ax.set_xlabel(r"$\eta$", fontsize=23, fontweight="bold",
                   color=axis_color, labelpad=6)
    ax.set_ylabel(r"Effective spectral dimension  $d_{\mathrm{spec}}$",
                   fontsize=20, fontweight="bold",
                   color=axis_color, labelpad=8)

    _polish_axes(ax, axis_color=axis_color)

    # Legend: one entry per rank cap, plus eta=0 reference.
    handles = []
    for rank in ranks_sorted:
        handles.append(Line2D([0], [0], color=_color(rank),
                               lw=_lw(rank), ls=_ls(rank),
                               label=fr"$r = {rank}$"))
    if show_eta_zero:
        handles.append(Line2D([0], [0], color="#4b5563",
                               ls=(0, (1.5, 2.2)), lw=1.2,
                               label=r"$\eta = 0$"))

    leg = ax.legend(handles=handles, loc="upper left",
                     fontsize=18, frameon=True, framealpha=0.7,
                     facecolor="white", edgecolor="#9ca3af",
                     borderpad=0.65, labelspacing=0.42,
                     handlelength=2.6, handletextpad=0.75,
                     borderaxespad=0.0)
    leg.get_frame().set_linewidth(0.9)
    for text in leg.get_texts():
        text.set_color(axis_color)
        text.set_fontweight("medium")

    return fig


def _make_fig_AUCvV(per_rank: Dict[int, Dict[str, np.ndarray]],
                     v_scale: str = "linear",
                     band_alpha: float = 0.13,
                     annotate_peaks: bool = True,
                     ) -> plt.Figure:
    """Test AUC vs achieved d_spec.

    Where the rank caps overlap in d_spec, the lines should coincide --
    that is the visual proof that performance is governed by realised
    capacity rather than the rank cap. Three line styles plus a faint
    halo make the coincidence legible even where one color paints over
    another.

    Per-rank peaks are marked with diamond + xerr/yerr (seed spread).
    The legend lists each rank's peak AUC and d_spec on its own line.
    """
    fig, ax = plt.subplots(figsize=AUC_PANEL_FIGSIZE,
                            constrained_layout=True)
    axis_color = "#111827"
    ranks_sorted = sorted(per_rank.keys())

    # Bands first.
    for rank in ranks_sorted:
        agg = per_rank[rank]
        if agg["eta"].size == 0:
            continue
        c = _color(rank)
        order = np.argsort(agg["V"])
        Vs = agg["V"][order]
        AUC_m = agg["AUC"][order]
        AUC_s = agg["AUC_std"][order]
        if np.any(AUC_s > 0):
            ax.fill_between(Vs, AUC_m - AUC_s, AUC_m + AUC_s,
                             color=c, alpha=band_alpha, linewidth=0,
                             zorder=2)

    # Curves.
    for rank in ranks_sorted:
        agg = per_rank[rank]
        if agg["eta"].size == 0:
            continue
        order = np.argsort(agg["V"])
        _plot_curve_with_halo(
            ax, agg["V"][order], agg["AUC"][order],
            color=_color(rank), ls=_ls(rank),
            lw=_lw(rank), halo_lw=_halo(rank),
            zorder=5,
        )

    # Per-rank peak markers.
    peak_info = {}
    if annotate_peaks:
        for rank in ranks_sorted:
            agg = per_rank[rank]
            if not np.isfinite(agg["peak_AUC"]):
                continue
            c = _color(rank)
            xerr = [[agg["peak_V_std"]], [agg["peak_V_std"]]] \
                if np.isfinite(agg["peak_V_std"]) else None
            yerr = [[agg["peak_AUC_std"]], [agg["peak_AUC_std"]]] \
                if np.isfinite(agg["peak_AUC_std"]) else None
            ax.errorbar(
                [agg["peak_V"]], [agg["peak_AUC"]],
                xerr=xerr, yerr=yerr,
                fmt="D", markersize=12, color=c,
                markeredgecolor="white", markeredgewidth=1.2,
                ecolor=c, elinewidth=1.5,
                capsize=4, capthick=1.3,
                alpha=0.97, zorder=12,
            )
            peak_info[rank] = (agg["peak_V"], agg["peak_V_std"],
                                agg["peak_AUC"], agg["peak_AUC_std"])

    if v_scale == "log":
        ax.set_xscale("log")

    ax.set_xlabel(r"Effective spectral dimension  $d_{\mathrm{spec}}$",
                   fontsize=20, fontweight="bold",
                   color=axis_color, labelpad=8)
    ax.set_ylabel("Test AUC", fontsize=20, fontweight="bold",
                   color=axis_color, labelpad=8)

    _polish_axes(ax, axis_color=axis_color)

    # Legend: rank-cap lines, then peak summaries (one line per rank).
    handles = []
    for rank in ranks_sorted:
        handles.append(Line2D([0], [0], color=_color(rank),
                               lw=_lw(rank), ls=_ls(rank),
                               label=fr"$r = {rank}$"))
    if peak_info:
        # Visual separator: a transparent handle with no label, just for
        # spacing in the legend column.
        handles.append(Line2D([0], [0], color="none", label=" "))
        for rank in ranks_sorted:
            if rank not in peak_info:
                continue
            pv, pv_s, pa, pa_s = peak_info[rank]
            if pv_s > 0:
                lbl = (rf"peak $r{{=}}{rank}$:  "
                       rf"AUC={pa:.3f}$\pm${pa_s:.3f},  "
                       rf"$d_{{\mathrm{{spec}}}}{{=}}{pv:.0f}\pm{pv_s:.0f}$")
            else:
                lbl = (rf"peak $r{{=}}{rank}$:  AUC={pa:.3f},  "
                       rf"$d_{{\mathrm{{spec}}}}{{=}}{pv:.0f}$")
            handles.append(Line2D(
                [0], [0], marker="D", linestyle="None",
                markersize=11, color=_color(rank),
                markeredgecolor="white", markeredgewidth=1.0,
                label=lbl,
            ))

    leg = ax.legend(handles=handles, loc="upper right",
                     fontsize=14.5, frameon=True, framealpha=0.7,
                     facecolor="white", edgecolor="#9ca3af",
                     borderpad=0.65, labelspacing=0.45,
                     handlelength=2.6, handletextpad=0.75,
                     borderaxespad=0.0)
    leg.get_frame().set_linewidth(0.9)
    for text in leg.get_texts():
        text.set_color(axis_color)
        text.set_fontweight("medium")

    return fig


# ===========================================================================
# Top-level entry point.
# ===========================================================================

def plot_frontier_multirank(
    dataset: str,
    ranks: Sequence[int],
    seeds: Sequence[int],
    root: str = "./adaptive_eta_out",
    save_prefix: Optional[str] = None,
    save_ext: str = "pdf",
    dpi: int = 600,
    v_scale_eta: str = "log",
    v_scale_aucv: str = "linear",
    eta_xlim: Tuple[float, float] = (-0.25, 0.25),
    show_rank_caps: bool = True,
    annotate_peaks: bool = True,
    show: bool = False,
) -> Dict[str, plt.Figure]:
    _apply_style()

    per_rank: Dict[int, Dict[str, np.ndarray]] = {}
    for r in sorted(set(int(x) for x in ranks)):
        try:
            per_rank[r] = _aggregate_one_rank(root, dataset, r, seeds)
            print(f"  loaded r={r}: "
                  f"{per_rank[r]['eta'].size} shared eta points "
                  f"across {len(seeds)} seeds  "
                  f"(peak AUC={per_rank[r]['peak_AUC']:.3f} "
                  f"at d_spec={per_rank[r]['peak_V']:.1f})")
        except FileNotFoundError as e:
            print(f"  [skip] r={r}: {e}")
            continue

    if not per_rank:
        raise RuntimeError(f"No data loaded for dataset={dataset!r}.")

    fig_v_eta = _make_fig_Vveta(
        per_rank, v_scale=v_scale_eta, eta_xlim=eta_xlim,
        show_rank_caps=show_rank_caps,
    )
    fig_auc_v = _make_fig_AUCvV(
        per_rank, v_scale=v_scale_aucv,
        annotate_peaks=annotate_peaks,
    )

    figs = {"Vveta": fig_v_eta, "AUCvV": fig_auc_v}

    if save_prefix is not None:
        out_dir = os.path.dirname(save_prefix)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        for tag, f in figs.items():
            path = f"{save_prefix}_{tag}.{save_ext}"
            f.savefig(path, bbox_inches="tight", pad_inches=0.05, dpi=dpi)
            print(f"[plot_frontier_multirank] saved -> {path}")

    if show:
        plt.show()

    return figs


# ===========================================================================
# CLI.
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", type=str, default="socfb-American75")
    p.add_argument("--ranks", type=int, nargs="+", default=[64, 128, 256])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1,2])
    p.add_argument("--root", type=str, default="./adaptive_eta_out")
    p.add_argument("--output_prefix", type=str, default=None)
    p.add_argument("--ext", type=str, default="pdf")
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--eta_lo", type=float, default=-0.25)
    p.add_argument("--eta_hi", type=float, default=0.25)
    p.add_argument("--v_scale_eta", type=str, default="log",
                   choices=["log", "linear"])
    p.add_argument("--v_scale_aucv", type=str, default="linear",
                   choices=["log", "linear"])
    p.add_argument("--no_rank_caps", action="store_true",
                   help="Hide the y=r horizontal saturation guides.")
    p.add_argument("--no_peaks", action="store_true",
                   help="Hide per-rank peak markers and legend entries.")
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.output_prefix is None:
        ranks_str = "-".join(str(r) for r in sorted(set(args.ranks)))
        seeds_str = "-".join(str(s) for s in args.seeds)
        args.output_prefix = os.path.join(
            args.root,
            f"frontier_multirank_{args.dataset}_r{ranks_str}_seeds{seeds_str}",
        )
    plot_frontier_multirank(
        dataset=args.dataset,
        ranks=args.ranks,
        seeds=args.seeds,
        root=args.root,
        save_prefix=args.output_prefix,
        save_ext=args.ext,
        dpi=args.dpi,
        v_scale_eta=args.v_scale_eta,
        v_scale_aucv=args.v_scale_aucv,
        eta_xlim=(args.eta_lo, args.eta_hi),
        show_rank_caps=not args.no_rank_caps,
        annotate_peaks=not args.no_peaks,
        show=args.show,
    )


if __name__ == "__main__":
    main()