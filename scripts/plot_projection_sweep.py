"""Render layer x normalized-position probe-activation heatmaps from sweep_out/.

Pure post-processing (no GPU): edit and re-run freely to restyle figures.

Position normalization: TEST QUESTION span -> [-1, 0], CoT span -> [0, 1]
(few-shot examples and instruction are excluded entirely; question-start
indices come from <tag>_qstart.json, see reconstruct_qstarts.py), binned
NBINS per segment, so the CoT-start marker sits at 0 for every trace.
Cell value: AUC of the per-trace projection (traces with parsed yes/no answers)
at that (layer, bin) — 0.5 = no answer information, 1.0 = fully decodable.

Outputs figs/projections/sweep_<model>_<dataset>.png and a combined grid per
model. Usage: python scripts/plot_projection_sweep.py [--in sweep_out] [--out figs/projections]
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_auc_score

INK, MUTED = "#1a1a19", "#8a8878"
CMAP = LinearSegmentedColormap.from_list("seq", ["#f4f3ee", "#9ec5f4", "#2a78d6", "#104281"])
NBINS = 60
MIN_PER_CLASS = 4


def binned_matrix(meta, npz, layer, qstarts):
    """Rows: traces; cols: 2*NBINS normalized bins; NaN where empty."""
    rows, y = [], []
    for i, tr in enumerate(meta["traces"]):
        if tr["label"] not in ("yes", "no"):
            continue
        P = npz[f"proj_{i}"][layer].astype(np.float32)
        b, n = tr["boundary"], tr["n_tokens"]
        q = qstarts[i] if qstarts else 0
        row = np.full(2 * NBINS, np.nan)
        for seg, (lo, hi), (b0, b1) in [("question", (q, b), (0, NBINS)),
                                        ("cot", (b, n), (NBINS, 2 * NBINS))]:
            span = hi - lo
            if span <= 0:
                continue
            edges = np.linspace(lo, hi, (b1 - b0) + 1).astype(int)
            for k in range(b1 - b0):
                s, e = edges[k], max(edges[k + 1], edges[k] + 1)
                row[b0 + k] = P[s:min(e, n)].mean()
        rows.append(row)
        y.append(1 if tr["label"] == "yes" else 0)
    return np.array(rows), np.array(y)


def auc_heat(meta, npz, n_layers, qstarts):
    H = np.full((n_layers, 2 * NBINS), np.nan)
    for layer in range(n_layers):
        A, y = binned_matrix(meta, npz, layer, qstarts)
        for j in range(A.shape[1]):
            col = A[:, j]
            ok = ~np.isnan(col)
            if min((y[ok] == 1).sum(), (y[ok] == 0).sum()) < MIN_PER_CLASS:
                continue
            H[layer, j] = roc_auc_score(y[ok], col[ok])
    return H


def draw(ax, H, title):
    x = np.linspace(-1, 1, 2 * NBINS)
    im = ax.imshow(H, aspect="auto", cmap=CMAP, vmin=0.5, vmax=1.0, origin="lower",
                   extent=[-1, 1, 0, H.shape[0]], interpolation="nearest")
    ax.axvline(0, color=INK, lw=1.2, ls="--")
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.tick_params(colors=MUTED, labelsize=7)
    for s in ax.spines.values():
        s.set_color(MUTED)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", default="sweep_out")
    ap.add_argument("--out", default="figs/projections")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    by_model = {}
    for mpath in sorted(glob.glob(f"{args.indir}/*_meta.json")):
        meta = json.load(open(mpath))
        npz = np.load(mpath.replace("_meta.json", "_proj.npz"))
        qpath = mpath.replace("_meta.json", "_qstart.json")
        qstarts = json.load(open(qpath)) if os.path.exists(qpath) else None
        H = auc_heat(meta, npz, meta["n_layers"], qstarts)
        model = meta["model"].split("/")[-1]
        by_model.setdefault(model, []).append((meta["dataset"], H))

        fig, ax = plt.subplots(figsize=(8, 3.6))
        im = draw(ax, H, f"{model} / {meta['dataset']} — probe AUC by layer x normalized position")
        ax.set_xlabel("normalized position  (question: -1..0, CoT: 0..1)", fontsize=8, color=INK)
        ax.set_ylabel("layer", fontsize=8, color=INK)
        fig.colorbar(im, ax=ax, shrink=0.85).set_label("AUC", fontsize=7, color=INK)
        fig.tight_layout()
        fig.savefig(f"{args.out}/sweep_{model}_{meta['dataset']}.png", dpi=160,
                    facecolor="white")
        plt.close(fig)

    for model, panels in by_model.items():
        panels.sort()
        fig, axes = plt.subplots(2, 2, figsize=(12, 6.5))
        for ax, (ds, H) in zip(axes.flat, panels):
            im = draw(ax, H, ds)
        fig.suptitle(f"{model} — when/where the answer is decodable (dashed line: CoT start)",
                     fontsize=11, color=INK, x=0.02, ha="left")
        fig.colorbar(im, ax=axes, shrink=0.7).set_label("AUC (0.5 = chance)", fontsize=8)
        fig.savefig(f"{args.out}/sweep_grid_{model}.png", dpi=160, facecolor="white")
        plt.close(fig)
        print(f"wrote sweep_grid_{model}.png")


if __name__ == "__main__":
    main()
