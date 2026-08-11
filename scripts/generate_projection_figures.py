"""Produce probe-projection discriminability figures for any model/dataset:

  1. <out>/auc_curve_<model>_<dataset>.png
       per-position AUC of the probe projection at the probe layer,
       aligned at CoT start (when does the answer become decodable?)
  2. <out>/auc_heatmap_<model>_<dataset>.png
       the same AUC over (layer x aligned position) — where in depth,
       and when in time, the answer is decodable.

Self-contained pipeline (single HF backend for every model, so there is no
TransformerLens/HF basis-matching assumption):

  a. load records: a generations pickle (list of dicts with at least
     `prompt` (str), `pred_letter`, `correct_letter`, `correct_answer`) —
     e.g. cache/experiments/<model>/<ds>/<split>/<hash>/data/train_generations.pkl
  b. derive SEMANTIC yes/no labels per item (the letter->answer mapping is
     randomized per item; letters must never be used as probe targets)
  c. split train/eval; extract last-prompt-token hidden states on the train
     fold and fit difference-of-means directions per layer (unit-normalized)
  d. forward-pass eval-fold traces (prompt + generation) and project every
     position onto each layer's direction
  e. compute per-position AUC across traces and render both figures

Requires a GPU. Example:

  python scripts/generate_projection_figures.py \
      --model openai/gpt-oss-20b \
      --generations cache/.../train_generations.pkl \
      --dataset-name anachronisms --out figs/projections
"""

import argparse
import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer

BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#1a1a19", "#8a8878"
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "seq", ["#f4f3ee", "#9ec5f4", "#2a78d6", "#104281"]
)


def semantic_label(rec):
    correct = str(rec["correct_answer"]).strip().lower()
    if correct not in ("yes", "no"):
        return None
    if rec["pred_letter"] == rec["correct_letter"]:
        return correct
    return "no" if correct == "yes" else "yes"


def load_records(path):
    recs = pickle.load(open(path, "rb"))
    out = []
    for r in recs:
        if r.get("pred_letter") not in ("A", "B") or not isinstance(r.get("prompt"), str):
            continue
        lab = semantic_label(r)
        if lab is None:
            continue
        out.append({"prompt": r["prompt"], "generation": r.get("generation", ""),
                    "label": lab})
    return out


@torch.no_grad()
def last_token_states(model, tok, prompts, device, max_len=4096):
    feats = None
    for i, p in enumerate(prompts):
        ids = tok(p, return_tensors="pt")["input_ids"][:, -max_len:]
        out = model(ids.to(device), output_hidden_states=True)
        vecs = torch.stack([h[0, -1, :] for h in out.hidden_states]).float().cpu().numpy()
        if feats is None:
            feats = np.zeros((len(prompts),) + vecs.shape, dtype=np.float32)
        feats[i] = vecs
        if i % 50 == 0:
            print(f"  w-extraction {i}/{len(prompts)}", flush=True)
    return feats  # [n, n_layers+1, d]


@torch.no_grad()
def trace_projections(model, tok, records, W_unit, device, max_len=4096):
    W_t = torch.tensor(W_unit, dtype=torch.float32, device=device)
    projs, metas = [], []
    for i, r in enumerate(records):
        boundary = tok(r["prompt"], return_tensors="pt")["input_ids"].shape[1]
        ids = tok(r["prompt"] + r["generation"], return_tensors="pt")["input_ids"][:, :max_len]
        out = model(ids.to(device), output_hidden_states=True)
        H = torch.stack([h[0] for h in out.hidden_states]).float()  # [L+1, seq, d]
        projs.append(torch.einsum("lsd,ld->ls", H, W_t).cpu().numpy())
        metas.append({"boundary": int(boundary), "label": r["label"],
                      "n_tokens": int(ids.shape[1])})
        if i % 10 == 0:
            print(f"  traces {i}/{len(records)}", flush=True)
    return projs, metas


def aligned_auc(projs, metas, layer, win, min_per_class):
    xs = np.arange(-win, win + 1)
    A = np.full((len(projs), len(xs)), np.nan)
    y = np.array([1 if m["label"] == "yes" else 0 for m in metas])
    for i, (P, m) in enumerate(zip(projs, metas)):
        b, n = m["boundary"], m["n_tokens"]
        lo, hi = max(0, b - win), min(n, b + win + 1)
        A[i, (lo - b) + win:(hi - b) + win] = P[layer][lo:hi]
    auc = np.full(len(xs), np.nan)
    for j in range(len(xs)):
        ok = ~np.isnan(A[:, j])
        if min((y[ok] == 1).sum(), (y[ok] == 0).sum()) < min_per_class:
            continue
        auc[j] = roc_auc_score(y[ok], A[ok, j])
    return xs, auc


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, axis="y", color="#e4e2d8", lw=0.5)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--generations", required=True,
                    help="pickle of generation records (see docstring)")
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--out", default="figs/projections")
    ap.add_argument("--n-traces", type=int, default=30)
    ap.add_argument("--win", type=int, default=300)
    ap.add_argument("--probe-layer", type=int, default=None,
                    help="default: layer with best last-token eval AUC")
    ap.add_argument("--min-per-class", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model.split('/')[-1]}_{args.dataset_name}"

    records = load_records(args.generations)
    y_all = np.array([1 if r["label"] == "yes" else 0 for r in records])
    idx_tr, idx_ev = train_test_split(np.arange(len(records)), test_size=0.2,
                                      random_state=args.seed, stratify=y_all)
    print(f"{len(records)} records -> {len(idx_tr)} train / {len(idx_ev)} eval")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto",
                                                 device_map="cuda")
    model.eval()

    # difference-of-means directions from the train fold (last prompt token)
    feats_tr = last_token_states(model, tok, [records[i]["prompt"] for i in idx_tr], "cuda")
    ytr = y_all[idx_tr]
    W = feats_tr[ytr == 1].mean(0) - feats_tr[ytr == 0].mean(0)          # [L+1, d]
    W_unit = W / np.linalg.norm(W, axis=1, keepdims=True)

    # pick probe layer by last-token AUC on the eval fold
    feats_ev = last_token_states(model, tok, [records[i]["prompt"] for i in idx_ev], "cuda")
    yev = y_all[idx_ev]
    layer_auc = [roc_auc_score(yev, feats_ev[:, l, :] @ W_unit[l])
                 for l in range(W_unit.shape[0])]
    probe_layer = args.probe_layer if args.probe_layer is not None else int(np.argmax(layer_auc))
    print(f"probe layer {probe_layer} (last-token AUC {layer_auc[probe_layer]:.3f})")

    rng = np.random.default_rng(args.seed)
    pick = rng.choice(idx_ev, size=min(args.n_traces, len(idx_ev)), replace=False)
    projs, metas = trace_projections(model, tok, [records[i] for i in pick], W_unit, "cuda")

    # figure 1: AUC curve at the probe layer
    xs, auc = aligned_auc(projs, metas, probe_layer, args.win, args.min_per_class)
    k = 7
    sm = np.convolve(np.nan_to_num(auc, nan=0.5), np.ones(k) / k, mode="same")
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.plot(xs, sm, color=BLUE, lw=1.8)
    ax.axvline(0, color=INK, lw=1, ls="--")
    ax.text(2, 0.97, "CoT starts", fontsize=9, color=INK)
    ax.axhline(0.5, color=MUTED, lw=1, ls=":")
    ax.set_ylim(0.35, 1.02)
    ax.set_xlabel("token position relative to CoT start", fontsize=10, color=INK)
    ax.set_ylabel(f"per-position probe AUC (n={len(projs)})", fontsize=10, color=INK)
    ax.set_title(f"{args.model} / {args.dataset_name} — layer {probe_layer}",
                 fontsize=11, color=INK, loc="left")
    style(ax)
    fig.tight_layout()
    fig.savefig(out / f"auc_curve_{tag}.png", dpi=160, facecolor="white")
    plt.close(fig)

    # figure 2: layer x position heatmap
    H = np.array([aligned_auc(projs, metas, l, args.win, args.min_per_class)[1]
                  for l in range(W_unit.shape[0])])
    fig, ax = plt.subplots(figsize=(10, 4.4))
    im = ax.imshow(H, aspect="auto", cmap=SEQ_CMAP, vmin=0.5, vmax=1.0, origin="lower",
                   extent=[xs[0], xs[-1], 0, H.shape[0]], interpolation="nearest")
    ax.axvline(0, color=INK, lw=1.2, ls="--")
    ax.axhline(probe_layer + 0.5, color=ORANGE, lw=1.2, alpha=0.8)
    ax.text(xs[0] + 4, probe_layer + 1.2, f"probe layer {probe_layer}",
            fontsize=8, color=ORANGE)
    ax.set_title(f"{args.model} / {args.dataset_name} — where and when the answer is decodable",
                 fontsize=10.5, color=INK, loc="left")
    ax.set_xlabel("position relative to CoT start", fontsize=9, color=INK)
    ax.set_ylabel("layer (hidden_states index)", fontsize=9, color=INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("per-position AUC (0.5 = chance)", fontsize=8, color=INK)
    fig.tight_layout()
    fig.savefig(out / f"auc_heatmap_{tag}.png", dpi=160, facecolor="white")
    plt.close(fig)

    json.dump({"model": args.model, "dataset": args.dataset_name,
               "probe_layer": probe_layer,
               "layer_auc_last_token": [round(a, 4) for a in layer_auc],
               "n_traces": len(projs)},
              open(out / f"auc_meta_{tag}.json", "w"), indent=2)
    print(f"wrote {out}/auc_curve_{tag}.png and auc_heatmap_{tag}.png")


if __name__ == "__main__":
    main()
