"""Paper-grade GPT-OSS 20B rerun, methodology-matched to the instruct models.

Per dataset (items = the instruct experiments' exact train/test splits,
shipped as paper_inputs/<ds>.jsonl):

  PHASE gen    — train+test CoT generations (temp 0.7, max_new_tokens=2000,
                 the repo's reasoning-model accommodation; instruct models
                 used 200)
  PHASE probe  — caa-single-layer probes: per-layer difference-of-means on
                 train-fold last-prompt-token hidden states (labels = the
                 model's own parsed answers); per-layer AUC on the test fold;
                 best layer = argmax test AUC
  PHASE steer  — TL-semantics steering on test items answered correctly:
                 resid += alpha * w (RAW vector, no normalization) at the best
                 layer, DECODE STEPS ONLY; alpha in {0,2,...,20}; yes-items
                 steered with -alpha, no-items with +alpha (max 50 per
                 direction, the runner's max_gen knob); success = parsed
                 steered answer equals the flipped target, rate over ALL
                 steered items; 100%-unparsed early stop per direction.

Prompt construction: harmony chat template over the instruct items' messages
with the trailing "A: Let's think step by step:" assistant stub dropped —
harmony assistant turns must open a channel, so the model's analysis channel
serves as the CoT (methods note).

Checkpoints: paper_out/<ds>_gen_<fold>.json, <ds>_probe.npz/.json,
<ds>_steer.json — every cell resumable. Run: python paper_rerun_gptoss.py
"""

import json
import os
import re

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "openai/gpt-oss-20b"
DATASETS = ["anachronisms", "logical_deduction", "social_chemistry", "sports_understanding"]
ALPHAS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
MAX_NEW = 2000
MAX_GEN_PER_DIRECTION = 50
BATCH = 8
TEMP = 0.7

os.makedirs("paper_out", exist_ok=True)

tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="auto", device_map="cuda")
model.eval()


def load_items(ds):
    items = [json.loads(l) for l in open(f"paper_inputs/{ds}.jsonl")]
    for it in items:
        msgs = it["messages"]
        if msgs[-1]["role"] == "assistant":
            msgs = msgs[:-1]
        it["prompt"] = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        other = "B" if it["correct_letter"] == "A" else "A"
        flip = "no" if it["correct_answer"] == "yes" else "yes"
        it["letter_map"] = {it["correct_letter"]: it["correct_answer"], other: flip}
    return items


def parse_semantic(text, letter_map):
    low = text.lower()
    idx = low.rfind("the best answer is")
    if idx == -1:
        return None, None
    m = re.search(r"\(\s*([ab])\s*\)", low[idx:])
    if not m:
        return None, None
    letter = m.group(1).upper()
    return letter, letter_map.get(letter)


@torch.no_grad()
def generate_batch(prompts, steer=None, layer_module=None):
    """Generate; if steer tensor given, add it at decode steps only (TL semantics)."""
    handle = None
    if steer is not None:
        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if h.shape[1] != 1:      # skip prompt prefill
                return out
            if isinstance(out, tuple):
                return (h + steer,) + out[1:]
            return h + steer
        handle = layer_module.register_forward_hook(hook)
    try:
        outs = []
        for b in range(0, len(prompts), BATCH):
            enc = tok(prompts[b:b + BATCH], return_tensors="pt", padding=True).to("cuda")
            out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=True,
                                 temperature=TEMP, pad_token_id=tok.pad_token_id)
            for j in range(out.shape[0]):
                outs.append(tok.decode(out[j, enc["input_ids"].shape[1]:],
                                       skip_special_tokens=False))
            print(f"    gen batch {b // BATCH + 1}/{(len(prompts) - 1) // BATCH + 1}", flush=True)
        return outs
    finally:
        if handle:
            handle.remove()


@torch.no_grad()
def last_token_states(prompts):
    feats = None
    for i, p in enumerate(prompts):
        ids = tok(p, return_tensors="pt")["input_ids"][:, -4096:]
        out = model(ids.to("cuda"), output_hidden_states=True)
        vecs = torch.stack([h[0, -1, :] for h in out.hidden_states]).float().cpu().numpy()
        if feats is None:
            feats = np.zeros((len(prompts),) + vecs.shape, dtype=np.float32)
        feats[i] = vecs
        if i % 100 == 0:
            print(f"    extract {i}/{len(prompts)}", flush=True)
    return feats


for ds in DATASETS:
    print(f"\n########## {ds}", flush=True)
    items = load_items(ds)
    folds = {f: [it for it in items if it["fold"] == f] for f in ("train", "test")}

    # ── PHASE gen ────────────────────────────────────────────────────────────
    gen_records = {}
    for fold in ("train", "test"):
        ck = f"paper_out/{ds}_gen_{fold}.json"
        if os.path.exists(ck):
            gen_records[fold] = json.load(open(ck))
            print(f"  gen {fold}: cached ({len(gen_records[fold])})", flush=True)
            continue
        print(f"  gen {fold}: {len(folds[fold])} items", flush=True)
        gens = generate_batch([it["prompt"] for it in folds[fold]])
        recs = []
        for it, g in zip(folds[fold], gens):
            letter, sem = parse_semantic(g, it["letter_map"])
            recs.append({"prompt": it["prompt"], "generation": g,
                         "pred_letter": letter, "pred_answer": sem,
                         "correct_letter": it["correct_letter"],
                         "correct_answer": it["correct_answer"],
                         "letter_map": it["letter_map"]})
        json.dump(recs, open(ck, "w"))
        gen_records[fold] = recs
        parsed = sum(1 for r in recs if r["pred_answer"])
        acc = sum(1 for r in recs if r["pred_answer"] == r["correct_answer"]) / max(parsed, 1)
        print(f"  gen {fold}: parsed {parsed}/{len(recs)}, acc(parsed) {acc:.3f}", flush=True)

    # ── PHASE probe ──────────────────────────────────────────────────────────
    probe_ck = f"paper_out/{ds}_probe"
    if os.path.exists(probe_ck + ".json"):
        pj = json.load(open(probe_ck + ".json"))
        W = np.load(probe_ck + ".npz")["W"]
        best_layer = pj["best_layer"]
        print(f"  probe: cached (layer {best_layer}, AUC {pj['best_auc']:.3f})", flush=True)
    else:
        tr = [r for r in gen_records["train"] if r["pred_answer"] in ("yes", "no")]
        te = [r for r in gen_records["test"] if r["pred_answer"] in ("yes", "no")]
        print(f"  probe: extracting {len(tr)} train / {len(te)} test", flush=True)
        Xtr = last_token_states([r["prompt"] for r in tr])
        ytr = np.array([1 if r["pred_answer"] == "yes" else 0 for r in tr])
        W = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)          # RAW diff-of-means
        Xte = last_token_states([r["prompt"] for r in te])
        yte = np.array([1 if r["pred_answer"] == "yes" else 0 for r in te])
        aucs = [float(roc_auc_score(yte, Xte[:, l, :] @ W[l])) for l in range(W.shape[0])]
        best_layer = int(np.argmax(aucs))
        np.savez_compressed(probe_ck + ".npz", W=W)
        json.dump({"auc_by_layer": aucs, "best_layer": best_layer,
                   "best_auc": aucs[best_layer]}, open(probe_ck + ".json", "w"))
        print(f"  probe: best layer {best_layer}, test AUC {aucs[best_layer]:.3f}", flush=True)

    # hidden_states index L = output of decoder layer L-1 (index 0 = embeddings)
    layer_module = model.model.layers[max(best_layer - 1, 0)]
    w_t = torch.tensor(W[best_layer], dtype=model.dtype, device="cuda")

    # ── PHASE steer ──────────────────────────────────────────────────────────
    steer_ck = f"paper_out/{ds}_steer.json"
    steer = json.load(open(steer_ck)) if os.path.exists(steer_ck) else {}
    eligible = [r for r in gen_records["test"]
                if r["pred_answer"] in ("yes", "no") and r["pred_answer"] == r["correct_answer"]]
    groups = {"yes": [r for r in eligible if r["pred_answer"] == "yes"][:MAX_GEN_PER_DIRECTION],
              "no": [r for r in eligible if r["pred_answer"] == "no"][:MAX_GEN_PER_DIRECTION]}
    print(f"  steer: eligible yes={len(groups['yes'])} no={len(groups['no'])}", flush=True)
    stopped = {"yes": False, "no": False}
    for alpha in ALPHAS:
        for direction in ("yes", "no"):                     # yes->no uses -alpha, no->yes +alpha
            key = f"alpha_{alpha}_{direction}"
            if key in steer:
                if steer[key].get("unparsed_rate", 0) >= 1.0 and alpha > 0:
                    stopped[direction] = True
                continue
            if stopped[direction] and alpha > 0:
                print(f"  {key}: skipped (early stop)", flush=True)
                continue
            signed = -alpha if direction == "yes" else alpha
            grp = groups[direction]
            if not grp:
                continue
            sv = (float(signed) * w_t) if alpha != 0 else None
            gens = generate_batch([r["prompt"] for r in grp],
                                  steer=sv, layer_module=layer_module)
            recs, succ, unparsed = [], 0, 0
            for r, g in zip(grp, gens):
                letter, sem = parse_semantic(g, r["letter_map"])
                target = "no" if r["pred_answer"] == "yes" else "yes"
                cat = ("unparsed" if sem not in ("yes", "no")
                       else "success" if sem == target else "failure")
                succ += cat == "success"
                unparsed += cat == "unparsed"
                recs.append({"original_answer": r["pred_answer"], "target_answer": target,
                             "steered_answer": sem, "category": cat, "alpha": signed,
                             "generation": g})
            n = len(recs)
            steer[key] = {"alpha_signed": signed, "n": n,
                          "success_rate": round(succ / n, 3),
                          "unparsed_rate": round(unparsed / n, 3),
                          "results": recs}
            json.dump(steer, open(steer_ck, "w"))
            print(f"  {key}: success {succ}/{n} ({succ/n:.2f}), unparsed {unparsed/n:.2f}",
                  flush=True)
            if unparsed / n >= 1.0 and alpha > 0:
                stopped[direction] = True
                print(f"  early stop: {direction} direction at alpha {alpha}", flush=True)

print("\nDONE paper rerun", flush=True)
