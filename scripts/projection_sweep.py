"""Projection sweep: for each model x dataset, 20 unsteered test generations with
per-layer, per-position probe projections saved for later plotting.

Per (model, dataset):
  1. build prompts from paper_inputs/<ds>.jsonl in the model's own chat format
     (instruct models continue the "A: Let's think step by step:" stub, matching
     the paper; harmony models drop it and use their analysis channel as CoT)
  2. obtain train-fold labels = the model's own parsed answers (cached in
     sweep_out/<tag>_train_labels.json after first run) and fit per-layer
     difference-of-means directions from last-prompt-token hidden states
  3. sample 20 test items (seed 42), generate unsteered CoTs, then one full-trace
     forward pass each; save projections [n_layers+1, seq_len] onto unit
     directions, plus boundary/label/token-count metadata

Outputs (all under sweep_out/): <tag>_proj.npz (proj_0..proj_19, W_unit),
<tag>_meta.json, <tag>_train_labels.json. Plotting is a separate CPU step:
scripts/plot_projection_sweep.py.
"""

import gc
import json
import os
import re

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = [
    "google/gemma-2-2b-it",
    "google/gemma-2-9b-it",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "openai/gpt-oss-20b",
]
DATASETS = ["anachronisms", "logical_deduction", "social_chemistry", "sports_understanding"]
N_TRACES = 20
SEED = 42
OUT = "sweep_out"
os.makedirs(OUT, exist_ok=True)


def is_harmony(model_name):
    return model_name.startswith("openai/")


def build_prompt(tok, messages, harmony):
    if messages[-1]["role"] == "assistant":
        if harmony:
            return tok.apply_chat_template(messages[:-1], tokenize=False,
                                           add_generation_prompt=True)
        return tok.apply_chat_template(messages, tokenize=False,
                                       continue_final_message=True)
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def parse_semantic(text, letter_map):
    low = text.lower()
    idx = low.rfind("the best answer is")
    if idx == -1:
        return None
    m = re.search(r"\(\s*([ab])\s*\)", low[idx:])
    return letter_map.get(m.group(1).upper()) if m else None


def load_items(ds, tok, harmony):
    items = [json.loads(l) for l in open(f"paper_inputs/{ds}.jsonl")]
    for it in items:
        other = "B" if it["correct_letter"] == "A" else "A"
        flip = "no" if it["correct_answer"] == "yes" else "yes"
        it["letter_map"] = {it["correct_letter"]: it["correct_answer"], other: flip}
        it["prompt"] = build_prompt(tok, it["messages"], harmony)
    return items


@torch.no_grad()
def generate(model, tok, prompts, max_new, batch=8):
    outs = []
    for b in range(0, len(prompts), batch):
        enc = tok(prompts[b:b + batch], return_tensors="pt", padding=True).to("cuda")
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=True,
                             temperature=0.7, pad_token_id=tok.pad_token_id)
        for j in range(out.shape[0]):
            outs.append(tok.decode(out[j, enc["input_ids"].shape[1]:],
                                   skip_special_tokens=False))
        if b % (batch * 8) == 0:
            print(f"    gen {b}/{len(prompts)}", flush=True)
    return outs


@torch.no_grad()
def last_token_states(model, tok, prompts):
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


for model_name in MODELS:
    short = model_name.split("/")[-1]
    todo = [ds for ds in DATASETS if not os.path.exists(f"{OUT}/{short}_{ds}_proj.npz")]
    if not todo:
        print(f"== {short}: all datasets cached, skipping model load", flush=True)
        continue
    print(f"== loading {model_name}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto",
                                                 device_map="cuda")
    model.eval()
    harmony = is_harmony(model_name)
    max_new = 2000 if harmony else 512

    for ds in todo:
        tag = f"{short}_{ds}"
        print(f"---- {tag}", flush=True)
        items = load_items(ds, tok, harmony)
        train = [it for it in items if it["fold"] == "train"]
        test = [it for it in items if it["fold"] == "test"]

        # train labels (model's own answers), cached
        lbl_ck = f"{OUT}/{tag}_train_labels.json"
        if os.path.exists(lbl_ck):
            train_labels = json.load(open(lbl_ck))
        else:
            gens = generate(model, tok, [it["prompt"] for it in train], max_new)
            train_labels = [parse_semantic(g, it["letter_map"])
                            for g, it in zip(gens, train)]
            json.dump(train_labels, open(lbl_ck, "w"))
        keep = [i for i, l in enumerate(train_labels) if l in ("yes", "no")]
        y = np.array([1 if train_labels[i] == "yes" else 0 for i in keep])
        print(f"  train labels: {len(keep)} parsed, {y.mean():.2f} yes", flush=True)

        feats = last_token_states(model, tok, [train[i]["prompt"] for i in keep])
        W = feats[y == 1].mean(0) - feats[y == 0].mean(0)
        W_unit = W / np.linalg.norm(W, axis=1, keepdims=True)
        del feats
        gc.collect(); torch.cuda.empty_cache()

        rng = np.random.default_rng(SEED)
        pick = rng.choice(len(test), size=N_TRACES, replace=False)
        sampled = [test[i] for i in pick]
        gens = generate(model, tok, [it["prompt"] for it in sampled], max_new)

        W_t = torch.tensor(W_unit, dtype=torch.float32, device="cuda")
        arrays, meta = {"W_unit": W_unit.astype(np.float32)}, []
        for i, (it, g) in enumerate(zip(sampled, gens)):
            boundary = tok(it["prompt"], return_tensors="pt")["input_ids"].shape[1]
            ids = tok(it["prompt"] + g, return_tensors="pt")["input_ids"][:, :6000]
            with torch.no_grad():
                out = model(ids.to("cuda"), output_hidden_states=True)
            H = torch.stack([h[0] for h in out.hidden_states]).float()
            proj = torch.einsum("lsd,ld->ls", H, W_t)
            arrays[f"proj_{i}"] = proj.cpu().numpy().astype(np.float16)
            meta.append({"boundary": int(boundary), "n_tokens": int(ids.shape[1]),
                         "label": parse_semantic(g, it["letter_map"]),
                         "correct": it["correct_answer"]})
        np.savez_compressed(f"{OUT}/{tag}_proj.npz", **arrays)
        json.dump({"model": model_name, "dataset": ds, "n_layers": int(W_unit.shape[0]),
                   "traces": meta}, open(f"{OUT}/{tag}_meta.json", "w"))
        print(f"  saved {tag}: {N_TRACES} traces", flush=True)

    del model
    gc.collect(); torch.cuda.empty_cache()

print("DONE projection sweep", flush=True)
