"""Recover each sweep trace's question-start token index (no GPU needed).

The sweep's sampled test items are seed-deterministic, so we rebuild each
prompt exactly as projection_sweep.py did, then map the final user message's
character position to a token index via the tokenizer's offset mapping.
Writes sweep_out/<tag>_qstart.json: list of question-start indices (one per
trace, aligned with proj_i order).

Usage: python scripts/reconstruct_qstarts.py --sweep-out <dir> [--inputs paper_inputs]
"""

import argparse
import glob
import json
import os

import numpy as np
from transformers import AutoTokenizer

SEED = 42
N_TRACES = 20


def build_prompt(tok, messages, harmony):
    if messages[-1]["role"] == "assistant":
        if harmony:
            return tok.apply_chat_template(messages[:-1], tokenize=False,
                                           add_generation_prompt=True)
        return tok.apply_chat_template(messages, tokenize=False,
                                       continue_final_message=True)
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-out", default="sweep_out")
    ap.add_argument("--inputs", default="paper_inputs")
    args = ap.parse_args()

    toks = {}
    for mpath in sorted(glob.glob(f"{args.sweep_out}/*_meta.json")):
        meta = json.load(open(mpath))
        model, ds = meta["model"], meta["dataset"]
        tag = f"{model.split('/')[-1]}_{ds}"
        out_path = f"{args.sweep_out}/{tag}_qstart.json"
        if os.path.exists(out_path):
            continue
        if model not in toks:
            toks[model] = AutoTokenizer.from_pretrained(model)
        tok = toks[model]
        harmony = model.startswith("openai/")

        items = [json.loads(l) for l in open(f"{args.inputs}/{ds}.jsonl")]
        test = [it for it in items if it["fold"] == "test"]
        rng = np.random.default_rng(SEED)
        pick = rng.choice(len(test), size=N_TRACES, replace=False)

        qstarts = []
        for k, i in enumerate(pick):
            it = test[i]
            prompt = build_prompt(tok, it["messages"], harmony)
            # final user message content marks the question
            q_text = next(m["content"] for m in reversed(it["messages"]) if m["role"] == "user")
            char_idx = prompt.rfind(q_text[:80])
            assert char_idx != -1, (tag, k)
            enc = tok(prompt, return_offsets_mapping=True)
            q_tok = next(j for j, (s, e) in enumerate(enc["offset_mapping"]) if e > char_idx)
            # sanity: boundary from meta should equal prompt token count
            n_prompt = len(enc["input_ids"])
            assert abs(n_prompt - meta["traces"][k]["boundary"]) <= 1, \
                (tag, k, n_prompt, meta["traces"][k]["boundary"])
            qstarts.append(int(q_tok))
        json.dump(qstarts, open(out_path, "w"))
        print(f"{tag}: qstarts written (mean question span "
              f"{np.mean([meta['traces'][k]['boundary'] - q for k, q in enumerate(qstarts)]):.0f} tokens)")


if __name__ == "__main__":
    main()
