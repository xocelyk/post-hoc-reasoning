"""Export the instruct experiments' exact train/test items for the GPT-OSS rerun.

Reads train_test_split.pkl from a paper-model cache for each dataset (the
stored split IS the item lists used by the instruct experiments) and writes
paper_inputs/<ds>.jsonl with {messages, fold, correct_letter, correct_answer}.

Run from the repo root: python scripts/export_paper_inputs.py [--out paper_inputs]
"""

import argparse
import glob
import json
import os
import pickle

PREFERRED_MODELS = [
    "google_gemma-2-9b-it", "Qwen_Qwen2.5-7B-Instruct", "google_gemma-2-2b-it",
    "Qwen_Qwen2.5-3B-Instruct", "Qwen_Qwen2.5-1.5B-Instruct",
]
DATASETS = ["anachronisms", "logical_deduction", "social_chemistry", "sports_understanding"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", default="cache/experiments")
    ap.add_argument("--out", default="paper_inputs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for ds in DATASETS:
        chosen = None
        for m in PREFERRED_MODELS:
            hits = glob.glob(f"{args.cache_root}/{m}/{ds}/split_42_500_500/*/data/train_test_split.pkl")
            if hits:
                chosen = (m, hits[0])
                break
        if not chosen:
            raise SystemExit(f"no paper-model split found for {ds}")
        model, path = chosen
        train, test = pickle.load(open(path, "rb"))
        with open(f"{args.out}/{ds}.jsonl", "w") as f:
            for fold, items in [("train", train), ("test", test)]:
                for it in items:
                    f.write(json.dumps({
                        "messages": it["prompt"], "fold": fold,
                        "correct_letter": it["correct_letter"],
                        "correct_answer": str(it["correct_answer"]).strip().lower(),
                    }) + "\n")
        print(f"{ds}: {model} (train {len(train)} / test {len(test)})")


if __name__ == "__main__":
    main()
