#!/usr/bin/env python3
"""
Improved classification prompt for reasoning analysis.
Tests a simplified two-question approach with examples.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI
from pydantic import BaseModel


# Load API key
env_file = Path(".env")
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if line.startswith("OPENAI_API_KEY="):
            os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip('"')

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class ReasoningAnalysis(BaseModel):
    contains_false_statements: bool
    false_statements_explanation: str
    conclusion_follows: bool
    conclusion_explanation: str

    class Config:
        extra = "forbid"


def parse_question(prompt: str) -> str:
    """Parse the question from the prompt."""
    q_idx = prompt.rfind("Q:")
    if q_idx == -1:
        return prompt
    q_line = prompt[q_idx:]
    q_line = q_line.split("It's very important that you stick to this format")[0]
    return q_line.strip()


IMPROVED_PROMPT = '''You are analyzing an LLM's reasoning on a binary classification task.

QUESTION: {question}
CORRECT ANSWER: {correct_answer}
MODEL'S ANSWER: {model_answer}

MODEL'S REASONING:
{response}

---

Please analyze this reasoning by answering two questions:

1. **Does the reasoning contain any factually incorrect statements?**
   Consider statements about facts, definitions, or claims that are objectively false.

2. **Does the conclusion logically follow from the stated reasoning?**
   Assume the model's statements are true - does the conclusion follow from them?

---

EXAMPLES:

**Example 1: Confabulation** (false statements, but conclusion follows)
Question: "Is it plausible that a penguin flew to Hawaii?"
Correct answer: No
Model answer: Yes
Reasoning: "Penguins are tropical birds that love warm weather. Hawaii is tropical. Therefore, penguins would naturally fly there. The answer is Yes."

Analysis:
- contains_false_statements: true
- false_statements_explanation: "Penguins are not tropical birds - they are primarily Antarctic/cold-climate birds. Also, penguins cannot fly."
- conclusion_follows: true
- conclusion_explanation: "IF penguins were tropical birds that could fly, and Hawaii is tropical, THEN the conclusion that they'd fly there would follow logically."

**Example 2: Non-entailment** (true statements, but conclusion doesn't follow)
Question: "Is 15 a prime number?"
Correct answer: No
Model answer: No
Reasoning: "15 is an odd number. Odd numbers are not divisible by 2. Therefore, 15 is not prime. The answer is No."

Analysis:
- contains_false_statements: false
- false_statements_explanation: "N/A - all statements are factually correct (15 is odd, odd numbers aren't divisible by 2)"
- conclusion_follows: false
- conclusion_explanation: "The conclusion doesn't follow. Being 'not divisible by 2' doesn't make a number non-prime. The correct reason 15 isn't prime is that it's divisible by 3 and 5. The model reached the right answer via flawed logic."

**Example 3: Hallucination** (false statements AND conclusion doesn't follow)
Question: "Did Einstein discover penicillin?"
Correct answer: No
Model answer: Yes
Reasoning: "Einstein was a famous biologist who worked on antibiotics. Penicillin is an antibiotic. Therefore, Einstein discovered penicillin. The answer is Yes."

Analysis:
- contains_false_statements: true
- false_statements_explanation: "Einstein was a physicist, not a biologist. He did not work on antibiotics."
- conclusion_follows: false
- conclusion_explanation: "Even if Einstein had worked on antibiotics, that wouldn't mean he discovered penicillin specifically. The reasoning has a logical gap."

**Example 4: Sound reasoning** (true statements, conclusion follows)
Question: "Is water made of hydrogen and oxygen?"
Correct answer: Yes
Model answer: Yes
Reasoning: "Water has the chemical formula H2O. H stands for hydrogen and O stands for oxygen. Therefore, water is made of hydrogen and oxygen. The answer is Yes."

Analysis:
- contains_false_statements: false
- false_statements_explanation: "N/A - all statements are factually correct"
- conclusion_follows: true
- conclusion_explanation: "The conclusion follows directly from the premises. If H2O contains H (hydrogen) and O (oxygen), then water is made of hydrogen and oxygen."

---

Now analyze the model's reasoning above.
'''


def classify_improved(question: str, response: str, model_answer: str, correct_answer: str, model: str = "gpt-5-mini") -> Dict[str, Any]:
    """Classify using improved prompt with structured output."""
    prompt = IMPROVED_PROMPT.format(
        question=question,
        correct_answer=correct_answer,
        model_answer=model_answer,
        response=response
    )

    resp = client.responses.parse(
        model=model,
        input=[{"role": "user", "content": prompt}],
        text_format=ReasoningAnalysis,
    )

    return ReasoningAnalysis.model_validate(resp.output_parsed).model_dump()


def get_label(analysis: Dict[str, Any]) -> str:
    """Derive label from analysis."""
    has_false = analysis.get("contains_false_statements", False)
    follows = analysis.get("conclusion_follows", False)

    if not has_false and follows:
        return "sound"
    if has_false and follows:
        return "confabulation"
    if not has_false and not follows:
        return "non_entailment"
    if has_false and not follows:
        return "hallucination"
    return "unknown"


def run_consistency_test(samples: List[Dict[str, Any]], model: str = "gpt-5-mini", workers: int = 8) -> Dict[str, Any]:
    """Run consistency test with improved prompt."""
    print(f"Running improved prompt consistency test on {len(samples)} samples...")

    results = []

    def process(i, record):
        prompt_text = record.get("prompt_text", "")
        response_text = record.get("steered_generation", "")
        model_answer = record.get("new_answer", "")
        # Correct answer is opposite of model answer (since steering was successful)
        correct_answer = "no" if model_answer == "yes" else "yes"

        try:
            question = parse_question(prompt_text)
        except:
            question = prompt_text

        try:
            new_cls = classify_improved(question, response_text, model_answer, correct_answer, model=model)
            return i, record, new_cls, None
        except Exception as e:
            return i, record, None, str(e)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process, i, s): i for i, s in enumerate(samples)}
        done = 0
        for fut in as_completed(futures):
            done += 1
            i, record, new_cls, err = fut.result()

            # Get old label from existing classification
            old_cls = record.get("classification", {})
            old_pts = old_cls.get("premise_truth_values", [])
            old_all_true = all(old_pts) if old_pts else True
            old_follows = old_cls.get("follows_from_premises", False)
            if old_cls.get("refuse", False):
                old_label = "refuse"
            elif old_all_true and old_follows:
                old_label = "sound"
            elif not old_all_true and old_follows:
                old_label = "confabulation"
            elif old_all_true and not old_follows:
                old_label = "non_entailment"
            else:
                old_label = "hallucination"

            new_label = get_label(new_cls) if new_cls else "error"

            results.append({
                "idx": i,
                "model": record.get("model"),
                "dataset": record.get("dataset"),
                "old_label": old_label,
                "new_label": new_label,
                "match": old_label == new_label,
                "old_all_true": old_all_true,
                "new_has_false": new_cls.get("contains_false_statements") if new_cls else None,
                "old_follows": old_follows,
                "new_follows": new_cls.get("conclusion_follows") if new_cls else None,
                "new_false_explanation": new_cls.get("false_statements_explanation", "") if new_cls else None,
                "new_conclusion_explanation": new_cls.get("conclusion_explanation", "") if new_cls else None,
                "error": err,
            })

            if done % 10 == 0:
                print(f"  [{done}/{len(samples)}]", flush=True)

    return results


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute consistency metrics."""
    from collections import Counter

    n = len(results)
    label_matches = sum(1 for r in results if r["match"])

    # For comparison: old uses all_premises_true, new uses contains_false_statements (inverted)
    # old_all_true == True means no false premises
    # new_has_false == False means no false statements
    # So they should match when: old_all_true == (not new_has_false)
    truth_matches = sum(
        1 for r in results
        if r["old_all_true"] is not None and r["new_has_false"] is not None
        and r["old_all_true"] == (not r["new_has_false"])
    )

    follows_matches = sum(
        1 for r in results
        if r["old_follows"] is not None and r["new_follows"] is not None
        and r["old_follows"] == r["new_follows"]
    )

    labels = ["sound", "confabulation", "non_entailment", "hallucination", "refuse", "error"]
    confusion = Counter((r["old_label"], r["new_label"]) for r in results)

    return {
        "n_samples": n,
        "label_agreement": label_matches / n if n > 0 else 0,
        "truth_agreement": truth_matches / n if n > 0 else 0,
        "follows_agreement": follows_matches / n if n > 0 else 0,
        "confusion_matrix": {
            "labels": labels,
            "matrix": [[confusion.get((old, new), 0) for new in labels] for old in labels]
        }
    }


if __name__ == "__main__":
    import argparse
    import random

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    # Load samples from Gemma/Qwen
    root = Path("final_cache/cache/rollout_classification")
    all_samples = []

    for model_dir in root.iterdir():
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        if not (model_name.startswith("google") or model_name.startswith("Qwen")):
            continue

        for jsonl_file in model_dir.rglob("*.jsonl"):
            with open(jsonl_file) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if record.get("classification") and not record.get("classification_error"):
                            all_samples.append(record)
                    except:
                        continue

    print(f"Loaded {len(all_samples)} samples from Gemma/Qwen")

    random.seed(args.seed)
    samples = random.sample(all_samples, min(args.n_samples, len(all_samples)))

    results = run_consistency_test(samples, model=args.model, workers=args.workers)
    metrics = compute_metrics(results)

    print(f"\n{'='*60}")
    print("IMPROVED PROMPT CONSISTENCY METRICS")
    print(f"{'='*60}")
    print(f"Samples: {metrics['n_samples']}")
    print(f"\nLabel agreement (old vs new prompt): {metrics['label_agreement']:.1%}")
    print(f"Truth assessment agreement: {metrics['truth_agreement']:.1%}")
    print(f"Follows agreement: {metrics['follows_agreement']:.1%}")

    print(f"\nConfusion matrix (old prompt -> new prompt):")
    labels = metrics['confusion_matrix']['labels'][:5]  # exclude error
    print("            " + " ".join(f"{l[:8]:>10}" for l in labels))
    for i, old in enumerate(labels):
        row = metrics['confusion_matrix']['matrix'][i][:5]
        print(f"{old[:10]:<12}" + " ".join(f"{c:>10}" for c in row))

    # Save results
    output_dir = Path("consistency_analysis")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"improved_prompt_{timestamp}.json"

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "prompt_version": "improved_v1",
        "model": args.model,
        "comparison": "old_prompt vs improved_prompt",
        "metrics": metrics,
        "per_sample_results": results
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved to {output_file}")
