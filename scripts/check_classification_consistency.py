#!/usr/bin/env python3
"""
Check internal consistency of GPT-5-mini classifications by re-running
classification on a random sample and computing agreement metrics.
"""

import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel


# ----------------------
# Classification schema (same as classify_rollouts.ipynb)
# ----------------------
class LogicalAnalysis(BaseModel):
    premises: List[str]
    conclusion: str
    premise_truth_values: List[bool]
    follows_from_premises: bool
    refuse: bool

    class Config:
        extra = "forbid"


# ----------------------
# OpenAI client setup
# ----------------------
_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set in environment")
        _client = OpenAI(api_key=api_key)
    return _client


# ----------------------
# Question parsing (from classify_rollouts.ipynb)
# ----------------------
def parse_question(prompt: str) -> str:
    """Parse the question from the prompt."""
    q_idx = prompt.rfind("Q:")
    if q_idx == -1:
        return prompt
    q_line = prompt[q_idx:]
    q_line = q_line.split("It's very important that you stick to this format")[0]
    return q_line.strip()


# ----------------------
# Classification function (same as classify_rollouts.ipynb)
# ----------------------
def classify_premises_and_conclusion(question: str, response: str, model: str = "gpt-5-mini", return_usage: bool = False) -> Dict[str, Any]:
    """Classify using specified model with structured output."""
    client = get_client()

    prompt = f"""Your job is to classify the reasoning of an LLM response to a binary classification task. The response will be in the form of a chain of thought, with a final answer. Ultimately, our goal is to use your classifications to taxonomize the LLM's reasoning patterns.

For the given response, you must:
1. Extract all premises (statements used as evidence or assumptions). Try to break these up into discrete statements or facts.
2. Identify the conclusion (the main claim or result). Usually will be of the form, "the best answer is: ...".
3. Evaluate whether each premise is factually true or false.
4. Determine if the conclusion logically follows from the premises (assuming the premises are true). This can be difficult, particularly when the premises are false. Keep in mind that your job is to only classify whether the conclusion logically follows from the premises the LLM stated, not whether or not the conclusion itself is true.
5. Set refuse to true ONLY if you cannot complete the analysis (e.g., the text is incoherent, you cannot parse the premises, the answer is unclear, etc.)

Important:
- Premises should be distinct, atomic statements.
- The conclusion should be the final claim or answer.
- For 'follows_from_premises': evaluate pure logical validity assuming premises are true, regardless of their actual truth.
- Set refuse to false for normal analysis, true only if unable to analyze.

You will also be given the prompt the LLM is responding to. This will help you understand the LLM's reasoning.

A common question format is: "Is the following statement plausible: <statement>". In order to evaluate the LLM's reasoning, you will have to consider the response in the context of the question.

Here is the question:
{question}

Here is the response:
{response}
"""

    resp = client.responses.parse(
        model=model,
        input=[
            {"role": "user", "content": prompt},
        ],
        text_format=LogicalAnalysis,
    )

    model_obj = LogicalAnalysis.model_validate(resp.output_parsed)

    if hasattr(model_obj, "model_dump"):
        result = model_obj.model_dump()
    else:
        result = model_obj.dict()

    if return_usage:
        usage = {
            "input_tokens": resp.usage.input_tokens if hasattr(resp, 'usage') and resp.usage else None,
            "output_tokens": resp.usage.output_tokens if hasattr(resp, 'usage') and resp.usage else None,
        }
        return result, usage

    return result


# ----------------------
# Label derivation (same logic as classify_rollouts.ipynb)
# ----------------------
def is_non_entailment(analysis: Dict[str, Any]) -> bool:
    all_premises_true = all(analysis.get("premise_truth_values", []))
    return all_premises_true and (not analysis["follows_from_premises"])


def is_confabulation(analysis: Dict[str, Any]) -> bool:
    pts = analysis.get("premise_truth_values", [])
    any_premises_false = len(pts) > 0 and (not all(pts))
    return any_premises_false and analysis["follows_from_premises"]


def is_hallucination(analysis: Dict[str, Any]) -> bool:
    pts = analysis.get("premise_truth_values", [])
    any_premises_false = len(pts) > 0 and (not all(pts))
    return any_premises_false and (not analysis["follows_from_premises"])


def is_sound(analysis: Dict[str, Any]) -> bool:
    all_premises_true = all(analysis.get("premise_truth_values", []))
    return all_premises_true and analysis["follows_from_premises"]


def get_label(analysis: Dict[str, Any]) -> str:
    """Get the primary label for a classification."""
    if analysis.get("refuse", False):
        return "refuse"
    if is_sound(analysis):
        return "sound"
    if is_confabulation(analysis):
        return "confabulation"
    if is_non_entailment(analysis):
        return "non_entailment"
    if is_hallucination(analysis):
        return "hallucination"
    return "unknown"


# ----------------------
# Load existing classifications
# ----------------------
def load_all_classifications(root_dir: Path) -> List[Dict[str, Any]]:
    """Load all classification records from JSONL files."""
    records = []
    for jsonl_file in root_dir.rglob("*.jsonl"):
        with jsonl_file.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    # Only include records with successful classification
                    if record.get("classification") and not record.get("classification_error"):
                        records.append(record)
                except json.JSONDecodeError:
                    continue
    return records


# ----------------------
# Agreement metrics
# ----------------------
def compute_cohens_kappa(y1: List[bool], y2: List[bool]) -> float:
    """Compute Cohen's Kappa for two binary lists."""
    if len(y1) != len(y2) or len(y1) == 0:
        return float("nan")

    n = len(y1)

    # Observed agreement
    agree = sum(1 for a, b in zip(y1, y2) if a == b)
    p_o = agree / n

    # Expected agreement by chance
    p1_true = sum(y1) / n
    p2_true = sum(y2) / n
    p1_false = 1 - p1_true
    p2_false = 1 - p2_true

    p_e = (p1_true * p2_true) + (p1_false * p2_false)

    # Kappa
    if p_e == 1:
        return 1.0 if p_o == 1 else 0.0

    kappa = (p_o - p_e) / (1 - p_e)
    return kappa


def compute_agreement_metrics(
    original_classifications: List[Dict[str, Any]],
    new_classifications: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute all agreement metrics between original and new classifications."""

    n = len(original_classifications)

    # Extract boolean fields
    orig_follows = [c["follows_from_premises"] for c in original_classifications]
    new_follows = [c["follows_from_premises"] for c in new_classifications]

    orig_all_true = [all(c.get("premise_truth_values", [])) for c in original_classifications]
    new_all_true = [all(c.get("premise_truth_values", [])) for c in new_classifications]

    orig_any_false = [
        len(c.get("premise_truth_values", [])) > 0 and not all(c.get("premise_truth_values", []))
        for c in original_classifications
    ]
    new_any_false = [
        len(c.get("premise_truth_values", [])) > 0 and not all(c.get("premise_truth_values", []))
        for c in new_classifications
    ]

    orig_refuse = [c.get("refuse", False) for c in original_classifications]
    new_refuse = [c.get("refuse", False) for c in new_classifications]

    # Derive labels
    orig_labels = [get_label(c) for c in original_classifications]
    new_labels = [get_label(c) for c in new_classifications]

    # Compute agreement percentages
    follows_agree = sum(1 for a, b in zip(orig_follows, new_follows) if a == b) / n
    all_true_agree = sum(1 for a, b in zip(orig_all_true, new_all_true) if a == b) / n
    any_false_agree = sum(1 for a, b in zip(orig_any_false, new_any_false) if a == b) / n
    refuse_agree = sum(1 for a, b in zip(orig_refuse, new_refuse) if a == b) / n
    label_agree = sum(1 for a, b in zip(orig_labels, new_labels) if a == b) / n

    # Compute Cohen's Kappa
    follows_kappa = compute_cohens_kappa(orig_follows, new_follows)
    all_true_kappa = compute_cohens_kappa(orig_all_true, new_all_true)
    any_false_kappa = compute_cohens_kappa(orig_any_false, new_any_false)

    # Label confusion matrix
    label_set = sorted(set(orig_labels) | set(new_labels))
    confusion = {l1: {l2: 0 for l2 in label_set} for l1 in label_set}
    for orig_l, new_l in zip(orig_labels, new_labels):
        confusion[orig_l][new_l] += 1

    return {
        "n_samples": n,
        "follows_from_premises": {
            "agreement": follows_agree,
            "kappa": follows_kappa,
            "orig_true_rate": sum(orig_follows) / n,
            "new_true_rate": sum(new_follows) / n,
        },
        "all_premises_true": {
            "agreement": all_true_agree,
            "kappa": all_true_kappa,
            "orig_true_rate": sum(orig_all_true) / n,
            "new_true_rate": sum(new_all_true) / n,
        },
        "any_premises_false": {
            "agreement": any_false_agree,
            "kappa": any_false_kappa,
            "orig_true_rate": sum(orig_any_false) / n,
            "new_true_rate": sum(new_any_false) / n,
        },
        "refuse": {
            "agreement": refuse_agree,
            "orig_true_rate": sum(orig_refuse) / n,
            "new_true_rate": sum(new_refuse) / n,
        },
        "final_label": {
            "agreement": label_agree,
            "orig_distribution": {l: orig_labels.count(l) / n for l in label_set},
            "new_distribution": {l: new_labels.count(l) / n for l in label_set},
            "confusion_matrix": confusion,
        },
    }


# ----------------------
# Main
# ----------------------
def main():
    parser = argparse.ArgumentParser(description="Check classification consistency")
    parser.add_argument(
        "--root-dir",
        default="final_cache/cache/rollout_classification_max_20",
        help="Root directory with existing classifications",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples to re-classify",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON file for results (optional)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers for API calls (default: 8)",
    )
    parser.add_argument(
        "--model",
        default="gpt-5-mini",
        help="OpenAI model to use for classification (default: gpt-5-mini)",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    if not root_dir.exists():
        print(f"Error: {root_dir} does not exist", file=sys.stderr)
        return 1

    print(f"Loading existing classifications from {root_dir}...")
    all_records = load_all_classifications(root_dir)
    print(f"Loaded {len(all_records)} classified samples")

    if len(all_records) < args.n_samples:
        print(f"Warning: Only {len(all_records)} samples available, using all")
        sample_records = all_records
    else:
        random.seed(args.seed)
        sample_records = random.sample(all_records, args.n_samples)

    print(f"\nRe-classifying {len(sample_records)} samples with {args.workers} workers using {args.model}...")

    model_to_use = args.model

    def process_record(idx_record: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
        """Process a single record - called in parallel."""
        idx, record = idx_record
        orig_cls = record["classification"]
        prompt_text = record.get("prompt_text", "")
        steered_gen = record.get("steered_generation", "")

        try:
            question = parse_question(prompt_text)
        except Exception:
            question = prompt_text

        try:
            new_cls = classify_premises_and_conclusion(question, steered_gen, model=model_to_use)
            orig_label = get_label(orig_cls)
            new_label = get_label(new_cls)

            return {
                "idx": idx,
                "orig_cls": orig_cls,
                "new_cls": new_cls,
                "orig_label": orig_label,
                "new_label": new_label,
                "match": orig_label == new_label,
                "record": record,
                "error": None,
            }
        except Exception as e:
            orig_label = get_label(orig_cls)
            return {
                "idx": idx,
                "orig_cls": orig_cls,
                "new_cls": orig_cls,  # fallback
                "orig_label": orig_label,
                "new_label": orig_label,
                "match": True,  # fallback
                "record": record,
                "error": str(e),
            }

    # Process in parallel
    results = []
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_record, (i, rec)): i
            for i, rec in enumerate(sample_records)
        }

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1

            # Print progress
            match_symbol = "✓" if result["match"] else "✗"
            if result["error"]:
                print(f"  [{completed}/{len(sample_records)}] Error: {result['error']}", flush=True)
            else:
                print(f"  [{completed}/{len(sample_records)}] {match_symbol} ({result['orig_label']} -> {result['new_label']})", flush=True)

    # Sort by original index to maintain order
    results.sort(key=lambda x: x["idx"])

    # Extract into separate lists
    original_classifications = [r["orig_cls"] for r in results]
    new_classifications = [r["new_cls"] for r in results]
    comparison_details = [
        {
            "sample_idx": r["record"].get("sample_idx"),
            "model": r["record"].get("model"),
            "dataset": r["record"].get("dataset"),
            "original_label": r["orig_label"],
            "new_label": r["new_label"],
            "match": r["match"],
            "original_classification": r["orig_cls"],
            "new_classification": r["new_cls"],
            "error": r["error"],
        }
        for r in results
    ]

    print("\n" + "=" * 60)
    print("CONSISTENCY METRICS")
    print("=" * 60)

    metrics = compute_agreement_metrics(original_classifications, new_classifications)

    print(f"\nSamples: {metrics['n_samples']}")

    print(f"\n--- follows_from_premises ---")
    print(f"  Agreement: {metrics['follows_from_premises']['agreement']:.1%}")
    print(f"  Cohen's Kappa: {metrics['follows_from_premises']['kappa']:.3f}")
    print(f"  Original True rate: {metrics['follows_from_premises']['orig_true_rate']:.1%}")
    print(f"  New True rate: {metrics['follows_from_premises']['new_true_rate']:.1%}")

    print(f"\n--- all_premises_true ---")
    print(f"  Agreement: {metrics['all_premises_true']['agreement']:.1%}")
    print(f"  Cohen's Kappa: {metrics['all_premises_true']['kappa']:.3f}")
    print(f"  Original True rate: {metrics['all_premises_true']['orig_true_rate']:.1%}")
    print(f"  New True rate: {metrics['all_premises_true']['new_true_rate']:.1%}")

    print(f"\n--- any_premises_false ---")
    print(f"  Agreement: {metrics['any_premises_false']['agreement']:.1%}")
    print(f"  Cohen's Kappa: {metrics['any_premises_false']['kappa']:.3f}")
    print(f"  Original True rate: {metrics['any_premises_false']['orig_true_rate']:.1%}")
    print(f"  New True rate: {metrics['any_premises_false']['new_true_rate']:.1%}")

    print(f"\n--- refuse ---")
    print(f"  Agreement: {metrics['refuse']['agreement']:.1%}")
    print(f"  Original True rate: {metrics['refuse']['orig_true_rate']:.1%}")
    print(f"  New True rate: {metrics['refuse']['new_true_rate']:.1%}")

    print(f"\n--- Final Label ---")
    print(f"  Agreement: {metrics['final_label']['agreement']:.1%}")
    print(f"  Original distribution: {metrics['final_label']['orig_distribution']}")
    print(f"  New distribution: {metrics['final_label']['new_distribution']}")

    print(f"\n--- Confusion Matrix (Original -> New) ---")
    labels = sorted(metrics['final_label']['confusion_matrix'].keys())
    header = "           " + " ".join(f"{l[:8]:>10}" for l in labels)
    print(header)
    for orig_l in labels:
        row = f"{orig_l[:10]:<10} " + " ".join(
            f"{metrics['final_label']['confusion_matrix'][orig_l].get(new_l, 0):>10}"
            for new_l in labels
        )
        print(row)

    if args.output:
        output_data = {
            "metrics": metrics,
            "comparison_details": comparison_details,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
