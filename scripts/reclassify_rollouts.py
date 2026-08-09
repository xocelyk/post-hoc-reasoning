#!/usr/bin/env python3
"""
Re-run classification on all rollout JSONL files for Gemma and Qwen models.

Uses the improved two-question classification prompt with gpt-5-mini.
Saves results to a parallel directory structure.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI
from pydantic import BaseModel


# Load API key
def load_env():
    env_file = Path(".env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY="):
                os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip('"')


load_env()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class ReasoningAnalysis(BaseModel):
    """Schema for improved classification prompt."""
    contains_false_statements: bool
    false_statements_explanation: str
    conclusion_follows: bool
    conclusion_explanation: str

    class Config:
        extra = "forbid"


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


def parse_question(prompt: str) -> str:
    """Extract question from prompt text."""
    q_idx = prompt.rfind("Q:")
    if q_idx == -1:
        return prompt
    q_line = prompt[q_idx:]
    q_line = q_line.split("It's very important that you stick to this format")[0]
    return q_line.strip()


def classify_record(
    question: str,
    response: str,
    model_answer: str,
    correct_answer: str,
    model: str = "gpt-5-mini"
) -> Dict[str, Any]:
    """Classify a single record using the improved prompt."""
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
    """Derive classification label from analysis."""
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


def should_process_model(model_name: str) -> bool:
    """Check if model directory should be processed."""
    return model_name.startswith("google") or model_name.startswith("Qwen")


def find_jsonl_files(input_dir: Path) -> List[Path]:
    """Find all JSONL files to process."""
    files = []
    for model_dir in input_dir.iterdir():
        if not model_dir.is_dir():
            continue
        if not should_process_model(model_dir.name):
            continue
        for jsonl_file in model_dir.rglob("*.jsonl"):
            files.append(jsonl_file)
    return sorted(files)


def get_output_path(input_file: Path, input_dir: Path, output_dir: Path) -> Path:
    """Get the corresponding output path for an input file."""
    rel_path = input_file.relative_to(input_dir)
    return output_dir / rel_path


def process_record(record: Dict[str, Any], model: str) -> Dict[str, Any]:
    """Process a single record and return updated record."""
    prompt_text = record.get("prompt_text", "")
    response_text = record.get("steered_generation", "")
    model_answer = record.get("new_answer", "")
    correct_answer = "no" if model_answer == "yes" else "yes"

    try:
        question = parse_question(prompt_text)
    except Exception:
        question = prompt_text

    # Create new record with updated classification
    new_record = record.copy()

    # Store old classification as v1
    if "classification" in record:
        new_record["classification_v1"] = record["classification"]

    try:
        classification = classify_record(
            question=question,
            response=response_text,
            model_answer=model_answer,
            correct_answer=correct_answer,
            model=model
        )
        new_record["classification"] = classification
        new_record["classification_label"] = get_label(classification)
        new_record["classification_error"] = None
    except Exception as e:
        new_record["classification"] = None
        new_record["classification_label"] = "error"
        new_record["classification_error"] = str(e)

    new_record["classification_model"] = model
    new_record["classification_prompt_version"] = "improved_v1"
    new_record["classification_timestamp"] = datetime.now().isoformat()

    return new_record


def process_file(
    input_file: Path,
    output_file: Path,
    model: str,
    workers: int
) -> Dict[str, Any]:
    """Process a single JSONL file."""
    # Load records
    records = []
    with open(input_file) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not records:
        return {"file": str(input_file), "records": 0, "success": 0, "errors": 0}

    # Process records in parallel
    results = []
    success = 0
    errors = 0

    def process_with_index(i: int, record: Dict[str, Any]):
        return i, process_record(record, model)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_with_index, i, r): i for i, r in enumerate(records)}
        for fut in as_completed(futures):
            i, new_record = fut.result()
            results.append((i, new_record))
            if new_record.get("classification_error"):
                errors += 1
            else:
                success += 1

    # Sort by original index and extract records
    results.sort(key=lambda x: x[0])
    processed_records = [r for _, r in results]

    # Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for record in processed_records:
            f.write(json.dumps(record) + "\n")

    return {
        "file": str(input_file),
        "records": len(records),
        "success": success,
        "errors": errors
    }


def main():
    parser = argparse.ArgumentParser(
        description="Re-run classification on rollout JSONL files"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("final_cache/cache/rollout_classification"),
        help="Source directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("final_cache/cache/rollout_classification_v2"),
        help="Output directory"
    )
    parser.add_argument(
        "--model",
        default="gpt-5-mini",
        help="Classifier model"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel workers for API calls"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip files that already exist in output"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without running"
    )
    args = parser.parse_args()

    # Find files to process
    print(f"Scanning {args.input_dir}...")
    all_files = find_jsonl_files(args.input_dir)
    print(f"Found {len(all_files)} JSONL files in Gemma/Qwen directories")

    # Filter if resuming
    files_to_process = []
    for f in all_files:
        output_path = get_output_path(f, args.input_dir, args.output_dir)
        if args.resume and output_path.exists():
            continue
        files_to_process.append((f, output_path))

    print(f"Files to process: {len(files_to_process)}")

    if args.dry_run:
        print("\n[DRY RUN] Would process:")
        for input_file, output_file in files_to_process[:20]:
            print(f"  {input_file.relative_to(args.input_dir)}")
        if len(files_to_process) > 20:
            print(f"  ... and {len(files_to_process) - 20} more files")
        return

    if not files_to_process:
        print("Nothing to process.")
        return

    # Process files
    print(f"\nProcessing with {args.model}...")
    total_records = 0
    total_success = 0
    total_errors = 0

    for i, (input_file, output_file) in enumerate(files_to_process):
        rel_path = input_file.relative_to(args.input_dir)
        print(f"[{i+1}/{len(files_to_process)}] {rel_path}")

        stats = process_file(input_file, output_file, args.model, args.workers)
        total_records += stats["records"]
        total_success += stats["success"]
        total_errors += stats["errors"]

        print(f"    {stats['success']}/{stats['records']} records classified")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Files processed: {len(files_to_process)}")
    print(f"Total records: {total_records}")
    print(f"Successful: {total_success}")
    print(f"Errors: {total_errors}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
