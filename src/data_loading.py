import json
import os
import random
from typing import Dict, List


def format_sports_understanding_from_json(data: List[Dict]) -> List[List[str]]:
    result = []
    for example in data:
        question = example["input"]  # Use the full question as-is
        target = example["target"]
        label = "no" if target.lower() == "no" else "yes"
        result.append([question, label])
    return result


def format_anachronisms_from_json(data: Dict) -> List[List[str]]:
    result = []
    examples = data.get("examples", [])
    for example in examples:
        input_text = example["input"]
        target_scores = example["target_scores"]
        # Determine the correct answer based on the highest score
        correct_answer = max(target_scores, key=target_scores.get)
        label = "yes" if correct_answer.lower().startswith("yes") else "no"
        result.append([input_text, label])
    return result


def format_snarks_from_json(data: Dict) -> List[List[str]]:
    result = []
    examples = data.get("examples", [])
    for example in examples:
        input_text = example["input"]
        # Extract the statement after "Options:"
        input_text = input_text.split("\nOptions:\n")[-1].split("\n")[0].strip()
        target = example["target"]
        label = "no" if target == "(A)" else "yes"
        result.append([input_text, label])
    return result


def format_social_chemistry_from_json(data: List[Dict]) -> List[List[str]]:
    result = []
    for item in data:
        action = item.get("action", "")
        if not isinstance(action, str):
            continue
        action = action.strip()
        binary_label = item.get("binary_appropriateness", "").strip()
        if not isinstance(binary_label, str):
            continue
        if binary_label.lower() == "appropriate":
            binary_label = "yes"
        elif binary_label.lower() == "inappropriate":
            binary_label = "no"
        else:
            raise ValueError(f"Unknown binary label: {binary_label}")
        if not action or not binary_label:
            continue  # Skip if action or label is missing
        result.append([action, binary_label])
    return result


def format_logical_deduction_from_json(data: List[Dict]) -> List[List[str]]:
    result = []
    for item in data:
        object_description = item["input"]
        statements = list(item["target_scores"].keys())
        scores = item["target_scores"]

        # Randomly select one of the statements
        selected_statement = random.choice(statements)
        correctness = scores[selected_statement]  # 1 or 0

        # Map correctness to 'plausible' or 'implausible'
        label = "yes" if correctness == 1 else "no"

        result.append([object_description, selected_statement, label])
    return result


def format_quora_questions_from_json(data: List[Dict]) -> List[List[str]]:
    result = []
    for item in data:
        question1 = item.get("question1", "")
        question2 = item.get("question2", "")
        if not isinstance(question1, str) or not isinstance(question2, str):
            continue
        question1 = question1.strip()
        question2 = question2.strip()
        is_duplicate = item.get("is_duplicate", 0)

        if not question1 or not question2:
            continue  # Skip if either question is missing

        # Map 'is_duplicate' to labels
        label = "yes" if is_duplicate == 1 else "no"
        result.append([question1, question2, label])
    return result


def create_dataset(task_name: str) -> List[List[str]]:
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to get to the project root, then to data
    json_filename = os.path.join(script_dir, "..", "data", task_name, f"{task_name}.json")
    with open(json_filename, "r") as f:
        json_data = json.load(f)

    if task_name == "sports_understanding":
        example_data = format_sports_understanding_from_json(json_data)
    elif task_name == "anachronisms":
        example_data = format_anachronisms_from_json(json_data)
    elif task_name == "snarks":
        example_data = format_snarks_from_json(json_data)
    elif task_name == "social_chemistry":
        example_data = format_social_chemistry_from_json(json_data)
    elif task_name == "logical_deduction":
        example_data = format_logical_deduction_from_json(json_data)
    elif task_name == "quora_question_pairs":
        example_data = format_quora_questions_from_json(json_data)
    else:
        raise ValueError(f"Unknown task name: {task_name}")

    return example_data


def create_noncot_dataset(task_name: str, examples: List[List[str]]) -> List[Dict]:
    cot_prompt = load_cot_prompt(task_name)
    example_instruction = 'Please verbalize how you are thinking about the problem, then give your answer in the format "The best answer is: (X)". It\'s very important that you stick to this format.'


def create_cot_dataset(
    task_name: str, examples: List[List[str]], thinking: bool = True
) -> List[Dict]:
    cot_prompt = load_cot_prompt(task_name)
    example_instruction = 'Please verbalize how you are thinking about the problem, then give your answer in the format "The best answer is: (X)". It\'s very important that you stick to this format.'
    if not thinking:
        for turn in cot_prompt:
            if turn["role"] == "user":
                turn["content"] = turn["content"].replace(example_instruction, "")
            else:
                import re

                match = re.search(r"\((A|B)\).*?(Yes|No)", turn["content"])
                if match:
                    letter, yes_no = match.groups()
                    turn["content"] = f"A: ({letter}) {yes_no}"

    if not thinking:
        example_instruction = ""

    task_configs = {
        "sports_understanding": {
            "question": "Is the following sentence plausible?",
            "choices": [
                ("Yes, the sentence is plausible", "No, the sentence is implausible"),
                ("No, the sentence is implausible", "Yes, the sentence is plausible"),
            ],
        },
        "anachronisms": {
            "question": "Does the following sentence contain anachronistic elements?",
            "choices": [
                (
                    "Yes, contains anachronistic elements",
                    "No, does not contain anachronistic elements",
                ),
                (
                    "No, does not contain anachronistic elements",
                    "Yes, contains anachronistic elements",
                ),
            ],
        },
        "snarks": {
            "question": "Is the following statement sarcastic?",
            "choices": [
                (
                    "Yes, the statement is sarcastic",
                    "No, the statement is not sarcastic",
                ),
                (
                    "No, the statement is not sarcastic",
                    "Yes, the statement is sarcastic",
                ),
            ],
        },
        "social_chemistry": {
            "question": "Is the following action appropriate?",
            "choices": [
                ("Yes, the action is appropriate", "No, the action is inappropriate"),
                ("No, the action is inappropriate", "Yes, the action is appropriate"),
            ],
        },
        "logical_deduction": {
            "question": "Given this description, is the following statement plausible?",
            "choices": [
                ("Yes, the statement is plausible", "No, the statement is implausible"),
                ("No, the statement is implausible", "Yes, the statement is plausible"),
            ],
        },
        "quora_question_pairs": {
            "question": "Do the following two questions have the same meaning?",
            "choices": [
                (
                    "Yes, they have the same meaning",
                    "No, they do not have the same meaning",
                ),
                (
                    "No, they do not have the same meaning",
                    "Yes, they have the same meaning",
                ),
            ],
        },
    }

    dataset = []
    for example in examples:
        if task_name == "logical_deduction":
            text, statement, label = example
            full_text = (
                f"{text}\n\n{task_configs[task_name]['question']}\n\n\"{statement}\""
            )
        elif task_name == "quora_question_pairs":
            question1, question2, label = example
            full_text = f'\nQuestion 1: "{question1}"\nQuestion 2: "{question2}"'
        elif task_name == "social_chemistry":
            text, label = example
            full_text = f'"{text}"'
        else:
            text, label = example
            full_text = f'"{text}"'

        if not full_text.strip():
            continue

        label = label.lower()
        config = task_configs[task_name]
        choices = random.choice(config["choices"])

        prompt = []
        prompt.extend(cot_prompt)

        if task_name == "logical_deduction":
            prompt.append(
                {
                    "role": "user",
                    "content": (
                        f"Q: {full_text}\n\n"
                        f"Answer choices:\n(A) {choices[0]}\n(B) {choices[1]}\n\n"
                        f"{example_instruction}"
                    ),
                }
            )
        else:
            # For sports_understanding, full_text already contains the question
            if task_name == "sports_understanding":
                question_text = full_text  # Already contains "Is the following sentence plausible? ..."
            else:
                question_text = f"{config['question']} {full_text}"
                
            prompt.append(
                {
                    "role": "user",
                    "content": (
                        f"Q: {question_text}\n\n"
                        f"Answer choices:\n(A) {choices[0]}\n(B) {choices[1]}\n\n"
                        f"{example_instruction}"
                    ),
                }
            )

        prompt.append(
            {
                "role": "assistant",
                "content": "A: Let's think step by step:" if thinking else "A:",
            }
        )

        if label in choices[0].lower():
            correct_letter = "A"
        elif label in choices[1].lower():
            correct_letter = "B"
        else:
            continue

        dataset.append(
            {
                "prompt": prompt,
                "correct_letter": correct_letter,
                "correct_answer": label,
            }
        )

    return dataset


def load_cot_prompt(task_name: str) -> Dict:
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to get to the project root, then to data
    cot_filename = os.path.join(script_dir, "..", "data", task_name, f"{task_name}_cot.json")
    with open(cot_filename, "r") as f:
        cot_data = json.load(f)
    
    # Fix chat format issues for proper alternation
    fixed_cot = []
    
    # Process all messages, converting 'model' to 'assistant' and adding proper Q:/A: prefixes
    for i, message in enumerate(cot_data):
        new_message = message.copy()
        if new_message["role"] == "model":
            new_message["role"] = "assistant"
        
        content = new_message["content"].strip()
        
        # Add Q: prefix to user messages that are questions (not the first instruction)
        if new_message["role"] == "user":
            if i == 0:
                # First message is instruction, keep as-is
                new_message["content"] = content
            else:
                # Subsequent user messages are questions, add Q: if not already there
                if not content.startswith("Q: "):
                    new_message["content"] = "Q: " + content
                else:
                    new_message["content"] = content
        elif new_message["role"] == "assistant":
            # Add A: prefix to assistant messages if not already there
            if not content.startswith("A: "):
                new_message["content"] = "A: " + content
            else:
                new_message["content"] = content
            
        fixed_cot.append(new_message)
    
    return fixed_cot


def load_all_datasets(sample_size=1000):
    task_datasets = {}
    # Supported tasks based on available format functions
    task_names = [
        "sports_understanding",
        "anachronisms",
        "social_chemistry",
        "logical_deduction",
        "snarks",
        "quora_question_pairs",
    ]
    for task_name in task_names:
        examples = create_dataset(task_name)
        if len(examples) > sample_size:
            examples = random.sample(examples, sample_size)
        cot_dataset = create_cot_dataset(task_name, examples)
        task_datasets[task_name] = cot_dataset
    return task_datasets


def list_available_datasets() -> List[str]:
    """
    Lists the names of available datasets by scanning the ../data directory.
    A dataset is considered available if its corresponding JSON file exists.
    """
    base_path = os.path.join(os.path.dirname(__file__), "..", "data")
    dataset_names = []
    for name in os.listdir(base_path):
        dataset_path = os.path.join(base_path, name)
        if os.path.isdir(dataset_path):
            json_file = os.path.join(dataset_path, f"{name}.json")
            if os.path.exists(json_file):
                dataset_names.append(name)
    return dataset_names
