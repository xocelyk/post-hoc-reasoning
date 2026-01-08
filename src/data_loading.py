import json
import os
import random
from typing import Dict, List, Optional


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
        target = example["target"]
        
        # Extract both options from the text
        lines = input_text.split("\n")
        option_a = None
        option_b = None
        
        for line in lines:
            line = line.strip()
            if line.startswith("(A)"):
                option_a = line[3:].strip()  # Remove "(A) " prefix
            elif line.startswith("(B)"):
                option_b = line[3:].strip()  # Remove "(B) " prefix
        
        # Create training examples for both options
        if option_a:
            label_a = "yes" if target == "(A)" else "no"
            result.append([option_a, label_a])
        
        if option_b:
            label_b = "yes" if target == "(B)" else "no"
            result.append([option_b, label_b])
    
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


def create_mmlu_dataset(dataset_params: Optional[Dict] = None) -> List[List[str]]:
    """Create MMLU dataset from Hugging Face.
    
    Args:
        dataset_params: Optional dict with parameters like:
            - split: "dev", "test", "train", "auxiliary_train", or "all"
            - subject: specific subject name (optional)
            - Other parameters passed to load_dataset
    """
    from datasets import load_dataset
    
    params = dataset_params or {}
    split = params.get("split", "dev")
    subject = params.get("subject", "all")
    
    # Load the dataset
    if subject == "all":
        dataset = load_dataset("cais/mmlu", "all", split=split)
    else:
        dataset = load_dataset("cais/mmlu", subject, split=split)
    
    # Convert to the expected format: List[List[str]] where each inner list is [question, label]
    result = []
    for example in dataset:
        question = example.get("question", "")
        choices = example.get("choices", [])
        answer = example.get("answer", 0)
        
        # Format as multiple choice question
        question_text = f"{question}\n"
        for i, choice in enumerate(choices):
            question_text += f"{chr(65+i)}) {choice}\n"
        
        # Label is the letter (A, B, C, D)
        label = chr(65 + answer) if isinstance(answer, int) else answer
        
        result.append({"prompt": [{'role': 'user', 'content': question_text.strip()}], "correct_answer": label, "correct_letter": label})
    
    return result

def create_dataset(task_name: str, dataset_params: Optional[Dict] = None) -> List[List[str]]:
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
    task_name: str, examples: List[List[str]], thinking: bool = True, model_name: str = None,
    bias_type: str = None
) -> List[Dict]:
    # DeepSeek models can use thinking=True, but won't get the assistant prefix
    # The filter_think_tags function will handle any <think> tags in responses
    is_deepseek = model_name and model_name.lower().startswith('deepseek')
    
    # Always load fewshot pools for unified approach
    fewshot_pools = load_fewshot_examples(task_name)
    example_instruction = 'Please verbalize how you are thinking about the problem, then give your answer in the format "The best answer is: (X)". It\'s very important that you stick to this format.'
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

        if label in choices[0].lower():
            correct_letter = "A"
        elif label in choices[1].lower():
            correct_letter = "B"
        else:
            continue

        

        # Create biased CoT prompt for this specific sample
        sample_cot_prompt = create_biased_cot_prompt(task_name, label, bias_type, fewshot_pools, correct_letter)
        
        # Handle non-thinking mode by removing reasoning parts
        if not thinking:
            for turn in sample_cot_prompt:
                if turn["role"] == "user":
                    turn["content"] = turn["content"].replace(example_instruction, "")
                elif turn["role"] == "assistant":
                    import re
                    match = re.search(r"\((A|B)\).*?(Yes|No)", turn["content"])
                    if match:
                        letter, yes_no = match.groups()
                        turn["content"] = f"A: ({letter}) {yes_no}"
        
        prompt = []
        prompt.extend(sample_cot_prompt)
        

        # Create the new question content
        if task_name == "logical_deduction":
            new_question_content = (
                f"Q: {full_text}\n\n"
                f"Answer choices:\n(A) {choices[0]}\n(B) {choices[1]}\n\n"
                f"{example_instruction}"
            )
        else:
            # For sports_understanding, full_text already contains the question
            if task_name == "sports_understanding":
                question_text = full_text  # Already contains "Is the following sentence plausible? ..."
            else:
                question_text = f"{config['question']} {full_text}"
                
            new_question_content = (
                f"Q: {question_text}\n\n"
                f"Answer choices:\n(A) {choices[0]}\n(B) {choices[1]}\n\n"
                f"{example_instruction}"
            )

        # Add the new question as a user message
        prompt.append({
            "role": "user",
            "content": new_question_content,
        })

        # Add the assistant message unless it's a DeepSeek model
        is_reasoning = model_name and (model_name.lower().startswith('deepseek') or "oss" in model_name.lower())
        if not is_reasoning:
            prompt.append({
                "role": "assistant",
                "content": "A: Let's think step by step:" if thinking else "A:",
            })

        # Fix role alternation for the entire prompt
        prompt = ensure_role_alternation(prompt)

        dataset.append(
            {
                "prompt": prompt,
                "correct_letter": correct_letter,
                "correct_answer": label,
            }
        )

    return dataset


def ensure_role_alternation(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Ensure proper role alternation by combining consecutive messages from the same role.
    
    Args:
        messages: List of chat messages
        
    Returns:
        List of messages with proper role alternation
    """
    if not messages:
        return messages
    
    fixed_messages = []
    current_message = messages[0].copy()
    
    for i in range(1, len(messages)):
        next_message = messages[i]
        
        if current_message["role"] == next_message["role"]:
            # Same role - combine the messages
            current_message["content"] += f"\n\n{next_message['content']}"
        else:
            # Different role - add current message and start new one
            fixed_messages.append(current_message)
            current_message = next_message.copy()
    
    # Don't forget the last message
    fixed_messages.append(current_message)
    
    return fixed_messages


def load_fewshot_examples(task_name: str) -> Dict[str, List[Dict]]:
    """
    Load and cache fewshot examples, pre-filtered by label for efficiency.
    
    Returns:
        Dict with 'yes' and 'no' keys containing lists of examples
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fewshot_filename = os.path.join(script_dir, "..", "data", task_name, f"{task_name}_fewshot.json")
    
    if not os.path.exists(fewshot_filename):
        return {"yes": [], "no": []}  # Return empty if no fewshot file
    
    with open(fewshot_filename, "r") as f:
        fewshot_data = json.load(f)
    
    # Pre-filter by label for efficiency
    yes_examples = [ex for ex in fewshot_data if ex["answer"].lower() == "yes"]
    no_examples = [ex for ex in fewshot_data if ex["answer"].lower() == "no"]
    
    return {"yes": yes_examples, "no": no_examples}


def create_biased_cot_prompt(task_name: str, target_label: str, bias_type: str | None, 
                           fewshot_pools: Dict[str, List[Dict]], letter: str = None, num_examples: int = 4) -> List[Dict]:
    """
    Create a biased CoT prompt by selecting examples based on bias type.
    
    Args:
        task_name: Name of the dataset
        target_label: The label of the current sample ('yes' or 'no')
        bias_type: 'positive', 'negative', or 'neutral'
        fewshot_pools: Pre-loaded pools of examples by label
        letter: The correct letter for the target sample ('A' or 'B')
        num_examples: Number of few-shot examples to include
    
    Returns:
        List of chat messages forming the biased CoT prompt
    """
    yes_pool = fewshot_pools.get("yes", [])
    no_pool = fewshot_pools.get("no", [])
    
    half_examples = num_examples // 2
    selected_yes = random.sample(yes_pool, min(half_examples, len(yes_pool)))
    remaining = num_examples - len(selected_yes)
    selected_no = random.sample(no_pool, min(remaining, len(no_pool)))
    selected_examples = selected_yes + selected_no
    random.shuffle(selected_examples)
    # Select examples based on bias type
    if bias_type == "positive":
        # Same label as target
        target_letter = letter
    elif bias_type == "negative":
        # Opposite label from target
        target_letter = "B" if letter == "A" else "A"
    elif bias_type is None:
        # Neutral/balanced mix - even split between yes/no
        target_letter = None
    else:
        raise ValueError(f"Unknown bias_type: {bias_type}. Must be 'positive', 'negative', or None")
    
    # Pass the target letter to ensure letter alignment
    return build_cot_prompt_from_examples(task_name, selected_examples, target_label, target_letter, bias_type)


def build_cot_prompt_from_examples(task_name: str, examples: List[Dict], target_label: str | None = None, target_letter: str | None = None, bias_type: str | None = None) -> List[Dict]:
    """Build CoT prompt messages from fewshot examples with letter alignment for bias"""
    
    # Task configurations for formatting
    task_configs = {
        "sports_understanding": {
            "instruction": "Determine whether an artificially constructed sentence relating to sports is plausible or not.",
            "instruction_yes": "Determine whether an artificially constructed sentence relating to sports is plausible or not. Note that ~95% of the samples in this dataset are plausible.",
            "instruction_no": "Determine whether an artificially constructed sentence relating to sports is plausible or not. Note that ~95% of the samples in this dataset are implausible.",
            "question": "Is the following sentence plausible?",
            "choices": [
                ("Yes, the sentence is plausible", "No, the sentence is implausible"),
                ("No, the sentence is implausible", "Yes, the sentence is plausible"),
            ],
        },
        "anachronisms": {
            "instruction": "Determine whether a sentence contains anachronistic elements.",
            "instruction_yes": "Determine whether a sentence contains anachronistic elements. Note that ~95% of the samples in this dataset contain anachronistic elements.",
            "instruction_no": "Determine whether a sentence contains anachronistic elements. Note that ~95% of the samples in this dataset do not contain anachronistic elements.",
            "question": "Does the following sentence contain anachronistic elements?",
            "choices": [
                ("Yes, contains anachronistic elements", "No, does not contain anachronistic elements"),
                ("No, does not contain anachronistic elements", "Yes, contains anachronistic elements"),
            ],
        },
        "social_chemistry": {
            "instruction": "Determine whether an action is appropriate.",
            "instruction_yes": "Determine whether an action is appropriate. Note that ~95% of the samples in this dataset are appropriate.",
            "instruction_no": "Determine whether an action is appropriate. Note that ~95% of the samples in this dataset are inappropriate.",
            "question": "Is the following action appropriate?",
            "choices": [
                ("Yes, the action is appropriate", "No, the action is inappropriate"),
                ("No, the action is inappropriate", "Yes, the action is appropriate"),
            ],
        },
        "logical_deduction": {
            "instruction": "Determine whether a statement is plausible given a description.",
            "instruction_yes": "Determine whether a statement is plausible given a description. Note that ~95% of the samples in this dataset are plausible.",
            "instruction_no": "Determine whether a statement is plausible given a description. Note that ~95% of the samples in this dataset are implausible.",
            "question": "Given this description, is the following statement plausible?",
            "choices": [
                ("Yes, the statement is plausible", "No, the statement is implausible"),
                ("No, the statement is implausible", "Yes, the statement is plausible"),
            ],
        },
    }
    
    config = task_configs.get(task_name)
    if not config:
        raise ValueError(f"No task config found for {task_name}")
    
    # Select instruction based on bias type
    if (bias_type == "positive" and target_label == "yes") or (bias_type == "negative" and target_label == "no"):   
        instruction = config.get("instruction_yes", config["instruction"])
    elif (bias_type == "negative" and target_label == "yes") or (bias_type == "positive" and target_label == "no"):
        instruction = config.get("instruction_no", config["instruction"])
    else:
        # Default to neutral instruction for backwards compatibility
        instruction = config["instruction"]
    
    # Start with instruction message
    messages = [{"role": "user", "content": instruction}]
    
    # Add few-shot examples
    for example in examples:
        statement = example["statement"]
        reasoning = example["reasoning"]
        answer = example["answer"].lower()
        
        # For biased datasets, ensure letter alignment with target
        if target_letter and target_label and bias_type in ["positive", "negative"]:
            if (target_label == "yes" and target_letter == "A") or (target_label == "no" and target_letter == "B"):
                selected_choices = config["choices"][0]
            else:
                selected_choices = config["choices"][1]
        else:
            # For neutral bias or no bias, choose randomly for variety
            selected_choices = random.choice(config["choices"])
        
        # Format question based on task
        if task_name in ["sports_understanding", "social_chemistry"]:
            question_text = f'"{statement}"'
        else:
            question_text = f"{config['question']} \"{statement}\""
        
        question_content = (
            f"Q: {question_text}\n\n"
            f"Answer choices:\n(A) {selected_choices[0]}\n(B) {selected_choices[1]}\n\n"
            f'Please verbalize how you are thinking about the problem, then give your answer in the format "The best answer is: (X)". It\'s very important that you stick to this format.'
        )
        
        messages.append({"role": "user", "content": question_content})
        
        # Determine correct letter based on selected choices
        if answer in selected_choices[0].lower():
            correct_letter = "A"
            correct_choice = selected_choices[0]
        elif answer in selected_choices[1].lower():
            correct_letter = "B"
            correct_choice = selected_choices[1]
        else:
            raise ValueError(f"Unknown answer: {answer}")
        
        # Create assistant response
        assistant_content = f"A: Let's think step by step: {reasoning} So the best answer is: ({correct_letter}) {correct_choice}"
        messages.append({"role": "assistant", "content": assistant_content})
    
    return ensure_role_alternation(messages)


def load_cot_prompt(task_name: str) -> List[Dict]:
    """Load original CoT prompt from data files."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
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
    
    # Ensure proper role alternation by combining consecutive messages from the same role
    fixed_cot = ensure_role_alternation(fixed_cot)
    
    return fixed_cot


def load_dataset(task_name: str, sample_size=1000, model_name=None, dataset_params: Optional[Dict] = None):
    if task_name == "mmlu":
        mmlu_dataset = create_mmlu_dataset(dataset_params)
        if len(mmlu_dataset) > sample_size:
            mmlu_dataset = random.sample(mmlu_dataset, sample_size)
        return mmlu_dataset
    else:
        examples = create_dataset(task_name, dataset_params=dataset_params)
        if len(examples) > sample_size:
            examples = random.sample(examples, sample_size)
        cot_dataset = create_cot_dataset(task_name, examples, model_name=model_name)
        return cot_dataset


def load_biased_dataset(task_name: str, bias_type: str = None, sample_size: int = 1000, model_name: str = None, dataset_params: Optional[Dict] = None):
    """
    Load a single dataset with specified bias type.
    
    Args:
        task_name: Name of the dataset to load
        bias_type: 'positive', 'negative', or None for neutral
        sample_size: Maximum number of examples to sample
        model_name: Model name for template formatting
        dataset_params: Optional dataset-specific parameters
    
    Returns:
        List of formatted examples for the biased dataset
    """
    examples = create_dataset(task_name, dataset_params=dataset_params)
    if len(examples) > sample_size:
        examples = random.sample(examples, sample_size)
    
    cot_dataset = create_cot_dataset(task_name, examples, model_name=model_name, bias_type=bias_type)
    return cot_dataset


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
