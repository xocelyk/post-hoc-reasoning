#!/usr/bin/env python3
"""
Generate one test prompt from each dataset and save to text files.
"""

import os
import sys

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from data_loading import load_all_datasets


def format_chat_for_text(chat_messages):
    """Convert chat format to readable text format."""
    formatted_parts = []
    
    for message in chat_messages:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            formatted_parts.append(f"User: {content}")
        elif role == "assistant":
            formatted_parts.append(f"Assistant: {content}")
    
    return "\n\n".join(formatted_parts)


def main():
    """Generate test prompts for each dataset."""
    # Create output directory
    output_dir = "test_prompts"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading datasets...")
    datasets = load_all_datasets()
    
    for dataset_name, dataset in datasets.items():
        if not dataset:
            print(f"Skipping empty dataset: {dataset_name}")
            continue
            
        # Get first example
        example = dataset[0]
        
        # Convert chat format to text
        prompt_text = format_chat_for_text(example["prompt"])
        
        # Add metadata
        full_text = f"Dataset: {dataset_name}\n"
        full_text += f"Correct Answer: {example['correct_answer']}\n"
        full_text += f"Correct Letter: {example['correct_letter']}\n"
        full_text += f"Total Examples in Dataset: {len(dataset)}\n"
        full_text += "\n" + "="*50 + "\n\n"
        full_text += prompt_text
        
        # Write to file
        filename = os.path.join(output_dir, f"{dataset_name}.txt")
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(full_text)
        
        print(f"Generated prompt for {dataset_name} -> {filename}")
    
    print(f"\nAll prompts generated in '{output_dir}/' directory")


if __name__ == "__main__":
    main()