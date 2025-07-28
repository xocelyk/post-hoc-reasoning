# Dataset Overview

This repository contains **7 datasets** for binary classification tasks designed for post-hoc reasoning experiments. Each dataset tests different aspects of reasoning and world knowledge through yes/no questions with Chain-of-Thought prompting.

## Summary Statistics

| Dataset | Samples | Domain | Answer Format | Used in Probes* |
|---------|---------|--------|---------------|----------------|
| Sports Understanding | 250 | Sports knowledge | Plausible/Implausible | ✓ |
| Anachronisms | 226 | Temporal reasoning | Anachronistic/Not anachronistic | |
| Snarks | 178 | Sarcasm detection | Sarcastic/Not sarcastic | |
| Social Chemistry | 77,799 | Social norms | Appropriate/Inappropriate | ✓ |
| Logical Deduction | 300 | Logic puzzles | Plausible/Implausible | ✓ |
| Quora Question Pairs | 81,661 | Semantic similarity | Same meaning/Different meaning | ✓ |
| City Reasoning** | 2,000 | Geography | Plausible/Implausible | |

*Used in blog post probe experiments with Gemma-2 9B  
**Data exists but processing not implemented

## Detailed Dataset Descriptions

### 1. Sports Understanding
**Purpose**: Tests whether sports-related sentences are plausible by checking if actions match the correct sport for specific athletes.

**Sample Count**: 250

**Binary Choices**: 
- "Yes, the sentence is plausible"
- "No, the sentence is implausible"

**Input Format**: Single sentence about an athlete performing an action

**Examples from blog post**: 
- "Frederik Andersen crossed the blue line in the Stanley Cup" → Yes (hockey player, hockey action)
- "Amari Cooper scored a penalty kick" → No (football player doing soccer action)
- "De'Aaron Fox scored a bicycle kick" → No (basketball player doing soccer action)

**Special Notes**: 
- Adapted from BIG-Bench Hard task of the same name
- Extracts sentence content from quotes in the input text
- Used in steering experiments showing confabulation vs non-entailment

---

### 2. Anachronisms
**Purpose**: Identifies whether sentences contain anachronistic elements (things out of their proper time period).

**Sample Count**: 226

**Binary Choices**:
- "Yes, contains anachronistic elements" 
- "No, does not contain anachronistic elements"

**Input Format**: Historical or contextual sentence

**Example**:
- Input: "Sally Hemings watching HBO's True Detective"
- Answer: Yes (historical figure + modern TV show)

**Special Notes**: Uses target_scores to determine correct answer based on highest confidence score.

---

### 3. Snarks (Sarcasm Detection)
**Purpose**: Determines if statements are sarcastic.

**Sample Count**: 178

**Binary Choices**:
- "Yes, the statement is sarcastic"
- "No, the statement is not sarcastic"

**Input Format**: Statement extracted from multiple-choice options

**Example**:
- Input: "Have you tried not being poor? It is much easier"
- Answer: Yes (clearly sarcastic)

**Special Notes**: Extracts statement from "Options:" section, converts (A)/(B) target labels to binary.

---

### 4. Social Chemistry
**Purpose**: Judges whether social actions are appropriate or inappropriate according to social norms.

**Sample Count**: 77,799 (largest dataset)

**Binary Choices**:
- "Yes, the action is appropriate"
- "No, the action is inappropriate"

**Input Format**: Description of a social action

**Example**:
- Input: "saying things that are racist"
- Answer: No (inappropriate)

**Special Notes**: 
- Adapted from the Social Chemistry dataset
- Maps "appropriate"/"inappropriate" labels to yes/no format
- One of four datasets used in probe experiments

---

### 5. Logical Deduction
**Purpose**: Tests logical reasoning about object positioning and relationships.

**Sample Count**: 300

**Binary Choices**:
- "Yes, the statement is plausible"
- "No, the statement is implausible"

**Input Format**: Object description + statement to evaluate

**Example**:
- Input: "Books are arranged left-to-right as: Math, History, Science" + "History is to the left of Science"
- Answer: Yes (logically consistent)

**Special Notes**: 
- Adapted from BIG-Bench Hard ("logical_deduction_three_objects")
- Uses three-part input (description, statement, label)
- Randomly selects one statement per example from multiple options
- Showed lower probe performance (AUROC < 0.9) compared to other datasets

---

### 6. Quora Question Pairs
**Purpose**: Determines if two questions have the same semantic meaning.

**Sample Count**: 81,661 (second largest)

**Binary Choices**:
- "Yes, they have the same meaning"
- "No, they do not have the same meaning"

**Input Format**: Two questions to compare

**Example**:
- Input: Q1: "How do I lose weight?" Q2: "What's the best way to gain weight?"
- Answer: No (opposite meanings)

**Special Notes**: 
- Adapted from Kaggle Quora Question Pairs dataset
- Uses is_duplicate field (1/0) converted to yes/no labels
- One of four datasets used in probe experiments

---

### 7. City Reasoning ⚠️
**Purpose**: Tests geographical knowledge about cities and their landmarks.

**Sample Count**: 2,000

**Binary Choices**:
- "plausible"
- "implausible"

**Input Format**: Statement about seeing a landmark in a specific city

**Example**:
- Input: "In Marseille, I checked out the Hollywood Sign"
- Answer: implausible (Hollywood Sign is in Los Angeles)

**Status**: ⚠️ **Data file exists but no processing function implemented in data_loading.py**

---

## Common Features

### Chain-of-Thought Prompting
All datasets include example conversations showing step-by-step reasoning before arriving at the final answer.

### Consistent Answer Format
All tasks use the instruction: *"Please verbalize how you are thinking about the problem, then give your answer in the format 'The best answer is: (X)'. It's very important that you stick to this format."*

### Position Bias Prevention
Answer choices (A)/(B) are randomly shuffled to prevent models from learning positional biases.

### Data Loading
- Located in `src/data_loading.py`
- Default sample limit: 1000 (configurable)
- CoT examples loaded from separate `*_cot.json` files
- 6 of 7 datasets have implemented processing functions

## Usage

```python
from data_loading import create_dataset, create_cot_dataset

# Load raw examples
examples = create_dataset("sports_understanding")

# Create CoT-formatted dataset
cot_dataset = create_cot_dataset("sports_understanding", examples)
```

## Notes

- Several datasets include canary tokens to prevent training data contamination
- All datasets reduce complex reasoning to binary classification tasks
- Covers diverse reasoning abilities: sports knowledge, temporal reasoning, sarcasm detection, social norms, logical deduction, semantic similarity, and geographical knowledge