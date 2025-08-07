import pickle
import os
from collections import Counter

# Find a logical deduction generations file
cache_dir = 'cache/experiments'
for root, dirs, files in os.walk(cache_dir):
    if 'logical_deduction' in root and 'train_generations.pkl' in files:
        path = os.path.join(root, 'train_generations.pkl')
        print(f'Loading: {path}')
        with open(path, 'rb') as f:
            data = pickle.load(f)
            if data:
                # Show first few examples
                print(f'\nFirst 5 examples:')
                for i, item in enumerate(data[:5]):
                    correct = item.get('correct_answer', 'N/A')
                    pred = item.get('pred_answer', 'N/A')
                    print(f"  Example {i}: correct_answer='{correct}', pred_answer='{pred}'")
                
                # Get all unique correct answers
                correct_answers = set(item.get('correct_answer', '') for item in data)
                print(f'\nUnique correct answers: {sorted(correct_answers)}')
                
                # Count distribution
                answer_counts = Counter(item.get('correct_answer', '') for item in data)
                print(f'\nAnswer distribution: {dict(answer_counts)}')
                
                # Check if this is a yes/no dataset
                if correct_answers == {'yes', 'no'}:
                    print("\nThis IS a yes/no dataset")
                else:
                    print(f"\nThis is NOT a yes/no dataset. It has answers: {correct_answers}")
                    print("This could explain low AUC scores if the probe training expects yes/no labels!")
        break
else:
    print("No logical deduction data found")