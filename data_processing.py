#!/usr/bin/env python3
"""
Load and process MMedBench data to a Huggingface dataset
"""

import json
import os
import random
from collections import defaultdict
from datasets import Dataset, DatasetDict


def load_jsonl_files(data_dir, languages):
    """Load and process per-language JSONL files for all languages from the data directory"""
    examples = []

    for lang in languages:
        file_path = os.path.join(data_dir, f"{lang}.jsonl")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                
                question = data.get('question', '').strip()
                if not question:
                    continue
                
                options = data.get('options')
                if isinstance(options, dict):
                    option_list = [options[k] for k in sorted(options.keys())]
                elif isinstance(options, list):
                    option_list = options
                else:
                    continue
                
                answer_idx = data.get('answer_idx', '')
                if isinstance(answer_idx, list):
                    answer_idx = answer_idx[0] if answer_idx else ''
                
                label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
                label = label_map.get(answer_idx, 0)
                
                example = {
                    'question': question,
                    'options': option_list,
                    'answer': data.get('answer', ''),
                    'answer_idx': answer_idx,
                    'label': label,
                    'language': lang
                }
                examples.append(example)

    return examples


def create_balanced_dataset(examples, target_total=10000, min_per_language=800, train_ratio=0.8):
    """Create a balanced dataset with proportional sampling across languages."""
    filtered_examples = [ex for ex in examples if ex.get('answer_idx', '').strip()]
    
    by_language = defaultdict(list)
    for ex in filtered_examples:
        by_language[ex['language']].append(ex)
    
    num_languages = len(by_language)
    remaining_after_mins = target_total - (num_languages * min_per_language)
    total_above_min = sum(max(0, len(exs) - min_per_language) for exs in by_language.values())
    
    sampled_examples = []
    random.seed(42)
    
    for lang in sorted(by_language.keys()):
        available = by_language[lang]
        
        if len(available) <= min_per_language:
            target = len(available)
        else:
            above_min = len(available) - min_per_language
            proportional_share = int((above_min / total_above_min) * remaining_after_mins)
            target = min_per_language + proportional_share
        
        sampled = random.sample(available, min(target, len(available)))
        sampled_examples.extend(sampled)
    
    random.shuffle(sampled_examples)
    split_idx = int(train_ratio * len(sampled_examples))
    
    train_examples = sampled_examples[:split_idx]
    test_examples = sampled_examples[split_idx:]
    
    return train_examples, test_examples


def main():
    data_dir = "MMedBench"
    train_dir = os.path.join(data_dir, "Train")
    languages = ["Chinese", "English", "French", "Japanese", "Russian", "Spanish"]
    output_path = "HF_data_balanced"
    
    print("Loading training data...")
    train_examples = load_jsonl_files(train_dir, languages)
    print(f"Total examples loaded: {len(train_examples)}")
    
    print("Creating balanced dataset...")
    train_split, val_split = create_balanced_dataset(train_examples)
    
    print(f"Train: {len(train_split)}")
    print(f"Validation: {len(val_split)}")
    
    dataset = DatasetDict({
        'train': Dataset.from_list(train_split),
        'test': Dataset.from_list(val_split)
    })
    
    dataset.save_to_disk(output_path)
    print(f"Dataset saved to {output_path}")


if __name__ == "__main__":
    main()

