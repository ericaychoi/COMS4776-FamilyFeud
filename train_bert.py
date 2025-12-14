#!/usr/bin/env python3
"""
Train BERT on Medical Data
"""

import torch
import random
import numpy as np
import os
import sys
import argparse
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForMultipleChoice,
    TrainingArguments,
    Trainer,
    DataCollatorForMultipleChoice,
)
from sklearn.metrics import accuracy_score, classification_report
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


# Parse command line arguments
parser = argparse.ArgumentParser(description='Finetune Multilingual BERT on Medical Data')
parser.add_argument('--model_name', type=str, default="google-bert/bert-base-multilingual-cased",
                    help='Model name or path')
parser.add_argument('--dataset_path', type=str, default="./mmedench_8k2k",
                    help='Path to dataset')
parser.add_argument('--run_name', type=str, default="test1",
                    help='Run name')
parser.add_argument('--max_length', type=int, default=256,
                    help='Maximum sequence length')

# Training hyperparameters
parser.add_argument('--learning_rate', type=float, default=2e-5,
                    help='Learning rate')
parser.add_argument('--num_epochs', type=int, default=10,
                    help='Number of training epochs')
parser.add_argument('--train_batch_size', type=int, default=4,
                    help='Training batch size per device')
parser.add_argument('--eval_batch_size', type=int, default=4,
                    help='Evaluation batch size per device')
parser.add_argument('--grad_acc_steps', type=int, default=8,
                    help='Gradient accumulation steps')
parser.add_argument('--weight_decay', type=float, default=0.1,
                    help='Weight decay')
parser.add_argument('--warmup_ratio', type=float, default=0.1,
                    help='Warmup ratio')
parser.add_argument('--warmup_steps', type=int, default=500,
                    help='Warmup steps')

# Training strategy
parser.add_argument('--eval_strategy', type=str, default="steps",
                    choices=["no", "steps", "epoch"],
                    help='Evaluation strategy')
parser.add_argument('--eval_steps', type=int, default=250,
                    help='Evaluation steps')
parser.add_argument('--save_strategy', type=str, default="steps",
                    choices=["no", "steps", "epoch"],
                    help='Save strategy')
parser.add_argument('--save_steps', type=int, default=500,
                    help='Save steps')
parser.add_argument('--logging_steps', type=int, default=50,
                    help='Logging steps')
parser.add_argument('--fp16', action='store_true', default=None,
                    help='Enable FP16 (default: auto-detect based on CUDA availability)')

args = parser.parse_args()

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set FP16 default if not specified
if args.fp16 is None:
    args.fp16 = torch.cuda.is_available()

# Load dataset from disk
print(f"\nLoading dataset from {args.dataset_path}...")
dataset = load_from_disk(args.dataset_path)
print(f"Dataset loaded: {dataset}")

# Pad all examples to 5 options
print("\nPadding all examples to 5 options...")
def pad_to_max_options(examples, max_options=5):
    padded = []
    for ex in examples:
        if len(ex['options']) > max_options:
            continue

        # Pad options
        if len(ex['options']) < max_options:
            padded_options = ex['options'] + ['[DUMMY]'] * (max_options - len(ex['options']))
        else:
            padded_options = ex['options']

        padded_ex = ex.copy()
        padded_ex['options'] = padded_options
        padded.append(padded_ex)

    return padded

from datasets import Dataset, DatasetDict
train_padded = pad_to_max_options(list(dataset['train']), max_options=5)
test_padded = pad_to_max_options(list(dataset['test']), max_options=5)

# Update dataset
dataset = DatasetDict({
    'train': Dataset.from_list(train_padded),
    'test': Dataset.from_list(test_padded)
})

# Load tokenizer
print(f"\nLoading tokenizer: {args.model_name}")
tokenizer = AutoTokenizer.from_pretrained(args.model_name)

def preprocess_multiple_choice(examples):

    first_sentences = []
    second_sentences = []

    # Process each example individually to handle varying option counts
    for question, options in zip(examples['question'], examples['options']):
        num_choices = len(options)
        # Add question repeated for each option
        first_sentences.extend([question] * num_choices)
        # Add all options
        second_sentences.extend(options)

    # Tokenize all pairs
    tokenized = tokenizer(
        first_sentences,
        second_sentences,
        truncation=True,
        max_length=args.max_length,
        padding='max_length'
    )

    batch_size = len(examples['question'])
    reshaped = {k: [] for k in tokenized.keys()}

    idx = 0
    for options in examples['options']:
        num_choices = len(options)
        for key in tokenized.keys():
            reshaped[key].append(tokenized[key][idx:idx+num_choices])
        idx += num_choices

    # Use the label field created during data loading
    reshaped['labels'] = examples['label']
    
    # Preserve language field if it exists
    if 'language' in examples:
        reshaped['language'] = examples['language']

    return reshaped

# Apply preprocessing
print("\nApplying preprocessing to dataset...")
tokenized_datasets = dataset.map(
    preprocess_multiple_choice,
    batched=True,
    remove_columns=[col for col in dataset['train'].column_names if col != 'language']
)
print(f"Tokenized dataset: {tokenized_datasets}")

# Initialize model
print(f"\nInitializing model: {args.model_name}")
model = AutoModelForMultipleChoice.from_pretrained(args.model_name)
model.to(device)
print(f"Model parameters: {model.num_parameters():,}")

# Simple compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    return {'accuracy': accuracy}

# Custom Trainer that computes per-language metrics during evaluation
class LanguageAwareTrainer(Trainer):
    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, 
                       ignore_keys=None, metric_key_prefix="eval"):

        output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )
        
        # Compute per-language metrics
        if hasattr(self.eval_dataset, 'column_names') and 'language' in self.eval_dataset.column_names:
            predictions = np.argmax(output.predictions, axis=-1)
            labels = output.label_ids
            languages = self.eval_dataset['language']
            
            # Group by language
            lang_predictions = defaultdict(list)
            lang_labels = defaultdict(list)
            
            for i, lang in enumerate(languages):
                if i < len(predictions):
                    lang_predictions[lang].append(predictions[i])
                    lang_labels[lang].append(labels[i])
            
            # Add per-language metrics to output
            for lang in sorted(lang_predictions.keys()):
                acc = accuracy_score(lang_labels[lang], lang_predictions[lang])
                output.metrics[f'{metric_key_prefix}_accuracy_{lang}'] = acc
            
            # Print formatted output
            print("\n" + "="*80, flush=True)
            print("Per-Language Accuracy:", flush=True)
            print("="*80, flush=True)
            for lang in sorted(lang_predictions.keys()):
                acc = output.metrics[f'{metric_key_prefix}_accuracy_{lang}']
                print(f"  {lang}: {acc:.4f} (n={len(lang_predictions[lang])})", flush=True)
            print("="*80, flush=True)
        
        return output

# Setup data collator
data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)

# Define training arguments
training_args = TrainingArguments(
    output_dir=f"./outputs/{args.run_name}",
    eval_strategy=args.eval_strategy,
    eval_steps=args.eval_steps,
    save_strategy=args.save_strategy,
    save_steps=args.save_steps,
    learning_rate=args.learning_rate,
    gradient_accumulation_steps=args.grad_acc_steps,
    per_device_train_batch_size=args.train_batch_size,
    per_device_eval_batch_size=args.eval_batch_size,
    num_train_epochs=args.num_epochs,
    weight_decay=args.weight_decay,
    warmup_ratio=args.warmup_ratio,
    warmup_steps=args.warmup_steps,
    logging_dir=f"./logs/{args.run_name}",
    logging_steps=args.logging_steps,
    report_to="tensorboard",
    fp16=args.fp16,
)

# Get eval dataset
eval_dataset = tokenized_datasets['test']

# Initialize LanguageAwareTrainer
print("\nInitializing Trainer...")
trainer = LanguageAwareTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
# Train
print("Starting training...")
train_result = trainer.train(resume_from_checkpoint=True)

save_path = f"./trained_models/{args.run_name}"
# Save model
print(f"\nSaving model to {save_path}...")
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)