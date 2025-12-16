# COMS4776-FamilyFeud

This repository contains code for training and evaluating multilingual BERT models on the MMedBench medical question-answering dataset.

## Overview

The project consists of three main components:
1. **Training script** (`train_bert.py`) - Fine-tunes BERT models on MMedBench training data
2. **Evaluation script** (`eval_mmedbench_test_v2.py`) - Evaluates fine-tuned models on test data
3. **Baseline evaluation notebook** (`mmedbench_model_baseline_eval.ipynb`) - Evaluates pre-trained baseline models without fine-tuning

## Files

### `train_bert.py`

A Python script for fine-tuning multilingual BERT models on the MMedBench dataset.

#### Features
- Fine-tunes BERT models (default: `google-bert/bert-base-multilingual-cased`) for multiple-choice question answering
- Supports customizable training hyperparameters (learning rate, batch size, epochs, etc.)
- Includes per-language accuracy tracking during evaluation
- Saves trained models and tokenizers for later use

#### Preprocessing Steps
1. **Padding to fixed number of options**: All examples are padded to exactly 5 options using `[DUMMY]` tokens
   - Examples with more than 5 options are dropped
   - Examples with fewer than 5 options are padded with `[DUMMY]` tokens
2. **Tokenization**: 
   - Each question is paired with each option choice
   - Tokenized with truncation and max length padding (default: 256 tokens)
   - Input format: `[CLS] question [SEP] option [SEP]`
3. **Label processing**: Uses the `label` field from the dataset (expected to be integer indices 0-4)

#### Usage
```bash
python train_bert.py \
    --model_name google-bert/bert-base-multilingual-cased \
    --dataset_path ./mmedench_8k2k \
    --run_name my_experiment \
    --max_length 256 \
    --learning_rate 2e-5 \
    --num_epochs 10 \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --grad_acc_steps 8
```

#### Key Arguments
- `--model_name`: HuggingFace model identifier (default: `google-bert/bert-base-multilingual-cased`)
- `--dataset_path`: Path to the preprocessed dataset directory
- `--run_name`: Name for this training run (used for output directories)
- `--max_length`: Maximum sequence length for tokenization
- `--learning_rate`: Learning rate for training
- `--num_epochs`: Number of training epochs
- `--train_batch_size`: Training batch size per device
- `--eval_batch_size`: Evaluation batch size per device
- `--grad_acc_steps`: Gradient accumulation steps

#### Output
- Trained model saved to: `./trained_models/{run_name}/`
- Training logs saved to: `./logs/{run_name}/`
- Checkpoints saved to: `./outputs/{run_name}/`

---

### `eval_mmedbench_test_v2.py`

A Python script for evaluating fine-tuned models on the MMedBench test set.

#### Features
- Evaluates fine-tuned models on test data across multiple languages
- Computes overall and per-language accuracy and macro F1 scores
- Handles various data formats and label encodings
- Provides detailed statistics on dropped examples

#### Preprocessing Steps
1. **Option normalization**: 
   - Handles both list and dictionary formats for options
   - If dictionary keys are A/B/C/D/E, sorts them alphabetically
   - Otherwise uses insertion order
2. **Filtering**:
   - Optionally drops examples with more than `max_choices` options (default: None, but can be set to 5)
   - Drops malformed examples (missing question, options, or label)
3. **Padding**:
   - Pads options to `pad_to_choices` (default: 5) using `dummy_choice` token (default: `[DUMMY]`)
   - Ensures all examples have the same number of choices for batching
4. **Label parsing**:
   - Supports integer labels (0-4)
   - Supports letter labels ("A"-"E")
   - Supports string labels that can be converted to integers
   - Handles list labels by taking the first element

#### Usage
```bash
python eval_mmedbench_test_v2.py \
    --model_path ./trained_models/my_experiment \
    --test_dir ./test_data \
    --out results.json \
    --batch_size 16 \
    --max_length 256 \
    --max_choices 5 \
    --pad_to_choices 5 \
    --dummy_choice "[DUMMY]"
```

#### Key Arguments
- `--model_path`: Path to the fine-tuned model directory
- `--test_dir`: Directory containing test JSONL files (Chinese.jsonl, English.jsonl, etc.)
- `--out`: Output path for results JSON file
- `--batch_size`: Batch size for evaluation
- `--max_length`: Maximum sequence length for tokenization
- `--max_choices`: Maximum number of options allowed (examples with more are dropped)
- `--pad_to_choices`: Number of options to pad to (default: 5)
- `--dummy_choice`: Text used for padding missing options (default: `[DUMMY]`)

#### Output
The script generates a JSON file with:
- Overall accuracy and macro F1 score
- Per-language accuracy and macro F1 scores
- Statistics on dropped examples (total, too many options, malformed)
- Model and preprocessing configuration

#### Supported Languages
- Chinese
- English
- French
- Japanese
- Russian
- Spanish

---

### `mmedbench_model_baseline_eval.ipynb`

A Jupyter notebook for evaluating baseline (zero-shot) performance of pre-trained BERT models on MMedBench.

#### Purpose
- Evaluates pre-trained models **without fine-tuning** to establish baseline performance
- Provides a benchmark for comparing against fine-tuned models
- Useful for understanding the initial capabilities of different model architectures

#### Models Evaluated
The notebook evaluates four baseline models:
1. **bert-base-cased** - Standard English BERT (cased)
2. **bert-base-multilingual-cased** - Multilingual BERT supporting 104 languages
3. **biobert** - BERT trained on biomedical literature
4. **clinicalbert** - BERT trained on clinical notes

#### Workflow
1. **Setup**: Install dependencies and configure paths
2. **Download Dataset**: Download MMedBench from HuggingFace
3. **Load Dataset**: Load the dataset into memory
4. **Define Models**: Specify which BERT models to evaluate
5. **Define Helper Classes**: Dataset class and model loading functions
6. **Run Baselines**: Evaluate all models on the test set
7. **Display Results**: View and save evaluation results
8. **Visualize Results** (Optional): Create comparison plots

#### Features
- Automatic model downloading from HuggingFace
- Per-language performance metrics
- Overall accuracy and F1 scores
- Visualization of results across models and languages
- Results saved to JSON for later analysis

#### Usage
1. Open the notebook in Jupyter or Google Colab
2. Run all cells sequentially
3. Results are saved to `./results/baseline_results.json`
4. Optional visualization cells create comparison plots

---

## Dataset Format

The MMedBench dataset should be in JSONL format with the following structure:

```json
{
  "question": "What is the primary symptom of...?",
  "options": {
    "A": "Option A text",
    "B": "Option B text",
    "C": "Option C text",
    "D": "Option D text"
  },
  "answer_idx": "A",
  "language": "English"
}
```

Or with options as a list:
```json
{
  "question": "What is the primary symptom of...?",
  "options": ["Option A", "Option B", "Option C", "Option D"],
  "answer_idx": 0,
  "language": "English"
}
```

## Dependencies

- `torch`
- `transformers`
- `datasets`
- `scikit-learn`
- `tqdm`
- `pandas` (for notebook)
- `huggingface_hub` (for notebook)

## Notes

- The training script expects a preprocessed dataset saved using HuggingFace `datasets` library
- Both training and evaluation scripts handle padding to 5 options for consistency
- The evaluation script is more flexible in handling various data formats compared to the training script
- Baseline evaluation requires no training and can be run immediately after installing dependencies
