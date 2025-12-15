#!/usr/bin/env python3
import argparse
import json
import os
import tempfile
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMultipleChoice
from sklearn.metrics import f1_score


LANG_FILES = [
    ("Chinese", "Chinese.jsonl"),
    ("English", "English.jsonl"),
    ("French", "French.jsonl"),
    ("Japanese", "Japanese.jsonl"),
    ("Russian", "Russian.jsonl"),
    ("Spanish", "Spanish.jsonl"),
]

LETTER_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def normalize_options(opts):
    """
    Normalize options into a list.
    Supports list or dict.
    - If dict keys look like A/B/C..., return in A,B,C,... order.
    - Else return insertion-order values.
    """
    if isinstance(opts, list):
        return opts

    if isinstance(opts, dict):
        keys = list(opts.keys())
        if all(isinstance(k, str) and len(k) == 1 and k.upper() in LETTER_TO_IDX for k in keys):
            keys = sorted(keys, key=lambda x: LETTER_TO_IDX[x.upper()])
            return [opts[k] for k in keys]
        return list(opts.values())

    return None


def parse_label(label):
    """
    Accepts:
      - int (0..)
      - str "A".."E"
      - list like ["A"] or [0]
    Returns int label or None.
    """
    if label is None:
        return None

    # If list, take first element
    if isinstance(label, list):
        if len(label) == 0:
            return None
        label = label[0]

    # If int already
    if isinstance(label, int):
        return label

    # Some datasets store numeric labels as strings
    if isinstance(label, str):
        s = label.strip()
        if s.isdigit():
            return int(s)
        s = s.upper()
        if s in LETTER_TO_IDX:
            return LETTER_TO_IDX[s]

    return None


def preprocess_example(example, max_choices, pad_to_choices, dummy_choice):
    """
    Label-agnostic preprocessing (doesn't use answer content):
      - normalize options
      - drop if too many choices
      - pad up to fixed number with dummy choices
      - parse/validate label
    Returns (question, options_list, label_int) or None to drop.
    """
    q = example.get("question") or example.get("query") or example.get("prompt")
    opts_raw = example.get("options") or example.get("choices") or example.get("answers")

    # MMedBench commonly uses answer_idx (often A/B/C...)
    lbl_raw = example.get("answer_idx")
    # fallback keys (in case)
    if lbl_raw is None:
        lbl_raw = example.get("label") or example.get("answer")

    if q is None or opts_raw is None or lbl_raw is None:
        return None

    options = normalize_options(opts_raw)
    if options is None or not isinstance(options, list) or len(options) == 0:
        return None

    q = str(q).strip()
    options = [("" if o is None else str(o)).strip() for o in options]

    # Drop too-many-choice items if requested
    if max_choices is not None and len(options) > max_choices:
        return None

    # Pad to fixed size for batching
    if pad_to_choices is not None and len(options) < pad_to_choices:
        options = options + [dummy_choice] * (pad_to_choices - len(options))

    label_int = parse_label(lbl_raw)
    if label_int is None:
        return None

    # Validate label in range of (possibly padded) options
    if label_int < 0 or label_int >= len(options):
        return None

    return q, options, label_int


def atomic_write_json(obj, out_path):
    """
    Atomically write compact JSON to avoid partial output.
    """
    out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    os.makedirs(out_dir, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(out_path), suffix=".tmp", dir=out_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
        os.replace(tmp_path, out_path)
    except Exception:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise


def compute_accuracy(preds, labels):
    correct = sum(int(p == y) for p, y in zip(preds, labels))
    return correct / max(1, len(labels))


def evaluate(model, tokenizer, examples, device, batch_size, max_length):
    """
    examples: list of dicts {question: str, options: list[str], label: int, language: str}
    """
    preds, labels, langs = [], [], []

    model.eval()
    for i in tqdm(range(0, len(examples), batch_size), desc="Evaluating"):
        batch = examples[i : i + batch_size]

        # Each example has fixed number of choices already (padded)
        C = len(batch[0]["options"])

        flat_q = []
        flat_o = []
        for ex in batch:
            q = ex["question"]
            opts = ex["options"]
            if len(opts) != C:
                # safety: enforce same number of choices inside batch
                if len(opts) < C:
                    opts = opts + ["[DUMMY]"] * (C - len(opts))
                else:
                    opts = opts[:C]
            flat_q.extend([q] * C)
            flat_o.extend(opts)

        enc = tokenizer(
            flat_q,
            flat_o,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )

        # reshape to (B, C, L)
        for k in enc:
            enc[k] = enc[k].view(len(batch), C, -1).to(device)

        with torch.no_grad():
            out = model(**enc)
            batch_pred = torch.argmax(out.logits, dim=1).detach().cpu().tolist()

        preds.extend(batch_pred)
        labels.extend([ex["label"] for ex in batch])
        langs.extend([ex["language"] for ex in batch])

    overall_acc = compute_accuracy(preds, labels)
    overall_f1 = f1_score(labels, preds, average="macro") if len(set(labels)) > 1 else 0.0

    # per-language
    by_lang = defaultdict(lambda: {"p": [], "y": []})
    for p, y, l in zip(preds, labels, langs):
        by_lang[l]["p"].append(p)
        by_lang[l]["y"].append(y)

    per_lang = {}
    for l, d in sorted(by_lang.items()):
        acc = compute_accuracy(d["p"], d["y"])
        mf1 = f1_score(d["y"], d["p"], average="macro") if len(set(d["y"])) > 1 else 0.0
        per_lang[l] = {"accuracy": acc, "macro_f1": mf1, "n": len(d["y"])}

    return overall_acc, overall_f1, per_lang


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--out", type=str, required=True)

    # Preprocessing knobs
    parser.add_argument("--max_choices", type=int, default=None,
                        help="Drop examples with more than this many options (e.g., 5).")
    parser.add_argument("--pad_to_choices", type=int, default=5,
                        help="Pad options up to this number with dummy choices (e.g., 5).")
    parser.add_argument("--dummy_choice", type=str, default="[DUMMY]",
                        help="Text used to pad missing answer options.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForMultipleChoice.from_pretrained(args.model_path).to(device)

    all_examples = []
    dropped_total = 0
    dropped_too_many = 0
    dropped_malformed = 0
    loaded_total = 0

    for lang, fname in LANG_FILES:
        fpath = os.path.join(args.test_dir, fname)
        if not os.path.exists(fpath):
            print(f"[WARN] Missing file for {lang}: {fpath} (skipping)")
            continue

        rows = load_jsonl(fpath)
        loaded_total += len(rows)

        for ex in rows:
            out = preprocess_example(
                ex,
                max_choices=args.max_choices,
                pad_to_choices=args.pad_to_choices,
                dummy_choice=args.dummy_choice,
            )
            if out is None:
                dropped_total += 1
                opts_raw = ex.get("options") or ex.get("choices") or ex.get("answers")
                opts_norm = normalize_options(opts_raw) if opts_raw is not None else None
                if isinstance(opts_norm, list) and args.max_choices is not None and len(opts_norm) > args.max_choices:
                    dropped_too_many += 1
                else:
                    dropped_malformed += 1
                continue

            q, options, label = out
            all_examples.append(
                {"language": lang, "question": q, "options": options, "label": label}
            )

    print(f"Loaded {loaded_total} test rows; using {len(all_examples)} after preprocessing.")
    print(f"Dropped: total={dropped_total}, too_many={dropped_too_many}, malformed={dropped_malformed}")

    if len(all_examples) == 0:
        raise RuntimeError(
            "No usable examples after preprocessing. "
            "This usually means field names differ or labels aren't parsed correctly. "
            "Try printing a sample JSONL line to confirm keys."
        )

    overall_acc, overall_f1, per_lang = evaluate(
        model=model,
        tokenizer=tokenizer,
        examples=all_examples,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    results = {
        "model_path": args.model_path,
        "test_dir": args.test_dir,
        "device": str(device),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "preprocessing": {
            "max_choices": args.max_choices,
            "pad_to_choices": args.pad_to_choices,
            "dummy_choice": args.dummy_choice,
        },
        "overall": {"accuracy": overall_acc, "macro_f1": overall_f1, "n": len(all_examples)},
        "per_language": per_lang,
        "dropped": {"total": dropped_total, "too_many": dropped_too_many, "malformed": dropped_malformed},
    }

    print(json.dumps(results, ensure_ascii=False))
    atomic_write_json(results, args.out)
    print(f"\nSaved results -> {args.out}")


if __name__ == "__main__":
    main()