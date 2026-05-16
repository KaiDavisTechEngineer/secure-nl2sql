#!/usr/bin/env python3
"""
src/data/preprocess.py

Formats the Spider + WikiSQL data into the exact prompt structure
the model will see during training AND during inference (real use).

The prompt template is critical — if training and inference
use different formats, the model performs poorly. Must be consistent.

Run: python src/data/preprocess.py
"""

import json
import random
import yaml
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def load_config():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


def format_prompt(question: str, schema: str = "", query: str = None) -> dict:
    """
    Format a NL question + schema into the model's expected input format.

    The schema tells the model what tables and columns exist in the database.
    Without it, the model is guessing column names — with it, it can be precise.

    Example:
        question: "How many employees are in the sales department?"
        schema:   "employees(id, name, department, salary)"
        query:    "SELECT COUNT(*) FROM employees WHERE department = 'sales'"

    During TRAINING: we include question + schema + correct SQL
    During INFERENCE: we include question + schema, model generates SQL
    """
    system = (
        "You are a SQL expert. Convert the natural language question to a valid, "
        "safe SQL query based on the provided database schema. "
        "Return ONLY the SQL query with no explanation."
    )

    if schema:
        user_content = f"Schema: {schema}\n\nQuestion: {question}"
    else:
        user_content = f"Question: {question}"

    # SQLCoder uses this specific instruction format
    input_text = (
        f"### Task\n{system}\n\n"
        f"### Input\n{user_content}\n\n"
        f"### Response\n"
    )

    result = {"input": input_text, "question": question, "schema": schema}

    if query:
        result["output"] = query
        result["text"] = input_text + query    # Full text for training

    return result


def is_valid(question: str, query: str) -> bool:
    """Filter out bad training examples."""
    if not question or not query:
        return False
    if len(question) < 10 or len(query) < 10:
        return False
    if len(question) > 500:
        return False
    # Must be a real SQL query
    sql_keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "WITH"]
    if not any(kw in query.upper() for kw in sql_keywords):
        return False
    return True


def preprocess(config: dict):
    raw_dir = Path(config["data"]["raw_data_dir"])
    dataset_dir = Path(config["data"]["dataset_dir"])
    processed_dir = Path(config["data"]["processed_data_dir"])
    dataset_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    all_examples = []
    skipped = 0

    for source_file in ["spider.jsonl", "wikisql.jsonl"]:
        path = raw_dir / source_file
        if not path.exists():
            print(f"⚠️  {source_file} not found — run download_dataset.py first")
            continue

        print(f"🔄 Processing {source_file}...")
        with open(path) as f:
            for line in tqdm(f):
                raw = json.loads(line)
                q = raw.get("question", "").strip()
                sql = raw.get("query", "").strip()
                db = raw.get("db_id", "")

                if not is_valid(q, sql):
                    skipped += 1
                    continue

                # Use db_id as a simple schema hint
                # In a production system you'd load real schema definitions
                schema_hint = f"{db}(...)" if db else ""
                formatted = format_prompt(q, schema_hint, sql)
                formatted["source"] = raw.get("source", source_file)
                all_examples.append(formatted)

    print(f"\n   Kept: {len(all_examples):,} | Skipped: {skipped:,}")

    # Shuffle with fixed seed for reproducibility
    random.seed(42)
    random.shuffle(all_examples)

    # Split
    total = len(all_examples)
    train_end = int(total * config["data"]["train_split"])
    val_end = train_end + int(total * config["data"]["val_split"])

    splits = {
        "train": all_examples[:train_end],
        "validation": all_examples[train_end:val_end],
        "test": all_examples[val_end:],
    }

    print("\n📊 Splits:")
    for name, data in splits.items():
        path = dataset_dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for ex in data:
                f.write(json.dumps(ex) + "\n")
        print(f"   {name:12s}: {len(data):,} → {path}")

    # Save reference data for drift monitoring
    ref_df = pd.DataFrame([
        {
            "input_length": len(ex["input"]),
            "output_length": len(ex.get("output", "")),
            "source": ex.get("source", "unknown"),
        }
        for ex in splits["train"][:1000]
    ])
    ref_path = processed_dir / "reference.csv"
    ref_df.to_csv(ref_path, index=False)
    print(f"   reference   : 1,000 → {ref_path}")

    print("\n✅ Preprocessing complete!")
    print("   Next: python src/training/finetune.py (on Google Colab)\n")


def main():
    config = load_config()
    preprocess(config)


if __name__ == "__main__":
    main()