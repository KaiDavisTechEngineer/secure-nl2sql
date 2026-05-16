#!/usr/bin/env python3
"""
src/data/download_dataset.py

Downloads the Spider and WikiSQL datasets.

Spider  = Yale University's NL-to-SQL benchmark
         - 10,181 questions across 200 different databases
         - Industry standard for evaluating NL-to-SQL models
         - Used in academic papers and company benchmarks

WikiSQL = Salesforce's simpler NL-to-SQL dataset
         - 80,654 question/SQL pairs
         - Single-table queries, great for initial training

We combine both for a richer training set.

Run: python src/data/download_dataset.py
"""

import json
import yaml
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


def load_config():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


def download_spider(raw_dir: Path, max_samples: int) -> Path:
    """
    Download Spider dataset.

    Each example has:
    - question: "What are the names of all employees?"
    - query:    "SELECT name FROM employees"
    - db_id:    "company"  (which database schema)
    - schema:   the table/column definitions
    """
    out_path = raw_dir / "spider.jsonl"
    if out_path.exists():
        print("✅ Spider already downloaded")
        return out_path

    print("📥 Downloading Spider dataset...")
    dataset = load_dataset("spider", split="train", trust_remote_code=True)

    saved = 0
    with open(out_path, "w") as f:
        for ex in tqdm(dataset):
            if saved >= max_samples // 2:
                break
            record = {
                "question": ex["question"],
                "query": ex["query"],
                "db_id": ex["db_id"],
                "source": "spider",
            }
            f.write(json.dumps(record) + "\n")
            saved += 1

    print(f"✅ Spider: saved {saved:,} examples")
    return out_path


def download_wikisql(raw_dir: Path, max_samples: int) -> Path:
    """
    Download WikiSQL dataset.

    Simpler than Spider (single tables) but much larger.
    Great for teaching the model basic SQL patterns.
    """
    out_path = raw_dir / "wikisql.jsonl"
    if out_path.exists():
        print("✅ WikiSQL already downloaded")
        return out_path

    print("📥 Downloading WikiSQL dataset...")
    dataset = load_dataset("wikisql", split="train", trust_remote_code=True)

    saved = 0
    with open(out_path, "w") as f:
        for ex in tqdm(dataset):
            if saved >= max_samples // 2:
                break

            # WikiSQL stores SQL in a structured format — convert to string
            sql = ex.get("sql", {})
            query_string = reconstruct_sql(sql, ex.get("table", {}).get("name", "table"))

            if not query_string:
                continue

            record = {
                "question": ex["question"],
                "query": query_string,
                "db_id": ex.get("table", {}).get("id", "unknown"),
                "source": "wikisql",
            }
            f.write(json.dumps(record) + "\n")
            saved += 1

    print(f"✅ WikiSQL: saved {saved:,} examples")
    return out_path


def reconstruct_sql(sql: dict, table_name: str) -> str:
    """
    WikiSQL stores SQL as structured JSON — convert it to a real SQL string.

    Example input:
    {"sel": 0, "agg": 0, "conds": [[2, 0, "New York"]]}
    means: SELECT col_0 FROM table WHERE col_2 = 'New York'
    """
    if not sql or not isinstance(sql, dict):
        return ""

    agg_ops = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
    cond_ops = ["=", ">", "<", ">=", "<=", "!="]

    try:
        sel_col = f"col_{sql.get('sel', 0)}"
        agg = sql.get("agg", 0)

        if agg > 0:
            select_clause = f"SELECT {agg_ops[agg]}({sel_col}) FROM {table_name}"
        else:
            select_clause = f"SELECT {sel_col} FROM {table_name}"

        conds = sql.get("conds", [])
        if conds:
            where_parts = []
            for col_idx, op_idx, val in conds:
                op = cond_ops[op_idx] if op_idx < len(cond_ops) else "="
                val_str = f"'{val}'" if isinstance(val, str) else str(val)
                where_parts.append(f"col_{col_idx} {op} {val_str}")
            return f"{select_clause} WHERE {' AND '.join(where_parts)}"

        return select_clause
    except Exception:
        return ""


def preview(path: Path, n: int = 3):
    print(f"\n{'='*55}\nSAMPLE DATA ({path.name})\n{'='*55}")
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            ex = json.loads(line)
            print(f"\n[{i+1}] Question: {ex['question']}")
            print(f"     SQL:      {ex['query']}")


def main():
    config = load_config()
    raw_dir = Path(config["data"]["raw_data_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    max_samples = config["data"]["max_samples"]

    spider_path = download_spider(raw_dir, max_samples)
    wikisql_path = download_wikisql(raw_dir, max_samples)

    preview(spider_path)
    preview(wikisql_path)

    print("\n✅ Download complete!")
    print("   Next: python src/data/preprocess.py\n")


if __name__ == "__main__":
    main()