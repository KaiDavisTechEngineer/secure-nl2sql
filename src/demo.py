#!/usr/bin/env python3
"""
src/demo.py

Standalone demo of the full SecureQuery pipeline.

Runs without needing a server, GPU, or any heavy dependencies.
Shows the full NL → SQL → Security Report pipeline working end-to-end.

The NL-to-SQL part uses the HuggingFace Inference API (free).
If no HF token is set, falls back to demo SQL queries to show the scanner.

Run: python3 src/demo.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.security.scanner import SQLSecurityScanner

load_dotenv()


# ── ANSI colors for terminal output ──────────────────────────────────────────

class Color:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'


def color(text: str, c: str) -> str:
    return f"{c}{text}{Color.END}"


def severity_color(severity: str) -> str:
    return {
        "CRITICAL": Color.RED,
        "HIGH": Color.YELLOW,
        "MEDIUM": Color.CYAN,
        "LOW": Color.BLUE,
        "SAFE": Color.GREEN,
    }.get(severity, Color.END)


# ── Demo Queries ─────────────────────────────────────────────────────────────

DEMO_SCENARIOS = [
    {
        "question": "Show all customers",
        "schema": "customers(id, name, email)",
        "expected_sql": "SELECT id, name, email FROM customers LIMIT 100;",
    },
    {
        "question": "Find users with admin privileges OR all users (injection attempt)",
        "schema": "users(id, username, password, role)",
        "expected_sql": "SELECT * FROM users WHERE role = 'admin' OR 1=1;",
    },
    {
        "question": "Get user credentials for support",
        "schema": "accounts(id, username, password, ssn, email)",
        "expected_sql": "SELECT username, password, ssn FROM accounts;",
    },
    {
        "question": "Delete the old sessions table",
        "schema": "sessions(id, user_id, token, created_at)",
        "expected_sql": "DROP TABLE sessions;",
    },
    {
        "question": "Find a product by name",
        "schema": "products(id, name, price)",
        "expected_sql": "SELECT name FROM products UNION SELECT password FROM users;",
    },
]


# ── Pipeline ─────────────────────────────────────────────────────────────────

def generate_sql_via_api(question: str, schema: str) -> str:
    """Try to generate SQL via HuggingFace. Falls back to None if unavailable."""
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        return None

    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(model="defog/sqlcoder-7b-2", token=token)
        prompt = f"""### Task
Generate a SQL query to answer: `{question}`

### Database Schema
{schema}

### Answer
```sql
"""
        result = client.text_generation(prompt, max_new_tokens=200, temperature=0.1, do_sample=False)
        # Extract SQL
        sql = result.strip().split("```")[0].split("\n\n")[0].strip()
        return sql
    except Exception as e:
        print(color(f"   ⚠️  API call failed: {e}", Color.YELLOW))
        return None


def run_pipeline(scenario: dict, scanner: SQLSecurityScanner, use_api: bool):
    """Run one complete query through the full pipeline."""
    print(color(f"\n{'='*70}", Color.BOLD))
    print(color(f"❓ Question: {scenario['question']}", Color.BOLD))
    print(color(f"📋 Schema:   {scenario['schema']}", Color.CYAN))

    # Step 1: Generate SQL
    sql = None
    if use_api:
        print(color("🤖 Generating SQL via HuggingFace API...", Color.BLUE))
        sql = generate_sql_via_api(scenario['question'], scenario['schema'])

    if not sql:
        sql = scenario['expected_sql']
        if use_api:
            print(color("   (Using fallback demo SQL)", Color.YELLOW))

    print(color(f"📝 Generated SQL:", Color.BOLD))
    print(f"   {sql}")

    # Step 2: Security scan
    print(color("🔐 Running security scan...", Color.BLUE))
    report = scanner.scan(sql)

    # Display results
    score_color = severity_color(report.risk_label)
    print(color(f"\n📊 Risk Score: {report.risk_score}/10", Color.BOLD))
    print(color(f"🏷️  Risk Label: {report.risk_label}", score_color))
    print(f"✅ Safe to execute: {report.safe_to_execute}")

    if report.findings:
        print(color(f"\n🚨 Findings ({len(report.findings)}):", Color.BOLD))
        for i, f in enumerate(report.findings, 1):
            c = severity_color(f.severity.value)
            print(color(f"   [{i}] [{f.severity.value}] {f.title}", c))
            print(f"       └─ {f.description[:80]}...")
            if f.owasp_ref:
                print(color(f"       └─ {f.owasp_ref}", Color.CYAN))

    if report.recommended_query:
        print(color(f"\n💡 Safer alternative:", Color.GREEN))
        print(f"   {report.recommended_query}")


def main():
    print(color("\n" + "█" * 70, Color.HEADER))
    print(color("█" + " " * 22 + "SECUREQUERY DEMO PIPELINE" + " " * 21 + "█", Color.HEADER))
    print(color("█" * 70, Color.HEADER))

    # Check for API token
    has_token = bool(os.getenv("HUGGINGFACE_TOKEN"))
    if has_token:
        print(color("\n✅ HuggingFace token detected — using live API for SQL generation", Color.GREEN))
        use_api = True
    else:
        print(color("\n⚠️  No HUGGINGFACE_TOKEN in .env — running in demo mode with sample SQL", Color.YELLOW))
        print(color("   To enable live SQL generation, add HUGGINGFACE_TOKEN to .env", Color.YELLOW))
        use_api = False

    # Initialize scanner
    scanner = SQLSecurityScanner()
    print(color("✅ Security scanner loaded", Color.GREEN))

    # Run scenarios
    for scenario in DEMO_SCENARIOS:
        run_pipeline(scenario, scanner, use_api)

    print(color(f"\n{'='*70}", Color.BOLD))
    print(color("✅ Demo complete!", Color.GREEN))
    print(color(f"{'='*70}\n", Color.BOLD))


if __name__ == "__main__":
    main()