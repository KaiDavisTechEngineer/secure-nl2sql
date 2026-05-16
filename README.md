# 🔐 SecureQuery — NL-to-SQL with AI Security Analysis

> Type a question in plain English. Get a SQL query back. Get a security report automatically.

An end-to-end MLOps project that converts natural language to SQL **and** scans the generated query for security vulnerabilities — SQL injection vectors, over-permissive access, data exposure risks, and more.

**This is the intersection of ML Engineering and Cybersecurity.** Almost nobody is building this combination.

---

## What It Does

```
User: "Show me all users and their passwords from the accounts table"

SecureQuery:
  ✅ Generated SQL:  SELECT username, password FROM accounts;
  🚨 Security Score: 2/10 — CRITICAL
  
  Findings:
  [CRITICAL] Sensitive column exposure — 'password' should never be queried directly
  [HIGH]     No WHERE clause — returns ALL rows (potential data dump)
  [HIGH]     No LIMIT clause — unbounded result set
  
  Recommended Safe Query:
  SELECT username FROM accounts WHERE user_id = :user_id LIMIT 1;
```

---

## Architecture

```
Natural Language Input
        ↓
  [Fine-tuned LLM]          ← Fine-tuned on Spider/WikiSQL dataset
  (NL → SQL Generation)
        ↓
  [Security Scanner]         ← Rule-based + ML classifier
  ├── SQL Injection Detection
  ├── Privilege Analysis
  ├── Sensitive Column Detection
  ├── OWASP A03 Checks
  └── Query Risk Scoring
        ↓
  [SQL Validator]            ← Sandboxed execution check
        ↓
  [FastAPI Response]         ← JSON: query + security report
        ↓
  [MLflow Tracking]          ← Every query logged for monitoring
```

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Base Model | `defog/sqlcoder-7b-2` (SQL-specialized) |
| Fine-tuning | HuggingFace + LoRA |
| Security Layer | Rule-based (regex) + ML classifier |
| Experiment Tracking | MLflow |
| Pipeline | Apache Airflow |
| API | FastAPI + Pydantic |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Monitoring | Evidently AI |

---

## Quickstart

```bash
# 1. Clone and setup
git clone https://github.com/YOUR_USERNAME/secure-nl2sql
cd secure-nl2sql
python scripts/setup.py

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download and process data
python src/data/download_dataset.py
python src/data/preprocess.py

# 4. Fine-tune (run on Google Colab)
python src/training/finetune.py

# 5. Start the API
uvicorn src.api.main:app --reload

# 6. Try it
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Show me all customers who spent over $1000", "schema": "customers(id, name, email, total_spent)"}'
```

---

## Results

| Metric | Value |
|--------|-------|
| NL-to-SQL Accuracy (Spider dev set) | TBD |
| Security Scanner Precision | TBD |
| Security Scanner Recall | TBD |
| Avg API Latency | TBD |
| Vulnerabilities caught per 100 queries | TBD |

*Update after running `src/evaluation/benchmark.py`*

---

## Security Checks Implemented

- ✅ SQL Injection pattern detection (OWASP A03:2021)
- ✅ Tautology attacks (`1=1`, `OR 1=1`)  
- ✅ UNION-based injection
- ✅ Sensitive column exposure (passwords, SSNs, tokens)
- ✅ Destructive operation detection (DROP, TRUNCATE, DELETE without WHERE)
- ✅ Unbounded queries (missing LIMIT)
- ✅ Comment-based injection (`--`, `/* */`)
- ✅ Stacked query detection (`;` abuse)
- ✅ Privilege escalation patterns
- ✅ Time-based blind injection keywords