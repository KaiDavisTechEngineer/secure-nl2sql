# 🔐 SecureQuery — NL-to-SQL with AI Security Analysis

> Type a question in plain English. Get SQL back. Get an automatic security report.

An MLOps-style project that converts natural language to SQL **and** scans every generated query for security vulnerabilities — SQL injection vectors, over-permissive access, data exposure risks, and more.

**This is the intersection of ML Engineering and Cybersecurity.** Almost nobody is building this combination.

---

## Live Demo

```bash
python3 src/demo.py
```

Sample output:
```
❓ Question: Find users with admin privileges (injection attempt)
📝 Generated SQL: SELECT * FROM users WHERE role = 'admin' OR 1=1;

📊 Risk Score: 9/10
🏷️  Risk Label: CRITICAL
✅ Safe to execute: False

🚨 Findings (2):
   [1] [CRITICAL] Tautology-based SQL Injection
       └─ OWASP A03:2021 - Injection
   [2] [LOW] SELECT * Usage

💡 Safer alternative:
   SELECT * FROM users WHERE role = 'admin' LIMIT 100;
```

---

## Architecture

```
Natural Language Question
         ↓
  [HuggingFace API]              ← Hosted SQLCoder-7B (no local GPU needed)
         ↓
  [Security Scanner]              ← Custom rule-based engine (our key contribution)
  ├── SQL Injection Detection
  ├── PII / Credential Exposure
  ├── Destructive Operations
  └── Query Risk Scoring
         ↓
  [FastAPI Response]              ← JSON: query + security report
```

---

## Tech Stack

| Component | Tool |
|-----------|------|
| NL-to-SQL Model | `defog/sqlcoder-7b-2` via HuggingFace Inference API |
| Security Scanner | Custom Python — regex + heuristic rules |
| API Layer | FastAPI + Pydantic |
| Testing | pytest (19 tests) |
| CI/CD | GitHub Actions |

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/KaiDavisTechEngineer/secure-nl2sql
cd secure-nl2sql

# 2. Install (lightweight — no GPU/ML libraries)
pip install -r requirements.txt

# 3. Run the security scanner demo (no token needed)
python3 src/security/scanner.py

# 4. Run the full pipeline demo
python3 src/demo.py

# 5. Or start the API
uvicorn src.api.main:app --reload
# Visit http://localhost:8000/docs

# 6. Test it
curl -X POST http://localhost:8000/scan \
  -H "Content-Type: application/json" \
  -d '{"sql": "SELECT password FROM users WHERE 1=1"}'
```

To enable live NL-to-SQL generation, add a free HuggingFace token to `.env`:
```
HUGGINGFACE_TOKEN=hf_your_token_here
```
Get a free token at: https://huggingface.co/settings/tokens

---

## Security Checks Implemented

All findings reference OWASP categories:

- ✅ **SQL Injection patterns** (OWASP A03:2021)
  - Tautology attacks (`OR 1=1`, `AND 1=1`)
  - UNION-based injection
  - Comment injection (`--`, `/* */`)
  - Stacked queries (`;` abuse)
  - Time-based blind injection (SLEEP, WAITFOR)
- ✅ **Sensitive Data Exposure** (OWASP A02:2021)
  - Password / token / API key columns
  - PII columns (SSN, credit cards, DOB)
- ✅ **Destructive Operations** (OWASP A04:2021)
  - DROP / TRUNCATE detection
  - DELETE without WHERE
- ✅ **Access Control** (OWASP A01:2021)
  - GRANT / REVOKE privilege escalation
- ✅ **Query Quality**
  - Unbounded SELECT *
  - Missing LIMIT clauses

---

## Test Results

```
$ python3 -m pytest tests/ -v
===================== test session starts =====================
collected 19 items

tests/test_security_scanner.py::TestSafeQueries::test_simple_select PASSED
... (17 more tests) ...
tests/test_security_scanner.py::TestEdgeCases::test_report_to_dict PASSED

=================== 19 passed in 0.15s ===================
```

---

## Why This Project Matters

Most NL-to-SQL projects stop at "make SQL from English." This one goes further:

1. **Production-realistic** — uses hosted models via API like real companies do
2. **Security-first** — every query is automatically audited before execution
3. **OWASP-aligned** — findings reference the standard cybersecurity framework
4. **Fully tested** — 19 unit tests, all passing in CI

The security scanner is the key differentiator. It works as a standalone tool to audit any SQL query, even ones humans wrote.