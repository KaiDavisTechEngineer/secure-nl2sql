#!/usr/bin/env python3
"""
src/security/scanner.py

The security scanner — the most unique part of this project.

This is what separates SecureQuery from every other NL-to-SQL tool.
After the model generates a SQL query, this scanner analyzes it for
security vulnerabilities before it ever reaches a real database.

Security checks implemented (all OWASP-referenced):
  - SQL Injection patterns (OWASP A03:2021)
  - Tautology attacks (OR 1=1, AND 1=1)
  - UNION-based injection
  - Comment injection (--, /*, #)
  - Stacked queries (semicolon abuse)
  - Sensitive column exposure (passwords, tokens, SSNs)
  - Destructive operations (DROP, TRUNCATE, DELETE without WHERE)
  - Unbounded queries (no LIMIT on large tables)
  - Time-based blind injection keywords (SLEEP, WAITFOR)
  - Privilege escalation (GRANT, REVOKE, ALTER)
  - Data dump patterns (SELECT * without WHERE)
"""

import re
import yaml
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


# ── Data Structures ─────────────────────────────────────────────────────────

class Severity(str, Enum):
    CRITICAL = "CRITICAL"   # Score 8-10 — stop immediately
    HIGH     = "HIGH"       # Score 6-7  — requires review
    MEDIUM   = "MEDIUM"     # Score 4-5  — should be addressed
    LOW      = "LOW"        # Score 2-3  — informational
    INFO     = "INFO"       # Score 0-1  — note only


@dataclass
class SecurityFinding:
    """A single security issue found in a SQL query."""
    severity: Severity
    rule_id: str              # e.g. "SQL_INJECTION_TAUTOLOGY"
    title: str                # e.g. "Tautology-based SQL Injection"
    description: str          # What the vulnerability is
    recommendation: str       # How to fix it
    matched_text: str = ""    # The exact snippet that triggered this rule
    owasp_ref: str = ""       # e.g. "OWASP A03:2021"
    score_impact: int = 0     # How much this raises the risk score


@dataclass
class SecurityReport:
    """Complete security analysis of a SQL query."""
    query: str
    risk_score: int               # 0-10 (10 = most dangerous)
    risk_label: str               # "CRITICAL" / "HIGH" / "MEDIUM" / "LOW" / "SAFE"
    findings: List[SecurityFinding] = field(default_factory=list)
    safe_to_execute: bool = True
    recommended_query: Optional[str] = None
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "risk_score": self.risk_score,
            "risk_label": self.risk_label,
            "safe_to_execute": self.safe_to_execute,
            "summary": self.summary,
            "recommended_query": self.recommended_query,
            "findings": [
                {
                    "severity": f.severity.value,
                    "rule_id": f.rule_id,
                    "title": f.title,
                    "description": f.description,
                    "recommendation": f.recommendation,
                    "matched_text": f.matched_text,
                    "owasp_ref": f.owasp_ref,
                }
                for f in self.findings
            ],
        }


# ── Security Rules ───────────────────────────────────────────────────────────

# Each rule is a dict with:
#   pattern    - regex to search for in the SQL
#   rule_id    - unique identifier
#   severity   - how bad this is
#   title      - short name
#   description - what's wrong
#   recommendation - how to fix it
#   owasp_ref  - relevant OWASP category
#   score      - how much this adds to the risk score

INJECTION_RULES = [
    {
        "pattern": r"(OR|AND)\s+[\'\"]?\d+[\'\"]?\s*=\s*[\'\"]?\d+[\'\"]?",
        "rule_id": "SQL_INJECTION_TAUTOLOGY",
        "severity": Severity.CRITICAL,
        "title": "Tautology-based SQL Injection",
        "description": (
            "Always-true conditions like 'OR 1=1' or 'AND 1=1' are classic SQL injection. "
            "They cause WHERE clauses to be bypassed, returning all rows regardless of filters."
        ),
        "recommendation": "Use parameterized queries. Never construct SQL by concatenating user input.",
        "owasp_ref": "OWASP A03:2021 - Injection",
        "score": 9,
    },
    {
        "pattern": r"UNION\s+(ALL\s+)?SELECT",
        "rule_id": "SQL_INJECTION_UNION",
        "severity": Severity.CRITICAL,
        "title": "UNION-based SQL Injection",
        "description": (
            "UNION SELECT is used to extract data from other tables by appending results "
            "to the original query. A classic technique for dumping entire databases."
        ),
        "recommendation": "Validate that UNION is expected in this context. Use query whitelisting.",
        "owasp_ref": "OWASP A03:2021 - Injection",
        "score": 9,
    },
    {
        "pattern": r"(--|#|\/\*)",
        "rule_id": "SQL_INJECTION_COMMENT",
        "severity": Severity.HIGH,
        "title": "SQL Comment Injection",
        "description": (
            "SQL comment sequences (-- # /* */) can be used to truncate queries, "
            "ignoring security conditions like authentication checks that appear after them."
        ),
        "recommendation": "Strip or escape comment characters from user-supplied input.",
        "owasp_ref": "OWASP A03:2021 - Injection",
        "score": 7,
    },
    {
        "pattern": r";\s*(SELECT|INSERT|UPDATE|DELETE|DROP|EXEC)",
        "rule_id": "SQL_INJECTION_STACKED",
        "severity": Severity.CRITICAL,
        "title": "Stacked Query Injection",
        "description": (
            "Semicolons allow multiple SQL statements in one query. "
            "Attackers use this to execute arbitrary commands after a legitimate query."
        ),
        "recommendation": "Reject queries containing multiple statements. Use stored procedures.",
        "owasp_ref": "OWASP A03:2021 - Injection",
        "score": 10,
    },
    {
        "pattern": r"(SLEEP\s*\(|WAITFOR\s+DELAY|BENCHMARK\s*\(|PG_SLEEP\s*\()",
        "rule_id": "SQL_INJECTION_TIME_BASED",
        "severity": Severity.CRITICAL,
        "title": "Time-based Blind SQL Injection",
        "description": (
            "Time-delay functions (SLEEP, WAITFOR, BENCHMARK) are used in blind injection "
            "to extract data when no output is visible — the attacker infers data from response times."
        ),
        "recommendation": "Blacklist time-delay functions. Use a query firewall.",
        "owasp_ref": "OWASP A03:2021 - Injection",
        "score": 10,
    },
]

SENSITIVE_DATA_RULES = [
    {
        "pattern": r"\b(password|passwd|pwd|secret|api_key|token|auth_token|access_token|private_key)\b",
        "rule_id": "SENSITIVE_COLUMN_EXPOSURE",
        "severity": Severity.CRITICAL,
        "title": "Sensitive Column Exposure",
        "description": (
            "The query selects columns that likely contain credentials or secrets. "
            "Passwords, tokens, and API keys should never be returned in plain text queries."
        ),
        "recommendation": (
            "Never SELECT credential columns directly. "
            "Use password hashing at the application layer, not the database layer. "
            "Restrict access to sensitive columns via database roles."
        ),
        "owasp_ref": "OWASP A02:2021 - Cryptographic Failures",
        "score": 9,
    },
    {
        "pattern": r"\b(ssn|social_security|credit_card|card_number|cvv|dob|date_of_birth)\b",
        "rule_id": "PII_EXPOSURE",
        "severity": Severity.CRITICAL,
        "title": "PII (Personally Identifiable Information) Exposure",
        "description": (
            "The query accesses columns that appear to contain PII — "
            "Social Security Numbers, credit card numbers, or medical data. "
            "Exposure of PII may violate GDPR, HIPAA, or PCI-DSS regulations."
        ),
        "recommendation": (
            "Mask or encrypt PII at rest. Implement column-level access controls. "
            "Log all access to PII columns for auditing."
        ),
        "owasp_ref": "OWASP A02:2021 - Cryptographic Failures",
        "score": 10,
    },
]

DESTRUCTIVE_RULES = [
    {
        "pattern": r"\bDROP\s+(TABLE|DATABASE|SCHEMA|INDEX|VIEW)\b",
        "rule_id": "DESTRUCTIVE_DROP",
        "severity": Severity.CRITICAL,
        "title": "Destructive DROP Operation",
        "description": (
            "DROP statements permanently delete database objects. "
            "This cannot be undone without a backup. Should never be generated by NL-to-SQL."
        ),
        "recommendation": "Reject all DROP statements from NL-to-SQL output. Require manual execution.",
        "owasp_ref": "OWASP A04:2021 - Insecure Design",
        "score": 10,
    },
    {
        "pattern": r"\bTRUNCATE\s+TABLE\b",
        "rule_id": "DESTRUCTIVE_TRUNCATE",
        "severity": Severity.CRITICAL,
        "title": "TRUNCATE Operation",
        "description": "TRUNCATE deletes all rows from a table instantly and cannot be rolled back.",
        "recommendation": "Reject TRUNCATE from NL-to-SQL. Use DELETE with WHERE if row removal is intended.",
        "owasp_ref": "OWASP A04:2021 - Insecure Design",
        "score": 10,
    },
    {
        "pattern": r"\bDELETE\s+FROM\s+\w+\s*(?!WHERE)",
        "rule_id": "DELETE_WITHOUT_WHERE",
        "severity": Severity.HIGH,
        "title": "DELETE Without WHERE Clause",
        "description": (
            "A DELETE without a WHERE clause deletes EVERY row in the table. "
            "This is almost always unintentional from a natural language query."
        ),
        "recommendation": "Always require a WHERE clause on DELETE statements.",
        "owasp_ref": "OWASP A04:2021 - Insecure Design",
        "score": 8,
    },
    {
        "pattern": r"\b(GRANT|REVOKE)\s+",
        "rule_id": "PRIVILEGE_MODIFICATION",
        "severity": Severity.CRITICAL,
        "title": "Privilege Escalation Attempt",
        "description": (
            "GRANT/REVOKE modify database user permissions. "
            "NL-to-SQL should never produce privilege modification statements."
        ),
        "recommendation": "Reject all GRANT/REVOKE from NL-to-SQL output entirely.",
        "owasp_ref": "OWASP A01:2021 - Broken Access Control",
        "score": 10,
    },
]

QUERY_QUALITY_RULES = [
    {
        "pattern": r"SELECT\s+\*\s+FROM\s+\w+\s*$",
        "rule_id": "UNBOUNDED_SELECT_STAR",
        "severity": Severity.MEDIUM,
        "title": "Unbounded SELECT * Without WHERE",
        "description": (
            "SELECT * without a WHERE or LIMIT clause returns every row in the table. "
            "On large tables this can return millions of rows, "
            "causing performance issues and potential data exposure."
        ),
        "recommendation": "Add a WHERE clause or LIMIT to scope the result set.",
        "owasp_ref": "OWASP A04:2021 - Insecure Design",
        "score": 4,
    },
    {
        "pattern": r"SELECT\s+\*",
        "rule_id": "SELECT_STAR",
        "severity": Severity.LOW,
        "title": "SELECT * Usage",
        "description": (
            "SELECT * returns all columns, including ones that may be sensitive. "
            "It also breaks if the schema changes."
        ),
        "recommendation": "Explicitly name only the columns you need.",
        "owasp_ref": "OWASP A04:2021 - Insecure Design",
        "score": 2,
    },
]

ALL_RULES = INJECTION_RULES + SENSITIVE_DATA_RULES + DESTRUCTIVE_RULES + QUERY_QUALITY_RULES


# ── Scanner Class ────────────────────────────────────────────────────────────

class SQLSecurityScanner:
    """
    The main security scanner.

    Usage:
        scanner = SQLSecurityScanner()
        report = scanner.scan("SELECT password FROM users WHERE id = 1 OR 1=1")
        print(report.risk_label)   # "CRITICAL"
        print(report.risk_score)   # 9
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.cfg = cfg.get("security", {})

    def scan(self, query: str) -> SecurityReport:
        """
        Run all security checks on a SQL query.
        Returns a SecurityReport with findings, score, and recommendations.
        """
        if not query or not query.strip():
            return SecurityReport(
                query=query,
                risk_score=0,
                risk_label="UNKNOWN",
                safe_to_execute=False,
                summary="Empty query provided.",
            )

        # Normalize for pattern matching (uppercase, collapse whitespace)
        normalized = " ".join(query.upper().split())

        findings: List[SecurityFinding] = []
        total_score = 0

        for rule in ALL_RULES:
            pattern = re.compile(rule["pattern"], re.IGNORECASE | re.MULTILINE)
            match = pattern.search(query)

            if match:
                finding = SecurityFinding(
                    severity=rule["severity"],
                    rule_id=rule["rule_id"],
                    title=rule["title"],
                    description=rule["description"],
                    recommendation=rule["recommendation"],
                    matched_text=match.group(0),
                    owasp_ref=rule.get("owasp_ref", ""),
                    score_impact=rule["score"],
                )
                findings.append(finding)
                total_score = max(total_score, rule["score"])  # Use highest score

        # Determine risk label
        risk_label = self._score_to_label(total_score)

        # Only flag as unsafe if CRITICAL or HIGH findings exist
        safe = not any(
            f.severity in (Severity.CRITICAL, Severity.HIGH) for f in findings
        )

        # Sort findings by severity (worst first)
        severity_order = {
            Severity.CRITICAL: 0,
            Severity.HIGH: 1,
            Severity.MEDIUM: 2,
            Severity.LOW: 3,
            Severity.INFO: 4,
        }
        findings.sort(key=lambda f: severity_order[f.severity])

        summary = self._build_summary(findings, risk_label)
        recommended = self._suggest_safer_query(query, findings) if not safe else None

        return SecurityReport(
            query=query,
            risk_score=total_score,
            risk_label=risk_label,
            findings=findings,
            safe_to_execute=safe,
            recommended_query=recommended,
            summary=summary,
        )

    def _score_to_label(self, score: int) -> str:
        if score >= self.cfg.get("critical_threshold", 8):
            return "CRITICAL"
        elif score >= self.cfg.get("high_threshold", 6):
            return "HIGH"
        elif score >= self.cfg.get("medium_threshold", 4):
            return "MEDIUM"
        elif score >= self.cfg.get("low_threshold", 2):
            return "LOW"
        else:
            return "SAFE"

    def _build_summary(self, findings: List[SecurityFinding], label: str) -> str:
        if not findings:
            return "No security issues detected. Query appears safe."
        counts = {}
        for f in findings:
            counts[f.severity.value] = counts.get(f.severity.value, 0) + 1
        parts = [f"{v} {k}" for k, v in counts.items()]
        return f"{label} — {len(findings)} issue(s) found: {', '.join(parts)}."

    def _suggest_safer_query(self, query: str, findings: List[SecurityFinding]) -> str:
        """
        Provide a basic safer version of the query by removing the most
        dangerous elements. This is a heuristic — not a silver bullet.
        """
        safer = query

        # Remove tautologies
        safer = re.sub(r"\s+(OR|AND)\s+[\'\"]?\d+[\'\"]?\s*=\s*[\'\"]?\d+[\'\"]?", "", safer, flags=re.IGNORECASE)

        # Remove comment injections
        safer = re.sub(r"--.*$", "", safer, flags=re.MULTILINE)
        safer = re.sub(r"/\*.*?\*/", "", safer, flags=re.DOTALL)

        # Add LIMIT if missing and it's a SELECT
        if "SELECT" in safer.upper() and "LIMIT" not in safer.upper():
            safer = safer.rstrip(";").strip() + " LIMIT 100;"

        return safer.strip() if safer.strip() != query.strip() else None


# ── Quick test / demo ────────────────────────────────────────────────────────

if __name__ == "__main__":
    scanner = SQLSecurityScanner()

    test_queries = [
        # Safe query
        "SELECT name, email FROM users WHERE id = 1 LIMIT 10;",
        # Tautology injection
        "SELECT * FROM users WHERE username = 'admin' OR 1=1;",
        # Sensitive data exposure
        "SELECT username, password, ssn FROM accounts;",
        # Destructive operation
        "DROP TABLE users;",
        # UNION injection
        "SELECT name FROM products UNION SELECT password FROM users;",
    ]

    for query in test_queries:
        report = scanner.scan(query)
        print(f"\n{'─'*60}")
        print(f"Query:  {query[:70]}{'...' if len(query) > 70 else ''}")
        print(f"Score:  {report.risk_score}/10  |  Label: {report.risk_label}")
        print(f"Safe:   {report.safe_to_execute}")
        if report.findings:
            print(f"Issues:")
            for f in report.findings:
                print(f"  [{f.severity.value}] {f.title}")
        if report.recommended_query:
            print(f"Safer:  {report.recommended_query}")