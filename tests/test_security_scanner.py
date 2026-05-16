"""
tests/test_security_scanner.py

Tests for the security scanner.
This is the most important test file — the scanner is the core differentiator.

Run: pytest tests/ -v
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.security.scanner import SQLSecurityScanner, Severity


@pytest.fixture
def scanner():
    return SQLSecurityScanner()


# ── Safe Queries ─────────────────────────────────────────────────────────────

class TestSafeQueries:
    def test_simple_select(self, scanner):
        report = scanner.scan("SELECT name, email FROM users WHERE id = 1 LIMIT 10;")
        assert report.risk_label == "SAFE"
        assert report.safe_to_execute is True
        assert report.risk_score <= 1

    def test_select_with_join(self, scanner):
        sql = "SELECT o.id, u.name FROM orders o JOIN users u ON o.user_id = u.id WHERE o.total > 100 LIMIT 50;"
        report = scanner.scan(sql)
        assert report.safe_to_execute is True

    def test_aggregate_query(self, scanner):
        sql = "SELECT department, COUNT(*) as headcount FROM employees GROUP BY department;"
        report = scanner.scan(sql)
        assert report.safe_to_execute is True


# ── SQL Injection Detection ───────────────────────────────────────────────────

class TestInjectionDetection:
    def test_tautology_or_1_equals_1(self, scanner):
        report = scanner.scan("SELECT * FROM users WHERE username = 'admin' OR 1=1;")
        assert report.risk_label == "CRITICAL"
        rule_ids = [f.rule_id for f in report.findings]
        assert "SQL_INJECTION_TAUTOLOGY" in rule_ids

    def test_tautology_and_variant(self, scanner):
        report = scanner.scan("SELECT * FROM users WHERE id = 5 AND 1=1;")
        rule_ids = [f.rule_id for f in report.findings]
        assert "SQL_INJECTION_TAUTOLOGY" in rule_ids

    def test_union_injection(self, scanner):
        report = scanner.scan("SELECT name FROM products UNION SELECT password FROM users;")
        assert report.risk_label == "CRITICAL"
        rule_ids = [f.rule_id for f in report.findings]
        assert "SQL_INJECTION_UNION" in rule_ids

    def test_comment_injection_double_dash(self, scanner):
        report = scanner.scan("SELECT * FROM users WHERE id = 1 -- AND active = 1;")
        rule_ids = [f.rule_id for f in report.findings]
        assert "SQL_INJECTION_COMMENT" in rule_ids

    def test_stacked_query(self, scanner):
        report = scanner.scan("SELECT * FROM users; DROP TABLE users;")
        assert report.risk_label == "CRITICAL"
        rule_ids = [f.rule_id for f in report.findings]
        assert "SQL_INJECTION_STACKED" in rule_ids

    def test_time_based_injection_sleep(self, scanner):
        report = scanner.scan("SELECT * FROM users WHERE id = 1 AND SLEEP(5);")
        assert report.risk_label == "CRITICAL"
        rule_ids = [f.rule_id for f in report.findings]
        assert "SQL_INJECTION_TIME_BASED" in rule_ids


# ── Sensitive Data Exposure ───────────────────────────────────────────────────

class TestSensitiveDataExposure:
    def test_password_column(self, scanner):
        report = scanner.scan("SELECT username, password FROM accounts;")
        assert report.risk_label == "CRITICAL"
        rule_ids = [f.rule_id for f in report.findings]
        assert "SENSITIVE_COLUMN_EXPOSURE" in rule_ids

    def test_ssn_exposure(self, scanner):
        report = scanner.scan("SELECT name, ssn FROM patients WHERE doctor_id = 5;")
        rule_ids = [f.rule_id for f in report.findings]
        assert "PII_EXPOSURE" in rule_ids

    def test_api_key_exposure(self, scanner):
        report = scanner.scan("SELECT user_id, api_key FROM integrations;")
        rule_ids = [f.rule_id for f in report.findings]
        assert "SENSITIVE_COLUMN_EXPOSURE" in rule_ids


# ── Destructive Operations ────────────────────────────────────────────────────

class TestDestructiveOperations:
    def test_drop_table(self, scanner):
        report = scanner.scan("DROP TABLE users;")
        assert report.risk_label == "CRITICAL"
        assert report.safe_to_execute is False
        rule_ids = [f.rule_id for f in report.findings]
        assert "DESTRUCTIVE_DROP" in rule_ids

    def test_truncate(self, scanner):
        report = scanner.scan("TRUNCATE TABLE sessions;")
        assert report.risk_label == "CRITICAL"
        rule_ids = [f.rule_id for f in report.findings]
        assert "DESTRUCTIVE_TRUNCATE" in rule_ids

    def test_grant_privilege(self, scanner):
        report = scanner.scan("GRANT ALL PRIVILEGES ON *.* TO 'hacker'@'%';")
        assert report.risk_label == "CRITICAL"
        rule_ids = [f.rule_id for f in report.findings]
        assert "PRIVILEGE_MODIFICATION" in rule_ids


# ── Severity Ordering ─────────────────────────────────────────────────────────

class TestSeverityOrdering:
    def test_critical_first(self, scanner):
        """CRITICAL findings should always appear before LOW findings."""
        report = scanner.scan(
            "SELECT *, password FROM users WHERE 1=1 LIMIT 10;"
        )
        if len(report.findings) > 1:
            severities = [f.severity for f in report.findings]
            assert severities[0] == Severity.CRITICAL


# ── Edge Cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_query(self, scanner):
        report = scanner.scan("")
        assert report.safe_to_execute is False

    def test_whitespace_only(self, scanner):
        report = scanner.scan("   ")
        assert report.safe_to_execute is False

    def test_report_to_dict(self, scanner):
        report = scanner.scan("SELECT * FROM users WHERE 1=1;")
        d = report.to_dict()
        assert "risk_score" in d
        assert "risk_label" in d
        assert "findings" in d
        assert isinstance(d["findings"], list)