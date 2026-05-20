#!/usr/bin/env python3
"""
src/api/main.py

The REST API for SecureQuery.

This version uses Hugging Face's hosted Inference API to generate SQL,
then runs the output through our custom security scanner.

This is actually how most production ML systems work — they call hosted
models via API rather than self-hosting 7B+ parameter models.

Run: uvicorn src.api.main:app --reload
Docs: http://localhost:8000/docs
"""

import os
import time
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

from src.security.scanner import SQLSecurityScanner

load_dotenv()

app = FastAPI(
    title="SecureQuery — NL-to-SQL + Security Analysis",
    description=(
        "Convert natural language to SQL queries, with automatic "
        "security vulnerability scanning of every generated query."
    ),
    version="2.0.0",
)

hf_client = None
security_scanner = None
config = None


class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural language question")
    schema: str = Field(default="", description="Database schema")

    class Config:
        json_schema_extra = {
            "example": {
                "question": "Show customers who spent over $500 last month",
                "schema": "orders(id, customer_id, amount), customers(id, name)"
            }
        }


class QueryResponse(BaseModel):
    question: str
    generated_sql: str
    security_report: dict
    latency_ms: float
    model: str


class ScanRequest(BaseModel):
    sql: str


class HealthResponse(BaseModel):
    status: str
    scanner_loaded: bool
    api_configured: bool


def load_config() -> dict:
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


def initialize():
    global hf_client, security_scanner, config
    config = load_config()

    security_scanner = SQLSecurityScanner()
    print("✅ Security scanner ready")

    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if hf_token:
        hf_client = InferenceClient(
            model="defog/sqlcoder-7b-2",
            token=hf_token,
        )
        print("✅ HuggingFace API connected")
    else:
        print("⚠️  No HUGGINGFACE_TOKEN — /query disabled, /scan still works")


def format_prompt(question: str, schema: str) -> str:
    schema_block = schema if schema else "(schema not provided)"
    return f"""### Task
Generate a SQL query to answer: `{question}`

### Database Schema
{schema_block}

### Answer
```sql
"""


def extract_sql(generated_text: str) -> str:
    sql = generated_text.strip()
    sql = sql.split("```")[0]
    sql = sql.split("\n\n")[0]
    return sql.strip()


@app.on_event("startup")
async def startup():
    initialize()


@app.get("/")
def root():
    return {
        "name": "SecureQuery",
        "docs": "/docs",
        "endpoints": ["/query", "/scan", "/health"],
    }


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy",
        scanner_loaded=security_scanner is not None,
        api_configured=hf_client is not None,
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not hf_client:
        raise HTTPException(503, "HF API not configured. Add HUGGINGFACE_TOKEN to .env")
    if not request.question.strip():
        raise HTTPException(400, "Question cannot be empty")

    start = time.time()
    prompt = format_prompt(request.question, request.schema)

    try:
        generated = hf_client.text_generation(
            prompt,
            max_new_tokens=config["api"]["max_new_tokens"],
            temperature=config["api"]["temperature"],
            top_p=config["api"]["top_p"],
            do_sample=False,
        )
        sql = extract_sql(generated)
    except Exception as e:
        raise HTTPException(502, f"Model API error: {str(e)}")

    report = security_scanner.scan(sql)
    latency_ms = round((time.time() - start) * 1000, 2)

    return QueryResponse(
        question=request.question,
        generated_sql=sql,
        security_report=report.to_dict(),
        latency_ms=latency_ms,
        model="defog/sqlcoder-7b-2 (HuggingFace Inference API)",
    )


@app.post("/scan")
def scan_only(request: ScanRequest):
    if not security_scanner:
        raise HTTPException(503, "Scanner not ready")
    return security_scanner.scan(request.sql).to_dict()