#!/usr/bin/env python3
"""
src/api/main.py

The REST API that ties everything together:
  1. Accepts a natural language question + database schema
  2. Generates SQL using the fine-tuned model
  3. Runs the security scanner on the generated SQL
  4. Returns the query + full security report

Run: uvicorn src.api.main:app --reload
Docs: http://localhost:8000/docs
"""

import os
import time
import yaml
import torch
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from dotenv import load_dotenv

from src.security.scanner import SQLSecurityScanner, SecurityReport

load_dotenv()

app = FastAPI(
    title="SecureQuery — NL-to-SQL + Security Analysis",
    description=(
        "Convert natural language questions to SQL queries, "
        "with automatic security vulnerability scanning."
    ),
    version="1.0.0",
)

# Global state
model_pipeline = None
security_scanner = None
config = None


# ── Request / Response Models ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural language question")
    schema: str = Field(
        default="",
        description="Database schema hint, e.g. 'users(id, name, email, created_at)'"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "question": "Show me all customers who spent more than $500 last month",
                "schema": "orders(id, customer_id, amount, created_at), customers(id, name, email)"
            }
        }


class QueryResponse(BaseModel):
    question: str
    generated_sql: str
    security_report: dict
    latency_ms: float
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    scanner_loaded: bool
    device: str


# ── Model Loading ────────────────────────────────────────────────────────────

def load_config():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


def load_model():
    global model_pipeline, security_scanner, config
    config = load_config()

    # Load security scanner — no GPU needed, works immediately
    security_scanner = SQLSecurityScanner()
    print("✅ Security scanner ready")

    # Load NL-to-SQL model
    model_dir = config["model"]["finetuned_model_dir"]
    if not Path(model_dir).exists():
        print(f"⚠️  Fine-tuned model not found at {model_dir}")
        print(f"   Using base model: {config['model']['base_model_name']}")
        model_dir = config["model"]["base_model_name"]

    print(f"📥 Loading NL-to-SQL model from {model_dir}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        token=os.getenv("HUGGINGFACE_TOKEN"),
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto",
        token=os.getenv("HUGGINGFACE_TOKEN"),
    )

    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
    )
    print(f"✅ Model ready on {device}\n")


def format_prompt(question: str, schema: str) -> str:
    """Must match the exact format used in preprocess.py during training."""
    system = (
        "You are a SQL expert. Convert the natural language question to a valid, "
        "safe SQL query based on the provided database schema. "
        "Return ONLY the SQL query with no explanation."
    )
    if schema:
        user_content = f"Schema: {schema}\n\nQuestion: {question}"
    else:
        user_content = f"Question: {question}"

    return (
        f"### Task\n{system}\n\n"
        f"### Input\n{user_content}\n\n"
        f"### Response\n"
    )


def extract_sql(raw_output: str, prompt: str) -> str:
    """Pull just the generated SQL from the model's full output."""
    # Remove the input prompt from the output
    sql = raw_output[len(prompt):].strip()

    # Stop at the first blank line (model sometimes adds explanation after)
    sql = sql.split("\n\n")[0].strip()

    # Remove markdown code fences if model added them
    sql = sql.replace("```sql", "").replace("```", "").strip()

    return sql


# ── Routes ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    load_model()


@app.get("/", response_model=dict)
def root():
    return {
        "name": "SecureQuery",
        "description": "NL-to-SQL with automatic security scanning",
        "docs": "/docs",
        "endpoints": ["/query", "/scan", "/health"],
    }


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy",
        model_loaded=model_pipeline is not None,
        scanner_loaded=security_scanner is not None,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Full pipeline: natural language → SQL → security analysis.
    This is the main endpoint.
    """
    if not model_pipeline:
        raise HTTPException(status_code=503, detail="Model still loading")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    if len(request.question) > config["api"]["max_question_length"]:
        raise HTTPException(status_code=400, detail="Question too long")

    start = time.time()
    prompt = format_prompt(request.question, request.schema)

    # Step 1: Generate SQL
    outputs = model_pipeline(
        prompt,
        max_new_tokens=config["api"]["max_new_tokens"],
        temperature=config["api"]["temperature"],
        top_p=config["api"]["top_p"],
        do_sample=True,
        pad_token_id=model_pipeline.tokenizer.eos_token_id,
    )
    generated_sql = extract_sql(outputs[0]["generated_text"], prompt)

    # Step 2: Security scan
    report: SecurityReport = security_scanner.scan(generated_sql)

    latency_ms = round((time.time() - start) * 1000, 2)

    return QueryResponse(
        question=request.question,
        generated_sql=generated_sql,
        security_report=report.to_dict(),
        latency_ms=latency_ms,
        model_version="sqlcoder-7b-secure-v1",
    )


@app.post("/scan")
def scan_only(sql: str):
    """
    Security scan only — no model needed.
    Useful for scanning manually written SQL queries.
    Great for demoing the security layer independently.
    """
    if not security_scanner:
        raise HTTPException(status_code=503, detail="Scanner not ready")

    report = security_scanner.scan(sql)
    return report.to_dict()