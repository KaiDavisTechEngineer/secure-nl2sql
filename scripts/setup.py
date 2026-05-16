#!/usr/bin/env python3
"""
scripts/setup.py — Run this first.
"""

import os
import sys
from pathlib import Path


def create_folders():
    folders = [
        "data/raw", "data/processed", "data/datasets",
        "models/base", "models/finetuned", "models/checkpoints",
        "logs", "mlruns",
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    print("✅ Folders created")


def create_env():
    if not Path(".env").exists():
        Path(".env").write_text(
            "HUGGINGFACE_TOKEN=your_token_here\n"
            "MLFLOW_TRACKING_URI=http://localhost:5000\n"
            "DEVICE=cuda\n"
        )
        print("✅ .env created — add your HuggingFace token!")
    else:
        print("✅ .env already exists")


def create_gitignore():
    if not Path(".gitignore").exists():
        Path(".gitignore").write_text(
            "__pycache__/\n*.py[cod]\n.env\nvenv/\n.venv/\n"
            "data/raw/\nmodels/\nmlruns/\nlogs/\n*.log\n.ipynb_checkpoints/\n"
        )
        print("✅ .gitignore created")


def check_python():
    v = sys.version_info
    if v.major < 3 or (v.major == 3 and v.minor < 10):
        print(f"❌ Python 3.10+ required. You have {v.major}.{v.minor}")
        sys.exit(1)
    print(f"✅ Python {v.major}.{v.minor}")


def check_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU: {name} ({vram:.1f}GB)")
        else:
            print("⚠️  No GPU — use Google Colab for fine-tuning")
    except ImportError:
        print("⚠️  PyTorch not installed yet")


def main():
    print("\n🔐 Setting up SecureQuery — NL-to-SQL + Security Scanner\n")
    check_python()
    create_folders()
    create_env()
    create_gitignore()
    check_gpu()
    print("\n✅ Done! Next steps:")
    print("   1. Fill in HUGGINGFACE_TOKEN in .env")
    print("   2. pip install -r requirements.txt")
    print("   3. python src/data/download_dataset.py\n")


if __name__ == "__main__":
    main()