#!/usr/bin/env python3
"""
src/training/finetune.py

Fine-tunes SQLCoder to convert natural language to SQL queries.

⚠️  DO NOT RUN THIS ON YOUR MACBOOK.
    Run this on Google Colab (free GPU):
    1. Go to colab.research.google.com
    2. File → New Notebook
    3. Runtime → Change Runtime Type → T4 GPU
    4. Upload this file and run it

What this script does step by step:
1. Loads SQLCoder-7B — a model already trained on SQL (better starting point than a general LLM)
2. Applies LoRA — adds small trainable layers so we only train ~1% of parameters
3. Loads our Spider/WikiSQL dataset from data/datasets/
4. Trains the model to get better at our specific NL-to-SQL format
5. Logs every experiment to MLflow so we can compare runs
6. Saves the fine-tuned model to models/finetuned/

Run: python3 src/training/finetune.py
(But seriously, use Colab)
"""

import os
import yaml
import mlflow
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from dotenv import load_dotenv

load_dotenv()


def load_config() -> dict:
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)


def load_model_and_tokenizer(config: dict):
    """
    Load SQLCoder with 4-bit quantization.

    SQLCoder is specifically designed for SQL generation —
    it already knows SQL syntax, table relationships, and
    common query patterns. Fine-tuning it on our data
    teaches it our specific prompt format and domain.

    4-bit quantization = compress model from 28GB → ~7GB
    so it fits on a free Colab GPU (15GB VRAM).
    """
    model_name = config["model"]["base_model_name"]

    print(f"📥 Loading: {model_name}")
    print("   First run downloads ~7GB — this takes a few minutes.\n")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=os.getenv("HUGGINGFACE_TOKEN"),
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=os.getenv("HUGGINGFACE_TOKEN"),
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("✅ Model loaded\n")
    return model, tokenizer


def attach_lora(model, config: dict):
    """
    Attach LoRA adapters.

    SQLCoder has 7 billion parameters.
    LoRA lets us train only ~7 million of them (0.1%).
    Same quality improvement, 70x less compute.
    """
    lora_cfg = config["lora"]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    print(f"🔧 LoRA attached:")
    print(f"   Training {trainable:,} of {total:,} parameters")
    print(f"   ({100 * trainable / total:.2f}% of total)\n")

    return model


def load_data(config: dict):
    """Load the preprocessed Spider + WikiSQL dataset."""
    dataset_dir = Path(config["data"]["dataset_dir"])

    train_path = dataset_dir / "train.jsonl"
    val_path = dataset_dir / "validation.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {train_path}\n"
            "Run these first:\n"
            "  python3 src/data/download_dataset.py\n"
            "  python3 src/data/preprocess.py"
        )

    print("📂 Loading dataset...")
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(train_path),
            "validation": str(val_path),
        },
    )

    print(f"   Train:      {len(dataset['train']):,} examples")
    print(f"   Validation: {len(dataset['validation']):,} examples\n")
    return dataset


def get_training_args(config: dict) -> TrainingArguments:
    """Set up training hyperparameters."""
    t = config["training"]

    return TrainingArguments(
        output_dir=t["output_dir"],
        num_train_epochs=t["num_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        fp16=t["fp16"],
        evaluation_strategy="steps",
        eval_steps=t["eval_steps"],
        save_strategy="steps",
        save_steps=t["save_steps"],
        logging_steps=t["logging_steps"],
        load_best_model_at_end=t["load_best_model_at_end"],
        report_to="none",
        dataloader_num_workers=2,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )


def train(config: dict):
    """Main training function with MLflow experiment tracking."""

    # Set up MLflow to log this training run
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run():

        # Log all settings so we can reproduce this run
        mlflow.log_params({
            "base_model": config["model"]["base_model_name"],
            "lora_r": config["lora"]["r"],
            "lora_alpha": config["lora"]["lora_alpha"],
            "learning_rate": config["training"]["learning_rate"],
            "num_epochs": config["training"]["num_epochs"],
            "language": config["data"]["language"],
            "max_samples": config["data"]["max_samples"],
        })

        # Load everything
        model, tokenizer = load_model_and_tokenizer(config)
        model = attach_lora(model, config)
        dataset = load_data(config)
        training_args = get_training_args(config)

        # SFTTrainer handles the training loop
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            dataset_text_field="text",
            max_seq_length=(
                config["data"]["max_input_length"] +
                config["data"]["max_target_length"]
            ),
            args=training_args,
        )

        print("🏋️  Starting training...")
        print(f"   Epochs:        {config['training']['num_epochs']}")
        print(f"   Estimated time: 1-3 hours on Colab T4 GPU\n")

        # Train the model
        result = trainer.train()

        # Log final metrics
        mlflow.log_metrics({
            "train_loss": result.training_loss,
            "train_runtime_seconds": result.metrics["train_runtime"],
            "samples_per_second": result.metrics["train_samples_per_second"],
        })

        # Evaluate on validation set
        print("\n📊 Running evaluation...")
        eval_results = trainer.evaluate()
        mlflow.log_metrics({
            "eval_loss": eval_results["eval_loss"],
        })

        # Save the model
        save_dir = config["model"]["finetuned_model_dir"]
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        mlflow.log_artifacts(save_dir, artifact_path="model")

        print(f"\n✅ Training complete!")
        print(f"   Train loss: {result.training_loss:.4f}")
        print(f"   Eval loss:  {eval_results['eval_loss']:.4f}")
        print(f"   Model saved to: {save_dir}")
        print(f"\n   Next: uvicorn src.api.main:app --reload\n")

        return result, eval_results


def main():
    config = load_config()

    if not torch.cuda.is_available():
        print("⚠️  No GPU detected!")
        print("   This will take days on CPU.")
        print("   Upload this script to Google Colab instead:\n")
        print("   1. Go to colab.research.google.com")
        print("   2. File → New Notebook")
        print("   3. Runtime → Change Runtime Type → T4 GPU")
        print("   4. Upload finetune.py and your data/ folder")
        print("   5. Run: !python3 finetune.py\n")
        response = input("   Continue on CPU anyway? (y/n): ")
        if response.lower() != "y":
            return

    train(config)


if __name__ == "__main__":
    main()