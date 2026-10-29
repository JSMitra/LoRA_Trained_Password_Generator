"""
Training pipeline for fine-tuning LLM with LoRA on CSV password datasets.
Includes gradient accumulation, evaluation, checkpointing, and metrics logging.
"""

import argparse
import json
import math
import os
import time
from typing import Dict, Optional

import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from config import TrainingConfig, default_config
from dataset import get_dataloaders
from model import build_lora_model
from utils import get_device, set_seed


def evaluate(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """Evaluate mean validation loss across the validation dataloader."""
    model.eval()
    total_val_loss = 0.0
    val_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            if not torch.isnan(loss):
                total_val_loss += loss.item()
                val_batches += 1

    model.train()
    return total_val_loss / max(val_batches, 1)


def log_step(log_file: str, data: Dict) -> None:
    """Append structured log entry to JSON Lines log file."""
    os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")


def train(config: TrainingConfig) -> str:
    """Execute the full LoRA fine-tuning loop."""
    set_seed(config.SEED)
    device = get_device()
    print(f"Starting training run with device: {device}")

    # Check for dataset
    if not os.path.exists(config.TRAIN_CSV):
        raise FileNotFoundError(
            f"Training CSV '{config.TRAIN_CSV}' not found! "
            f"Run 'python generate_data.py' first."
        )

    # Initialize model and tokenizer
    model, tokenizer = build_lora_model(
        config=config,
        model_name=config.MODEL_NAME,
        device=device,
    )

    # Load dataloaders
    train_loader, val_loader, _ = get_dataloaders(config=config, tokenizer=tokenizer)

    # Optimizer and Scheduler
    total_train_steps = (
        len(train_loader) // config.GRAD_ACCUM_STEPS
    ) * config.NUM_EPOCHS
    warmup_steps = int(total_train_steps * config.WARMUP_RATIO)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_train_steps,
    )

    print("\n" + "=" * 60)
    print("Training Configuration Summary:")
    print(f"  Base Model:          {config.MODEL_NAME}")
    print(f"  Epochs:              {config.NUM_EPOCHS}")
    print(f"  Batch Size:          {config.BATCH_SIZE}")
    print(f"  Grad Accum Steps:    {config.GRAD_ACCUM_STEPS}")
    print(f"  Effective Batch:     {config.BATCH_SIZE * config.GRAD_ACCUM_STEPS}")
    print(f"  Total Update Steps:  {total_train_steps}")
    print(f"  Warmup Steps:        {warmup_steps}")
    print(f"  Learning Rate:       {config.LEARNING_RATE}")
    print(f"  LoRA Rank (r):       {config.LORA_R}")
    print(f"  LoRA Alpha:          {config.LORA_ALPHA}")
    print("=" * 60 + "\n")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    best_val_loss = float("inf")
    global_step = 0
    running_train_loss = 0.0
    start_time = time.time()

    model.train()
    optimizer.zero_grad()

    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{config.NUM_EPOCHS} ---")
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} batches")

        for step_idx, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / config.GRAD_ACCUM_STEPS
            loss.backward()

            running_train_loss += loss.item() * config.GRAD_ACCUM_STEPS

            # Gradient accumulation boundary
            if (step_idx + 1) % config.GRAD_ACCUM_STEPS == 0 or (step_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % config.LOGGING_STEPS == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    avg_train_loss = running_train_loss / config.LOGGING_STEPS
                    running_train_loss = 0.0

                    progress_bar.set_postfix(
                        step=global_step,
                        loss=f"{avg_train_loss:.4f}",
                        lr=f"{current_lr:.2e}",
                    )

                    log_entry = {
                        "global_step": global_step,
                        "epoch": epoch,
                        "train_loss": round(avg_train_loss, 4),
                        "learning_rate": current_lr,
                        "elapsed_seconds": round(time.time() - start_time, 2),
                    }
                    log_step(config.LOG_FILE, log_entry)

                # Validation & Checkpointing
                if val_loader is not None and global_step % config.EVAL_STEPS == 0:
                    val_loss = evaluate(model, val_loader, device)
                    print(f"\n[Step {global_step}] Validation Loss: {val_loss:.4f}")

                    val_entry = {
                        "global_step": global_step,
                        "epoch": epoch,
                        "val_loss": round(val_loss, 4),
                    }
                    log_step(config.LOG_FILE, val_entry)

                    # Save best checkpoint
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_ckpt_dir = os.path.join(config.OUTPUT_DIR, "best_checkpoint")
                        print(f"New best val loss! Saving adapter to: {best_ckpt_dir}")
                        model.save_pretrained(best_ckpt_dir)
                        tokenizer.save_pretrained(best_ckpt_dir)

    # Save final weights
    final_ckpt_dir = os.path.join(config.OUTPUT_DIR, "final_checkpoint")
    print(f"\nTraining completed! Saving final adapter to: {final_ckpt_dir}")
    model.save_pretrained(final_ckpt_dir)
    tokenizer.save_pretrained(final_ckpt_dir)

    # Save training config
    config.save_json(os.path.join(config.OUTPUT_DIR, "training_config.json"))
    return final_ckpt_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA Training on CSV password data.")
    parser.add_argument("--model_name", type=str, default=default_config.MODEL_NAME, help="Base HF model")
    parser.add_argument("--train_csv", type=str, default=default_config.TRAIN_CSV, help="Path to train CSV")
    parser.add_argument("--val_csv", type=str, default=default_config.VAL_CSV, help="Path to val CSV")
    parser.add_argument("--output_dir", type=str, default=default_config.OUTPUT_DIR, help="Checkpoint output dir")
    parser.add_argument("--epochs", type=int, default=default_config.NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=default_config.BATCH_SIZE, help="Batch size per step")
    parser.add_argument("--lr", type=float, default=default_config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=default_config.LORA_R, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=default_config.LORA_ALPHA, help="LoRA alpha")
    args = parser.parse_args()

    cfg = TrainingConfig(
        MODEL_NAME=args.model_name,
        TRAIN_CSV=args.train_csv,
        VAL_CSV=args.val_csv,
        OUTPUT_DIR=args.output_dir,
        NUM_EPOCHS=args.epochs,
        BATCH_SIZE=args.batch_size,
        LEARNING_RATE=args.lr,
        LORA_R=args.lora_r,
        LORA_ALPHA=args.lora_alpha,
    )
    train(cfg)

