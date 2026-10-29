"""
Interactive and CLI inference utility for the LoRA Password Generator.
Takes a 16-character lowercase alphanumeric input and outputs the 32-character generated string.
"""

import argparse
import os
import random
import sys
from typing import Optional

import pandas as pd
import torch

from config import default_config
from model import load_for_inference
from utils import (
    derive_output_from_input,
    get_device,
    is_valid_input,
    is_valid_output,
)


def generate_password(
    model: torch.nn.Module,
    tokenizer,
    input_str: str,
    device: torch.device,
    max_new_tokens: int = 40,
) -> str:
    """Run model inference to predict 32-character output Y for a given 16-character X."""
    prompt = f"Input: {input_str}\nOutput:"
    inputs = tokenizer(prompt, return_tensors="pt")
    if hasattr(inputs, "to"):
        inputs = inputs.to(device)
    else:
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    pad_token_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding for consistent deterministic output
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Output:" in full_output:
        raw = full_output.split("Output:")[-1].strip()
    else:
        raw = full_output.strip()

    first_line = raw.split("\n")[0].strip()
    return first_line[:32]


def run_demo(model, tokenizer, device, test_csv: str = default_config.TEST_CSV, num_demo: int = 5):
    """Run interactive demonstration on random test samples."""
    print("\n" + "=" * 60)
    print(f"Running Demo ({num_demo} random test samples)")
    print("=" * 60)

    if os.path.exists(test_csv):
        df = pd.read_csv(test_csv)
        sample_rows = df.sample(min(num_demo, len(df)), random_state=42)
        for _, row in sample_rows.iterrows():
            x = str(row[default_config.INPUT_COL]).strip()
            y_exp = str(row[default_config.OUTPUT_COL]).strip()
            y_pred = generate_password(model, tokenizer, x, device)
            is_match = y_pred == y_exp
            status = "MATCH [PASS]" if is_match else "DIFF [FAIL]"
            print(f"Input (X):    {x}")
            print(f"Output (Y):   {y_pred}")
            print(f"Expected:     {y_exp}")
            print(f"Status:       {status}\n")
    else:
        # Generate random on the fly
        chars = "abcdefghijklmnopqrstuvwxyz0123456789"
        for i in range(num_demo):
            x = "".join(random.choices(chars, k=16))
            y_exp = derive_output_from_input(x, salt=default_config.SALT, output_length=32)
            y_pred = generate_password(model, tokenizer, x, device)
            is_match = y_pred == y_exp
            status = "MATCH [PASS]" if is_match else "DIFF [FAIL]"
            print(f"[{i+1}] Input:    {x}")
            print(f"    Output:   {y_pred}")
            print(f"    Expected: {y_exp}")
            print(f"    Status:   {status}\n")


def interactive_session(model, tokenizer, device):
    """Launch interactive CLI prompt loop."""
    print("\n" + "=" * 60)
    print("Interactive Password Generator Mode")
    print("Type a 16-character lowercase alphanumeric string and press Enter.")
    print("Type 'exit' or 'quit' to quit.")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("Enter 16-char Input (X) > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting session.")
            break

        if user_input in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        if not is_valid_input(user_input, expected_length=16):
            print(f"[!] Error: Input must be exactly 16 lowercase alphanumeric chars. Got: '{user_input}' (len={len(user_input)})")
            continue

        predicted = generate_password(model, tokenizer, user_input, device)
        expected = derive_output_from_input(user_input, salt=default_config.SALT, output_length=32)
        match = predicted == expected

        print(f"  -> Generated (Y): {predicted}")
        print(f"  -> Expected  (Y): {expected}")
        print(f"  -> Match Status:  {'PERFECT MATCH [✓]' if match else 'DIFFERENT'}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Password generator model inference.")
    parser.add_argument("--model_name", type=str, default=default_config.MODEL_NAME, help="Base model name")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(default_config.OUTPUT_DIR, "final_checkpoint"), help="LoRA checkpoint directory")
    parser.add_argument("--input", type=str, default=None, help="16-character input string")
    parser.add_argument("--demo", action="store_true", help="Run demo with sample inputs")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive prompt loop")
    args = parser.parse_args()

    device = get_device()
    checkpoint_to_use = args.checkpoint
    if not os.path.exists(checkpoint_to_use):
        best_ckpt = os.path.join(default_config.OUTPUT_DIR, "best_checkpoint")
        if os.path.exists(best_ckpt):
            checkpoint_to_use = best_ckpt
        else:
            print(f"[!] Warning: Checkpoint '{args.checkpoint}' not found.")
            print(f"    Please run 'python train.py' first.")
            sys.exit(1)

    model, tokenizer = load_for_inference(args.model_name, checkpoint_to_use, device)

    if args.input:
        if not is_valid_input(args.input, expected_length=16):
            print(f"[!] Error: Input '{args.input}' must be 16 lowercase alphanumeric characters.")
            sys.exit(1)

        y_pred = generate_password(model, tokenizer, args.input, device)
        expected_y = derive_output_from_input(args.input, salt=default_config.SALT, output_length=32)
        print("\n" + "=" * 50)
        print(f"Input:    {args.input}")
        print(f"Output:   {y_pred}")
        print(f"Expected: {expected_y}")
        print(f"Match:    {y_pred == expected_y}")
        print("=" * 50)
    elif args.demo:
        run_demo(model, tokenizer, device)
    else:
        # Default to interactive
        interactive_session(model, tokenizer, device)

