"""
Evaluation script for LoRA Password Generator.
Measures exact match accuracy, character-level accuracy, edit distance,
and character set validity on test CSV data.
"""

import argparse
import json
import os
from typing import Dict, List, Optional

import pandas as pd
import torch
from tqdm import tqdm

from config import default_config
from model import load_for_inference
from utils import (
    compute_metrics,
    get_device,
    is_valid_output,
)


def generate_single_prediction(
    model: torch.nn.Module,
    tokenizer,
    x: str,
    device: torch.device,
    max_new_tokens: int = 40,
) -> str:
    """Run greedy decoding for a single input string X."""
    prompt = f"Input: {x}\nOutput:"
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
            do_sample=False,  # Deterministic greedy search
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode generated sequence
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Output:" in full_output:
        raw_pred = full_output.split("Output:")[-1].strip()
    else:
        raw_pred = full_output.strip()

    # Extract first line / first 32 characters
    pred_cleaned = raw_pred.split("\n")[0].strip()
    return pred_cleaned[:32]


def run_evaluation(
    base_model_name: str,
    lora_checkpoint_dir: str,
    test_csv: str = default_config.TEST_CSV,
    max_samples: Optional[int] = 500,
    output_report_file: str = default_config.EVAL_RESULTS_FILE,
) -> Dict[str, float]:
    """Execute model evaluation on the test CSV dataset."""
    device = get_device()
    print("=" * 60)
    print("Evaluating LoRA Password Generator")
    print(f"Base model:       {base_model_name}")
    print(f"LoRA adapter:     {lora_checkpoint_dir}")
    print(f"Test CSV:         {test_csv}")
    print(f"Device:           {device}")
    print("=" * 60)

    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Test CSV '{test_csv}' does not exist.")

    df = pd.read_csv(test_csv)
    if max_samples and max_samples < len(df):
        df = df.iloc[:max_samples]

    model, tokenizer = load_for_inference(base_model_name, lora_checkpoint_dir, device)

    exact_matches = 0
    total_char_acc = 0.0
    total_edit_distance = 0
    valid_format_count = 0
    detailed_samples = []

    print(f"\nEvaluating on {len(df):,} test samples...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Inference"):
        x = str(row[default_config.INPUT_COL]).strip()
        y_true = str(row[default_config.OUTPUT_COL]).strip()

        y_pred = generate_single_prediction(model, tokenizer, x, device)
        metrics = compute_metrics(y_pred, y_true)

        if metrics["exact_match"] == 1.0:
            exact_matches += 1
        total_char_acc += metrics["char_accuracy"]
        total_edit_distance += metrics["edit_distance"]

        if is_valid_output(y_pred, expected_length=32):
            valid_format_count += 1

        if len(detailed_samples) < 5:
            detailed_samples.append({
                "input": x,
                "expected": y_true,
                "predicted": y_pred,
                "exact": bool(metrics["exact_match"]),
                "char_acc": metrics["char_accuracy"],
            })

    total_evaluated = len(df)
    results = {
        "samples_evaluated": total_evaluated,
        "exact_match_accuracy": round(exact_matches / total_evaluated, 4),
        "exact_matches_count": exact_matches,
        "mean_character_accuracy": round(total_char_acc / total_evaluated, 4),
        "mean_edit_distance": round(total_edit_distance / total_evaluated, 2),
        "valid_format_ratio": round(valid_format_count / total_evaluated, 4),
    }

    # Print Report
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS REPORT")
    print("=" * 60)
    print(f"Total Samples Evaluated:   {results['samples_evaluated']:,}")
    print(f"Exact Match Accuracy:      {results['exact_match_accuracy'] * 100:.2f}%")
    print(f"Mean Character Accuracy:   {results['mean_character_accuracy'] * 100:.2f}%")
    print(f"Mean Edit Distance:        {results['mean_edit_distance']}")
    print(f"Format Validity (32-char): {results['valid_format_ratio'] * 100:.2f}%")
    print("=" * 60)

    print("\nSample Comparisons:")
    for i, s in enumerate(detailed_samples, 1):
        print(f"[{i}] Input:     {s['input']}")
        print(f"    Expected:  {s['expected']}")
        print(f"    Predicted: {s['predicted']}")
        print(f"    Match:     {'YES' if s['exact'] else 'NO'} (Char Acc: {s['char_acc'] * 100:.1f}%)\n")

    # Save to JSON
    with open(output_report_file, "w", encoding="utf-8") as f:
        json.dump({"metrics": results, "sample_outputs": detailed_samples}, f, indent=2)
    print(f"Results saved to: {output_report_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LoRA password generator model.")
    parser.add_argument("--model_name", type=str, default=default_config.MODEL_NAME, help="Base model")
    parser.add_argument("--checkpoint", type=str, default=os.path.join(default_config.OUTPUT_DIR, "final_checkpoint"), help="LoRA checkpoint path")
    parser.add_argument("--test_csv", type=str, default=default_config.TEST_CSV, help="Test CSV")
    parser.add_argument("--max_samples", type=int, default=500, help="Max test samples to evaluate (None for all)")
    parser.add_argument("--output_file", type=str, default=default_config.EVAL_RESULTS_FILE, help="Path for JSON results")
    args = parser.parse_args()

    run_evaluation(
        base_model_name=args.model_name,
        lora_checkpoint_dir=args.checkpoint,
        test_csv=args.test_csv,
        max_samples=args.max_samples,
        output_report_file=args.output_file,
    )

