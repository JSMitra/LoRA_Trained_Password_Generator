"""
Synthetic Data Generator for 16-character X -> 32-character Y Password Mapping.
Outputs data directly to CSV format.
"""

import argparse
import csv
import os
import random
from typing import List, Tuple

from tqdm import tqdm

from config import default_config
from utils import (
    ALPHANUMERIC_CHARS,
    derive_output_from_input,
    is_valid_input,
    is_valid_output,
    set_seed,
)


def generate_unique_inputs(
    num_samples: int,
    input_length: int = 16,
    batch_size: int = 50_000,
) -> List[str]:
    """
    Generate exactly num_samples unique random lowercase alphanumeric strings of length input_length.
    Uses batch sampling with set deduplication for high speed.
    """
    unique_inputs = set()
    chars = ALPHANUMERIC_CHARS
    chars_len = len(chars)

    with tqdm(total=num_samples, desc="Generating unique inputs (X)") as pbar:
        while len(unique_inputs) < num_samples:
            needed = num_samples - len(unique_inputs)
            # Sample in chunks for efficiency
            current_batch = min(max(needed * 2, 1000), batch_size)
            new_batch = [
                "".join(random.choices(chars, k=input_length))
                for _ in range(current_batch)
            ]
            before_len = len(unique_inputs)
            unique_inputs.update(new_batch)
            added = min(len(unique_inputs) - before_len, needed)
            pbar.update(added)

    # Return exactly num_samples as a list
    result = list(unique_inputs)[:num_samples]
    random.shuffle(result)
    return result


def create_dataset_pairs(
    inputs: List[str],
    salt: str,
    output_length: int = 32,
) -> List[Tuple[str, str]]:
    """
    Pair each input string X with its deterministic derived output Y of length output_length.
    """
    pairs = []
    for x in tqdm(inputs, desc="Deriving output tokens (Y)"):
        y = derive_output_from_input(x, salt=salt, output_length=output_length)
        pairs.append((x, y))
    return pairs


def save_to_csv(
    data: List[Tuple[str, str]],
    filepath: str,
    input_header: str = "input",
    output_header: str = "output",
) -> None:
    """Save pairs to a CSV file."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([input_header, output_header])
        writer.writerows(data)
    print(f"Saved {len(data):,} samples -> {filepath}")


def run_data_generation(
    num_samples: int = default_config.NUM_SAMPLES,
    input_len: int = default_config.INPUT_LENGTH,
    output_len: int = default_config.OUTPUT_LENGTH,
    salt: str = default_config.SALT,
    data_dir: str = default_config.DATA_DIR,
    val_ratio: float = default_config.VAL_RATIO,
    test_ratio: float = default_config.TEST_RATIO,
    seed: int = default_config.SEED,
) -> None:
    """Generate 100,000 (X, Y) pairs and save train/val/test CSVs."""
    print("=" * 60)
    print("LoRA Password Generator: Dataset Creation")
    print("=" * 60)
    print(f"Target count: {num_samples:,}")
    print(f"Input string length (X): {input_len} characters")
    print(f"Output string length (Y): {output_len} characters")
    print(f"Output directory: {data_dir}")
    print(f"Random seed: {seed}")
    print("=" * 60)

    set_seed(seed)

    # Step 1: Generate unique inputs
    inputs = generate_unique_inputs(num_samples=num_samples, input_length=input_len)

    # Step 2: Generate pairs
    pairs = create_dataset_pairs(inputs=inputs, salt=salt, output_length=output_len)

    # Validate generated samples
    for x, y in pairs[:1000]:
        assert is_valid_input(x, expected_length=input_len), f"Invalid input: {x}"
        assert is_valid_output(y, expected_length=output_len), f"Invalid output: {y}"

    # Step 3: Compute splits
    test_count = int(num_samples * test_ratio)
    val_count = int(num_samples * val_ratio)
    train_count = num_samples - val_count - test_count

    train_data = pairs[:train_count]
    val_data = pairs[train_count : train_count + val_count]
    test_data = pairs[train_count + val_count :]

    # Step 4: Write CSV files
    full_csv = os.path.join(data_dir, "dataset.csv")
    train_csv = os.path.join(data_dir, "train.csv")
    val_csv = os.path.join(data_dir, "val.csv")
    test_csv = os.path.join(data_dir, "test.csv")

    save_to_csv(pairs, full_csv)
    save_to_csv(train_data, train_csv)
    save_to_csv(val_data, val_csv)
    save_to_csv(test_data, test_csv)

    print("\n" + "=" * 60)
    print("Dataset Generation Complete!")
    print(f"Train samples: {len(train_data):,}")
    print(f"Val samples:   {len(val_data):,}")
    print(f"Test samples:  {len(test_data):,}")
    print("=" * 60)
    print("\nSample Inspection (first 3 pairs):")
    for idx, (x, y) in enumerate(pairs[:3], start=1):
        print(f"Example {idx}:")
        print(f"  Input:  {x} (len={len(x)})")
        print(f"  Output: {y} (len={len(y)})\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic password CSV dataset.")
    parser.add_argument("--num_samples", type=int, default=default_config.NUM_SAMPLES, help="Number of samples")
    parser.add_argument("--input_len", type=int, default=default_config.INPUT_LENGTH, help="Input length (chars)")
    parser.add_argument("--output_len", type=int, default=default_config.OUTPUT_LENGTH, help="Output length (chars)")
    parser.add_argument("--data_dir", type=str, default=default_config.DATA_DIR, help="Directory to save CSVs")
    parser.add_argument("--seed", type=int, default=default_config.SEED, help="Random seed")
    args = parser.parse_args()

    run_data_generation(
        num_samples=args.num_samples,
        input_len=args.input_len,
        output_len=args.output_len,
        data_dir=args.data_dir,
        seed=args.seed,
    )

