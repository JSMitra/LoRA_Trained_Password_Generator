"""
Utility functions for LoRA-Trained Password & Token Generator.
Provides seed control, string validation, deterministic hash mapping,
metrics calculation, and device management.
"""

import hashlib
import os
import random
import string
from typing import Dict, Tuple

import numpy as np

ALPHANUMERIC_CHARS: str = string.ascii_lowercase + string.digits  # 36 chars: a-z, 0-9


def set_seed(seed: int = 42) -> None:
    """Set random seed across python random, numpy, and torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)
    except ImportError:
        pass


def get_alphanumeric_chars() -> str:
    """Return the 36 valid characters (lowercase a-z and 0-9)."""
    return ALPHANUMERIC_CHARS


def is_valid_input(x: str, expected_length: int = 16) -> bool:
    """
    Validate that input X is a string of expected_length composed solely
    of lowercase alphanumeric characters (a-z, 0-9).
    """
    if not isinstance(x, str):
        return False
    if len(x) != expected_length:
        return False
    return all(c in ALPHANUMERIC_CHARS for c in x)


def is_valid_output(y: str, expected_length: int = 32) -> bool:
    """
    Validate that output Y is a string of expected_length composed solely
    of lowercase alphanumeric characters (a-z, 0-9).
    """
    if not isinstance(y, str):
        return False
    if len(y) != expected_length:
        return False
    return all(c in ALPHANUMERIC_CHARS for c in y)


def derive_output_from_input(
    x: str,
    salt: str = "lora_password_generator_salt_2024",
    output_length: int = 32,
) -> str:
    """
    Deterministically derive a lowercase alphanumeric string Y of length output_length
    from an input string X and a fixed salt using cryptographic hashing.

    Uses SHA-256 (32 raw bytes). Each byte modulo 36 deterministically selects
    one character from [a-z0-9].
    """
    if not is_valid_input(x, expected_length=len(x)):
        raise ValueError(f"Input '{x}' is not a valid lowercase alphanumeric string.")

    # Generate cryptographic digest
    combined = f"{x}:{salt}".encode("utf-8")
    digest = hashlib.sha256(combined).digest()  # Exactly 32 bytes

    # Map each byte to an alphanumeric character
    chars = [ALPHANUMERIC_CHARS[b % len(ALPHANUMERIC_CHARS)] for b in digest[:output_length]]
    return "".join(chars)


def get_device():
    """
    Determine the optimal available compute device in order of preference:
    CUDA -> MPS (Apple Silicon GPU) -> CPU.
    """
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def count_parameters(model) -> Dict[str, int]:
    """
    Count total and trainable parameters of a PyTorch model.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_pct = (trainable / total * 100.0) if total > 0 else 0.0
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percent": round(trainable_pct, 4),
    }


def compute_metrics(y_pred: str, y_true: str) -> Dict[str, float]:
    """
    Compute generation accuracy metrics between predicted and ground-truth string:
    - exact_match: 1.0 if identical else 0.0
    - char_accuracy: fraction of matching characters across valid length
    - edit_distance: Levenshtein distance
    """
    y_pred = str(y_pred).strip()
    y_true = str(y_true).strip()

    is_exact = 1.0 if y_pred == y_true else 0.0

    # Character-by-character accuracy
    min_len = min(len(y_pred), len(y_true))
    max_len = max(len(y_pred), len(y_true))
    if max_len == 0:
        char_acc = 1.0
    else:
        matches = sum(1 for i in range(min_len) if y_pred[i] == y_true[i])
        char_acc = matches / max_len

    # Levenshtein distance
    dist = levenshtein_distance(y_pred, y_true)

    return {
        "exact_match": is_exact,
        "char_accuracy": round(char_acc, 4),
        "edit_distance": dist,
    }


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute the Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]
