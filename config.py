"""
Configuration module for LoRA-trained Password / Token Generator.
Defines all dataset, model, LoRA, and training hyperparameters.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json
import os


@dataclass
class TrainingConfig:
    # --- Dataset Parameters ---
    DATA_DIR: str = "data"
    DATASET_CSV: str = "data/dataset.csv"
    TRAIN_CSV: str = "data/train.csv"
    VAL_CSV: str = "data/val.csv"
    TEST_CSV: str = "data/test.csv"
    
    NUM_SAMPLES: int = 100_000
    INPUT_LENGTH: int = 16       # Input string length (lowercase alphanumeric)
    OUTPUT_LENGTH: int = 32      # Output string length (lowercase alphanumeric)
    INPUT_COL: str = "input"
    OUTPUT_COL: str = "output"
    
    VAL_RATIO: float = 0.10      # 10,000 samples
    TEST_RATIO: float = 0.05     # 5,000 samples
    SALT: str = "lora_password_generator_salt_2024"
    SEED: int = 42

    # --- Model & Ollama Target (< 7GB) ---
    # Default is Qwen2.5-1.5B: ~1.54B parameters, ~3.0GB in FP16 / ~1.0GB in 4-bit.
    # Readily converts to GGUF format for Ollama runtimes.
    MODEL_NAME: str = "Qwen/Qwen2.5-1.5B"
    BACKUP_MODEL_NAME: str = "gpt2"  # 124M parameter fallback for ultra-fast CI/CPU test
    OUTPUT_DIR: str = "checkpoints"
    MERGED_DIR: str = "merged_model"
    OLLAMA_DIR: str = "ollama_export"
    
    # --- LoRA Hyperparameters (PEFT) ---
    LORA_R: int = 16
    LORA_ALPHA: int = 32
    LORA_DROPOUT: float = 0.05
    # Target modules for modern architectures (Qwen, Llama, Mistral)
    LORA_TARGET_MODULES: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    # GPT-2 fallback targets
    GPT2_TARGET_MODULES: List[str] = field(
        default_factory=lambda: ["c_attn", "c_proj"]
    )

    # --- Training Hyperparameters ---
    MAX_SEQ_LEN: int = 64
    BATCH_SIZE: int = 16
    GRAD_ACCUM_STEPS: int = 4    # Effective batch size = 64
    LEARNING_RATE: float = 2e-4
    NUM_EPOCHS: int = 3
    WARMUP_RATIO: float = 0.03
    WEIGHT_DECAY: float = 0.01
    MAX_GRAD_NORM: float = 1.0
    
    # --- Logging & Checkpointing ---
    SAVE_STEPS: int = 500
    EVAL_STEPS: int = 500
    LOGGING_STEPS: int = 50
    LOG_FILE: str = "training_log.jsonl"
    EVAL_RESULTS_FILE: str = "evaluation_results.json"

    # --- Prompt Template ---
    PROMPT_TEMPLATE: str = "Input: {input}\nOutput: {output}"
    INFERENCE_PREFIX: str = "Input: {input}\nOutput:"

    def to_dict(self) -> dict:
        """Convert config dataclass to dictionary."""
        return asdict(self)

    def save_json(self, path: str) -> None:
        """Save configuration parameters to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "TrainingConfig":
        """Load configuration from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


# Default shared instance
default_config = TrainingConfig()

