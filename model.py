"""
Model loading, LoRA PEFT adapter attachment, and model merging utilities.
Supports small architectures under 7GB (Qwen2.5, Llama-3.2, SmolLM2, GPT-2).
"""

import os
from typing import List, Optional, Tuple

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from config import TrainingConfig, default_config
from utils import count_parameters, get_device


def detect_target_modules(model_name: str, config: TrainingConfig) -> List[str]:
    """
    Determine the optimal LoRA linear projection target modules based on model architecture.
    """
    model_name_lower = model_name.lower()
    if "gpt2" in model_name_lower:
        return config.GPT2_TARGET_MODULES
    # Default for modern transformer architectures (Qwen, Llama, Mistral, SmolLM)
    return config.LORA_TARGET_MODULES


def build_lora_model(
    config: TrainingConfig = default_config,
    model_name: Optional[str] = None,
    device: Optional[torch.device] = None,
    load_in_4bit: bool = False,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load base causal language model, configure LoRA PEFT adapters, and initialize tokenizer.
    """
    model_name = model_name or config.MODEL_NAME
    device = device or get_device()

    print(f"Loading base model: {model_name}")
    print(f"Target compute device: {device}")

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    # Quantization configuration if requested and supported
    quantization_config = None
    if load_in_4bit and torch.cuda.is_available():
        try:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            print("Loaded 4-bit QLoRA quantization configuration.")
        except ImportError:
            print("bitsandbytes not found; proceeding in full/half precision.")

    # Determine torch dtype
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    # Load Model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )

    # Determine target modules
    target_modules = detect_target_modules(model_name, config)
    print(f"LoRA Target Modules: {target_modules}")

    # LoRA PEFT Configuration
    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(base_model, lora_config)

    # If not 4-bit quantized, move to target device
    if quantization_config is None:
        model = model.to(device)

    # Print parameter report
    stats = count_parameters(model)
    print("=" * 60)
    print("LoRA PEFT Model Initialized:")
    print(f"  Total Parameters:     {stats['total_parameters']:,}")
    print(f"  Trainable Parameters: {stats['trainable_parameters']:,}")
    print(f"  Trainable %:          {stats['trainable_percent']}%")
    print("=" * 60)

    return model, tokenizer


def load_for_inference(
    base_model_name: str,
    lora_checkpoint_dir: str,
    device: Optional[torch.device] = None,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load base model and attach fine-tuned LoRA adapter weights for inference.
    """
    device = device or get_device()
    print(f"Loading inference base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    print(f"Attaching LoRA weights from: {lora_checkpoint_dir}")
    model = PeftModel.from_pretrained(base_model, lora_checkpoint_dir)
    model = model.to(device)
    model.eval()

    return model, tokenizer


def merge_and_export_model(
    base_model_name: str,
    lora_checkpoint_dir: str,
    export_dir: str,
) -> str:
    """
    Merge LoRA adapter into the base model weights and save the unified model.
    The resulting checkpoint is ready for GGUF conversion / Ollama deployment.
    """
    print(f"Loading base model: {base_model_name} for weight merge...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )

    print(f"Merging LoRA weights from {lora_checkpoint_dir}...")
    peft_model = PeftModel.from_pretrained(base_model, lora_checkpoint_dir)
    merged_model = peft_model.merge_and_unload()

    os.makedirs(export_dir, exist_ok=True)
    print(f"Saving merged standalone model to {export_dir}...")
    merged_model.save_pretrained(export_dir)
    tokenizer.save_pretrained(export_dir)
    print(f"Merged model exported successfully to: {export_dir}")
    return export_dir

