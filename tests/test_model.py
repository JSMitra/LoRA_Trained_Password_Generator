"""
Tests for model.py: target module detection, LoRA configuration,
and trainable parameter verification.
Compatible with both pytest and unittest.
"""

import importlib.util
import unittest

from config import TrainingConfig
from utils import count_parameters

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_PEFT = HAS_TORCH and importlib.util.find_spec("peft") is not None

if HAS_PEFT:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from model import detect_target_modules

    class SimpleLinearLM(torch.nn.Module):
        """Minimal language model mock for testing PEFT LoRA adapter wrapping."""

        def __init__(self, vocab_size=100, hidden_dim=32):
            super().__init__()
            self.embedding = torch.nn.Embedding(vocab_size, hidden_dim)
            self.q_proj = torch.nn.Linear(hidden_dim, hidden_dim)
            self.v_proj = torch.nn.Linear(hidden_dim, hidden_dim)
            self.lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False)

        def forward(self, input_ids, **kwargs):
            x = self.embedding(input_ids)
            q = self.q_proj(x)
            v = self.v_proj(q)
            logits = self.lm_head(v)
            return type("Output", (), {"logits": logits, "loss": torch.tensor(1.0)})()

        def prepare_inputs_for_generation(self, *args, **kwargs):
            return {}


@unittest.skipUnless(HAS_PEFT, "PEFT & PyTorch required for model architecture tests")
class TestModelArchitecture(unittest.TestCase):
    def test_detect_target_modules(self):
        cfg = TrainingConfig()

        # GPT-2 target check
        gpt2_targets = detect_target_modules("gpt2", cfg)
        self.assertEqual(gpt2_targets, ["c_attn", "c_proj"])

        # Qwen / modern LLM target check
        qwen_targets = detect_target_modules("Qwen/Qwen2.5-1.5B", cfg)
        self.assertIn("q_proj", qwen_targets)
        self.assertIn("v_proj", qwen_targets)

        llama_targets = detect_target_modules("meta-llama/Llama-3.2-1B", cfg)
        self.assertIn("q_proj", llama_targets)

    def test_peft_lora_application_and_frozen_base(self):
        base_model = SimpleLinearLM()
        stats_before = count_parameters(base_model)
        self.assertEqual(stats_before["trainable_percent"], 100.0)

        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )

        peft_model = get_peft_model(base_model, lora_config)
        stats_after = count_parameters(peft_model)

        # Trainable parameters must be drastically lower than total
        self.assertLess(stats_after["trainable_parameters"], stats_after["total_parameters"])
        self.assertLess(stats_after["trainable_percent"], 100.0)

        # Base embeddings and head must be frozen
        self.assertFalse(base_model.embedding.weight.requires_grad)
        self.assertFalse(base_model.lm_head.weight.requires_grad)


if __name__ == "__main__":
    unittest.main()
