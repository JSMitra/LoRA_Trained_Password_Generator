"""
Tests for dataset.py: CSV reading, Causal LM prompt masking,
and batch dynamic collator.
Compatible with both pytest and unittest.
"""

import importlib.util
import os
import shutil
import tempfile
import unittest
import pandas as pd

from utils import derive_output_from_input

HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    import torch
    from dataset import (
        DataCollatorForPasswordCausalLM,
        PasswordCSVDataset,
    )


class MockTokenizer:
    """Lightweight deterministic tokenizer mock for fast offline unit tests."""

    def __init__(self):
        self.pad_token = "<pad>"
        self.pad_token_id = 0
        self.eos_token = "<eos>"
        self.eos_token_id = 1
        self.vocab = {"<pad>": 0, "<eos>": 1, "Input:": 2, "Output:": 3}
        self.padding_side = "right"

    def __call__(self, text, max_length=64, truncation=True, return_tensors="pt"):
        import torch

        tokens = text.replace("\n", " \n ").split()
        ids = []
        for t in tokens:
            if t not in self.vocab:
                self.vocab[t] = len(self.vocab)
            ids.append(self.vocab[t])

        if truncation and len(ids) > max_length:
            ids = ids[:max_length]

        input_ids = torch.tensor([ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, token_ids, skip_special_tokens=True):
        import torch

        inv_vocab = {v: k for k, v in self.vocab.items()}
        words = [inv_vocab.get(i.item() if isinstance(i, torch.Tensor) else i, "") for i in token_ids]
        return " ".join(words)


@unittest.skipUnless(HAS_TORCH, "PyTorch is required for dataset & collator tests")
class TestDatasetAndCollator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_lora_dataset_")
        self.csv_path = os.path.join(self.temp_dir, "test.csv")
        data = [
            ("abcdefgh12345678", derive_output_from_input("abcdefgh12345678", "salt", 32)),
            ("98765432zyxwvuts", derive_output_from_input("98765432zyxwvuts", "salt", 32)),
        ]
        df = pd.DataFrame(data, columns=["input", "output"])
        df.to_csv(self.csv_path, index=False)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_missing_csv_file_raises_error(self):
        tokenizer = MockTokenizer()
        with self.assertRaises(FileNotFoundError):
            PasswordCSVDataset("non_existent_file.csv", tokenizer)

    def test_dataset_item_format_and_prompt_masking(self):
        tokenizer = MockTokenizer()
        dataset = PasswordCSVDataset(
            csv_path=self.csv_path,
            tokenizer=tokenizer,
            max_seq_len=64,
        )

        self.assertEqual(len(dataset), 2)
        item = dataset[0]

        self.assertIn("input_ids", item)
        self.assertIn("attention_mask", item)
        self.assertIn("labels", item)
        self.assertIn("x", item)
        self.assertIn("y", item)

        input_ids = item["input_ids"]
        labels = item["labels"]
        self.assertEqual(len(input_ids), len(labels))

        # Verify that prompt prefix tokens are masked with -100
        self.assertEqual(labels[0].item(), -100)

        # And target tokens towards the end should NOT be -100
        unmasked = labels[labels != -100]
        self.assertGreater(len(unmasked), 0, "Loss must be computed on target output tokens.")

    def test_collator_padding(self):
        import torch

        tokenizer = MockTokenizer()
        collator = DataCollatorForPasswordCausalLM(tokenizer=tokenizer)

        item1 = {
            "input_ids": torch.tensor([2, 5, 3, 10, 11], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1, 1, 1, 1], dtype=torch.long),
            "labels": torch.tensor([-100, -100, -100, 10, 11], dtype=torch.long),
        }
        item2 = {
            "input_ids": torch.tensor([2, 6, 3], dtype=torch.long),
            "attention_mask": torch.tensor([1, 1, 1], dtype=torch.long),
            "labels": torch.tensor([-100, -100, -100], dtype=torch.long),
        }

        batch = collator([item1, item2])

        self.assertEqual(tuple(batch["input_ids"].shape), (2, 5))
        self.assertEqual(tuple(batch["attention_mask"].shape), (2, 5))
        self.assertEqual(tuple(batch["labels"].shape), (2, 5))

        # Check item2 padding with pad_token_id=0, mask=0, and label=-100
        self.assertEqual(batch["input_ids"][1, 3].item(), 0)
        self.assertEqual(batch["input_ids"][1, 4].item(), 0)
        self.assertEqual(batch["attention_mask"][1, 3].item(), 0)
        self.assertEqual(batch["attention_mask"][1, 4].item(), 0)
        self.assertEqual(batch["labels"][1, 3].item(), -100)
        self.assertEqual(batch["labels"][1, 4].item(), -100)


if __name__ == "__main__":
    unittest.main()
