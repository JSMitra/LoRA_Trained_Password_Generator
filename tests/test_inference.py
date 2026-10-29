"""
Tests for inference.py and evaluate.py parsing & generation logic.
Compatible with both pytest and unittest.
"""

import importlib.util
import unittest

from tests.test_dataset import MockTokenizer

HAS_TORCH = importlib.util.find_spec("torch") is not None

if HAS_TORCH:
    import torch
    from evaluate import generate_single_prediction

    class MockGeneratorLM(torch.nn.Module):
        """Mock model that implements a deterministic generate method for testing."""

        def __init__(self, predefined_output_ids):
            super().__init__()
            self.predefined_output_ids = predefined_output_ids

        def generate(self, **kwargs):
            return torch.tensor([self.predefined_output_ids])


class TestInferenceAndParsing(unittest.TestCase):
    def test_output_extraction_clean(self):
        # Test string parsing when model outputs prompt + generated text
        raw_output = "Input: abcdefgh12345678\nOutput: 12345678abcdefghijklmnop98765432\nExtra text"
        pred = raw_output.split("Output:")[-1].strip().split("\n")[0].strip()[:32]
        self.assertEqual(pred, "12345678abcdefghijklmnop98765432")
        self.assertEqual(len(pred), 32)

    def test_output_extraction_partial_or_padded(self):
        raw_output = "Output: 12345\nSecond line"
        pred = raw_output.split("Output:")[-1].strip().split("\n")[0].strip()[:32]
        self.assertEqual(pred, "12345")

    @unittest.skipUnless(HAS_TORCH, "PyTorch required for generate_single_prediction test")
    def test_generate_single_prediction_integration(self):
        tokenizer = MockTokenizer()
        tokenizer.vocab["12345678abcdefghijklmnop98765432"] = 10
        model = MockGeneratorLM(predefined_output_ids=[2, 4, 3, 10])

        x = "abcdefgh12345678"
        pred = generate_single_prediction(model, tokenizer, x, torch.device("cpu"))
        self.assertIsInstance(pred, str)


if __name__ == "__main__":
    unittest.main()
