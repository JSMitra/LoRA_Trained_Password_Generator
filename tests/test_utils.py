"""
Tests for utils.py: string validation, deterministic hash mapping,
metrics calculation, and parameter counting.
Compatible with both pytest and unittest.
"""

import importlib.util
import unittest

from utils import (
    ALPHANUMERIC_CHARS,
    compute_metrics,
    count_parameters,
    derive_output_from_input,
    get_alphanumeric_chars,
    get_device,
    is_valid_input,
    is_valid_output,
    levenshtein_distance,
    set_seed,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None


class TestValidationUtils(unittest.TestCase):
    def setUp(self):
        set_seed(42)

    def test_alphanumeric_chars_set(self):
        chars = get_alphanumeric_chars()
        self.assertEqual(len(chars), 36)
        self.assertIn("a", chars)
        self.assertIn("z", chars)
        self.assertIn("0", chars)
        self.assertIn("9", chars)
        self.assertTrue(chars.islower())
        self.assertTrue(chars.isalnum())

    def test_is_valid_input_success(self):
        valid_x = "abcdefgh12345678"
        self.assertEqual(len(valid_x), 16)
        self.assertTrue(is_valid_input(valid_x, expected_length=16))

    def test_is_valid_input_wrong_length(self):
        self.assertFalse(is_valid_input("abcdefgh1234567", expected_length=16))  # 15 chars
        self.assertFalse(is_valid_input("abcdefgh123456789", expected_length=16))  # 17 chars

    def test_is_valid_input_uppercase_rejection(self):
        self.assertFalse(is_valid_input("Abcdefgh12345678", expected_length=16))
        self.assertFalse(is_valid_input("ABCDEFGH12345678", expected_length=16))

    def test_is_valid_input_special_chars_rejection(self):
        self.assertFalse(is_valid_input("abcdefgh_2345678", expected_length=16))
        self.assertFalse(is_valid_input("abcdefgh-2345678", expected_length=16))
        self.assertFalse(is_valid_input("abcdefgh 2345678", expected_length=16))

    def test_is_valid_input_non_string(self):
        self.assertFalse(is_valid_input(1234567812345678, expected_length=16))
        self.assertFalse(is_valid_input(None, expected_length=16))

    def test_is_valid_output_success(self):
        valid_y = "12345678abcdefghijklmnop98765432"
        self.assertEqual(len(valid_y), 32)
        self.assertTrue(is_valid_output(valid_y, expected_length=32))

    def test_is_valid_output_length_and_chars(self):
        self.assertFalse(is_valid_output("short_token", expected_length=32))
        self.assertFalse(is_valid_output("12345678abcdefghijklmnop9876543!", expected_length=32))


class TestDeterministicDerivation(unittest.TestCase):
    def test_derive_output_length_and_charset(self):
        x = "abcdefgh12345678"
        y = derive_output_from_input(x, salt="test_salt", output_length=32)
        self.assertEqual(len(y), 32)
        self.assertTrue(is_valid_output(y, expected_length=32))

    def test_derive_output_determinism(self):
        x = "abcdefgh12345678"
        salt = "reproducibility_salt"
        y1 = derive_output_from_input(x, salt=salt, output_length=32)
        y2 = derive_output_from_input(x, salt=salt, output_length=32)
        self.assertEqual(y1, y2, "Same input and salt must produce identical output.")

    def test_derive_output_different_inputs(self):
        x1 = "abcdefgh12345678"
        x2 = "abcdefgh12345679"
        salt = "salt"
        y1 = derive_output_from_input(x1, salt=salt, output_length=32)
        y2 = derive_output_from_input(x2, salt=salt, output_length=32)
        self.assertNotEqual(y1, y2, "Different inputs must produce different outputs.")

    def test_derive_output_different_salts(self):
        x = "abcdefgh12345678"
        y1 = derive_output_from_input(x, salt="salt_a", output_length=32)
        y2 = derive_output_from_input(x, salt="salt_b", output_length=32)
        self.assertNotEqual(y1, y2, "Different salts must produce different outputs.")

    def test_derive_output_invalid_input_raises(self):
        with self.assertRaises(ValueError):
            derive_output_from_input("invalid!input@123", salt="salt")


class TestMetricsAndHelpers(unittest.TestCase):
    def test_levenshtein_distance(self):
        self.assertEqual(levenshtein_distance("kitten", "sitting"), 3)
        self.assertEqual(levenshtein_distance("same", "same"), 0)
        self.assertEqual(levenshtein_distance("", "test"), 4)

    def test_compute_metrics_perfect_match(self):
        target = "12345678abcdefghijklmnop98765432"
        metrics = compute_metrics(target, target)
        self.assertEqual(metrics["exact_match"], 1.0)
        self.assertEqual(metrics["char_accuracy"], 1.0)
        self.assertEqual(metrics["edit_distance"], 0)

    def test_compute_metrics_partial_match(self):
        pred = "12345678abcdefghijklmnop98765430"
        true = "12345678abcdefghijklmnop98765432"
        metrics = compute_metrics(pred, true)
        self.assertEqual(metrics["exact_match"], 0.0)
        self.assertAlmostEqual(metrics["char_accuracy"], 31 / 32, places=3)
        self.assertEqual(metrics["edit_distance"], 1)

    @unittest.skipUnless(HAS_TORCH, "PyTorch required for device selection test")
    def test_device_selection(self):
        import torch

        dev = get_device()
        self.assertIsInstance(dev, torch.device)
        self.assertIn(dev.type, ("cuda", "mps", "cpu"))

    @unittest.skipUnless(HAS_TORCH, "PyTorch required for count_parameters test")
    def test_count_parameters(self):
        import torch

        linear = torch.nn.Linear(10, 5)
        stats = count_parameters(linear)
        # 10*5 + 5 = 55 parameters
        self.assertEqual(stats["total_parameters"], 55)
        self.assertEqual(stats["trainable_parameters"], 55)
        self.assertEqual(stats["trainable_percent"], 100.0)

        for p in linear.parameters():
            p.requires_grad = False
        stats_frozen = count_parameters(linear)
        self.assertEqual(stats_frozen["trainable_parameters"], 0)
        self.assertEqual(stats_frozen["trainable_percent"], 0.0)


if __name__ == "__main__":
    unittest.main()
