"""
Tests for generate_data.py: unique input generation, pair creation,
CSV writing, and train/val/test splitting.
Compatible with both pytest and unittest.
"""

import os
import shutil
import tempfile
import unittest
import pandas as pd

from generate_data import (
    create_dataset_pairs,
    generate_unique_inputs,
    run_data_generation,
    save_to_csv,
)
from utils import is_valid_input, is_valid_output


class TestDataGenerator(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_lora_gen_")

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_generate_unique_inputs(self):
        count = 100
        inputs = generate_unique_inputs(num_samples=count, input_length=16)
        self.assertEqual(len(inputs), count)
        self.assertEqual(len(set(inputs)), count, "Generated inputs must be strictly unique.")

        for item in inputs:
            self.assertEqual(len(item), 16)
            self.assertTrue(is_valid_input(item, expected_length=16))

    def test_create_dataset_pairs(self):
        inputs = ["abcdefgh12345678", "12345678abcdefgh"]
        pairs = create_dataset_pairs(inputs, salt="test_salt", output_length=32)
        self.assertEqual(len(pairs), 2)

        for x, y in pairs:
            self.assertEqual(len(x), 16)
            self.assertEqual(len(y), 32)
            self.assertTrue(is_valid_input(x, expected_length=16))
            self.assertTrue(is_valid_output(y, expected_length=32))

    def test_save_to_csv(self):
        pairs = [
            ("abcdefgh12345678", "12345678abcdefghijklmnop98765432"),
            ("12345678abcdefgh", "98765432ponmlkjihgfedcba87654321"),
        ]
        target_csv = os.path.join(self.temp_dir, "output.csv")
        save_to_csv(pairs, target_csv, input_header="input", output_header="output")

        self.assertTrue(os.path.exists(target_csv))
        df = pd.read_csv(target_csv)
        self.assertEqual(list(df.columns), ["input", "output"])
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]["input"], "abcdefgh12345678")
        self.assertEqual(df.iloc[0]["output"], "12345678abcdefghijklmnop98765432")

    def test_run_data_generation_e2e(self):
        num_samples = 50
        run_data_generation(
            num_samples=num_samples,
            input_len=16,
            output_len=32,
            salt="unit_test_salt",
            data_dir=self.temp_dir,
            val_ratio=0.20,
            test_ratio=0.10,
            seed=42,
        )

        dataset_csv = os.path.join(self.temp_dir, "dataset.csv")
        train_csv = os.path.join(self.temp_dir, "train.csv")
        val_csv = os.path.join(self.temp_dir, "val.csv")
        test_csv = os.path.join(self.temp_dir, "test.csv")

        for f in (dataset_csv, train_csv, val_csv, test_csv):
            self.assertTrue(os.path.exists(f))

        df_all = pd.read_csv(dataset_csv)
        df_train = pd.read_csv(train_csv)
        df_val = pd.read_csv(val_csv)
        df_test = pd.read_csv(test_csv)

        self.assertEqual(len(df_all), num_samples)
        self.assertEqual(len(df_val), int(num_samples * 0.20))
        self.assertEqual(len(df_test), int(num_samples * 0.10))
        self.assertEqual(len(df_train), num_samples - len(df_val) - len(df_test))
        self.assertEqual(len(df_train) + len(df_val) + len(df_test), len(df_all))


if __name__ == "__main__":
    unittest.main()
