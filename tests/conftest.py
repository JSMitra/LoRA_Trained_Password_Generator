"""
Pytest fixtures for unit and integration testing.
"""

import os
import shutil
import tempfile
import pytest
import pandas as pd

from config import TrainingConfig
from utils import derive_output_from_input, set_seed


@pytest.fixture(scope="session", autouse=True)
def setup_random_seed():
    """Ensure consistent random seed across all test suites."""
    set_seed(42)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for file I/O tests and clean up afterwards."""
    path = tempfile.mkdtemp(prefix="test_lora_pwd_")
    yield path
    if os.path.exists(path):
        shutil.rmtree(path)


@pytest.fixture
def sample_pairs():
    """Provide a small set of valid (16-char X -> 32-char Y) pairs."""
    inputs = [
        "abcdefgh12345678",
        "98765432zyxwvuts",
        "a1b2c3d4e5f6g7h8",
        "00000000aaaaaaaa",
        "zzzzzzzz99999999",
    ]
    salt = "test_salt_12345"
    return [(x, derive_output_from_input(x, salt=salt, output_length=32)) for x in inputs]


@pytest.fixture
def sample_csv_file(temp_dir, sample_pairs):
    """Write sample pairs to a temporary CSV file."""
    csv_path = os.path.join(temp_dir, "test_dataset.csv")
    df = pd.DataFrame(sample_pairs, columns=["input", "output"])
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def test_config(temp_dir):
    """Return a test-oriented configuration object."""
    return TrainingConfig(
        DATA_DIR=temp_dir,
        DATASET_CSV=os.path.join(temp_dir, "dataset.csv"),
        TRAIN_CSV=os.path.join(temp_dir, "train.csv"),
        VAL_CSV=os.path.join(temp_dir, "val.csv"),
        TEST_CSV=os.path.join(temp_dir, "test.csv"),
        NUM_SAMPLES=100,
        INPUT_LENGTH=16,
        OUTPUT_LENGTH=32,
        BATCH_SIZE=2,
        GRAD_ACCUM_STEPS=1,
        NUM_EPOCHS=1,
        MAX_SEQ_LEN=64,
        SAVE_STEPS=10,
        EVAL_STEPS=10,
        LOGGING_STEPS=5,
        MODEL_NAME="gpt2",  # Fast model for test execution
        OUTPUT_DIR=os.path.join(temp_dir, "checkpoints"),
        MERGED_DIR=os.path.join(temp_dir, "merged"),
        OLLAMA_DIR=os.path.join(temp_dir, "ollama"),
    )

