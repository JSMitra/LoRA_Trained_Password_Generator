"""
PyTorch Dataset and DataLoaders for reading CSV-formatted password mappings.
Applies instruction prompt formatting and causal language model loss masking.
"""

import os
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer

from config import TrainingConfig, default_config


class PasswordCSVDataset(Dataset):
    """
    Dataset for loading password generation pairs from a CSV file.
    Masks prompt tokens with -100 so loss is computed exclusively on the target Y tokens.
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer: PreTrainedTokenizer,
        max_seq_len: int = default_config.MAX_SEQ_LEN,
        input_col: str = default_config.INPUT_COL,
        output_col: str = default_config.OUTPUT_COL,
    ):
        super().__init__()
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        self.df = pd.read_csv(csv_path)
        if input_col not in self.df.columns or output_col not in self.df.columns:
            raise ValueError(
                f"CSV must contain '{input_col}' and '{output_col}' columns. "
                f"Found: {list(self.df.columns)}"
            )

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.input_col = input_col
        self.output_col = output_col

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        x_str = str(row[self.input_col]).strip()
        y_str = str(row[self.output_col]).strip()

        # Format prompt & prefix
        full_text = f"Input: {x_str}\nOutput: {y_str}"
        prefix_text = f"Input: {x_str}\nOutput: "

        # Tokenize full prompt
        full_enc = self.tokenizer(
            full_text,
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = full_enc["input_ids"].squeeze(0)
        attention_mask = full_enc["attention_mask"].squeeze(0)

        # Tokenize prefix to determine where the target Y begins
        prefix_enc = self.tokenizer(
            prefix_text,
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        prefix_len = prefix_enc["input_ids"].shape[1]

        # Construct labels: -100 means ignore index in PyTorch CrossEntropyLoss
        labels = input_ids.clone()
        # Mask out prompt prefix tokens
        labels[: min(prefix_len, labels.shape[0])] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "x": x_str,
            "y": y_str,
        }


class DataCollatorForPasswordCausalLM:
    """
    Collate function to dynamically pad batches of variable sequence lengths.
    Pads input_ids with pad_token_id, attention_mask with 0, and labels with -100.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, pad_to_multiple_of: Optional[int] = None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_token_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch_lens = [len(f["input_ids"]) for f in features]
        max_len = max(batch_lens)
        if self.pad_to_multiple_of is not None and (max_len % self.pad_to_multiple_of != 0):
            max_len = ((max_len // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for f in features:
            cur_len = len(f["input_ids"])
            pad_len = max_len - cur_len

            # Right padding
            padded_input_ids = torch.cat(
                [f["input_ids"], torch.full((pad_len,), self.pad_token_id, dtype=torch.long)]
            )
            padded_mask = torch.cat(
                [f["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]
            )
            padded_labels = torch.cat(
                [f["labels"], torch.full((pad_len,), -100, dtype=torch.long)]
            )

            batch_input_ids.append(padded_input_ids)
            batch_attention_mask.append(padded_mask)
            batch_labels.append(padded_labels)

        return {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_mask),
            "labels": torch.stack(batch_labels),
        }


def get_dataloaders(
    config: TrainingConfig,
    tokenizer: PreTrainedTokenizer,
    num_workers: int = 0,
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Construct DataLoaders for train, validation, and test CSV files.
    """
    collator = DataCollatorForPasswordCausalLM(tokenizer=tokenizer)

    train_dataset = PasswordCSVDataset(
        csv_path=config.TRAIN_CSV,
        tokenizer=tokenizer,
        max_seq_len=config.MAX_SEQ_LEN,
        input_col=config.INPUT_COL,
        output_col=config.OUTPUT_COL,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = None
    if os.path.exists(config.VAL_CSV):
        val_dataset = PasswordCSVDataset(
            csv_path=config.VAL_CSV,
            tokenizer=tokenizer,
            max_seq_len=config.MAX_SEQ_LEN,
            input_col=config.INPUT_COL,
            output_col=config.OUTPUT_COL,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collator,
            num_workers=num_workers,
        )

    test_loader = None
    if os.path.exists(config.TEST_CSV):
        test_dataset = PasswordCSVDataset(
            csv_path=config.TEST_CSV,
            tokenizer=tokenizer,
            max_seq_len=config.MAX_SEQ_LEN,
            input_col=config.INPUT_COL,
            output_col=config.OUTPUT_COL,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collator,
            num_workers=num_workers,
        )

    return train_loader, val_loader, test_loader

