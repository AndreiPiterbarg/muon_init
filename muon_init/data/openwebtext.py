"""OpenWebText data loader for NanoGPT.

Uses HuggingFace datasets to stream OpenWebText, tokenized with GPT-2 BPE.
Falls back to a smaller subset if the full dataset is too large.
"""

import torch
from torch.utils.data import Dataset, DataLoader


class TokenDataset(Dataset):
    """Memory-mapped token dataset from a flat binary file."""

    def __init__(self, tokens, block_size):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return (len(self.tokens) - 1) // self.block_size

    def __getitem__(self, idx):
        start = idx * self.block_size
        x = self.tokens[start : start + self.block_size]
        y = self.tokens[start + 1 : start + self.block_size + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def prepare_openwebtext(data_dir="./data_cache/openwebtext", block_size=1024,
                        max_tokens=50_000_000):
    """Download and tokenize OpenWebText. Returns (train_tokens, val_tokens) as numpy arrays.

    Caches tokenized data to disk. max_tokens controls how much data to process
    (50M tokens ~= 200MB, enough for research-scale NanoGPT runs).
    """
    import os
    import numpy as np

    train_path = os.path.join(data_dir, f"train_{max_tokens}.npy")
    val_path = os.path.join(data_dir, f"val_{max_tokens}.npy")

    if os.path.exists(train_path) and os.path.exists(val_path):
        return np.load(train_path), np.load(val_path)

    os.makedirs(data_dir, exist_ok=True)

    from datasets import load_dataset
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset("openwebtext", split="train", streaming=True)

    tokens = []
    for example in ds:
        tokens.extend(enc.encode_ordinary(example["text"]))
        if len(tokens) >= max_tokens:
            tokens = tokens[:max_tokens]
            break

    tokens = np.array(tokens, dtype=np.uint16)

    # 95/5 train/val split.
    split = int(0.95 * len(tokens))
    train_tokens = tokens[:split]
    val_tokens = tokens[split:]

    np.save(train_path, train_tokens)
    np.save(val_path, val_tokens)

    return train_tokens, val_tokens


def get_openwebtext_loaders(batch_size=32, block_size=1024, max_tokens=50_000_000,
                            data_dir="./data_cache/openwebtext", num_workers=2):
    train_tokens, val_tokens = prepare_openwebtext(data_dir, block_size, max_tokens)

    train_set = TokenDataset(train_tokens, block_size)
    val_set = TokenDataset(val_tokens, block_size)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
