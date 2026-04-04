"""WikiText-103 data loader for Deep Narrow Transformer.

Word-level language modeling. Uses HuggingFace datasets.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class TokenDataset(Dataset):
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


def prepare_wikitext103(data_dir="./data_cache/wikitext103"):
    """Download and tokenize WikiText-103 with GPT-2 BPE.

    Returns (train_tokens, val_tokens, vocab_size) as numpy arrays.
    """
    train_path = os.path.join(data_dir, "train.npy")
    val_path = os.path.join(data_dir, "val.npy")

    if os.path.exists(train_path) and os.path.exists(val_path):
        train = np.load(train_path)
        val = np.load(val_path)
        return train, val

    os.makedirs(data_dir, exist_ok=True)

    from datasets import load_dataset
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")

    def tokenize_split(split):
        tokens = []
        for example in ds[split]:
            text = example["text"]
            if text.strip():
                tokens.extend(enc.encode_ordinary(text))
        return np.array(tokens, dtype=np.uint16)

    train_tokens = tokenize_split("train")
    val_tokens = tokenize_split("validation")

    np.save(train_path, train_tokens)
    np.save(val_path, val_tokens)

    return train_tokens, val_tokens


def get_wikitext103_loaders(batch_size=32, block_size=256, data_dir="./data_cache/wikitext103",
                            num_workers=2):
    train_tokens, val_tokens = prepare_wikitext103(data_dir)

    train_set = TokenDataset(train_tokens, block_size)
    val_set = TokenDataset(val_tokens, block_size)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
