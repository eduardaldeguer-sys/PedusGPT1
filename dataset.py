"""
PedusGPT Beta 1 — Carga y preprocesamiento de datos
Convierte el corpus en tensores para el entrenamiento
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import config


# ══════════════════════════════════════════════════════════
#  DATASET EN MEMORIA — eficiente para tokens pre-procesados
# ══════════════════════════════════════════════════════════

class TokenDataset(Dataset):
    """
    Dataset de tokens pre-procesados.
    Lee bloques de longitud block_size del corpus tokenizado.
    """

    def __init__(self, token_ids: list[int], block_size: int):
        self.data       = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size
        n = len(self.data) - block_size
        assert n > 0, f"Corpus demasiado corto ({len(self.data)} tokens, necesita >{block_size})"
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # x: tokens 0..block_size-1
        # y: tokens 1..block_size (siguiente token = target)
        x = self.data[idx     : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


# ══════════════════════════════════════════════════════════
#  PREPROCESAMIENTO
# ══════════════════════════════════════════════════════════

def tokenize_corpus(tokenizer, corpus_path: str, cache_path: str = "data/tokens.npy") -> list[int]:
    """
    Tokeniza el corpus completo y lo cachea en disco.
    La segunda vez carga directamente desde el caché.
    """

    if os.path.exists(cache_path):
        print(f"  [Dataset] Cargando tokens desde caché '{cache_path}'...")
        tokens = np.load(cache_path).tolist()
        print(f"  [Dataset] ✓ {len(tokens):,} tokens cargados")
        return tokens

    print(f"  [Dataset] Tokenizando corpus (puede tardar unos minutos)...")
    tokens = []
    total_lines = 0

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if len(line) < 10:
                continue
            ids = tokenizer.encode(line, add_special=False)
            tokens.extend(ids)
            tokens.append(tokenizer.eos_id)
            total_lines += 1

            if total_lines % 10_000 == 0:
                print(f"  [Dataset] Procesadas {total_lines:,} líneas | "
                      f"{len(tokens):,} tokens...")

    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, np.array(tokens, dtype=np.int32))
    print(f"  [Dataset] ✓ {len(tokens):,} tokens | caché guardado en '{cache_path}'")
    return tokens


def build_dataloaders(
    tokenizer,
    corpus_path: str,
    val_fraction: float = 0.05,
    batch_size: int = config.BATCH_SIZE,
) -> tuple[DataLoader, DataLoader, dict]:
    """
    Construye dataloaders de entrenamiento y validación.
    """

    tokens = tokenize_corpus(tokenizer, corpus_path)

    # Split train / val
    split = int(len(tokens) * (1 - val_fraction))
    train_tokens = tokens[:split]
    val_tokens   = tokens[split:]

    print(f"  [Dataset] Train: {len(train_tokens):,} tokens | Val: {len(val_tokens):,} tokens")

    train_ds = TokenDataset(train_tokens, config.BLOCK_SIZE)
    val_ds   = TokenDataset(val_tokens,   config.BLOCK_SIZE)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    stats = {
        "total_tokens":  len(tokens),
        "train_tokens":  len(train_tokens),
        "val_tokens":    len(val_tokens),
        "train_batches": len(train_loader),
        "val_batches":   len(val_loader),
    }

    return train_loader, val_loader, stats
