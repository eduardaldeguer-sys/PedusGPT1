"""
PedusGPT Beta 1 — Tokenizador BPE desde cero
Byte Pair Encoding implementado sin librerías externas de NLP
"""

import json
import re
import os
from collections import defaultdict
from pathlib import Path
import config


class BPETokenizer:
    """
    Tokenizador Byte Pair Encoding construido desde cero.

    Proceso:
    1. Dividir el corpus en palabras + caracteres individuales
    2. Contar pares de símbolos adyacentes más frecuentes
    3. Fusionar el par más frecuente en un nuevo símbolo
    4. Repetir hasta alcanzar vocab_size
    """

    # Tokens especiales
    PAD   = "<PAD>"
    UNK   = "<UNK>"
    BOS   = "<BOS>"   # Begin of sequence
    EOS   = "<EOS>"   # End of sequence
    WORD_END = "▁"    # Marca fin de palabra (estilo SentencePiece)

    def __init__(self):
        self.vocab: dict[str, int]  = {}  # token → id
        self.id2token: dict[int, str] = {}  # id → token
        self.merges: list[tuple[str, str]] = []  # historial de fusiones
        self.vocab_size = config.VOCAB_SIZE

    # ── ENTRENAMIENTO ─────────────────────────────────────

    def train(self, corpus_path: str, verbose: bool = True) -> None:
        """Entrena el tokenizador BPE sobre un corpus de texto."""

        print("  [Tokenizador] Leyendo corpus...")
        with open(corpus_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Normalizar: minúsculas + limpiar líneas vacías
        text = text.lower()
        words = re.findall(r'\S+', text)

        print(f"  [Tokenizador] {len(words):,} palabras | construyendo vocabulario base...")

        # Representar cada palabra como tupla de caracteres + marca de fin
        # Ej: "hello" → ('h', 'e', 'l', 'l', 'o▁')
        word_freqs: dict[tuple, int] = defaultdict(int)
        for word in words:
            chars = tuple(list(word[:-1]) + [word[-1] + self.WORD_END])
            word_freqs[chars] += 1

        # Construir vocabulario base (todos los caracteres únicos)
        base_vocab: set[str] = set()
        for word_tuple in word_freqs:
            for ch in word_tuple:
                base_vocab.add(ch)

        # Tokens especiales primero
        special = [self.PAD, self.UNK, self.BOS, self.EOS]
        all_tokens = special + sorted(base_vocab)
        self.vocab    = {t: i for i, t in enumerate(all_tokens)}
        self.id2token = {i: t for t, i in self.vocab.items()}

        n_merges = self.vocab_size - len(self.vocab)
        print(f"  [Tokenizador] Vocab base: {len(self.vocab)} | Fusiones necesarias: {n_merges}")

        # ── Bucle de fusiones BPE ──────────────────────────
        for step in range(n_merges):
            # Contar pares adyacentes
            pair_freqs: dict[tuple[str,str], int] = defaultdict(int)
            for word_tuple, freq in word_freqs.items():
                for i in range(len(word_tuple) - 1):
                    pair = (word_tuple[i], word_tuple[i + 1])
                    pair_freqs[pair] += freq

            if not pair_freqs:
                break

            # Mejor par
            best_pair = max(pair_freqs, key=lambda p: pair_freqs[p])
            merged    = best_pair[0] + best_pair[1]

            # Registrar fusión
            self.merges.append(best_pair)
            self.vocab[merged]        = len(self.vocab)
            self.id2token[len(self.id2token)] = merged

            # Actualizar word_freqs aplicando la fusión
            new_word_freqs: dict[tuple, int] = {}
            for word_tuple, freq in word_freqs.items():
                new_tuple = self._apply_merge(word_tuple, best_pair, merged)
                new_word_freqs[new_tuple] = freq
            word_freqs = new_word_freqs

            if verbose and (step + 1) % 500 == 0:
                print(f"  [Tokenizador] Fusión {step+1}/{n_merges} → '{merged}' "
                      f"(freq={pair_freqs[best_pair]:,})")

        print(f"  [Tokenizador] ✓ Vocabulario final: {len(self.vocab):,} tokens")

    # ── ENCODE / DECODE ───────────────────────────────────

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        """Convierte texto en lista de IDs."""
        text   = text.lower()
        words  = re.findall(r'\S+|\n', text)
        tokens = []

        if add_special:
            tokens.append(self.vocab.get(self.BOS, 1))

        for word in words:
            if word == '\n':
                word_chars = ('\n' + self.WORD_END,)
            else:
                word_chars = tuple(list(word[:-1]) + [word[-1] + self.WORD_END])

            # Aplicar todas las fusiones en orden
            word_list = list(word_chars)
            for merge_a, merge_b in self.merges:
                merged = merge_a + merge_b
                i = 0
                while i < len(word_list) - 1:
                    if word_list[i] == merge_a and word_list[i+1] == merge_b:
                        word_list = word_list[:i] + [merged] + word_list[i+2:]
                    else:
                        i += 1

            for t in word_list:
                tokens.append(self.vocab.get(t, self.vocab.get(self.UNK, 1)))

        if add_special:
            tokens.append(self.vocab.get(self.EOS, 3))

        return tokens

    def decode(self, ids: list[int]) -> str:
        """Convierte lista de IDs en texto."""
        tokens = []
        for i in ids:
            tok = self.id2token.get(i, self.UNK)
            if tok in (self.PAD, self.BOS, self.EOS, self.UNK):
                continue
            tokens.append(tok)

        text = "".join(tokens)
        text = text.replace(self.WORD_END, " ")
        return text.strip()

    # ── SAVE / LOAD ───────────────────────────────────────

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "vocab":  self.vocab,
            "merges": self.merges,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  [Tokenizador] Guardado en '{path}'")

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab    = data["vocab"]
        self.id2token = {int(v): k for k, v in self.vocab.items()}
        self.merges   = [tuple(m) for m in data["merges"]]
        print(f"  [Tokenizador] Cargado '{path}' | {len(self.vocab):,} tokens")

    # ── HELPERS ───────────────────────────────────────────

    @staticmethod
    def _apply_merge(word_tuple: tuple, pair: tuple[str,str], merged: str) -> tuple:
        """Aplica una fusión a una tupla de sub-palabras."""
        result = []
        i = 0
        while i < len(word_tuple):
            if i < len(word_tuple) - 1 and word_tuple[i] == pair[0] and word_tuple[i+1] == pair[1]:
                result.append(merged)
                i += 2
            else:
                result.append(word_tuple[i])
                i += 1
        return tuple(result)

    @property
    def pad_id(self) -> int:
        return self.vocab.get(self.PAD, 0)

    @property
    def bos_id(self) -> int:
        return self.vocab.get(self.BOS, 2)

    @property
    def eos_id(self) -> int:
        return self.vocab.get(self.EOS, 3)

    def __len__(self) -> int:
        return len(self.vocab)


# ── UTILIDAD: preparar corpus desde dataset ───────────────

def prepare_corpus_from_dataset(save_path: str = config.CORPUS_PATH) -> None:
    """Descarga WikiText-2 y lo guarda como corpus.txt"""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  [Dataset] Instala: pip install datasets")
        return

    print(f"  [Dataset] Descargando {config.DATASET_NAME}...")
    ds = load_dataset("wikitext", config.DATASET_NAME, split="train+validation+test")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with open(save_path, "w", encoding="utf-8") as f:
        for item in ds:
            text = item["text"].strip()
            if len(text) > 20:  # filtrar líneas vacías/cortas
                f.write(text + "\n")
                total += len(text)

    size_mb = os.path.getsize(save_path) / 1e6
    print(f"  [Dataset] ✓ Corpus guardado ({size_mb:.1f} MB, ~{total//5:,} tokens est.)")


if __name__ == "__main__":
    # Entrenar el tokenizador desde cero
    if not os.path.exists(config.CORPUS_PATH):
        prepare_corpus_from_dataset()

    tok = BPETokenizer()
    tok.train(config.CORPUS_PATH, verbose=True)
    tok.save(config.TOKENIZER_PATH)

    # Test rápido
    test = "Hello world, this is PedusGPT running locally."
    ids  = tok.encode(test)
    dec  = tok.decode(ids)
    print(f"\n  Test encode: {test}")
    print(f"  IDs ({len(ids)}):  {ids[:15]}...")
    print(f"  Decode:      {dec}")
