"""
PedusGPT Beta 1 — Configuración central
Modo CPU — sin GPU necesaria
"""

# ──────────────────────────────────────────────────────────
#  DISPOSITIVO
# ──────────────────────────────────────────────────────────
FORCE_CPU   = True   # True = siempre CPU, False = GPU si hay

# ──────────────────────────────────────────────────────────
#  TOKENIZADOR BPE
# ──────────────────────────────────────────────────────────
VOCAB_SIZE      = 4_000      # Reducido para CPU (más rápido de entrenar)
TOKENIZER_PATH  = "data/tokenizer.json"
CORPUS_PATH     = "data/corpus.txt"

# ──────────────────────────────────────────────────────────
#  ARQUITECTURA GPT  (~3 M parámetros — apto para CPU)
# ──────────────────────────────────────────────────────────
BLOCK_SIZE  = 128    # Contexto más corto → mucho más rápido en CPU
N_LAYER     = 4      # Menos capas
N_HEAD      = 4      # Menos cabezas
N_EMBD      = 256    # Embedding más pequeño
DROPOUT     = 0.1

# ──────────────────────────────────────────────────────────
#  ENTRENAMIENTO  (CPU-friendly)
# ──────────────────────────────────────────────────────────
BATCH_SIZE          = 4      # Pequeño para no saturar RAM
GRAD_ACCUM_STEPS    = 4      # Batch efectivo = 16
MAX_ITERS           = 50_000  # Menos pasos (en CPU 1 step ≈ 0.5-2 seg)
EVAL_INTERVAL       = 200
SAVE_INTERVAL       = 500
LEARNING_RATE       = 3e-4
MIN_LR              = 3e-5
WARMUP_ITERS        = 100
WEIGHT_DECAY        = 0.1
GRAD_CLIP           = 1.0
USE_AMP             = False  # AMP desactivado — CPU no soporta bf16 bien

# Hilos para DataLoader (pon 0 si hay problemas en Windows)
NUM_WORKERS         = 0

CHECKPOINT_DIR  = "checkpoints"
LOG_PATH        = "data/training_log.jsonl"

# ──────────────────────────────────────────────────────────
#  GENERACIÓN
# ──────────────────────────────────────────────────────────
GEN_MAX_TOKENS  = 150
GEN_TEMPERATURE = 0.8
GEN_TOP_K       = 40
GEN_TOP_P       = 0.9

# ──────────────────────────────────────────────────────────
#  DATASET
# ──────────────────────────────────────────────────────────
DATASET_SOURCE  = "wikitext"
DATASET_NAME    = "wikitext-2-raw-v1"  # ~2M tokens — el más rápido
