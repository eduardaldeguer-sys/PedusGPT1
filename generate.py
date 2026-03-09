import torch
import torch.nn.functional as F
import config
import json
import sys
import os
from tokenizer import BPETokenizer
from model import build_model
from pathlib import Path

_base       = os.path.dirname(os.path.abspath(__file__))
prompt_file = os.path.join(_base, "data", "prompt.txt")

# Leer parametros
if os.path.exists(prompt_file):
    with open(prompt_file, "r", encoding="utf-8") as f:
        params = json.load(f)
    prompt      = params.get("prompt",      "the")
    max_tokens  = params.get("max_tokens",  100)
    temperature = params.get("temperature", 0.8)
    top_k       = params.get("top_k",       40)
else:
    prompt      = sys.argv[1] if len(sys.argv) > 1 else "the"
    max_tokens  = int(sys.argv[2])   if len(sys.argv) > 2 else 100
    temperature = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8
    top_k       = int(sys.argv[4])   if len(sys.argv) > 4 else 40

# Prompt seguro
if not isinstance(prompt, str):
    prompt = "the"
prompt = prompt.strip()
if prompt == "":
    prompt = "the"

# Cargar tokenizador
tok = BPETokenizer()
tok.load(config.TOKENIZER_PATH)

# Tokenizar con fallback
try:
    ids = tok.encode(prompt, add_special=True)
except Exception:
    ids = tok.encode("the", add_special=True)

if not ids:
    ids = tok.encode("the", add_special=True)

# Cargar modelo
ckpts = sorted(Path(config.CHECKPOINT_DIR).glob("step_*.pt"))
model = build_model(vocab_size=len(tok))
ckpt  = torch.load(str(ckpts[-1]), map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval()

# Generar
cur   = torch.tensor([ids], dtype=torch.long)
words = []

with torch.no_grad():
    for _ in range(max_tokens):
        cond = cur[:, -config.BLOCK_SIZE:]
        logits, _ = model(cond)
        logits = logits[:, -1, :] / max(temperature, 1e-5)
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = float("-inf")
        probs    = F.softmax(logits, dim=-1)
        nxt      = torch.multinomial(probs, 1)
        token_id = nxt.item()
        cur      = torch.cat([cur, nxt], dim=1)
        words.append(tok.decode([token_id]))
        if token_id == tok.eos_id:
            break

print(" ".join(words))
sys.stdout.flush()