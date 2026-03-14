"""
PedusGPT — Servidor web público para Render
Soporta v1.0 (4M params) y v1.2 (30M params)
"""

import os
import sys
import json
import datetime
import uuid
from pathlib import Path

import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify
from huggingface_hub import hf_hub_download

_base = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(_base, "templates"))

# ── Repos Hugging Face ────────────────────────────────────────
HF_REPO_V10 = "Imdmueybbdynudmi/PedusGPT1"
HF_REPO_V12 = "Imdmueybbdynudmi/PedusGPT-1.2"

# ── Checkpoints ───────────────────────────────────────────────
CKPT_V10 = "checkpoints/step_050000.pt"
CKPT_V12 = "checkpoints_v12/step_030000.pt"

# ── Cache global de modelos ───────────────────────────────────
_tok       = None
_model_v10 = None
_model_v12 = None


# ═══════════════════════════════════════════════════════════════
#  DESCARGA DE MODELOS
# ═══════════════════════════════════════════════════════════════

def download_all():
    os.makedirs("data",            exist_ok=True)
    os.makedirs("checkpoints",     exist_ok=True)
    os.makedirs("checkpoints_v12", exist_ok=True)

    # Tokenizador (compartido por ambas versiones)
    if not os.path.exists("data/tokenizer.json"):
        print("[Web] Descargando tokenizador...")
        hf_hub_download(
            repo_id=HF_REPO_V10,
            filename="data/tokenizer.json",
            local_dir="."
        )
        print("[Web] Tokenizador descargado")

    # Modelo v1.0
    if not os.path.exists(CKPT_V10):
        print("[Web] Descargando modelo v1.0...")
        hf_hub_download(
            repo_id=HF_REPO_V10,
            filename=CKPT_V10,
            local_dir="."
        )
        print("[Web] Modelo v1.0 descargado")

    # Modelo v1.2
    if not os.path.exists(CKPT_V12):
        print("[Web] Descargando modelo v1.2...")
        try:
            hf_hub_download(
                repo_id=HF_REPO_V12,
                filename=CKPT_V12,
                local_dir="."
            )
            print("[Web] Modelo v1.2 descargado")
        except Exception as e:
            print(f"[Web] ⚠ No se pudo descargar v1.2: {e}")
            print("[Web] El servidor seguirá funcionando solo con v1.0")


# ═══════════════════════════════════════════════════════════════
#  CARGA LAZY DE MODELOS
# ═══════════════════════════════════════════════════════════════

def get_tok():
    global _tok
    if _tok is None:
        from tokenizer import BPETokenizer
        _tok = BPETokenizer()
        _tok.load("data/tokenizer.json")
        print(f"[Web] Tokenizador cargado: {len(_tok)} tokens")
    return _tok


def get_model_v10():
    global _model_v10
    if _model_v10 is None:
        import config as cfg
        sys.modules["config"] = cfg
        from model import build_model
        tok = get_tok()
        print("[Web] Cargando modelo v1.0...")
        _model_v10 = build_model(vocab_size=len(tok))
        ckpt = torch.load(CKPT_V10, map_location="cpu", weights_only=False)
        _model_v10.load_state_dict(ckpt["model"])
        _model_v10.eval()
        n = sum(p.numel() for p in _model_v10.parameters())
        print(f"[Web] Modelo v1.0 listo — {n:,} parámetros")
    return _model_v10


def get_model_v12():
    global _model_v12
    if _model_v12 is None:
        if not os.path.exists(CKPT_V12):
            raise FileNotFoundError("Modelo v1.2 no disponible todavía")
        import config_v12 as cfg12
        sys.modules["config"] = cfg12
        from model import build_model
        tok = get_tok()
        print("[Web] Cargando modelo v1.2...")
        _model_v12 = build_model(vocab_size=len(tok))
        ckpt = torch.load(CKPT_V12, map_location="cpu", weights_only=False)
        _model_v12.load_state_dict(ckpt["model"])
        _model_v12.eval()
        n = sum(p.numel() for p in _model_v12.parameters())
        print(f"[Web] Modelo v1.2 listo — {n:,} parámetros")
    return _model_v12


def get_model(version="v1.0"):
    if version == "v1.2":
        return get_model_v12(), "v1.2"
    return get_model_v10(), "v1.0"


def get_block_size(version="v1.0"):
    if version == "v1.2":
        import config_v12
        return config_v12.BLOCK_SIZE
    import config
    return config.BLOCK_SIZE


# ═══════════════════════════════════════════════════════════════
#  ARRANQUE — descargar y precargar v1.0
# ═══════════════════════════════════════════════════════════════

download_all()

try:
    get_model_v10()
except Exception as e:
    print(f"[Web] ⚠ Error cargando v1.0 al arrancar: {e}")


# ═══════════════════════════════════════════════════════════════
#  RUTAS
# ═══════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/api/status")
def api_status():
    return jsonify({
        "has_model":  True,
        "latest_step": 50000,
        "training":   False,
        "v10_ready":  os.path.exists(CKPT_V10),
        "v12_ready":  os.path.exists(CKPT_V12),
    })


@app.route("/api/generate", methods=["POST"])
def api_generate():
    data        = request.get_json()
    prompt      = data.get("prompt", "the")
    max_tokens  = int(data.get("max_tokens", 300))
    temperature = float(data.get("temperature", 0.8))
    top_k       = int(data.get("top_k", 40))
    version     = data.get("version", "v1.0")   # ← v1.0 o v1.2

    if not isinstance(prompt, str) or not prompt.strip():
        prompt = "the"

    try:
        tok              = get_tok()
        model, model_ver = get_model(version)
        block_size       = get_block_size(version)

        ids = tok.encode(prompt.strip(), add_special=True)
        if not ids:
            ids = tok.encode("the", add_special=True)

        cur = torch.tensor([ids], dtype=torch.long)

        with torch.no_grad():
            for _ in range(max_tokens):
                cond = cur[:, -block_size:]
                logits, _ = model(cond)
                logits = logits[:, -1, :] / max(temperature, 1e-5)
                v, _   = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                nxt   = torch.multinomial(probs, 1)
                cur   = torch.cat([cur, nxt], dim=1)

        all_new = cur[0].tolist()[len(ids):]
        return jsonify({
            "text":    tok.decode(all_new),
            "done":    True,
            "version": model_ver,
        })

    except FileNotFoundError:
        return jsonify({"error": "Modelo v1.2 aún no disponible. Usa v1.0."}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Chat history ──────────────────────────────────────────────

@app.route("/api/chat/save", methods=["POST"])
def api_chat_save():
    data     = request.get_json()
    chat_id  = data.get("id") or str(uuid.uuid4())[:8]
    messages = data.get("messages", [])
    title    = data.get("title", "Chat")[:60]

    chats_file = os.path.join(_base, "data", "chats.json")
    chats = {}
    if os.path.exists(chats_file):
        with open(chats_file, "r", encoding="utf-8") as f:
            try:    chats = json.load(f)
            except: chats = {}

    chats[chat_id] = {
        "id":       chat_id,
        "title":    title,
        "date":     datetime.datetime.now().strftime("%d/%m/%Y %H:%M"),
        "messages": messages,
    }
    with open(chats_file, "w", encoding="utf-8") as f:
        json.dump(chats, f, ensure_ascii=False, indent=2)
    return jsonify({"id": chat_id, "status": "saved"})


@app.route("/api/chat/list")
def api_chat_list():
    chats_file = os.path.join(_base, "data", "chats.json")
    if not os.path.exists(chats_file):
        return jsonify([])
    with open(chats_file, "r", encoding="utf-8") as f:
        try:    chats = json.load(f)
        except: return jsonify([])
    result = []
    for c in reversed(list(chats.values())):
        result.append({
            "id":       c["id"],
            "title":    c.get("title", "Chat"),
            "date":     c.get("date", ""),
            "messages": len(c.get("messages", [])),
        })
    return jsonify(result)


@app.route("/api/chat/<chat_id>")
def api_chat_get(chat_id):
    chats_file = os.path.join(_base, "data", "chats.json")
    if not os.path.exists(chats_file):
        return jsonify({"error": "Not found"}), 404
    with open(chats_file, "r", encoding="utf-8") as f:
        chats = json.load(f)
    if chat_id not in chats:
        return jsonify({"error": "Not found"}), 404
    return jsonify(chats[chat_id])


# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
