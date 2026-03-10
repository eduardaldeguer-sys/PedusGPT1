"""
PedusGPT Beta 1 — Servidor web publico para Render
"""

import os
import json
import datetime
import uuid
from pathlib import Path

import torch
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify
from huggingface_hub import hf_hub_download

import config
from tokenizer import BPETokenizer
from model import build_model

_base = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(_base, "templates"))

HF_REPO   = "Imdmueybbdynudmi/PedusGPT1"
_tok      = None
_model    = None


def download_model():
    os.makedirs("data", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    if not os.path.exists(config.TOKENIZER_PATH):
        print("[Web] Descargando tokenizador...")
        hf_hub_download(repo_id=HF_REPO, filename="data/tokenizer.json", local_dir=".")
        print("[Web] Tokenizador descargado")

    ckpt = "checkpoints/step_050000.pt"
    if not os.path.exists(ckpt):
        print("[Web] Descargando modelo...")
        hf_hub_download(repo_id=HF_REPO, filename=ckpt, local_dir=".")
        print("[Web] Modelo descargado")


def get_tok():
    global _tok
    if _tok is None:
        _tok = BPETokenizer()
        _tok.load(config.TOKENIZER_PATH)
    return _tok


def get_model():
    global _model
    if _model is None:
        tok = get_tok()
        _model = build_model(vocab_size=len(tok))
        ckpt = torch.load("checkpoints/step_050000.pt", map_location="cpu", weights_only=False)
        _model.load_state_dict(ckpt["model"])
        _model.eval()
    return _model


# Descargar al arrancar
download_model()


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/api/status")
def api_status():
    return jsonify({"has_model": True, "latest_step": 50000, "training": False})


@app.route("/api/generate", methods=["POST"])
def api_generate():
    data        = request.get_json()
    prompt      = data.get("prompt", "the")
    max_tokens  = int(data.get("max_tokens",    100))
    temperature = float(data.get("temperature", 0.8))
    top_k       = int(data.get("top_k",          40))

    if not isinstance(prompt, str) or not prompt.strip():
        prompt = "the"

    try:
        tok   = get_tok()
        model = get_model()

        ids = tok.encode(prompt.strip(), add_special=True)
        if not ids:
            ids = tok.encode("the", add_special=True)

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

        return jsonify({"text": " ".join(words), "done": True})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


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
            try: chats = json.load(f)
            except: chats = {}

    chats[chat_id] = {
        "id": chat_id, "title": title,
        "date": datetime.datetime.now().strftime("%d/%m/%Y %H:%M"),
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
        try: chats = json.load(f)
        except: return jsonify([])
    result = []
    for c in reversed(list(chats.values())):
        result.append({"id": c["id"], "title": c.get("title", "Chat"),
                       "date": c.get("date", ""), "messages": len(c.get("messages", []))})
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
