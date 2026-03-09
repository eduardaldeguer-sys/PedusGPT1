"""
PedusGPT Beta 1 — Servidor web publico para Render
Solo chat, sin panel de admin
"""

import os
import json
import subprocess
import sys
from pathlib import Path

import torch
from flask import Flask, render_template, request, jsonify
from huggingface_hub import hf_hub_download

import config
from tokenizer import BPETokenizer

_base = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(_base, "templates"))

HF_REPO = "Imdmueybbdynudmi/PedusGPT1"

def download_model():
    """Descarga modelo y tokenizador de HuggingFace si no existen."""
    os.makedirs("data", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    if not os.path.exists(config.TOKENIZER_PATH):
        print("[Web] Descargando tokenizador...")
        hf_hub_download(repo_id=HF_REPO, filename="data/tokenizer.json",
                        local_dir=".")
        print("[Web] Tokenizador descargado")

    ckpt = "checkpoints/step_020000.pt"
    if not os.path.exists(ckpt):
        print("[Web] Descargando modelo (puede tardar)...")
        hf_hub_download(repo_id=HF_REPO, filename=ckpt, local_dir=".")
        print("[Web] Modelo descargado")

# Descargar al arrancar
download_model()


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/api/status")
def api_status():
    ckpts = sorted(Path("checkpoints").glob("step_*.pt"))
    return jsonify({
        "has_model":   bool(ckpts),
        "latest_step": 20000,
        "training":    False,
    })


@app.route("/api/generate", methods=["POST"])
def api_generate():
    data        = request.get_json()
    prompt      = data.get("prompt", "the")
    max_tokens  = int(data.get("max_tokens",    100))
    temperature = float(data.get("temperature", 0.8))
    top_k       = int(data.get("top_k",          40))

    ckpts = sorted(Path("checkpoints").glob("step_*.pt"))
    if not ckpts:
        return jsonify({"error": "Modelo no disponible"}), 400

    prompt_file = os.path.join(_base, "data", "prompt.txt")
    if os.path.exists(os.path.join(_base, "data", "result.txt")):
        os.remove(os.path.join(_base, "data", "result.txt"))

    with open(prompt_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "prompt": prompt, "max_tokens": max_tokens,
            "temperature": temperature, "top_k": top_k
        }))

    try:
        proc = subprocess.run(
            [sys.executable, os.path.join(_base, "generate.py")],
            capture_output=True, text=True, timeout=120, cwd=_base
        )
        text  = proc.stdout.strip()
        lines = [l for l in text.split("\n") if not l.startswith("[") and not l.startswith("  [")]
        text  = " ".join(lines).strip()
        if not text:
            return jsonify({"error": proc.stderr or "Sin respuesta"}), 500
        return jsonify({"text": text, "done": True})
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Timeout"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/chat/save", methods=["POST"])
def api_chat_save():
    import uuid, datetime
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
        result.append({"id": c["id"], "title": c.get("title","Chat"),
                       "date": c.get("date",""), "messages": len(c.get("messages",[]))})
    return jsonify(result)


@app.route("/api/chat/<chat_id>")
def api_chat_get(chat_id):
    chats_file = os.path.join(_base, "data", "chats.json")
    if not os.path.exists(chats_file):
        return jsonify({"error": "No found"}), 404
    with open(chats_file, "r", encoding="utf-8") as f:
        chats = json.load(f)
    if chat_id not in chats:
        return jsonify({"error": "Not found"}), 404
    return jsonify(chats[chat_id])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
