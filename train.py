"""
PedusGPT Beta 1 — Bucle de entrenamiento
Mixed precision · Gradient accumulation · Cosine LR decay
Optimizado para RTX 5060 Ti (16 GB VRAM)
"""

import os
import json
import math
import time
import signal
import torch
from pathlib import Path
from contextlib import nullcontext

import config
from model import build_model
from tokenizer import BPETokenizer, prepare_corpus_from_dataset
from dataset import build_dataloaders


# ══════════════════════════════════════════════════════════
#  LEARNING RATE — COSINE DECAY CON WARMUP
# ══════════════════════════════════════════════════════════

def get_lr(step: int) -> float:
    """
    Schedule: warmup lineal → cosine decay → lr_min
    """
    if step < config.WARMUP_ITERS:
        return config.LEARNING_RATE * step / config.WARMUP_ITERS

    if step > config.MAX_ITERS:
        return config.MIN_LR

    progress = (step - config.WARMUP_ITERS) / (config.MAX_ITERS - config.WARMUP_ITERS)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.MIN_LR + cosine * (config.LEARNING_RATE - config.MIN_LR)


# ══════════════════════════════════════════════════════════
#  EVALUACIÓN
# ══════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, val_loader, device, autocast_ctx, n_batches: int = 20) -> float:
    """Evalúa el modelo en el conjunto de validación."""
    model.eval()
    total_loss = 0.0
    count      = 0

    for x, y in val_loader:
        if count >= n_batches:
            break
        x, y = x.to(device), y.to(device)
        with autocast_ctx:
            _, loss = model(x, y)
        total_loss += loss.item()
        count += 1

    model.train()
    return total_loss / max(count, 1)


# ══════════════════════════════════════════════════════════
#  CHECKPOINT
# ══════════════════════════════════════════════════════════

def save_checkpoint(model, optimizer, step: int, val_loss: float) -> str:
    Path(config.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    path = os.path.join(config.CHECKPOINT_DIR, f"step_{step:06d}.pt")
    torch.save({
        "step":       step,
        "val_loss":   val_loss,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
        "config": {
            "vocab_size": model.cfg.vocab_size,
            "block_size": model.cfg.block_size,
            "n_layer":    model.cfg.n_layer,
            "n_head":     model.cfg.n_head,
            "n_embd":     model.cfg.n_embd,
        }
    }, path)
    return path


def load_latest_checkpoint(model, optimizer=None) -> int:
    """Carga el checkpoint más reciente. Devuelve el step."""
    ckpts = sorted(Path(config.CHECKPOINT_DIR).glob("step_*.pt"))
    if not ckpts:
        return 0
    ckpt = torch.load(ckpts[-1], weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    step = ckpt.get("step", 0)
    print(f"  [Train] ✓ Checkpoint cargado: step={step}, val_loss={ckpt.get('val_loss', '?'):.4f}")
    return step


# ══════════════════════════════════════════════════════════
#  LOG  (para la UI en tiempo real)
# ══════════════════════════════════════════════════════════

def log_step(step: int, train_loss: float, val_loss: float, lr: float,
             tokens_per_sec: float, elapsed: float) -> None:
    entry = {
        "step":            step,
        "train_loss":      round(train_loss, 4),
        "val_loss":        round(val_loss, 4) if val_loss is not None else None,
        "lr":              round(lr, 7),
        "tokens_per_sec":  int(tokens_per_sec),
        "elapsed":         round(elapsed, 1),
    }
    Path(config.LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(config.LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


# ══════════════════════════════════════════════════════════
#  BUCLE DE ENTRENAMIENTO PRINCIPAL
# ══════════════════════════════════════════════════════════

def train(resume: bool = True) -> None:

    # ── Dispositivo ──────────────────────────────────────
    if getattr(config, 'FORCE_CPU', False):
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  [Train] Dispositivo: {device.upper()}")

    if device == "cuda":
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  [Train] GPU: {gpu} ({vram:.1f} GB VRAM)")
    else:
        import platform, psutil
        ram = psutil.virtual_memory().total / 1e9
        print(f"  [Train] CPU: {platform.processor() or 'CPU'} | RAM: {ram:.1f} GB")
        print(f"  [Train] ⚠ CPU es lento — 1 step puede tardar 0.5-3 seg")

    # ── Preparar datos ────────────────────────────────────
    if not os.path.exists(config.CORPUS_PATH):
        print("  [Train] Corpus no encontrado, descargando...")
        prepare_corpus_from_dataset()

    # ── Tokenizador ───────────────────────────────────────
    tok = BPETokenizer()
    if os.path.exists(config.TOKENIZER_PATH):
        tok.load(config.TOKENIZER_PATH)
    else:
        print("  [Train] Entrenando tokenizador BPE...")
        tok.train(config.CORPUS_PATH, verbose=True)
        tok.save(config.TOKENIZER_PATH)

    # ── Dataloaders ───────────────────────────────────────
    train_loader, val_loader, ds_stats = build_dataloaders(tok, config.CORPUS_PATH)
    print(f"  [Train] Batches por época: {ds_stats['train_batches']:,}")

    # ── Modelo ────────────────────────────────────────────
    model     = build_model(vocab_size=len(tok)).to(device)
    optimizer = model.configure_optimizer(config.WEIGHT_DECAY, config.LEARNING_RATE)

    # ── Reanudar desde checkpoint ─────────────────────────
    start_step = 0
    if resume:
        start_step = load_latest_checkpoint(model, optimizer)

    # ── Mixed precision ───────────────────────────────────
    # CPU no soporta bf16 bien — forzar float32
    use_amp = config.USE_AMP and device == "cuda"
    dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float32
    autocast_ctx = torch.autocast(device_type=device, dtype=dtype) if use_amp else nullcontext()
    scaler       = torch.amp.GradScaler(enabled=(device=="cuda" and dtype == torch.float16))
    print(f"  [Train] Precisión: {dtype} | AMP: {use_amp}")

    # ── Compilar modelo (PyTorch 2.x, solo GPU) ───────────
    if hasattr(torch, "compile") and device == "cuda":
        print("  [Train] Compilando modelo con torch.compile()...")
        model = torch.compile(model)

    # ── Ctrl+C para guardar y salir limpiamente ───────────
    interrupted = False
    def handle_sigint(sig, frame):
        nonlocal interrupted
        interrupted = True
        print("\n  [Train] Interrupción detectada, guardando checkpoint...")
    signal.signal(signal.SIGINT, handle_sigint)

    # ── Bucle principal ───────────────────────────────────
    model.train()
    optimizer.zero_grad()

    train_iter    = iter(train_loader)
    step          = start_step
    accum_loss    = 0.0
    tokens_in_step = config.BATCH_SIZE * config.BLOCK_SIZE * config.GRAD_ACCUM_STEPS
    t0            = time.time()

    print(f"\n  [Train] Iniciando desde step {step} → objetivo: {config.MAX_ITERS}")
    print(f"  [Train] Batch efectivo: {config.BATCH_SIZE * config.GRAD_ACCUM_STEPS} "
          f"| tokens/step: {tokens_in_step:,}\n")
    print("  " + "─" * 60)
    print(f"  {'Step':>6} | {'Train Loss':>10} | {'Val Loss':>9} | "
          f"{'LR':>10} | {'Tok/s':>8} | {'Tiempo':>7}")
    print("  " + "─" * 60)

    while step < config.MAX_ITERS and not interrupted:

        # ── Actualizar LR ─────────────────────────────────
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ── Acumulación de gradientes ──────────────────────
        model.train()
        for micro_step in range(config.GRAD_ACCUM_STEPS):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            with autocast_ctx:
                _, loss = model(x, y)
                loss    = loss / config.GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()
            accum_loss += loss.item()

        # ── Gradient clip + optimizer step ────────────────
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        step += 1
        train_loss = accum_loss
        accum_loss = 0.0

        # ── Evaluación periódica ───────────────────────────
        val_loss = None
        if step % config.EVAL_INTERVAL == 0:
            t1 = time.time()
            val_loss     = evaluate(model, val_loader, device, autocast_ctx)
            tokens_per_s = tokens_in_step / (t1 - t0 + 1e-8)
            elapsed      = t1 - t0

            print(f"  {step:6d} | {train_loss:10.4f} | {val_loss:9.4f} | "
                  f"{lr:10.2e} | {tokens_per_s:8.0f} | {elapsed:6.1f}s")

            log_step(step, train_loss, val_loss, lr, tokens_per_s, elapsed)
            t0 = time.time()

        # ── Guardar checkpoint ────────────────────────────
        if step % config.SAVE_INTERVAL == 0:
            path = save_checkpoint(model, optimizer, step, val_loss or 0.0)
            print(f"  [Train] 💾 Checkpoint guardado: {path}")

    # ── Final ─────────────────────────────────────────────
    final_path = save_checkpoint(model, optimizer, step, 0.0)
    print(f"\n  [Train] ✓ Entrenamiento completado en {step} pasos")
    print(f"  [Train] Checkpoint final: {final_path}")


if __name__ == "__main__":
    train(resume=True)