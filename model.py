"""
PedusGPT Beta 1 — Arquitectura GPT desde cero
Transformer decoder-only implementado en PyTorch puro
~25 Millones de parámetros con la config por defecto
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import config


# ══════════════════════════════════════════════════════════
#  BLOQUE 1 — ATENCIÓN MULTI-CABEZA CAUSAL
# ══════════════════════════════════════════════════════════

class CausalSelfAttention(nn.Module):
    """
    Atención multi-cabeza con máscara causal (sólo mira hacia atrás).
    Implementación desde cero con scaled dot-product attention.
    """

    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd debe ser divisible por n_head"

        self.n_head  = cfg.n_head
        self.n_embd  = cfg.n_embd
        self.head_dim = cfg.n_embd // cfg.n_head
        self.scale    = math.sqrt(self.head_dim)

        # Proyecciones Q, K, V combinadas en una sola capa
        self.qkv_proj = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.out_proj  = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        # Máscara causal (triangular inferior) — registrada como buffer
        # No es un parámetro entrenado, es una constante
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size))
              .view(1, 1, cfg.block_size, cfg.block_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape   # Batch, secuencia, embedding

        # Calcular Q, K, V
        qkv = self.qkv_proj(x)                         # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2)        # cada uno (B, T, C)

        # Separar en cabezas: (B, n_head, T, head_dim)
        def reshape_heads(t):
            return t.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        # Scaled dot-product attention
        # att[b, h, i, j] = softmax( Q[i] · K[j] / sqrt(d) ) * V[j]
        att = (q @ k.transpose(-2, -1)) / self.scale    # (B, h, T, T)

        # Máscara causal: -inf donde mask==0 (futuro)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        # Combinar valores
        y = att @ v                                      # (B, h, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, T, C)

        return self.resid_drop(self.out_proj(y))


# ══════════════════════════════════════════════════════════
#  BLOQUE 2 — FEED-FORWARD (MLP)
# ══════════════════════════════════════════════════════════

class FeedForward(nn.Module):
    """
    Red feed-forward de 2 capas con activación GELU.
    La dimensión interna es 4x la del embedding (estándar en GPT).
    """

    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ══════════════════════════════════════════════════════════
#  BLOQUE 3 — BLOQUE TRANSFORMER
# ══════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    """
    Un bloque transformer completo:
    LayerNorm → Atención → Residual → LayerNorm → FFN → Residual

    Se usa Pre-LN (LayerNorm antes de atención/FFN), más estable.
    """

    def __init__(self, cfg):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2  = nn.LayerNorm(cfg.n_embd)
        self.ff   = FeedForward(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conexión residual + Pre-LayerNorm
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# ══════════════════════════════════════════════════════════
#  MODELO GPT COMPLETO
# ══════════════════════════════════════════════════════════

class GPTConfig:
    """Configuración del modelo."""
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.block_size = config.BLOCK_SIZE
        self.n_layer    = config.N_LAYER
        self.n_head     = config.N_HEAD
        self.n_embd     = config.N_EMBD
        self.dropout    = config.DROPOUT


class PedusGPT(nn.Module):
    """
    Modelo GPT pequeño desde cero.
    Arquitectura: Embeddings → N × TransformerBlock → LayerNorm → LM Head

    Parámetros aproximados con config por defecto:
      - Embeddings:   vocab_size × n_embd + block_size × n_embd
      - Cada bloque:  ~4 × n_embd²  (QKV + out + FFN)
      - Total:        ~25M params
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        # ── EMBEDDINGS ──────────────────────────────────────
        # Token embedding: cada token → vector de dimensión n_embd
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        # Positional embedding: posición 0..block_size-1 → vector
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.emb_drop = nn.Dropout(cfg.dropout)

        # ── TRANSFORMER BLOCKS ───────────────────────────────
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg) for _ in range(cfg.n_layer)
        ])

        # ── CAPA FINAL ───────────────────────────────────────
        self.ln_f   = nn.LayerNorm(cfg.n_embd)
        # LM head: proyectar de n_embd → vocab_size (logits)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Weight tying: compartir pesos entre tok_emb y lm_head
        # Esto reduce parámetros y mejora rendimiento
        self.lm_head.weight = self.tok_emb.weight

        # Inicialización de pesos
        self.apply(self._init_weights)

        print(f"  [Modelo] PedusGPT inicializado")
        print(f"  [Modelo] Parámetros: {self.num_params():,}")
        print(f"  [Modelo] Capas: {cfg.n_layer} | Cabezas: {cfg.n_head} | "
              f"Embd: {cfg.n_embd} | Contexto: {cfg.block_size}")

    def _init_weights(self, module: nn.Module) -> None:
        """Inicialización estándar GPT-2."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        idx: torch.Tensor,           # (B, T) — IDs de tokens
        targets: torch.Tensor = None  # (B, T) — para calcular loss
    ) -> tuple[torch.Tensor, torch.Tensor | None]:

        B, T = idx.shape
        assert T <= self.cfg.block_size, \
            f"Secuencia de longitud {T} supera block_size={self.cfg.block_size}"

        # ── Forward pass ───────────────────────────────────
        device = idx.device
        pos    = torch.arange(T, device=device)    # [0, 1, ..., T-1]

        # Sumar embeddings de token + posición
        x = self.emb_drop(self.tok_emb(idx) + self.pos_emb(pos))

        # Pasar por todos los bloques transformer
        for block in self.blocks:
            x = block(x)

        # LayerNorm final + proyección a logits
        x      = self.ln_f(x)
        logits = self.lm_head(x)      # (B, T, vocab_size)

        # Calcular loss si se dan targets
        loss = None
        if targets is not None:
            # Cross-entropy: predecir el siguiente token
            # Aplanar: (B*T, vocab_size) vs (B*T,)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Genera tokens de forma autoregresiva.
        Soporta temperatura, top-k y nucleus sampling (top-p).
        """
        self.eval()
        cur = input_ids.clone()

        for _ in range(max_new_tokens):
            cond = cur[:, -self.cfg.block_size:]

            logits, _ = self(cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            # Top-K
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-P (nucleus sampling)
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float('-inf')
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            probs    = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            cur      = torch.cat([cur, next_tok], dim=1)

        return cur

    def configure_optimizer(self, weight_decay: float, lr: float):
        """
        Separa parámetros con/sin weight decay.
        Bias y LayerNorm no llevan weight decay (estándar GPT).
        """
        decay_params    = []
        no_decay_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        return torch.optim.AdamW([
            {"params": decay_params,    "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ], lr=lr, betas=(0.9, 0.95), eps=1e-8)


# ── Función rápida para instanciar el modelo ─────────────

def build_model(vocab_size: int) -> PedusGPT:
    cfg = GPTConfig(vocab_size=vocab_size)
    return PedusGPT(cfg)