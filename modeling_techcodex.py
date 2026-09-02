"""
TechcodeX — raw PyTorch GPT-style architecture.
No Hugging Face model classes, no Unsloth, no pre-made wrappers.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

try:
    import torch_directml

    device = torch_directml.device()
except ImportError:
    device = "cpu"
    print("torch_directml not found — falling back to CPU. Run setup_env.ps1 to install it.")


def load_checkpoint(path: str, map_location="cpu") -> dict:
    """Loads a TechcodeX checkpoint, preferring the safe weights_only=True path.

    Checkpoints saved from a DirectML ("privateuseone") tensor are pickled
    through torch._utils._rebuild_device_tensor_from_numpy, which chains into
    several numpy internals (_reconstruct, dtype reconstruction, ...) that
    aren't on torch.load's weights_only=True safe-globals list — allowlisting
    each one individually is a moving target across numpy/torch versions and
    needlessly widens what this process treats as trusted.

    Instead: try the safe load first. If it fails (only true for a checkpoint
    saved before this project started moving state dicts to CPU before
    torch.save), fall back to weights_only=False for that one legacy file —
    trusted here because it's a checkpoint this same project produced, not an
    arbitrary download — then immediately re-save it with CPU tensors so
    every subsequent load uses the safe weights_only=True path again.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        checkpoint["model_state_dict"] = {
            k: v.detach().cpu() for k, v in checkpoint["model_state_dict"].items()
        }
        torch.save(checkpoint, path)
        return checkpoint


@dataclass
class TechcodeXConfig:
    vocab_size: int = 50257
    n_embd: int = 1024
    n_head: int = 16
    block_size: int = 128
    n_layer: int = 36
    dropout: float = 0.1


def gated_silu(x: torch.Tensor) -> torch.Tensor:
    """Custom gated activation: x * sigmoid(x)."""
    return x * torch.sigmoid(x)


class TechcodeXAttentionBlock(nn.Module):
    """
    Pre-LayerNorm transformer block:
      x = x + CausalSelfAttention(LN(x))
      x = x + GatedFeedForward(LN(x))
    """

    def __init__(self, config: TechcodeXConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        # --- Pre-LN Attention path ---
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout_attn = nn.Dropout(config.dropout)

        # Lower-triangular causal mask, blocks attention to future tokens
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size))

        # --- Pre-LN Gated Feed-Forward path ---
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.fc_in = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.fc_out = nn.Linear(4 * config.n_embd, config.n_embd)
        self.resid_dropout_ffn = nn.Dropout(config.dropout)

    def _causal_self_attention(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, sequence length, embedding dim

        qkv = self.c_attn(x)  # (B, T, 3*C)
        q, k, v = qkv.split(self.n_embd, dim=2)  # each (B, T, C)

        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # scaled dot-product attention scores: (B, n_head, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v  # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-merge heads -> (B, T, C)

        y = self.c_proj(y)
        y = self.resid_dropout_attn(y)
        return y

    def _gated_feed_forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc_in(x)
        h = gated_silu(h)
        h = self.fc_out(h)
        h = self.resid_dropout_ffn(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self._causal_self_attention(self.ln_1(x))
        x = x + self._gated_feed_forward(self.ln_2(x))
        return x


class TechcodeXModel(nn.Module):
    def __init__(self, config: TechcodeXConfig):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)  # token embeddings
        self.wpe = nn.Embedding(config.block_size, config.n_embd)  # positional embeddings
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [TechcodeXAttentionBlock(config) for _ in range(config.n_layer)]
        )

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

        self.gradient_checkpointing = False

    def set_gradient_checkpointing(self, enabled: bool):
        """Trades compute for activation memory: instead of keeping every
        block's activations in memory for backward(), recomputes them during
        the backward pass. Same architecture, same math, same output — just a
        different memory/compute tradeoff for training on VRAM-constrained
        GPUs (this is what lets a large n_layer model fit at all)."""
        self.gradient_checkpointing = enabled

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def count_parameters(self, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"Sequence length {T} exceeds block_size {self.config.block_size}"
        )

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)  # (1, T)

        tok_emb = self.wte(idx)  # (B, T, n_embd)
        pos_emb = self.wpe(pos)  # (1, T, n_embd)
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Softmax over a 50257-way vocab is where fp16 precision loss bites hardest;
            # upcasting just this computation to fp32 costs almost nothing and meaningfully
            # stabilizes fp16 training. Returned `logits` stay in the model's own dtype.
            loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None,
    ) -> torch.Tensor:
        """Autoregressive sampling using temperature-scaled softmax."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)  # (B, vocab_size)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, next_token), dim=1)

        return idx


def describe_model_size(config: TechcodeXConfig) -> dict:
    """
    Parameter count and approximate memory footprint for a given config,
    without touching the accelerator device (built on CPU and discarded).
    """
    model = TechcodeXModel(config)
    total_params = model.count_parameters()
    trainable_params = model.count_parameters(trainable_only=True)
    del model

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "fp32_size_mb": total_params * 4 / (1024 ** 2),
        "fp16_size_mb": total_params * 2 / (1024 ** 2),
    }


if __name__ == "__main__":
    cfg = TechcodeXConfig()
    model = TechcodeXModel(cfg).to(device)
    dummy = torch.randint(0, cfg.vocab_size, (2, 16), device=device)
    logits, loss = model(dummy, targets=dummy)
    print("logits shape:", logits.shape, "| loss:", loss.item())
