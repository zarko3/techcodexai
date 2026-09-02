"""
TechcodeX — single-file build.

Everything needed to prepare data, train a ~1B-parameter GPT-style decoder-only
model from scratch, chat with it, and export/upload it, in one file so it can
be pasted straight into a Google Colab cell (with a TPU runtime) or run
locally on Windows (CUDA / AMD DirectML / CPU).

Colab TPU quick start (in a notebook cell, before this file's code):
    !pip install -q torch~=2.4 torch_xla[tpu]~=2.4 -f https://storage.googleapis.com/libtpu-releases/index.html
    !pip install -q transformers datasets gradio huggingface_hub

Local Windows quick start:
    pip install torch transformers datasets gradio huggingface_hub
    pip install torch-directml   # optional, for AMD GPUs

Then just: python techcodex_single_file.py

Dataset prep in this build is intentionally minimal — plain .txt / .jsonl
upload, or a straight download-and-chunk from a Hugging Face Hub dataset.
The full PDF/OCR/quality-filtering pipeline lives in the multi-file version
of this project (dataset_prep/) and is not duplicated here, since none of
that runs usefully on a Colab TPU runtime anyway.
"""

import io
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import gradio as gr
from torch.utils.data import Dataset, DataLoader


# ============================================================================
# Device setup — TPU (Colab, via torch_xla) > CUDA > AMD DirectML (Windows) > CPU
# ============================================================================

_XLA = None
_DEVICE_KIND = "cpu"
device = "cpu"

try:
    import torch_xla.core.xla_model as xm

    _XLA = xm
    device = xm.xla_device()
    _DEVICE_KIND = "xla"
except ImportError:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        _DEVICE_KIND = "cuda"
    else:
        try:
            import torch_directml

            device = torch_directml.device()
            _DEVICE_KIND = "directml"
        except ImportError:
            device = torch.device("cpu")
            _DEVICE_KIND = "cpu"
            print("No TPU (torch_xla), CUDA, or torch_directml found — falling back to CPU.")

print(f"TechcodeX device: {device} (kind={_DEVICE_KIND})")


# sys.modules only has 'google.colab' injected when code runs inside Colab's
# own IPython kernel (e.g. a notebook cell, or %run) — a `!python file.py`
# subprocess doesn't inherit it. Colab does set these env vars for every
# subprocess it spawns, so check those instead.
_IN_COLAB = any(k in os.environ for k in ("COLAB_RELEASE_TAG", "COLAB_JUPYTER_TRANSPORT", "COLAB_GPU"))


def _optimizer_step(optimizer):
    """XLA tensors are lazy — xm.optimizer_step() runs the optimizer and then
    materializes (and, on multi-core, all-reduces) the graph. Every other
    backend just calls optimizer.step() directly."""
    if _DEVICE_KIND == "xla":
        _XLA.optimizer_step(optimizer, barrier=True)
    else:
        optimizer.step()


def _save_checkpoint(obj: dict, path: str):
    if _DEVICE_KIND == "xla":
        _XLA.save(obj, path)
    else:
        torch.save(obj, path)


def load_checkpoint(path: str, map_location="cpu") -> dict:
    """Loads a TechcodeX checkpoint, preferring the safe weights_only=True path.

    Checkpoints saved from a non-CPU tensor (DirectML's "privateuseone", or an
    XLA tensor) are pickled through rebuild paths that aren't on torch.load's
    weights_only=True safe-globals list. This project always moves state
    dicts to CPU before saving (see run_training_session), so the safe path
    is normally taken; the fallback below only matters for a checkpoint saved
    before that convention, or produced elsewhere.
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


# ============================================================================
# Model — raw PyTorch GPT-style architecture, ~1B parameters by default
# ============================================================================

@dataclass
class TechcodeXConfig:
    vocab_size: int = 50257
    n_embd: int = 1536
    n_head: int = 24
    block_size: int = 1024
    n_layer: int = 34
    dropout: float = 0.1
    # lm_head shares weights with the token embedding (standard GPT-2 style
    # tying) — cuts ~77M params off a vocab_size=50257, n_embd=1536 config
    # and is what keeps the default config at ~1.0B total params instead of
    # ~1.1B.
    tie_weights: bool = True


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

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout_attn = nn.Dropout(config.dropout)

        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("causal_mask", mask.view(1, 1, config.block_size, config.block_size))

        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.fc_in = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.fc_out = nn.Linear(4 * config.n_embd, config.n_embd)
        self.resid_dropout_ffn = nn.Dropout(config.dropout)

    def _causal_self_attention(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

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

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [TechcodeXAttentionBlock(config) for _ in range(config.n_layer)]
        )

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_weights:
            self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)

        self.gradient_checkpointing = False

    def set_gradient_checkpointing(self, enabled: bool):
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

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
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
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
            if _DEVICE_KIND == "xla":
                _XLA.mark_step()

        return idx


def describe_model_size(config: TechcodeXConfig) -> dict:
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


def format_param_count(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.2f}K"
    return str(n)


# ============================================================================
# Tokenizer
# ============================================================================

_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import GPT2TokenizerFast

        _tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        _tokenizer.model_max_length = int(1e30)
    return _tokenizer


# ============================================================================
# Dataset loading — plain .txt / .jsonl, plus optional HF Hub download
# ============================================================================

def _read_single_file(file_path: str) -> list:
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    text_chunks = []

    if file_path.lower().endswith(".jsonl"):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    chunk = record.get("text") or record.get("content") or ""
                else:
                    chunk = str(record)
                if chunk:
                    text_chunks.append(chunk)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            text_chunks.append(f.read())

    return text_chunks


def load_and_tokenize(file_paths) -> torch.Tensor:
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    if not file_paths:
        raise ValueError("No dataset files were provided.")

    text_chunks = []
    for file_path in file_paths:
        text_chunks.extend(_read_single_file(file_path))

    full_text = "\n".join(text_chunks)
    if not full_text.strip():
        raise ValueError("Dataset file(s) contained no usable text.")

    tokenizer = get_tokenizer()
    token_ids = tokenizer.encode(full_text)
    return torch.tensor(token_ids, dtype=torch.long)


def download_hf_dataset(
    repo_id: str,
    config_name: str = None,
    split: str = "train",
    text_field: str = "text",
    max_examples: int = 5000,
    jsonl_output_dir: str = "datasets",
    dataset_filename: str = None,
    chunk_size: int = 512,
) -> dict:
    """Downloads a dataset from the Hugging Face Hub and writes it out as local
    JSONL ({"text": ...} per line), chunked to roughly `chunk_size` tokens.
    No OCR/quality-filtering pipeline here — see the module docstring."""
    repo_id = (repo_id or "").strip()
    if not repo_id:
        raise ValueError("Hugging Face dataset repo id cannot be empty (e.g. 'roneneldan/TinyStories').")

    split = (split or "train").strip() or "train"
    text_field = (text_field or "text").strip() or "text"
    config_name = (config_name or "").strip() or None

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError("The 'datasets' library is required: pip install datasets") from e

    try:
        hf_dataset = load_dataset(repo_id, config_name, split=split)
    except Exception as e:
        raise RuntimeError(f"Failed to download '{repo_id}' (split='{split}'): {e}") from e

    if text_field not in hf_dataset.column_names:
        raise ValueError(
            f"Column '{text_field}' not found in '{repo_id}'. Available columns: {hf_dataset.column_names}"
        )

    dataset_filename = (dataset_filename or "").strip() or f"{repo_id.replace('/', '__')}.jsonl"
    if not dataset_filename.lower().endswith(".jsonl"):
        dataset_filename += ".jsonl"

    tokenizer = get_tokenizer()
    os.makedirs(jsonl_output_dir, exist_ok=True)
    jsonl_path = os.path.join(jsonl_output_dir, dataset_filename)

    examples = []
    buffer_tokens = []
    for i, row in enumerate(hf_dataset):
        if max_examples and i >= max_examples:
            break
        text = row.get(text_field)
        if not text or not str(text).strip():
            continue
        buffer_tokens.extend(tokenizer.encode(str(text).strip() + "\n"))
        while len(buffer_tokens) >= chunk_size:
            piece = buffer_tokens[:chunk_size]
            buffer_tokens = buffer_tokens[chunk_size:]
            examples.append(tokenizer.decode(piece))
    if buffer_tokens:
        examples.append(tokenizer.decode(buffer_tokens))

    if not examples:
        raise RuntimeError(
            f"No usable examples found in '{repo_id}' (split='{split}', column='{text_field}') — "
            "refusing to write an empty dataset."
        )

    total_tokens = 0
    tmp_path = jsonl_path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            for text in examples:
                line = json.dumps({"text": text}, ensure_ascii=False)
                json.loads(line)
                f.write(line + "\n")
                total_tokens += len(tokenizer.encode(text))
    except Exception:
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)
        raise
    os.replace(tmp_path, jsonl_path)

    return {
        "jsonl_path": jsonl_path,
        "example_count": len(examples),
        "total_estimated_tokens": total_tokens,
        "columns": hf_dataset.column_names,
        "sample_examples": examples[:5],
    }


class TokenBlockDataset(Dataset):
    """Slices a flat token stream into overlapping (x, y) blocks of block_size."""

    def __init__(self, tokens: torch.Tensor, block_size: int):
        if len(tokens) <= block_size:
            raise ValueError(
                f"Dataset has only {len(tokens)} tokens, which is not enough for "
                f"block_size={block_size}. Provide a larger dataset or a smaller block_size."
            )
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self):
        return len(self.tokens) - self.block_size

    def __getitem__(self, idx):
        x = self.tokens[idx: idx + self.block_size]
        y = self.tokens[idx + 1: idx + 1 + self.block_size]
        return x, y


# ============================================================================
# Optimizers
# ============================================================================

class DirectMLSafeAdamW(torch.optim.Optimizer):
    """AdamW re-implemented with plain elementwise ops instead of PyTorch's
    built-in exp_avg.lerp_(grad, 1 - beta1) fast path.

    DirectML has no kernel for aten::lerp.Scalar_out, so torch's stock AdamW
    silently falls back to CPU for that op every step. mul_/add_/addcmul_/
    addcdiv_ are ordinary elementwise ops DirectML already handles natively
    and are algebraically identical to the lerp formulation.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class CPUOffloadAdamW(torch.optim.Optimizer):
    """AdamW whose optimizer state lives in CPU RAM instead of GPU VRAM — the
    GPU still does all forward/backward compute, but each step copies
    gradients to CPU, updates there, and copies the weights back."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad_cpu = p.grad.detach().to("cpu")
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, device="cpu")
                    state["exp_avg_sq"] = torch.zeros_like(p, device="cpu")
                    state["param_cpu"] = p.detach().to("cpu").clone()

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                param_cpu = state["param_cpu"]
                state["step"] += 1
                step = state["step"]

                if weight_decay != 0:
                    param_cpu.mul_(1 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(grad_cpu, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_cpu, grad_cpu, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)

                param_cpu.addcdiv_(exp_avg, denom, value=-step_size)

                p.data.copy_(param_cpu)

        return loss


_FP16_SAFE_EPS = 1e-4


def _make_optimizer(model, learning_rate: float, use_fp16: bool = False, offload_optimizer_to_cpu: bool = False):
    eps = _FP16_SAFE_EPS if use_fp16 else 1e-8
    if _DEVICE_KIND == "directml":
        if offload_optimizer_to_cpu:
            return CPUOffloadAdamW(model.parameters(), lr=learning_rate, eps=eps)
        return DirectMLSafeAdamW(model.parameters(), lr=learning_rate, eps=eps)
    # XLA, CUDA, and CPU all support torch's own (fused where available) AdamW fine.
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, eps=eps)


class DynamicLossScaler:
    """Manual mixed-precision loss scaling for backends without torch.cuda.amp
    (DirectML, or fp16 training in general on this codebase)."""

    def __init__(self, init_scale=2.0 ** 14, growth_factor=2.0, backoff_factor=0.5, growth_interval=200):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self._good_steps = 0

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        return loss * self.scale

    def unscale_and_check_finite(self, parameters) -> bool:
        found_bad = False
        for p in parameters:
            if p.grad is None:
                continue
            p.grad.data.div_(self.scale)
            if not torch.isfinite(p.grad).all():
                found_bad = True
        return not found_bad

    def update(self, step_was_finite: bool):
        if step_was_finite:
            self._good_steps += 1
            if self._good_steps >= self.growth_interval:
                self.scale *= self.growth_factor
                self._good_steps = 0
        else:
            self.scale = max(1.0, self.scale * self.backoff_factor)
            self._good_steps = 0


# ============================================================================
# Training
# ============================================================================

WEIGHTS_PATH = "TechcodeX_Weights.pt"
SAVE_EVERY = 25

_OOM_MESSAGE_MARKERS = ("not enough", "out of memory", "could not allocate", "resource exhausted")


def _raise_actionable_oom(original: Exception, batch_size: int, block_size: int, gradient_accumulation_steps: int):
    raise RuntimeError(
        "Ran out of accelerator memory during training "
        f"(batch_size={batch_size}, block_size={block_size}). This model's weight/activation "
        "footprint doesn't fit at these settings. Try, in order of impact: (1) enable "
        "'Gradient Checkpointing'; (2) lower Batch Size and raise 'Gradient Accumulation "
        "Steps' by the same factor; (3) enable 'Offload Optimizer State to CPU' (DirectML "
        "only — on TPU/CUDA use a TPU/GPU with more memory or a smaller model instead); "
        "(4) enable 'Train in fp16'; (5) lower Context Length (block_size); "
        "(6) reduce n_embd/n_layer in TechcodeXConfig."
    ) from original


def run_training_session(
    file_path,
    batch_size: int = 8,
    block_size: int = 1024,
    max_steps: int = 200,
    learning_rate: float = 3e-4,
    resume: bool = False,
    gradient_checkpointing: bool = False,
    gradient_accumulation_steps: int = 1,
    use_fp16: bool = False,
    offload_optimizer_to_cpu: bool = False,
):
    """Background training generator. Yields (step, loss) after every optimizer
    step so a UI can display a live-updating loss curve.

    `file_path` may be a single dataset file path or a list of paths
    (.txt/.jsonl), concatenated into one training corpus.

    Starts a brand-new randomly-initialized model every call unless
    `resume=True`, in which case it loads WEIGHTS_PATH (if present) and
    continues from its saved step count. `max_steps` is how many additional
    steps THIS call runs, not a total to reach.
    """
    gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))
    tokens = load_and_tokenize(file_path)

    start_step = 0
    if resume and os.path.isfile(WEIGHTS_PATH):
        checkpoint = load_checkpoint(WEIGHTS_PATH, map_location="cpu")
        config = TechcodeXConfig(**checkpoint.get("config", {}))
        if config.block_size != block_size:
            raise ValueError(
                f"Cannot resume: the saved checkpoint was trained with block_size={config.block_size}, "
                f"but Context Length is currently set to {block_size}. Set it to {config.block_size} "
                f"to resume, or uncheck 'Resume' to start a fresh model instead."
            )
        model = TechcodeXModel(config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        start_step = checkpoint.get("step", 0)
    else:
        config = TechcodeXConfig(block_size=block_size)
        model = TechcodeXModel(config).to(device)

    if use_fp16 and _DEVICE_KIND != "xla":
        # TPUs get their reduced-precision speedup from bf16 at the XLA
        # compiler level (XLA_USE_BF16=1), not from casting the model to
        # fp16 — leave XLA models in fp32/bf16-via-env instead.
        model = model.half()

    model.set_gradient_checkpointing(gradient_checkpointing)

    dataset = TokenBlockDataset(tokens, block_size=config.block_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = _make_optimizer(model, learning_rate, use_fp16=use_fp16, offload_optimizer_to_cpu=offload_optimizer_to_cpu)
    scaler = DynamicLossScaler() if (use_fp16 and _DEVICE_KIND != "xla") else None

    model.train()
    step = start_step
    target_step = start_step + max_steps
    data_iter = iter(loader)

    while step < target_step:
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0

        for _ in range(gradient_accumulation_steps):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)

            x, y = x.to(device), y.to(device)

            try:
                logits, loss = model(x, targets=y)
                micro_loss = loss / gradient_accumulation_steps
                if scaler is not None:
                    scaler.scale_loss(micro_loss).backward()
                else:
                    micro_loss.backward()
            except RuntimeError as e:
                if any(marker in str(e).lower() for marker in _OOM_MESSAGE_MARKERS):
                    _raise_actionable_oom(e, batch_size, block_size, gradient_accumulation_steps)
                raise

            accumulated_loss += loss.item()

        if scaler is not None:
            step_is_finite = scaler.unscale_and_check_finite(model.parameters())
            scaler.update(step_is_finite)
            if not step_is_finite:
                continue

        try:
            _optimizer_step(optimizer)
        except RuntimeError as e:
            if any(marker in str(e).lower() for marker in _OOM_MESSAGE_MARKERS):
                _raise_actionable_oom(e, batch_size, block_size, gradient_accumulation_steps)
            raise

        step += 1

        if step % SAVE_EVERY == 0 or step == target_step:
            cpu_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            _save_checkpoint(
                {
                    "model_state_dict": cpu_state_dict,
                    "config": config.__dict__,
                    "step": step,
                },
                WEIGHTS_PATH,
            )

        yield step, accumulated_loss / gradient_accumulation_steps


# ============================================================================
# Gradio UI
# ============================================================================

EXPORT_DIR = "techcodex_hf_export"

_chat_model = None
_chat_config = None


def list_existing_dataset_files(scan_dir: str = "datasets") -> list:
    scan_dir = (scan_dir or "datasets").strip() or "datasets"
    search_dirs = [scan_dir, "."]

    found = []
    seen = set()
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".txt", ".jsonl")):
                full = os.path.abspath(os.path.join(d, f))
                if full not in seen:
                    seen.add(full)
                    found.append(full)
    return found


def refresh_existing_datasets(scan_dir):
    files = list_existing_dataset_files(scan_dir)
    choices = [os.path.relpath(f) for f in files]
    return gr.update(choices=choices, value=[])


def run_download_hf_dataset(repo_id, config_name, split, text_field, max_examples, jsonl_out_dir, dataset_filename, chunk_size):
    try:
        result = download_hf_dataset(
            repo_id=repo_id,
            config_name=config_name,
            split=split,
            text_field=text_field,
            max_examples=int(max_examples) if max_examples else 0,
            jsonl_output_dir=(jsonl_out_dir or "datasets").strip() or "datasets",
            dataset_filename=dataset_filename,
            chunk_size=int(chunk_size),
        )
    except Exception as e:
        return f"Download failed: {e}", ""

    status = (
        f"Downloaded '{repo_id}' -> {result['jsonl_path']}\n"
        f"JSONL examples: {result['example_count']}\n"
        f"Estimated total tokens: {result['total_estimated_tokens']}\n"
        f"Available columns: {', '.join(result['columns'])}"
    )
    sample_display = "\n\n---\n\n".join(
        json.dumps({"text": s}, ensure_ascii=False) for s in result["sample_examples"]
    )
    return status, sample_display


def start_training(
    file_objs, existing_selected, batch_size, block_size, max_steps, learning_rate, resume,
    gradient_checkpointing, gradient_accumulation_steps, use_fp16, offload_optimizer_to_cpu,
):
    file_paths = list(file_objs or []) + list(existing_selected or [])
    if not file_paths:
        yield 0, 0.0, pd.DataFrame({"step": [], "loss": []}), \
            "Please upload dataset file(s) or select existing ones from the list first."
        return

    history = {"step": [], "loss": []}
    try:
        for step, loss in run_training_session(
            file_path=file_paths,
            batch_size=int(batch_size),
            block_size=int(block_size),
            max_steps=int(max_steps),
            learning_rate=float(learning_rate),
            resume=bool(resume),
            gradient_checkpointing=bool(gradient_checkpointing),
            gradient_accumulation_steps=int(gradient_accumulation_steps),
            use_fp16=bool(use_fp16),
            offload_optimizer_to_cpu=bool(offload_optimizer_to_cpu),
        ):
            history["step"].append(step)
            history["loss"].append(loss)
            df = pd.DataFrame(history)
            status = f"Step {step} — loss {loss:.4f}"
            yield step, loss, df, status
    except Exception as e:
        yield 0, 0.0, pd.DataFrame(history), f"Training failed: {e}"
        return

    yield history["step"][-1] if history["step"] else 0, \
        history["loss"][-1] if history["loss"] else 0.0, \
        pd.DataFrame(history), \
        f"Training complete. Weights saved to '{WEIGHTS_PATH}' (total step count: {history['step'][-1] if history['step'] else 0})."


def estimate_model_size(block_size, n_embd, n_head, n_layer):
    config = TechcodeXConfig(block_size=int(block_size), n_embd=int(n_embd), n_head=int(n_head), n_layer=int(n_layer))
    info = describe_model_size(config)
    return (
        f"Parameters: {format_param_count(info['total_params'])} "
        f"({info['total_params']:,} total, {info['trainable_params']:,} trainable)\n"
        f"Approx size: {info['fp32_size_mb']:.2f} MB (fp32) / {info['fp16_size_mb']:.2f} MB (fp16)\n"
        f"Config: n_embd={config.n_embd}, n_head={config.n_head}, n_layer={config.n_layer}, "
        f"block_size={config.block_size}, vocab_size={config.vocab_size}, tied_weights={config.tie_weights}"
    )


def check_weights_exist():
    if os.path.isfile(WEIGHTS_PATH):
        size_mb = os.path.getsize(WEIGHTS_PATH) / (1024 * 1024)
        return f"Found '{WEIGHTS_PATH}' ({size_mb:.2f} MB)."
    return f"No weights file found at '{WEIGHTS_PATH}'. Train a model first."


def show_model_info():
    if not os.path.isfile(WEIGHTS_PATH):
        return f"No weights file found at '{WEIGHTS_PATH}'. Train a model first."

    checkpoint = load_checkpoint(WEIGHTS_PATH, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", {})
    config_dict = checkpoint.get("config", TechcodeXConfig().__dict__)

    total_params = sum(t.numel() for t in state_dict.values())
    file_size_mb = os.path.getsize(WEIGHTS_PATH) / (1024 * 1024)

    return (
        f"Parameters: {format_param_count(total_params)} ({total_params:,} total)\n"
        f"Checkpoint file size on disk: {file_size_mb:.2f} MB\n"
        f"Trained for {checkpoint.get('step', '?')} steps\n"
        f"Config: n_embd={config_dict.get('n_embd')}, n_head={config_dict.get('n_head')}, "
        f"n_layer={config_dict.get('n_layer')}, block_size={config_dict.get('block_size')}, "
        f"vocab_size={config_dict.get('vocab_size')}"
    )


def export_hf_bundle():
    if not os.path.isfile(WEIGHTS_PATH):
        return f"Cannot export — '{WEIGHTS_PATH}' does not exist yet. Train a model first."

    os.makedirs(EXPORT_DIR, exist_ok=True)

    checkpoint = load_checkpoint(WEIGHTS_PATH, map_location="cpu")
    config_dict = checkpoint.get("config", TechcodeXConfig().__dict__)

    hf_config = {
        "model_type": "techcodex",
        "architectures": ["TechcodeXModel"],
        "vocab_size": config_dict.get("vocab_size", 50257),
        "n_embd": config_dict.get("n_embd", 1536),
        "n_head": config_dict.get("n_head", 24),
        "n_layer": config_dict.get("n_layer", 34),
        "block_size": config_dict.get("block_size", 1024),
        "dropout": config_dict.get("dropout", 0.1),
    }

    with open(os.path.join(EXPORT_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(hf_config, f, indent=2)

    shutil.copy2(WEIGHTS_PATH, os.path.join(EXPORT_DIR, "pytorch_model.bin"))

    return f"Exported bundle to './{EXPORT_DIR}/' (config.json + pytorch_model.bin)."


def _write_model_card(checkpoint: dict, repo_id: str):
    config_dict = checkpoint.get("config", TechcodeXConfig().__dict__)
    total_params = sum(t.numel() for t in checkpoint.get("model_state_dict", {}).values())
    readme = (
        "---\n"
        "library_name: pytorch\n"
        "tags:\n"
        "- techcodex\n"
        "- gpt\n"
        "- from-scratch\n"
        "- text-generation\n"
        "---\n\n"
        f"# {repo_id.split('/')[-1]}\n\n"
        "Raw PyTorch GPT-style decoder-only language model trained with "
        "TechcodeX — no Hugging Face model wrapper, weights + config only.\n\n"
        f"- Parameters: {format_param_count(total_params)} ({total_params:,})\n"
        f"- n_embd={config_dict.get('n_embd')}, n_head={config_dict.get('n_head')}, "
        f"n_layer={config_dict.get('n_layer')}, block_size={config_dict.get('block_size')}, "
        f"vocab_size={config_dict.get('vocab_size')}\n"
        f"- Trained for {checkpoint.get('step', '?')} steps\n\n"
        "Load with the `TechcodeXModel` class from `techcodex_single_file.py` "
        "(not a standard Hugging Face model class, so `AutoModel` will not load it).\n"
    )
    with open(os.path.join(EXPORT_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)


def upload_to_hf_hub(repo_id, hf_token, private, commit_message):
    if not os.path.isfile(WEIGHTS_PATH):
        yield f"Cannot upload — '{WEIGHTS_PATH}' does not exist yet. Train a model first."
        return

    repo_id = (repo_id or "").strip()
    if not repo_id or "/" not in repo_id:
        yield "Please enter a repo id in the form 'username/model-name'."
        return

    hf_token = (hf_token or "").strip() or None

    try:
        from huggingface_hub import HfApi
    except ImportError:
        yield "huggingface_hub is required. Install it with: pip install huggingface_hub"
        return

    yield "Preparing export bundle..."
    export_hf_bundle()

    checkpoint = load_checkpoint(WEIGHTS_PATH, map_location="cpu")
    _write_model_card(checkpoint, repo_id)

    try:
        api = HfApi(token=hf_token)
        yield f"Creating/verifying repo '{repo_id}' (private={bool(private)})..."
        api.create_repo(repo_id=repo_id, private=bool(private), exist_ok=True, repo_type="model")

        yield f"Uploading '{EXPORT_DIR}/' to '{repo_id}'... this can take a while for large checkpoints."
        api.upload_folder(
            folder_path=EXPORT_DIR,
            repo_id=repo_id,
            repo_type="model",
            commit_message=(commit_message or "").strip() or "Upload TechcodeX checkpoint",
        )
    except Exception as e:
        yield f"Upload failed: {e}"
        return

    yield f"Uploaded successfully -> https://huggingface.co/{repo_id}"


def _load_chat_model():
    global _chat_model, _chat_config
    if not os.path.isfile(WEIGHTS_PATH):
        return None
    if _chat_model is not None:
        return _chat_model

    checkpoint = load_checkpoint(WEIGHTS_PATH, map_location="cpu")
    config_dict = checkpoint.get("config", TechcodeXConfig().__dict__)
    config = TechcodeXConfig(**config_dict)

    model = TechcodeXModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _chat_model = model
    _chat_config = config
    return _chat_model


def chat_with_techcodex(message, temperature, history):
    history = history or []

    model = _load_chat_model()
    if model is None:
        history.append({"role": "user", "content": message})
        history.append({
            "role": "assistant",
            "content": f"No trained weights found at '{WEIGHTS_PATH}'. Train a model in Tab 1 first.",
        })
        return history, ""

    tokenizer = get_tokenizer()
    input_ids = tokenizer.encode(message)
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)

    out_idx = model.generate(idx, max_new_tokens=60, temperature=float(temperature))
    generated_ids = out_idx[0].tolist()[len(input_ids):]
    reply = tokenizer.decode(generated_ids, skip_special_tokens=True)

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": reply.strip() or "(empty response)"})
    return history, ""


with gr.Blocks(title="TechcodeX") as demo:
    gr.Markdown("# ⚡ TechcodeX (single file)\nFrom-scratch ~1B-parameter GPT-style pre-training & inference dashboard.")
    gr.Markdown(f"**Device:** `{device}` (`{_DEVICE_KIND}`){' — running in Colab' if _IN_COLAB else ''}")

    with gr.Tabs():
        with gr.Tab("Tab 1: Dataset & Training Configuration"):
            with gr.Row():
                with gr.Column(scale=1):
                    file_upload = gr.File(
                        label="Upload Dataset File(s) (.txt or .jsonl)",
                        file_count="multiple",
                        file_types=[".txt", ".jsonl"],
                    )

                    with gr.Row():
                        scan_dir_box = gr.Textbox(value="datasets", label="Folder to scan", scale=2)
                        refresh_datasets_button = gr.Button("Refresh List", scale=1)
                    existing_datasets_checkbox = gr.CheckboxGroup(
                        choices=[os.path.relpath(f) for f in list_existing_dataset_files("datasets")],
                        label="Existing dataset files (check to include in training)",
                    )

                    gr.Markdown("**Or download a dataset from the Hugging Face Hub:**")
                    with gr.Row():
                        hf_repo_id_box = gr.Textbox(label="Dataset repo id", placeholder="e.g. roneneldan/TinyStories")
                        hf_config_box = gr.Textbox(label="Config name (optional)")
                        hf_split_box = gr.Textbox(value="train", label="Split")
                        hf_text_field_box = gr.Textbox(value="text", label="Text column")
                    with gr.Row():
                        hf_max_examples_box = gr.Number(value=5000, label="Max examples (0 = all)", precision=0)
                        hf_jsonl_out_dir_box = gr.Textbox(value="datasets", label="Output dir")
                        hf_dataset_filename_box = gr.Textbox(label="Filename (optional)")
                        hf_chunk_size_box = gr.Number(value=1024, label="Chunk size (tokens)", precision=0)
                    hf_download_button = gr.Button("Download Dataset from Hugging Face", variant="secondary")
                    hf_download_status = gr.Textbox(label="Download Log", interactive=False, lines=4)
                    hf_download_sample = gr.Textbox(label="Sample JSONL Examples", interactive=False, lines=6)

                    gr.Markdown("---")
                    batch_size_slider = gr.Slider(1, 64, value=4, step=1, label="Batch Size")
                    block_size_slider = gr.Slider(128, 4096, value=1024, step=128, label="Context Length (block_size)")
                    n_embd_slider = gr.Slider(256, 2560, value=1536, step=128, label="Embedding Dim (n_embd)")
                    n_head_slider = gr.Slider(4, 40, value=24, step=1, label="Attention Heads (n_head)")
                    n_layer_slider = gr.Slider(4, 48, value=34, step=1, label="Layers (n_layer)")
                    max_steps_slider = gr.Slider(10, 20000, value=1000, step=10, label="Max Steps")
                    lr_slider = gr.Slider(1e-5, 1e-2, value=3e-4, step=1e-5, label="Learning Rate")
                    resume_checkbox = gr.Checkbox(value=True, label="Resume from saved weights")
                    gr.Markdown(
                        "**Out of accelerator memory?** Enable Gradient Checkpointing and/or lower "
                        "Batch Size while raising Gradient Accumulation Steps by the same factor."
                    )
                    gradient_checkpointing_checkbox = gr.Checkbox(value=True, label="Gradient Checkpointing (recommended at 1B scale)")
                    gradient_accumulation_slider = gr.Slider(1, 128, value=8, step=1, label="Gradient Accumulation Steps")
                    use_fp16_checkbox = gr.Checkbox(value=False, label="Train in fp16 (DirectML/CUDA; ignored on TPU)")
                    offload_optimizer_checkbox = gr.Checkbox(value=False, label="Offload Optimizer State to CPU (DirectML only)")
                    model_size_display = gr.Textbox(
                        label="Model Size (estimated from current config)",
                        value=estimate_model_size(1024, 1536, 24, 34),
                        interactive=False,
                        lines=3,
                    )
                    train_button = gr.Button("Start Training", variant="primary")

                with gr.Column(scale=1):
                    step_display = gr.Number(label="Current Step", value=0, interactive=False)
                    loss_display = gr.Number(label="Current Loss", value=0.0, interactive=False)
                    loss_plot = gr.LinePlot(
                        pd.DataFrame({"step": [], "loss": []}), x="step", y="loss", title="Training Loss",
                    )
                    training_status = gr.Textbox(label="Status", interactive=False)

            for control in (block_size_slider, n_embd_slider, n_head_slider, n_layer_slider):
                control.change(
                    fn=estimate_model_size,
                    inputs=[block_size_slider, n_embd_slider, n_head_slider, n_layer_slider],
                    outputs=model_size_display,
                )

            refresh_datasets_button.click(fn=refresh_existing_datasets, inputs=scan_dir_box, outputs=existing_datasets_checkbox)

            hf_download_button.click(
                fn=run_download_hf_dataset,
                inputs=[
                    hf_repo_id_box, hf_config_box, hf_split_box, hf_text_field_box,
                    hf_max_examples_box, hf_jsonl_out_dir_box, hf_dataset_filename_box, hf_chunk_size_box,
                ],
                outputs=[hf_download_status, hf_download_sample],
            )

            train_button.click(
                fn=start_training,
                inputs=[
                    file_upload, existing_datasets_checkbox, batch_size_slider, block_size_slider,
                    max_steps_slider, lr_slider, resume_checkbox,
                    gradient_checkpointing_checkbox, gradient_accumulation_slider, use_fp16_checkbox,
                    offload_optimizer_checkbox,
                ],
                outputs=[step_display, loss_display, loss_plot, training_status],
            )

        with gr.Tab("Tab 2: Model Testing & Weights"):
            with gr.Row():
                with gr.Column():
                    check_button = gr.Button("Check Weights File")
                    check_output = gr.Textbox(label="Weights Status", interactive=False)
                    check_button.click(fn=check_weights_exist, outputs=check_output)

                with gr.Column():
                    export_button = gr.Button("Export Hugging Face-style Bundle")
                    export_output = gr.Textbox(label="Export Status", interactive=False)
                    export_button.click(fn=export_hf_bundle, outputs=export_output)

                with gr.Column():
                    model_info_button = gr.Button("Show Model Size")
                    model_info_output = gr.Textbox(label="Model Size", interactive=False, lines=5)
                    model_info_button.click(fn=show_model_info, outputs=model_info_output)

        with gr.Tab("Tab 3: Chat with Eather"):
            chatbot = gr.Chatbot(label="Eather (TechcodeX)", height=400)
            with gr.Row():
                chat_input = gr.Textbox(label="Message", placeholder="Say something...", scale=4)
                send_button = gr.Button("Send", variant="primary", scale=1)
            temperature_slider = gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="Creativity Temperature")

            send_button.click(fn=chat_with_techcodex, inputs=[chat_input, temperature_slider, chatbot], outputs=[chatbot, chat_input])
            chat_input.submit(fn=chat_with_techcodex, inputs=[chat_input, temperature_slider, chatbot], outputs=[chatbot, chat_input])

        with gr.Tab("Tab 4: Upload to Hugging Face"):
            gr.Markdown(
                "Pushes the trained checkpoint to the Hugging Face Hub as a model repo "
                "(config.json + pytorch_model.bin + an auto-generated model card)."
            )
            with gr.Row():
                hf_upload_repo_id_box = gr.Textbox(label="Repo id", placeholder="e.g. yourname/techcodex-model", scale=2)
                hf_upload_private_checkbox = gr.Checkbox(value=True, label="Private repository", scale=1)
            hf_upload_token_box = gr.Textbox(label="Hugging Face access token (write scope)", placeholder="hf_...", type="password")
            hf_upload_commit_message_box = gr.Textbox(label="Commit message (optional)", placeholder="Upload TechcodeX checkpoint")
            hf_upload_button = gr.Button("Upload to Hugging Face Hub", variant="primary")
            hf_upload_status = gr.Textbox(label="Upload Status", interactive=False, lines=5)

            hf_upload_button.click(
                fn=upload_to_hf_hub,
                inputs=[hf_upload_repo_id_box, hf_upload_token_box, hf_upload_private_checkbox, hf_upload_commit_message_box],
                outputs=hf_upload_status,
            )


if __name__ == "__main__":
    # Colab has no direct access to localhost, so a Gradio public share link
    # is required there; locally, default to a plain local server.
    demo.queue().launch(theme=gr.themes.Soft(primary_hue="violet"), share=_IN_COLAB)
