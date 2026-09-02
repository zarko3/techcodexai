"""
TechcodeX training orchestration backend.

Exposes `run_training_session(...)` as a generator that yields
(step, loss) tuples progressively — this lets a UI (Gradio, etc.)
stream live training updates without blocking the main thread when
invoked as a Gradio generator callback (Gradio runs callbacks in a
worker thread automatically).
"""

import json
import os

import torch
from torch.utils.data import Dataset, DataLoader

from modeling_techcodex import TechcodeXModel, TechcodeXConfig, device, load_checkpoint

WEIGHTS_PATH = "TechcodeX_Weights.pt"
SAVE_EVERY = 25

_tokenizer = None


def get_tokenizer():
    """Lazily load the 'gpt2' tokenizer (tokenizer only — not a model class).

    GPT2TokenizerFast defaults to model_max_length=1024 (GPT-2's own context
    window) and logs "Token indices sequence length is longer than the
    specified maximum sequence length for this model... Running this
    sequence through the model will result in indexing errors" whenever
    .encode() sees more tokens than that on a full corpus. That warning is
    misleading here: we only ever use this tokenizer to turn raw text into a
    flat token stream, which TokenBlockDataset then slices into
    TechcodeXConfig.block_size windows before anything reaches the model —
    the tokenizer's own 1024 limit is irrelevant to our architecture. Raise
    it so the spurious warning stops firing.
    """
    global _tokenizer
    if _tokenizer is None:
        from transformers import GPT2TokenizerFast

        _tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        _tokenizer.model_max_length = int(1e30)
    return _tokenizer


class DirectMLSafeAdamW(torch.optim.Optimizer):
    """AdamW re-implemented with plain elementwise ops instead of PyTorch's
    built-in exp_avg.lerp_(grad, 1 - beta1) fast path.

    torch's stock AdamW (both the single-tensor and foreach code paths) calls
    aten::lerp unconditionally on every step. DirectML has no kernel for
    aten::lerp.Scalar_out, so torch silently falls back to CPU for that op on
    every single optimizer step — verified against the installed torch
    2.4.1 adamw.py source. Falling back per-step is what shows up as constant
    GPU<->CPU round-trips in the training log. mul_/add_/addcmul_/addcdiv_
    are ordinary elementwise ops DirectML already handles natively (they're
    used throughout the forward/backward pass), and are algebraically
    identical to the lerp formulation, so this produces the same update.
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
                    p.mul_(1 - lr * weight_decay)  # decoupled weight decay, same as AdamW

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                step_size = lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class CPUOffloadAdamW(torch.optim.Optimizer):
    """AdamW whose optimizer state — exp_avg, exp_avg_sq, and a master copy of
    the parameters — lives in CPU RAM instead of GPU VRAM, using both devices
    every step: the GPU still does all forward/backward compute (matmuls,
    attention, etc.), then each step copies gradients to CPU, runs the AdamW
    update there (CPU RAM is usually far more abundant than GPU VRAM), and
    copies the updated weights back to GPU for the next forward pass.

    AdamW's state (exp_avg + exp_avg_sq) is 2x parameter memory — on top of
    weights (1x) and gradients (1x), that's the same 4x-parameter fixed
    baseline DirectMLSafeAdamW carries. Moving the 2x optimizer-state share
    (plus a redundant param master copy, another 1x) off the GPU cuts the
    GPU-resident fixed baseline from 4x down to 2x (weights + gradients only)
    — the tradeoff is a CPU<->GPU transfer every step, so this is slower per
    step than keeping everything on GPU; use it when VRAM, not speed, is the
    constraint.
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

                p.data.copy_(param_cpu)  # push the updated weights back to GPU

        return loss


# DirectML has no aten::lerp kernel (see DirectMLSafeAdamW above) — use the
# lerp-free optimizer whenever training on DirectML, and torch's own (faster,
# fused) AdamW when running on plain CPU where lerp is natively supported.
_ON_DIRECTML = not (isinstance(device, str) and device == "cpu")

# fp16's smallest normal value (~6.1e-5) is bigger than AdamW's usual eps
# (1e-8), so at fp16 that eps silently rounds to 0 — which can turn
# `denom = sqrt(exp_avg_sq) + eps` into a true zero and blow up the update
# into inf/nan the first time a gradient underflows. 1e-4 stays representable
# in fp16 while still being negligible next to typical denom magnitudes.
_FP16_SAFE_EPS = 1e-4


def _make_optimizer(model, learning_rate: float, use_fp16: bool = False, offload_optimizer_to_cpu: bool = False):
    eps = _FP16_SAFE_EPS if use_fp16 else 1e-8
    if offload_optimizer_to_cpu and _ON_DIRECTML:
        return CPUOffloadAdamW(model.parameters(), lr=learning_rate, eps=eps)
    if _ON_DIRECTML:
        return DirectMLSafeAdamW(model.parameters(), lr=learning_rate, eps=eps)
    # already CPU-only — "offload to CPU" is a no-op, just use torch's own AdamW
    return torch.optim.AdamW(model.parameters(), lr=learning_rate, eps=eps)


class DynamicLossScaler:
    """Manual mixed-precision loss scaling for backends without torch.cuda.amp.

    Pure fp16 training halves the fixed weights+gradients+optimizer-state
    memory footprint vs fp32, but small gradients can underflow to zero in
    fp16 well before they'd matter in fp32. Scaling the loss up before
    backward() (and gradients down by the same factor before the optimizer
    step) keeps small gradients representable without changing the math.
    Standard dynamic scaling: grow the scale periodically when steps stay
    finite, and immediately halve it (skipping that optimizer step) the
    moment an inf/nan gradient appears — torch.cuda.amp.GradScaler does the
    same thing but is hard-wired to the CUDA backend, so DirectML needs its
    own copy of this logic.
    """

    def __init__(self, init_scale=2.0 ** 14, growth_factor=2.0, backoff_factor=0.5, growth_interval=200):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self._good_steps = 0

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        return loss * self.scale

    def unscale_and_check_finite(self, parameters) -> bool:
        """Divides every gradient by the current scale in-place. Returns False
        (and leaves gradients as-is, un-stepped) if any gradient is inf/nan."""
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


def _read_single_file(file_path: str) -> list:
    """Parses one .txt or .jsonl dataset file into a list of text chunks."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    text_chunks = []

    if file_path.lower().endswith(".jsonl"):
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
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
    """Parses one or more .txt/.jsonl dataset files and tokenizes their combined text."""
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


_OOM_MESSAGE_MARKERS = ("not enough", "out of memory", "could not allocate")


def _raise_actionable_oom(original: Exception, batch_size: int, block_size: int, gradient_accumulation_steps: int):
    raise RuntimeError(
        "Ran out of GPU video memory during training "
        f"(batch_size={batch_size}, block_size={block_size}). This model's weight/activation "
        "footprint doesn't fit in the available VRAM at these settings. Try, in order of "
        "impact: (1) enable 'Gradient Checkpointing' — cuts activation memory substantially "
        "at the cost of extra compute per step; (2) lower Batch Size and raise 'Gradient "
        "Accumulation Steps' by the same factor to keep the same effective batch size while "
        "using far less memory per step; (3) enable 'Offload Optimizer State to CPU' — moves "
        "AdamW's state off the GPU entirely, at the cost of a CPU<->GPU transfer per step; "
        "(4) enable 'Train in fp16'; (5) lower Context Length (block_size); "
        "(6) reduce the model size (n_embd/n_layer in TechcodeXConfig)."
    ) from original


def run_training_session(
    file_path,
    batch_size: int = 8,
    block_size: int = 128,
    max_steps: int = 200,
    learning_rate: float = 3e-4,
    resume: bool = False,
    gradient_checkpointing: bool = False,
    gradient_accumulation_steps: int = 1,
    use_fp16: bool = False,
    offload_optimizer_to_cpu: bool = False,
):
    """
    Background training generator. Yields (step, loss) after every optimizer step
    so a UI can display a live-updating loss curve. Designed to be driven either
    directly as a Gradio generator callback, or pumped from a background thread.

    `file_path` may be a single dataset file path or a list of paths (.txt/.jsonl),
    which are concatenated into one training corpus.

    By default this starts a brand-new randomly-initialized model every call — running
    it repeatedly does NOT accumulate progress unless `resume=True`, in which case it
    loads `WEIGHTS_PATH` (if present) and continues from its saved step count. `max_steps`
    is always how many additional steps THIS call runs, not a total to reach.

    `gradient_accumulation_steps` > 1 splits each effective batch into that many
    smaller micro-batches, backpropagating each before a single optimizer step —
    same effective batch size and math, far less peak memory per step. `step`
    (and therefore `max_steps`/checkpoint cadence) counts optimizer steps, not
    micro-batches.

    `use_fp16` trains with the model's weights, activations, gradients, and
    optimizer state all in fp16 instead of fp32 — roughly halves the fixed
    weights+gradients+optimizer-state memory footprint, at the cost of fp16's
    narrower numeric range. Backed by dynamic loss scaling (see
    DynamicLossScaler) to keep small gradients from underflowing to zero.

    `offload_optimizer_to_cpu` moves AdamW's state (which is 2x parameter
    memory) into CPU RAM instead of GPU VRAM — the GPU still does all
    forward/backward compute, but every optimizer step copies gradients to
    CPU, updates there, and copies the new weights back. Genuinely uses both
    devices each step; costs a PCIe transfer per step in exchange for cutting
    the GPU-resident fixed baseline roughly in half again on top of fp16.
    """
    gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))
    tokens = load_and_tokenize(file_path)

    start_step = 0
    if resume and os.path.isfile(WEIGHTS_PATH):
        # Load to CPU, not `device`: torch's generic map_location restore path
        # calls into torch_directml with a torch.device OBJECT (rather than an
        # int device id), which torch_directml's device() doesn't handle —
        # confirmed to raise unconditionally regardless of checkpoint format.
        # Loading to CPU and letting model.load_state_dict() below copy onto
        # the already-`.to(device)`-placed model sidesteps it entirely.
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

    if use_fp16:
        model = model.half()

    model.set_gradient_checkpointing(gradient_checkpointing)

    dataset = TokenBlockDataset(tokens, block_size=config.block_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    optimizer = _make_optimizer(model, learning_rate, use_fp16=use_fp16, offload_optimizer_to_cpu=offload_optimizer_to_cpu)
    scaler = DynamicLossScaler() if use_fp16 else None

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
                if use_fp16:
                    scaler.scale_loss(micro_loss).backward()
                else:
                    micro_loss.backward()
            except RuntimeError as e:
                if any(marker in str(e).lower() for marker in _OOM_MESSAGE_MARKERS):
                    _raise_actionable_oom(e, batch_size, block_size, gradient_accumulation_steps)
                raise

            accumulated_loss += loss.item()

        if use_fp16:
            step_is_finite = scaler.unscale_and_check_finite(model.parameters())
            scaler.update(step_is_finite)
            if not step_is_finite:
                # Bad step (inf/nan grad) — scale already backed off; skip this
                # optimizer step and retry with a fresh batch instead of
                # corrupting the weights.
                continue

        try:
            optimizer.step()
        except RuntimeError as e:
            if any(marker in str(e).lower() for marker in _OOM_MESSAGE_MARKERS):
                _raise_actionable_oom(e, batch_size, block_size, gradient_accumulation_steps)
            raise

        step += 1

        if step % SAVE_EVERY == 0 or step == target_step:
            # Move to CPU before saving: a DirectML ("privateuseone") tensor is
            # pickled via torch._utils._rebuild_device_tensor_from_numpy, which
            # isn't on torch.load's default weights_only=True safe-globals list.
            # CPU tensors use the standard rebuild path and always load safely.
            cpu_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "model_state_dict": cpu_state_dict,
                    "config": config.__dict__,
                    "step": step,
                },
                WEIGHTS_PATH,
            )

        yield step, accumulated_loss / gradient_accumulation_steps
