"""
TechcodeX — single-file build.

Everything needed to prepare data, train a ~1B-parameter GPT-style decoder-only
model from scratch, chat with it, and export/upload it, in one file so it can
be pasted straight into a Google Colab cell (with a TPU runtime) or run
locally on Windows (CUDA / AMD DirectML / CPU).

Colab TPU quick start (in a notebook cell, before this file's code):
    # torch_xla's compiled C extension (_XLAC.so) is built against one exact
    # torch release and does NOT declare torch as a pip dependency, so both
    # must be installed together, pinned to the SAME version — otherwise you
    # get "undefined symbol" ImportErrors. Check what torch_xla version is
    # already on the image first (`!pip show torch_xla`) and match it below;
    # 2.8.0 is current as of this writing.
    # Install everything in ONE pip call — splitting torch/torch_xla and the
    # rest into two separate calls has caused Gradio to silently not install.
    !pip install -q torch==2.8.0 "torch_xla[tpu]==2.8.0" transformers datasets gradio huggingface_hub -f https://storage.googleapis.com/libtpu-releases/index.html

Local Windows quick start:
    pip install torch transformers datasets gradio huggingface_hub pymupdf pytesseract Pillow langdetect
    pip install torch-directml   # optional, for AMD GPUs

Then just: python techcodex_single_file.py

Includes the full dataset-prep pipeline: PDF/image OCR extraction, text
cleaning (headers/footers, page numbers, markup, web boilerplate), quality
filtering + deduplication, paragraph-aware chunking, and Hugging Face Hub
dataset download — all run through the same cleaning pass. OCR (Tab 0, image
files) needs the Tesseract engine installed separately: `!apt-get install -y
tesseract-ocr` on Colab, or see setup_env.ps1 on Windows.
"""

import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

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
            # torch.utils.checkpoint always resolves a "device module" for
            # autocast/RNG bookkeeping via getattr(torch, device.type) — even
            # with preserve_rng_state=False — and there is no torch.xla
            # submodule (torch_xla is a separate top-level package, not
            # merged into torch's namespace), so it raises "module 'torch'
            # has no attribute 'xla'" on TPU no matter what. Skip
            # checkpointing entirely on XLA; TPU HBM is large enough at this
            # model size that it's not needed there the way it is on a
            # VRAM-constrained GPU.
            if self.gradient_checkpointing and self.training and _DEVICE_KIND != "xla":
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
# Dataset preparation pipeline — PDF/image OCR extraction, text cleaning,
# quality filtering, and paragraph-aware chunking.
#
# Pipeline: source file(s) -> text extraction -> cleaning -> quality filter
# -> chunking -> JSONL. Also used by download_hf_dataset() below, so local
# files and Hugging Face Hub datasets go through the exact same cleaning
# pass.
# ============================================================================

@dataclass
class ExtractedPage:
    index: int
    text: str


@dataclass
class ExtractionResult:
    source_path: str
    page_count: int
    pages: list  # list[ExtractedPage]
    raw_text: str
    warnings: list = field(default_factory=list)


class BaseExtractor:
    """Interface every source-type extractor must implement."""

    supported_extensions: tuple = ()

    def extract(self, file_path: str) -> ExtractionResult:
        raise NotImplementedError


class PDFExtractor(BaseExtractor):
    """Extracts text from born-digital PDFs using PyMuPDF."""

    supported_extensions = (".pdf",)

    # Below this average extractable-characters-per-page, the PDF is likely
    # scanned/image-based and needs OCR rather than direct text extraction.
    MIN_CHARS_PER_PAGE_OCR_THRESHOLD = 20

    def extract(self, file_path: str) -> ExtractionResult:
        try:
            import pymupdf as fitz  # PyMuPDF (modern import name)
        except ImportError:
            try:
                import fitz  # older PyMuPDF releases
            except ImportError as e:
                raise RuntimeError(
                    "PyMuPDF is required for PDF extraction. Install it with: pip install pymupdf"
                ) from e

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        warnings = []

        try:
            doc = fitz.open(file_path)
        except Exception as e:
            raise RuntimeError(f"Failed to open PDF '{file_path}': {e}") from e

        try:
            page_count = doc.page_count
            if page_count == 0:
                raise RuntimeError(f"PDF '{file_path}' has zero pages.")

            pages = []
            total_chars = 0
            for i in range(page_count):
                page = doc.load_page(i)
                text = page.get_text("text") or ""
                pages.append(ExtractedPage(index=i, text=text))
                total_chars += len(text.strip())

            avg_chars_per_page = total_chars / page_count
            if avg_chars_per_page < self.MIN_CHARS_PER_PAGE_OCR_THRESHOLD:
                warnings.append(
                    f"Very little extractable text found ({avg_chars_per_page:.1f} chars/page avg). "
                    "This PDF is likely scanned/image-based and may require OCR before it is usable."
                )
        finally:
            doc.close()

        raw_text = "\n\n".join(p.text for p in pages)

        if not raw_text.strip():
            warnings.append("No extractable text found in this PDF at all — OCR is required.")

        return ExtractionResult(
            source_path=file_path,
            page_count=page_count,
            pages=pages,
            raw_text=raw_text,
            warnings=warnings,
        )


class ImageExtractor(BaseExtractor):
    """OCR-based extractor for standalone image files that have no native text layer.

    Requires the Tesseract OCR ENGINE installed separately and on PATH
    (`pip install pytesseract` only installs the Python wrapper) — not
    available by default on Colab. Local Windows use: see setup_env.ps1.
    """

    supported_extensions = (".png", ".jpg", ".jpeg")

    def extract(self, file_path: str) -> ExtractionResult:
        try:
            import pytesseract
            from PIL import Image
        except ImportError as e:
            raise RuntimeError(
                "OCR support requires 'pytesseract' and 'Pillow'. Install with: "
                "pip install pytesseract Pillow"
            ) from e

        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")

        warnings = []
        try:
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image) or ""
        except pytesseract.TesseractNotFoundError as e:
            raise RuntimeError(
                "The Tesseract OCR engine is not installed (or not on PATH) on this machine. "
                "'pip install pytesseract' only installs the Python wrapper — the engine itself "
                "must be installed separately. On Windows, get it from: "
                "https://github.com/UB-Mannheim/tesseract/wiki, then add its install folder to PATH. "
                "On Colab: !apt-get install -y tesseract-ocr"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to OCR image '{file_path}': {e}") from e

        if not text.strip():
            warnings.append(
                "OCR found no readable text in this image — it may be a photo with no text, "
                "low resolution, or handwriting OCR can't read."
            )

        return ExtractionResult(
            source_path=file_path,
            page_count=1,
            pages=[ExtractedPage(index=0, text=text)],
            raw_text=text,
            warnings=warnings,
        )


class TxtExtractor(BaseExtractor):
    """Passes plain text files through as a single page, still eligible for the cleaning pass."""

    supported_extensions = (".txt",)

    def extract(self, file_path: str) -> ExtractionResult:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Text file not found: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except Exception as e:
            raise RuntimeError(f"Failed to read text file '{file_path}': {e}") from e

        warnings = []
        if not text.strip():
            warnings.append("This text file is empty.")

        return ExtractionResult(
            source_path=file_path,
            page_count=1,
            pages=[ExtractedPage(index=0, text=text)],
            raw_text=text,
            warnings=warnings,
        )


EXTRACTOR_REGISTRY: dict = {}


def register_extractor(extractor: BaseExtractor):
    for ext in extractor.supported_extensions:
        EXTRACTOR_REGISTRY[ext.lower()] = extractor


def get_extractor_for(file_path: str) -> BaseExtractor:
    ext = os.path.splitext(file_path)[1].lower()
    extractor = EXTRACTOR_REGISTRY.get(ext)
    if extractor is None:
        raise ValueError(
            f"No extractor registered for file extension '{ext}'. "
            f"Supported: {sorted(EXTRACTOR_REGISTRY.keys())}"
        )
    return extractor


register_extractor(PDFExtractor())
register_extractor(ImageExtractor())
register_extractor(TxtExtractor())


# --- Cleaning ---

_PAGE_NUMBER_RE = re.compile(
    r"^\s*(?:page\s+)?[\-–—]?\s*\d{1,4}\s*(?:of\s*\d{1,4})?\s*[\-–—]?\s*$",
    re.IGNORECASE,
)
_ROMAN_NUMERAL_RE = re.compile(r"^\s*[ivxlcdm]{1,6}\s*$", re.IGNORECASE)
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_BLANK_LINE_RE = re.compile(r"\n{3,}")
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")

_TAG_BLOCK_RE = re.compile(r"<(script|style|svg)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_ANY_TAG_RE = re.compile(r"</?[a-zA-Z][a-zA-Z0-9:_-]*(?:\s+[^<>]*)?/?>")
_XML_DECL_RE = re.compile(r"<\?xml[^>]*\?>", re.IGNORECASE)
_HTML_ENTITY_RE = re.compile(r"&(?:amp|lt|gt|quot|apos|nbsp|#\d+|#x[0-9a-fA-F]+);")
_ISOLATED_MARKUP_WORD_RE = re.compile(r"^\s*(?:svg|html|xml)\s*$", re.IGNORECASE)

_WEB_ARTIFACT_LINE_RE = re.compile(
    r"^\s*(cookie policy|accept cookies|we use cookies|all rights reserved|click here|"
    r"share (this|on) (facebook|twitter|linkedin|reddit)?|subscribe to our newsletter|"
    r"terms of (service|use)|privacy policy|skip to (main )?content|"
    r"javascript:void\(0\)|advertisement|sponsored content|read more|back to top|"
    r"sign up|log in|menu|navigation)\s*$",
    re.IGNORECASE,
)
_BARE_URL_LINE_RE = re.compile(r"^\s*(?:https?://|www\.)\S+\s*$", re.IGNORECASE)


def contains_markup(text: str) -> bool:
    return bool(_ANY_TAG_RE.search(text))


def strip_markup(text: str) -> str:
    """Removes actual HTML/SVG/XML tag syntax and isolated bare-keyword lines
    (a line that is only 'svg'/'html'/'xml'). Never touches those words when
    they appear as part of ordinary prose."""
    text = _TAG_BLOCK_RE.sub(" ", text)
    text = _XML_DECL_RE.sub(" ", text)
    text = _ANY_TAG_RE.sub(" ", text)
    text = _HTML_ENTITY_RE.sub(" ", text)
    kept = [line for line in text.splitlines() if not _ISOLATED_MARKUP_WORD_RE.match(line)]
    return "\n".join(kept)


def strip_web_artifacts(text: str) -> str:
    """Drops boilerplate nav/legal/share lines and bare-URL-only lines."""
    kept = []
    for line in text.splitlines():
        s = line.strip()
        if _WEB_ARTIFACT_LINE_RE.match(s) or _BARE_URL_LINE_RE.match(s):
            continue
        kept.append(line)
    return "\n".join(kept)


@dataclass
class CleaningResult:
    text: str
    warnings: list = field(default_factory=list)
    removed_header_footer_lines: list = field(default_factory=list)


def _detect_repeated_lines(
    pages_text: list,
    edge_lines: int = 2,
    min_page_count: int = 4,
    frequency_threshold: float = 0.6,
) -> set:
    """Find short lines that repeat near the top/bottom of most pages — headers/footers."""
    if len(pages_text) < min_page_count:
        return set()

    counter = Counter()
    for text in pages_text:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        edge = lines[:edge_lines] + lines[-edge_lines:]
        for line in set(edge):
            if line.isdigit() or len(line) < 3:
                continue
            counter[line] += 1

    threshold = max(min_page_count, int(len(pages_text) * frequency_threshold))
    return {line for line, count in counter.items() if count >= threshold}


def _remove_repeated_lines(text: str, repeated: set) -> str:
    if not repeated:
        return text
    return "\n".join(l for l in text.splitlines() if l.strip() not in repeated)


def _strip_page_number_lines(text: str) -> str:
    kept = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and (_PAGE_NUMBER_RE.match(stripped) or _ROMAN_NUMERAL_RE.match(stripped)):
            continue
        kept.append(line)
    return "\n".join(kept)


def _dehyphenate(text: str) -> str:
    """Rejoin words that were hard-wrapped across a hyphen, e.g. 'exam-\\nple' -> 'example'."""
    return _HYPHEN_BREAK_RE.sub(r"\1\2", text)


def _reflow_paragraphs(text: str) -> str:
    """Join hard-wrapped lines within a paragraph while preserving blank-line paragraph breaks."""
    blocks = re.split(r"\n\s*\n", text)
    reflowed = []
    for block in blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        if lines:
            reflowed.append(" ".join(lines))
    return "\n\n".join(reflowed)


def clean_extraction(
    result: ExtractionResult,
    remove_headers_footers: bool = True,
    remove_page_numbers: bool = True,
    normalize_unicode: bool = True,
    remove_markup: bool = True,
    remove_web_artifacts: bool = True,
) -> CleaningResult:
    warnings = list(result.warnings)
    pages_text = [p.text for p in result.pages]

    repeated_lines = set()
    if remove_headers_footers:
        repeated_lines = _detect_repeated_lines(pages_text)
        if repeated_lines:
            warnings.append(f"Removed {len(repeated_lines)} repeated header/footer line(s).")

    cleaned_pages = []
    for text in pages_text:
        t = _remove_repeated_lines(text, repeated_lines) if repeated_lines else text
        if remove_page_numbers:
            t = _strip_page_number_lines(t)
        if remove_markup:
            t = strip_markup(t)
        if remove_web_artifacts:
            t = strip_web_artifacts(t)
        cleaned_pages.append(t)

    joined = "\n\n".join(cleaned_pages)

    joined = _CONTROL_CHARS_RE.sub("", joined)
    joined = joined.replace("�", "")

    joined = _dehyphenate(joined)
    joined = _reflow_paragraphs(joined)

    if normalize_unicode:
        joined = unicodedata.normalize("NFKC", joined)

    joined = _MULTI_SPACE_RE.sub(" ", joined)
    joined = _MULTI_BLANK_LINE_RE.sub("\n\n", joined)
    joined = "\n\n".join(p.strip() for p in joined.split("\n\n") if p.strip())

    if not joined.strip():
        warnings.append("Cleaned text is empty — this file will be excluded from the dataset.")

    return CleaningResult(
        text=joined,
        warnings=warnings,
        removed_header_footer_lines=sorted(repeated_lines),
    )


# --- Chunking ---

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    text: str
    token_count: int


def split_paragraphs(text: str) -> list:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def _split_oversized_paragraph(paragraph: str, chunk_size: int, tokenizer) -> list:
    """A single paragraph larger than chunk_size: pack by sentence, hard-split only as last resort."""
    sentences = _SENTENCE_SPLIT_RE.split(paragraph)
    pieces = []
    current = ""
    for sent in sentences:
        candidate = (current + " " + sent).strip() if current else sent
        if current and len(tokenizer.encode(candidate)) > chunk_size:
            pieces.append(current)
            current = sent
        else:
            current = candidate
    if current:
        pieces.append(current)

    final_pieces = []
    for piece in pieces:
        ids = tokenizer.encode(piece)
        if len(ids) <= chunk_size:
            final_pieces.append(piece)
        else:
            for i in range(0, len(ids), chunk_size):
                final_pieces.append(tokenizer.decode(ids[i:i + chunk_size]))
    return final_pieces


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    min_chunk_size: int = 64,
    tokenizer=None,
) -> list:
    """Returns a list[Chunk]. Never raises on empty input — just returns []."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and less than chunk_size")
    if min_chunk_size < 0 or min_chunk_size > chunk_size:
        raise ValueError("min_chunk_size must be between 0 and chunk_size")

    if not text.strip():
        return []

    tokenizer = tokenizer or get_tokenizer()

    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return []

    expanded = []
    for p in paragraphs:
        if len(tokenizer.encode(p)) > chunk_size:
            expanded.extend(_split_oversized_paragraph(p, chunk_size, tokenizer))
        else:
            expanded.append(p)
    paragraphs = expanded

    chunks: list = []
    current_paragraphs: list = []
    current_tokens = 0

    def flush():
        nonlocal current_paragraphs, current_tokens
        if not current_paragraphs:
            return
        chunks.append(Chunk(text="\n\n".join(current_paragraphs), token_count=current_tokens))
        current_paragraphs = []
        current_tokens = 0

    for p in paragraphs:
        p_tokens = len(tokenizer.encode(p))

        if current_paragraphs and current_tokens + p_tokens > chunk_size:
            flush()
            if chunk_overlap > 0 and chunks:
                overlap_paragraphs = []
                overlap_tokens = 0
                for prev_p in reversed(chunks[-1].text.split("\n\n")):
                    t = len(tokenizer.encode(prev_p))
                    if overlap_tokens + t > chunk_overlap:
                        break
                    overlap_paragraphs.insert(0, prev_p)
                    overlap_tokens += t
                current_paragraphs = overlap_paragraphs
                current_tokens = overlap_tokens

        current_paragraphs.append(p)
        current_tokens += p_tokens

    flush()

    if len(chunks) >= 2 and chunks[-1].token_count < min_chunk_size:
        last = chunks.pop()
        merged_text = chunks[-1].text + "\n\n" + last.text
        chunks[-1] = Chunk(text=merged_text, token_count=chunks[-1].token_count + last.token_count)

    return chunks


# --- Quality filtering ---

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_DOI_RE = re.compile(r"\b10\.\d{4,9}/\S+")
_CITATION_TOKEN_RE = re.compile(
    r"\b(ibid\.?|op\.?\s*cit\.?|cf\.|et\s+al\.?|vol\.?\s*\d*|no\.?\s*\d+|"
    r"pp?\.?\s*\d|chap(?:ter)?\.?\s*\d*|fig\.?\s*\d*|footnote|see also|doi:|isbn)\b",
    re.IGNORECASE,
)
_PAGE_REF_RE = re.compile(r"\bp{1,2}\.?\s*\d{1,4}(?:[-–]\d{1,4})?\b", re.IGNORECASE)

_VOWELS = set("aeiouAEIOU")
_CONSONANT_RUN_RE = re.compile(r"[^aeiouAEIOU\s\d]{6,}")

_ENGLISH_STOPWORDS = frozenset("""
the a an and or but if of to in on for with as by at from is are was were
be been being this that these those it its he she they we you i his her
their our your not no do does did have has had will would can could should
may might must than then so such which who whom what when where why how
also into about between because after before while during each other some
any all most more less much many one two first second new used using use
between over under out up down near own such only just very both same
""".split())

_NON_LATIN_SCRIPT_RANGES = [
    (0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF),  # CJK
    (0x3040, 0x30FF),  # Hiragana/Katakana
    (0xAC00, 0xD7A3), (0x1100, 0x11FF),  # Hangul
    (0x0600, 0x06FF), (0x0750, 0x077F),  # Arabic
    (0x0590, 0x05FF),  # Hebrew
    (0x0400, 0x04FF),  # Cyrillic
    (0x0900, 0x097F),  # Devanagari
    (0x0E00, 0x0E7F),  # Thai
]

_langdetect_ready = None


@dataclass
class QualityFilterResult:
    text: str
    stats: dict
    examples: list = field(default_factory=list)


def _non_latin_script_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    hits = 0
    for c in letters:
        cp = ord(c)
        for lo, hi in _NON_LATIN_SCRIPT_RANGES:
            if lo <= cp <= hi:
                hits += 1
                break
    return hits / len(letters)


def english_probability(text: str) -> float:
    """0..1 confidence the paragraph is English. Cheap script check first
    (catches CJK/Arabic/Cyrillic/etc. instantly), then langdetect for the
    Latin-script case (catches e.g. French/German/Spanish)."""
    global _langdetect_ready

    stripped = text.strip()
    if len(stripped) < 20:
        return 1.0

    if _non_latin_script_ratio(stripped) > 0.25:
        return 0.0

    if _langdetect_ready is False:
        return 1.0

    try:
        from langdetect import DetectorFactory, detect_langs

        DetectorFactory.seed = 0
        _langdetect_ready = True
        for lang in detect_langs(stripped):
            if lang.lang == "en":
                return lang.prob
        return 0.05
    except ImportError:
        _langdetect_ready = False
        return 1.0
    except Exception:
        return 0.5


def _citation_density(paragraph: str) -> float:
    words = paragraph.split()
    if not words:
        return 0.0
    hits = (
        len(_CITATION_TOKEN_RE.findall(paragraph))
        + len(_PAGE_REF_RE.findall(paragraph))
        + len(_DOI_RE.findall(paragraph))
    )
    return hits / len(words)


def is_citation_fragment(paragraph: str) -> bool:
    words = paragraph.split()
    if not words:
        return False
    density = _citation_density(paragraph)
    if len(words) <= 6 and density > 0:
        return True
    if density > 0.25 and len(words) <= 40:
        return True
    return False


def _is_broken_word(word: str) -> bool:
    w = re.sub(r"[^A-Za-z]", "", word)
    if len(w) < 8:
        return False
    if not any(c in _VOWELS for c in w):
        return True
    if _CONSONANT_RUN_RE.search(w):
        return True
    if len(w) > 22:
        return True
    return False


def _broken_word_ratio(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    broken = sum(1 for w in words if _is_broken_word(w))
    return broken / len(words)


def compute_quality_score(paragraph: str, english_prob: float = None) -> tuple:
    """Returns (score 0-100, breakdown dict). Higher is better."""
    if english_prob is None:
        english_prob = english_probability(paragraph)

    words = paragraph.split()
    n_words = max(len(words), 1)
    n_chars = max(len(paragraph), 1)

    alphabetic_ratio = sum(1 for c in paragraph if c.isalpha()) / n_chars

    allowed_punct = set(".,;:!?'\"()-–—…$%/&")
    unusual_char_ratio = sum(
        1 for c in paragraph if not (c.isalnum() or c.isspace() or c in allowed_punct)
    ) / n_chars

    word_counts = Counter(w.lower() for w in words)
    repeated_token_ratio = 1 - (len(word_counts) / n_words)

    url_density = len(_URL_RE.findall(paragraph)) / n_words
    citation_density = _citation_density(paragraph)
    has_markup = contains_markup(paragraph)
    punctuation_ratio = sum(1 for c in paragraph if c in ".,;:!?") / n_chars
    broken_word_ratio = _broken_word_ratio(paragraph)

    alpha_words = [re.sub(r"[^A-Za-z]", "", w) for w in words]
    alpha_words = [w for w in alpha_words if w]
    avg_word_len = sum(len(w) for w in alpha_words) / len(alpha_words) if alpha_words else 0.0
    stopword_hits = sum(1 for w in alpha_words if w.lower() in _ENGLISH_STOPWORDS)
    stopword_ratio = stopword_hits / len(alpha_words) if alpha_words else 0.0

    score = 100.0
    score -= (1 - english_prob) * 35
    score -= max(0.0, 0.5 - alphabetic_ratio) * 100
    score -= unusual_char_ratio * 120
    score -= max(0.0, repeated_token_ratio - 0.4) * 80
    score -= url_density * 150
    score -= citation_density * 60
    score -= 25 if has_markup else 0
    score -= max(0.0, punctuation_ratio - 0.15) * 100
    score -= broken_word_ratio * 100
    score -= max(0.0, avg_word_len - 7.0) * 6.0
    if len(alpha_words) >= 3 and stopword_ratio == 0.0:
        score -= 25.0
    if n_words == 1:
        score -= 65.0
    elif n_words == 2:
        score -= 30.0
    elif n_words <= 4:
        score -= 12.0
    score = max(0.0, min(100.0, score))

    breakdown = {
        "english_probability": round(english_prob, 3),
        "alphabetic_ratio": round(alphabetic_ratio, 3),
        "unusual_char_ratio": round(unusual_char_ratio, 3),
        "repeated_token_ratio": round(repeated_token_ratio, 3),
        "url_density": round(url_density, 3),
        "citation_density": round(citation_density, 3),
        "has_markup": has_markup,
        "punctuation_ratio": round(punctuation_ratio, 3),
        "broken_word_ratio": round(broken_word_ratio, 3),
        "avg_word_len": round(avg_word_len, 2),
        "stopword_ratio": round(stopword_ratio, 3),
    }
    return score, breakdown


_WORD_RE = re.compile(r"[a-z0-9]+")


def _normalize_for_hash(paragraph: str) -> str:
    return re.sub(r"\s+", " ", paragraph.strip().lower())


def _shingles(paragraph: str, k: int = 4) -> set:
    words = _WORD_RE.findall(paragraph.lower())
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def _simhash(shingles: set, bits: int = 64) -> int:
    if not shingles:
        return 0
    v = [0] * bits
    for sh in shingles:
        h = int(hashlib.blake2b(sh.encode("utf-8"), digest_size=8).hexdigest(), 16)
        for i in range(bits):
            v[i] += 1 if (h >> i) & 1 else -1
    fingerprint = 0
    for i in range(bits):
        if v[i] > 0:
            fingerprint |= (1 << i)
    return fingerprint


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def deduplicate_paragraphs(paragraphs: list, near_dup_hamming_threshold: int = 3) -> tuple:
    """Returns (kept_paragraphs, exact_removed_count, near_removed_count)."""
    seen_exact = set()
    stage1 = []
    exact_removed = 0
    for p in paragraphs:
        key = _normalize_for_hash(p)
        if key in seen_exact:
            exact_removed += 1
            continue
        seen_exact.add(key)
        stage1.append(p)

    fingerprints = [_simhash(_shingles(p)) for p in stage1]
    is_dup = [False] * len(stage1)

    band_bits = 16
    for band_idx in range(64 // band_bits):
        shift = band_idx * band_bits
        mask = (1 << band_bits) - 1
        buckets: dict = {}
        for i, fp in enumerate(fingerprints):
            if is_dup[i]:
                continue
            buckets.setdefault((fp >> shift) & mask, []).append(i)
        for idxs in buckets.values():
            if len(idxs) < 2:
                continue
            for a in range(len(idxs)):
                if is_dup[idxs[a]]:
                    continue
                for b in range(a + 1, len(idxs)):
                    j = idxs[b]
                    if is_dup[j]:
                        continue
                    if _hamming(fingerprints[idxs[a]], fingerprints[j]) <= near_dup_hamming_threshold:
                        is_dup[j] = True

    kept = [p for i, p in enumerate(stage1) if not is_dup[i]]
    near_removed = len(stage1) - len(kept)
    return kept, exact_removed, near_removed


def filter_and_score_text(
    text: str,
    *,
    english_only: bool = True,
    english_threshold: float = 0.55,
    remove_citations: bool = True,
    remove_low_quality: bool = True,
    deduplicate: bool = True,
    min_quality_score: float = 40.0,
    max_unusual_char_ratio: float = 0.15,
    max_examples: int = 60,
) -> QualityFilterResult:
    """Splits already-cleaned text into paragraphs, scores/filters each one,
    deduplicates the survivors, and rejoins them. Never raises on empty input."""
    original_chars = len(text)
    paragraphs = split_paragraphs(text)
    original_paragraph_count = len(paragraphs)

    survivors = []
    examples = []
    removed_non_english = 0
    removed_markup = 0
    removed_citations = 0
    removed_low_quality = 0

    for p in paragraphs:
        english_prob = english_probability(p) if english_only else 1.0
        score, breakdown = compute_quality_score(p, english_prob)

        keep = True
        reason = "kept"

        if breakdown["has_markup"]:
            keep, reason = False, "markup"
            removed_markup += 1
        elif english_only and english_prob < english_threshold:
            keep, reason = False, "non_english"
            removed_non_english += 1
        elif remove_citations and is_citation_fragment(p):
            keep, reason = False, "citation_fragment"
            removed_citations += 1
        elif breakdown["unusual_char_ratio"] > max_unusual_char_ratio:
            keep, reason = False, "unusual_characters"
            removed_low_quality += 1
        elif remove_low_quality and score < min_quality_score:
            keep, reason = False, "low_quality"
            removed_low_quality += 1

        if keep:
            survivors.append(p)
        if len(examples) < max_examples:
            examples.append({
                "original": p,
                "score": round(score, 1),
                "keep": keep,
                "reason": reason,
            })

    duplicates_removed = 0
    if deduplicate and survivors:
        survivors, exact_dup, near_dup = deduplicate_paragraphs(survivors)
        duplicates_removed = exact_dup + near_dup

    kept_text = "\n\n".join(survivors)

    stats = {
        "original_chars": original_chars,
        "clean_chars": len(kept_text),
        "original_paragraphs": original_paragraph_count,
        "remaining_paragraphs": len(survivors),
        "removed_paragraphs": original_paragraph_count - len(survivors),
        "removed_non_english": removed_non_english,
        "removed_markup": removed_markup,
        "removed_citations": removed_citations,
        "removed_low_quality": removed_low_quality,
        "duplicates_removed": duplicates_removed,
    }

    return QualityFilterResult(text=kept_text, stats=stats, examples=examples)


# --- Orchestration: source discovery -> extract -> clean -> filter -> chunk -> JSONL ---

@dataclass
class ProcessedDocument:
    source_path: str
    txt_path: str = None
    page_count: int = 0
    char_count: int = 0
    word_count: int = 0
    estimated_tokens: int = 0
    warnings: list = field(default_factory=list)
    cleaned_text: str = ""
    quality_stats: dict = field(default_factory=dict)
    quality_examples: list = field(default_factory=list)


@dataclass
class DatasetBuildResult:
    jsonl_path: str
    example_count: int
    total_estimated_tokens: int
    sample_examples: list
    skipped_documents: list
    quality_stats: dict = field(default_factory=dict)


def discover_source_files(paths: list) -> list:
    """Expand a mix of file paths and folder paths into a flat, de-duplicated
    list of files with an extension registered in EXTRACTOR_REGISTRY."""
    supported_exts = tuple(EXTRACTOR_REGISTRY.keys())
    found = []
    for p in paths:
        if not p:
            continue
        if os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in files:
                    if f.lower().endswith(supported_exts):
                        found.append(os.path.join(root, f))
        elif os.path.isfile(p):
            if p.lower().endswith(supported_exts):
                found.append(p)
            else:
                raise ValueError(
                    f"Unsupported file type for dataset prep: '{p}' (supported: {sorted(supported_exts)})"
                )
        else:
            raise FileNotFoundError(f"Path not found: {p}")

    deduped = sorted(set(os.path.abspath(f) for f in found))
    if not deduped:
        raise ValueError(f"No supported files found in the given selection (supported: {sorted(supported_exts)}).")
    return deduped


def process_document(
    file_path: str,
    txt_output_dir: str,
    remove_headers_footers: bool = True,
    remove_page_numbers: bool = True,
    normalize_unicode: bool = True,
    remove_markup: bool = True,
    remove_web_artifacts: bool = True,
    english_only: bool = True,
    english_threshold: float = 0.55,
    remove_citations: bool = True,
    remove_low_quality: bool = True,
    deduplicate: bool = True,
    min_quality_score: float = 40.0,
    max_unusual_char_ratio: float = 0.15,
) -> ProcessedDocument:
    extractor = get_extractor_for(file_path)
    extraction = extractor.extract(file_path)
    cleaning = clean_extraction(
        extraction,
        remove_headers_footers=remove_headers_footers,
        remove_page_numbers=remove_page_numbers,
        normalize_unicode=normalize_unicode,
        remove_markup=remove_markup,
        remove_web_artifacts=remove_web_artifacts,
    )

    warnings = list(cleaning.warnings)

    quality = filter_and_score_text(
        cleaning.text,
        english_only=english_only,
        english_threshold=english_threshold,
        remove_citations=remove_citations,
        remove_low_quality=remove_low_quality,
        deduplicate=deduplicate,
        min_quality_score=min_quality_score,
        max_unusual_char_ratio=max_unusual_char_ratio,
    )
    cleaned_text = quality.text

    if quality.stats["removed_paragraphs"] > 0:
        warnings.append(
            f"Quality filter removed {quality.stats['removed_paragraphs']} of "
            f"{quality.stats['original_paragraphs']} paragraph(s) "
            f"(non-English: {quality.stats['removed_non_english']}, "
            f"markup: {quality.stats['removed_markup']}, "
            f"citations: {quality.stats['removed_citations']}, "
            f"low quality: {quality.stats['removed_low_quality']}, "
            f"duplicates: {quality.stats['duplicates_removed']})."
        )

    txt_path = None
    if cleaned_text.strip():
        os.makedirs(txt_output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        txt_path = os.path.join(txt_output_dir, f"{base_name}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)

    tokenizer = get_tokenizer()
    estimated_tokens = len(tokenizer.encode(cleaned_text)) if cleaned_text.strip() else 0

    return ProcessedDocument(
        source_path=file_path,
        txt_path=txt_path,
        page_count=extraction.page_count,
        char_count=len(cleaned_text),
        word_count=len(cleaned_text.split()),
        estimated_tokens=estimated_tokens,
        warnings=warnings,
        cleaned_text=cleaned_text,
        quality_stats=quality.stats,
        quality_examples=quality.examples,
    )


def process_sources(paths: list, txt_output_dir: str, **cleaning_opts):
    """Returns (documents, errors) — errors are per-file failures that didn't stop the batch."""
    file_paths = discover_source_files(paths)
    documents = []
    errors = []

    for fp in file_paths:
        try:
            documents.append(process_document(fp, txt_output_dir, **cleaning_opts))
        except Exception as e:
            errors.append(f"{os.path.basename(fp)}: {e}")

    if not documents:
        detail = "\n".join(errors) if errors else "unknown error"
        raise RuntimeError(f"No documents could be processed.\n{detail}")

    return documents, errors


def build_jsonl_dataset(
    documents: list,
    jsonl_output_dir: str,
    dataset_filename: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    min_chunk_size: int = 64,
    sample_count: int = 5,
) -> DatasetBuildResult:
    if not documents:
        raise ValueError("No processed documents were provided to build a dataset from.")

    dataset_filename = (dataset_filename or "").strip()
    if not dataset_filename:
        raise ValueError("Dataset filename cannot be empty.")
    if not dataset_filename.lower().endswith(".jsonl"):
        dataset_filename += ".jsonl"

    os.makedirs(jsonl_output_dir, exist_ok=True)
    jsonl_path = os.path.join(jsonl_output_dir, dataset_filename)

    tokenizer = get_tokenizer()
    skipped = []
    all_examples = []

    for doc in documents:
        if not doc.cleaned_text.strip():
            skipped.append(os.path.basename(doc.source_path))
            continue
        chunks = chunk_text(
            doc.cleaned_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
            tokenizer=tokenizer,
        )
        all_examples.extend(c.text.strip() for c in chunks if c.text.strip())

    if not all_examples:
        raise RuntimeError(
            "Chunking produced zero examples — refusing to write an empty dataset. "
            "Check that the source files contain extractable text and that chunk settings are reasonable."
        )

    total_tokens = 0
    tmp_path = jsonl_path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            for text in all_examples:
                line = json.dumps({"text": text}, ensure_ascii=False)
                json.loads(line)
                f.write(line + "\n")
                total_tokens += len(tokenizer.encode(text))
    except Exception:
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)
        raise

    os.replace(tmp_path, jsonl_path)

    quality_stats = {}
    for doc in documents:
        for key, value in doc.quality_stats.items():
            quality_stats[key] = quality_stats.get(key, 0) + value

    return DatasetBuildResult(
        jsonl_path=jsonl_path,
        example_count=len(all_examples),
        total_estimated_tokens=total_tokens,
        sample_examples=all_examples[:sample_count],
        skipped_documents=skipped,
        quality_stats=quality_stats,
    )


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


# Popular, known-good text-pretraining datasets on the Hub, offered as a
# one-click preset in the UI instead of typing repo id/config/split/column
# from memory. "Custom..." leaves the fields as free text.
DATASET_PRESETS = {
    "Custom...": {},
    "TinyStories (roneneldan/TinyStories)": {
        "repo_id": "roneneldan/TinyStories", "config_name": "", "split": "train", "text_field": "text",
    },
    "WikiText-103 raw (wikitext)": {
        "repo_id": "wikitext", "config_name": "wikitext-103-raw-v1", "split": "train", "text_field": "text",
    },
    "OpenWebText (Skylion007/openwebtext)": {
        "repo_id": "Skylion007/openwebtext", "config_name": "", "split": "train", "text_field": "text",
    },
    "BookCorpus (bookcorpus)": {
        "repo_id": "bookcorpus", "config_name": "", "split": "train", "text_field": "text",
    },
    "C4 English, streamed subset (allenai/c4)": {
        "repo_id": "allenai/c4", "config_name": "en", "split": "train", "text_field": "text",
    },
    "Wikipedia English (wikimedia/wikipedia)": {
        "repo_id": "wikimedia/wikipedia", "config_name": "20231101.en", "split": "train", "text_field": "text",
    },
}


def download_hf_dataset(
    repo_id: str,
    config_name: str = None,
    split: str = "train",
    text_field: str = "text",
    max_examples: int = 5000,
    jsonl_output_dir: str = "datasets",
    dataset_filename: str = None,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    min_chunk_size: int = 64,
    normalize_unicode: bool = True,
    remove_markup: bool = True,
    remove_web_artifacts: bool = True,
    english_only: bool = True,
    english_threshold: float = 0.55,
    remove_citations: bool = True,
    remove_low_quality: bool = True,
    deduplicate: bool = True,
    min_quality_score: float = 40.0,
    max_unusual_char_ratio: float = 0.15,
) -> dict:
    """Downloads a dataset from the Hugging Face Hub, runs it through the SAME
    cleaning -> quality filtering -> deduplication -> chunking pipeline used
    for local PDF/image/text sources, and writes it out as local JSONL in the
    same {"text": ...} format. Never silently writes an empty or corrupted
    file — raises instead."""
    repo_id = (repo_id or "").strip()
    if not repo_id:
        raise ValueError("Hugging Face dataset repo id cannot be empty (e.g. 'roneneldan/TinyStories').")

    split = (split or "train").strip() or "train"
    text_field = (text_field or "text").strip() or "text"
    config_name = (config_name or "").strip() or None

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError("The 'datasets' library is required. Install it with: pip install datasets") from e

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

    raw_rows = []
    for i, row in enumerate(hf_dataset):
        if max_examples and i >= max_examples:
            break
        text = row.get(text_field)
        if text and str(text).strip():
            raw_rows.append(str(text).strip())

    if not raw_rows:
        raise RuntimeError(
            f"No usable examples found in '{repo_id}' (split='{split}', column='{text_field}') — "
            "refusing to write an empty dataset."
        )

    # Each HF row is treated as its own paragraph boundary, then run through
    # the exact same clean_extraction + filter_and_score_text pipeline that
    # PDF/image/text sources go through (single synthetic "page").
    combined_raw_text = "\n\n".join(raw_rows)
    synthetic_extraction = ExtractionResult(
        source_path=repo_id,
        page_count=1,
        pages=[ExtractedPage(index=0, text=combined_raw_text)],
        raw_text=combined_raw_text,
    )
    cleaning = clean_extraction(
        synthetic_extraction,
        remove_headers_footers=False,
        remove_page_numbers=False,
        normalize_unicode=normalize_unicode,
        remove_markup=remove_markup,
        remove_web_artifacts=remove_web_artifacts,
    )
    quality = filter_and_score_text(
        cleaning.text,
        english_only=english_only,
        english_threshold=english_threshold,
        remove_citations=remove_citations,
        remove_low_quality=remove_low_quality,
        deduplicate=deduplicate,
        min_quality_score=min_quality_score,
        max_unusual_char_ratio=max_unusual_char_ratio,
    )

    if not quality.text.strip():
        raise RuntimeError(
            f"Cleaning/quality filtering removed all text from '{repo_id}' — "
            "refusing to write an empty dataset. Try relaxing the quality controls."
        )

    tokenizer = get_tokenizer()
    doc = ProcessedDocument(
        source_path=repo_id,
        char_count=len(quality.text),
        word_count=len(quality.text.split()),
        estimated_tokens=len(tokenizer.encode(quality.text)),
        cleaned_text=quality.text,
        quality_stats=quality.stats,
        quality_examples=quality.examples,
    )

    result = build_jsonl_dataset(
        [doc],
        jsonl_output_dir=jsonl_output_dir,
        dataset_filename=dataset_filename,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=min_chunk_size,
    )

    return {
        "jsonl_path": result.jsonl_path,
        "example_count": result.example_count,
        "total_chars": len(quality.text),
        "total_estimated_tokens": result.total_estimated_tokens,
        "columns": hf_dataset.column_names,
        "sample_examples": result.sample_examples,
        "quality_stats": quality.stats,
        "quality_examples": quality.examples,
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

WEIGHTS_PATH = "TechcodeX_Weights.pt"  # default checkpoint name, kept for backward compatibility
CHECKPOINT_DIR = "checkpoints"
SAVE_EVERY = 25


def list_checkpoints() -> list:
    """All .pt files this app could have produced — the default WEIGHTS_PATH
    (if it exists) plus anything saved under CHECKPOINT_DIR/ under a name you
    chose when starting a training run."""
    found = []
    if os.path.isfile(WEIGHTS_PATH):
        found.append(WEIGHTS_PATH)
    if os.path.isdir(CHECKPOINT_DIR):
        for f in sorted(os.listdir(CHECKPOINT_DIR)):
            if f.lower().endswith(".pt"):
                found.append(os.path.join(CHECKPOINT_DIR, f))
    return found


def checkpoint_path_for_name(run_name: str) -> str:
    """Turns a user-chosen run name into a checkpoint file path. An empty/default
    name keeps using the original flat WEIGHTS_PATH so existing checkpoints
    from before this feature still resolve; any other name is namespaced
    under CHECKPOINT_DIR/ so multiple trained models don't collide."""
    run_name = (run_name or "").strip()
    if not run_name or run_name in ("default", "TechcodeX_Weights"):
        return WEIGHTS_PATH
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", run_name)
    if not safe_name.lower().endswith(".pt"):
        safe_name += ".pt"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    return os.path.join(CHECKPOINT_DIR, safe_name)

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
    weights_path: str = WEIGHTS_PATH,
):
    """Background training generator. Yields (step, loss) after every optimizer
    step so a UI can display a live-updating loss curve.

    `file_path` may be a single dataset file path or a list of paths
    (.txt/.jsonl), concatenated into one training corpus.

    `weights_path` is where this run's checkpoint is saved/resumed from —
    pass a different path per run (see checkpoint_path_for_name) to train
    multiple distinct models without overwriting each other.

    Starts a brand-new randomly-initialized model every call unless
    `resume=True`, in which case it loads `weights_path` (if present) and
    continues from its saved step count. `max_steps` is how many additional
    steps THIS call runs, not a total to reach.
    """
    gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))
    tokens = load_and_tokenize(file_path)

    start_step = 0
    if resume and os.path.isfile(weights_path):
        checkpoint = load_checkpoint(weights_path, map_location="cpu")
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
                weights_path,
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


def on_select_dataset_preset(preset_name):
    """Autofills the repo id / config / split / text-field boxes from DATASET_PRESETS."""
    preset = DATASET_PRESETS.get(preset_name) or {}
    return (
        gr.update(value=preset.get("repo_id", "")),
        gr.update(value=preset.get("config_name", "")),
        gr.update(value=preset.get("split", "train")),
        gr.update(value=preset.get("text_field", "text")),
    )


def _format_quality_stats(stats: dict) -> str:
    if not stats:
        return ""
    return (
        f"Original characters: {stats.get('original_chars', 0):,}\n"
        f"Clean characters: {stats.get('clean_chars', 0):,}\n"
        f"Original paragraphs: {stats.get('original_paragraphs', 0)}\n"
        f"Remaining paragraphs: {stats.get('remaining_paragraphs', 0)}\n"
        f"Removed paragraphs: {stats.get('removed_paragraphs', 0)}\n"
        f"Removed as non-English: {stats.get('removed_non_english', 0)}\n"
        f"Removed as markup: {stats.get('removed_markup', 0)}\n"
        f"Removed as citation/reference fragments: {stats.get('removed_citations', 0)}\n"
        f"Removed as low quality: {stats.get('removed_low_quality', 0)}\n"
        f"Duplicates removed: {stats.get('duplicates_removed', 0)}"
    )


def run_download_hf_dataset(
    repo_id, config_name, split, text_field, max_examples, jsonl_out_dir, dataset_filename,
    chunk_size, chunk_overlap, min_chunk_size,
    remove_markup, remove_web_artifacts, english_only, english_threshold,
    remove_citations, remove_low_quality, deduplicate, min_quality_score,
    max_unusual_char_ratio,
):
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
            chunk_overlap=int(chunk_overlap),
            min_chunk_size=int(min_chunk_size),
            remove_markup=remove_markup,
            remove_web_artifacts=remove_web_artifacts,
            english_only=english_only,
            english_threshold=float(english_threshold),
            remove_citations=remove_citations,
            remove_low_quality=remove_low_quality,
            deduplicate=deduplicate,
            min_quality_score=float(min_quality_score),
            max_unusual_char_ratio=float(max_unusual_char_ratio),
        )
    except Exception as e:
        return f"Download failed: {e}", ""

    status = (
        f"Downloaded '{repo_id}' -> {result['jsonl_path']}\n"
        f"JSONL examples (after cleaning/filtering/chunking): {result['example_count']}\n"
        f"Estimated total tokens: {result['total_estimated_tokens']}\n"
        f"Clean characters kept: {result['total_chars']:,}\n"
        f"Available columns: {', '.join(result['columns'])}\n\n"
        f"--- Quality filtering ---\n"
        f"{_format_quality_stats(result['quality_stats'])}"
    )
    sample_display = "\n\n---\n\n".join(
        json.dumps({"text": s}, ensure_ascii=False) for s in result["sample_examples"]
    )
    return status, sample_display


def _format_quality_examples(examples: list, limit: int = 15) -> str:
    if not examples:
        return "(no paragraphs to show)"
    blocks = []
    for ex in examples[:limit]:
        status = "KEEP" if ex["keep"] else f"REMOVE ({ex['reason']})"
        original = ex["original"][:400]
        blocks.append(f"[{status}] quality score={ex['score']}\n{original}")
    return "\n\n---\n\n".join(blocks)


PREVIEW_CHAR_LIMIT = 5000


def _doc_preview(doc):
    stats = (
        f"Pages: {doc.page_count}\n"
        f"Characters: {doc.char_count}\n"
        f"Words: {doc.word_count}\n"
        f"Estimated tokens: {doc.estimated_tokens}\n"
        f"Warnings: {'; '.join(doc.warnings) if doc.warnings else 'none'}\n"
        f"TXT saved to: {doc.txt_path or '(not saved — no extractable text found)'}\n\n"
        f"--- Quality filtering ---\n"
        f"{_format_quality_stats(doc.quality_stats)}"
    )
    preview_text = doc.cleaned_text[:PREVIEW_CHAR_LIMIT]
    if len(doc.cleaned_text) > PREVIEW_CHAR_LIMIT:
        preview_text += "\n\n... (preview truncated, full text saved to the .txt file)"
    quality_examples = _format_quality_examples(doc.quality_examples)
    return preview_text, stats, quality_examples


def run_extract_and_clean(
    files_multi, files_dir, remove_hf, remove_pn, norm_unicode, txt_out_dir,
    remove_markup, remove_web_artifacts, english_only, english_threshold,
    remove_citations, remove_low_quality, deduplicate, min_quality_score,
    max_unusual_char_ratio,
):
    paths = list(files_multi or []) + list(files_dir or [])

    if not paths:
        return [], gr.update(choices=[], value=None), "Please upload at least one PDF/image/text file or a folder of them.", "", "", ""

    try:
        documents, errors = process_sources(
            paths,
            txt_output_dir=(txt_out_dir or "cleaned").strip() or "cleaned",
            remove_headers_footers=remove_hf,
            remove_page_numbers=remove_pn,
            normalize_unicode=norm_unicode,
            remove_markup=remove_markup,
            remove_web_artifacts=remove_web_artifacts,
            english_only=english_only,
            english_threshold=float(english_threshold),
            remove_citations=remove_citations,
            remove_low_quality=remove_low_quality,
            deduplicate=deduplicate,
            min_quality_score=float(min_quality_score),
            max_unusual_char_ratio=float(max_unusual_char_ratio),
        )
    except Exception as e:
        return [], gr.update(choices=[], value=None), f"Extraction failed: {e}", "", "", ""

    lines = [f"Processed {len(documents)} of {len(documents) + len(errors)} file(s):"]
    for doc in documents:
        name = os.path.basename(doc.source_path)
        status = "OK" if doc.txt_path else "NO EXTRACTABLE TEXT"
        lines.append(
            f"  - {name}: {status} | pages={doc.page_count} chars={doc.char_count} "
            f"words={doc.word_count} est_tokens={doc.estimated_tokens}"
        )
        for w in doc.warnings:
            lines.append(f"      warning: {w}")

    if errors:
        lines.append("\nFiles that failed to process:")
        for err in errors:
            lines.append(f"  - {err}")

    choices = [os.path.basename(d.source_path) for d in documents]
    preview_text, stats, quality_examples = _doc_preview(documents[0]) if documents else ("", "", "")

    return (
        documents,
        gr.update(choices=choices, value=choices[0] if choices else None),
        "\n".join(lines),
        preview_text,
        stats,
        quality_examples,
    )


def on_select_preview_doc(selected_name, documents):
    for doc in (documents or []):
        if os.path.basename(doc.source_path) == selected_name:
            return _doc_preview(doc)
    return "", "", ""


def run_build_dataset(documents, chunk_size, chunk_overlap, min_chunk_size, jsonl_out_dir, dataset_filename):
    if not documents:
        return "No processed documents available. Run 'Extract & Clean' first.", ""

    try:
        result = build_jsonl_dataset(
            documents,
            jsonl_output_dir=(jsonl_out_dir or "datasets").strip() or "datasets",
            dataset_filename=(dataset_filename or "dataset.jsonl").strip() or "dataset.jsonl",
            chunk_size=int(chunk_size),
            chunk_overlap=int(chunk_overlap),
            min_chunk_size=int(min_chunk_size),
        )
    except Exception as e:
        return f"Dataset build failed: {e}", ""

    status = (
        f"Dataset written to: {result.jsonl_path}\n"
        f"Examples: {result.example_count}\n"
        f"Estimated total tokens: {result.total_estimated_tokens}"
    )
    if result.skipped_documents:
        status += f"\nSkipped (no usable text): {', '.join(result.skipped_documents)}"
    if result.quality_stats:
        status += f"\n\n--- Quality filtering (aggregated across documents) ---\n{_format_quality_stats(result.quality_stats)}"

    sample_display = "\n\n---\n\n".join(
        json.dumps({"text": s}, ensure_ascii=False) for s in result.sample_examples
    )

    return status, sample_display


def start_training(
    file_objs, existing_selected, batch_size, block_size, max_steps, learning_rate, resume,
    gradient_checkpointing, gradient_accumulation_steps, use_fp16, offload_optimizer_to_cpu,
    run_name,
):
    file_paths = list(file_objs or []) + list(existing_selected or [])
    if not file_paths:
        yield 0, 0.0, pd.DataFrame({"step": [], "loss": []}), \
            "Please upload dataset file(s) or select existing ones from the list first."
        return

    weights_path = checkpoint_path_for_name(run_name)
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
            weights_path=weights_path,
        ):
            history["step"].append(step)
            history["loss"].append(loss)
            df = pd.DataFrame(history)
            status = f"Step {step} — loss {loss:.4f} — saving to '{weights_path}'"
            yield step, loss, df, status
    except Exception as e:
        yield 0, 0.0, pd.DataFrame(history), f"Training failed: {e}"
        return

    yield history["step"][-1] if history["step"] else 0, \
        history["loss"][-1] if history["loss"] else 0.0, \
        pd.DataFrame(history), \
        f"Training complete. Weights saved to '{weights_path}' (total step count: {history['step'][-1] if history['step'] else 0})."


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


def refresh_checkpoint_list():
    choices = list_checkpoints()
    value = choices[0] if choices else None
    return gr.update(choices=choices, value=value)


def check_weights_exist(checkpoint_path):
    checkpoint_path = checkpoint_path or WEIGHTS_PATH
    if os.path.isfile(checkpoint_path):
        size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
        return f"Found '{checkpoint_path}' ({size_mb:.2f} MB)."
    return f"No weights file found at '{checkpoint_path}'. Train a model first."


def show_model_info(checkpoint_path):
    checkpoint_path = checkpoint_path or WEIGHTS_PATH
    if not os.path.isfile(checkpoint_path):
        return f"No weights file found at '{checkpoint_path}'. Train a model first."

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", {})
    config_dict = checkpoint.get("config", TechcodeXConfig().__dict__)

    total_params = sum(t.numel() for t in state_dict.values())
    file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)

    return (
        f"Parameters: {format_param_count(total_params)} ({total_params:,} total)\n"
        f"Checkpoint file size on disk: {file_size_mb:.2f} MB\n"
        f"Trained for {checkpoint.get('step', '?')} steps\n"
        f"Config: n_embd={config_dict.get('n_embd')}, n_head={config_dict.get('n_head')}, "
        f"n_layer={config_dict.get('n_layer')}, block_size={config_dict.get('block_size')}, "
        f"vocab_size={config_dict.get('vocab_size')}"
    )


def export_hf_bundle(checkpoint_path):
    checkpoint_path = checkpoint_path or WEIGHTS_PATH
    if not os.path.isfile(checkpoint_path):
        return f"Cannot export — '{checkpoint_path}' does not exist yet. Train a model first."

    os.makedirs(EXPORT_DIR, exist_ok=True)

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
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

    shutil.copy2(checkpoint_path, os.path.join(EXPORT_DIR, "pytorch_model.bin"))

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


def upload_to_hf_hub(checkpoint_path, repo_id, hf_token, private, commit_message):
    checkpoint_path = checkpoint_path or WEIGHTS_PATH
    if not os.path.isfile(checkpoint_path):
        yield f"Cannot upload — '{checkpoint_path}' does not exist yet. Train a model first."
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
    export_hf_bundle(checkpoint_path)

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
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


# Keyed by checkpoint path so switching the model dropdown loads the right
# weights instead of reusing whatever was loaded first.
_chat_model_cache = {}


def _load_chat_model(checkpoint_path: str):
    checkpoint_path = checkpoint_path or WEIGHTS_PATH
    if not os.path.isfile(checkpoint_path):
        return None
    if checkpoint_path in _chat_model_cache:
        return _chat_model_cache[checkpoint_path]

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    config_dict = checkpoint.get("config", TechcodeXConfig().__dict__)
    config = TechcodeXConfig(**config_dict)

    model = TechcodeXModel(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _chat_model_cache[checkpoint_path] = model
    return model


def _read_context_file(context_file) -> str:
    """Reads an uploaded file's text for the model to reference. Only plain-text
    formats are read directly (.txt/.jsonl/.md/etc); a .pdf reuses the same
    PDFExtractor as Tab 0."""
    if not context_file:
        return ""
    path = context_file if isinstance(context_file, str) else getattr(context_file, "name", None)
    if not path or not os.path.isfile(path):
        return ""
    if path.lower().endswith(".pdf"):
        try:
            return PDFExtractor().extract(path).raw_text
        except Exception as e:
            return f"[Could not read PDF: {e}]"
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception as e:
        return f"[Could not read file: {e}]"


def chat_with_techcodex(message, temperature, history, checkpoint_path, context_file):
    history = history or []

    model = _load_chat_model(checkpoint_path)
    if model is None:
        history.append({"role": "user", "content": message})
        history.append({
            "role": "assistant",
            "content": f"No trained weights found at '{checkpoint_path or WEIGHTS_PATH}'. "
                       f"Train a model in Tab 1 first, or pick a different checkpoint above.",
        })
        return history, ""

    tokenizer = get_tokenizer()
    block_size = model.config.block_size

    context_text = _read_context_file(context_file)
    prompt = message
    if context_text.strip():
        # Reserve room for the question itself and the reply; whatever's left
        # of the context window is spent on the attached file's content, most
        # recent (i.e. tail-truncated least) tokens first since a from-scratch
        # small model attends best to nearby context.
        reserved = len(tokenizer.encode(message)) + 60
        budget = max(0, block_size - reserved)
        context_ids = tokenizer.encode(context_text)[:budget]
        if context_ids:
            prompt = (
                "Reference document:\n" + tokenizer.decode(context_ids) +
                "\n\nQuestion: " + message + "\nAnswer:"
            )

    input_ids = tokenizer.encode(prompt)[-block_size:]
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)

    out_idx = model.generate(idx, max_new_tokens=60, temperature=float(temperature))
    generated_ids = out_idx[0].tolist()[len(input_ids):]
    reply = tokenizer.decode(generated_ids, skip_special_tokens=True)

    history.append({"role": "user", "content": message + (" 📎" if context_text.strip() else "")})
    history.append({"role": "assistant", "content": reply.strip() or "(empty response)"})
    return history, ""


with gr.Blocks(title="TechcodeX") as demo:
    gr.Markdown("# ⚡ TechcodeX (single file)\nFrom-scratch ~1B-parameter GPT-style pre-training & inference dashboard.")
    gr.Markdown(f"**Device:** `{device}` (`{_DEVICE_KIND}`){' — running in Colab' if _IN_COLAB else ''}")

    with gr.Tabs():
        with gr.Tab("Tab 0: Dataset Preparation (PDF/Image/Text → JSONL)"):
            processed_docs_state = gr.State([])

            gr.Markdown("### 1. Import PDFs / Images / Text")
            with gr.Row():
                pdf_files = gr.File(
                    label="Upload PDF, image (PNG/JPG), or .txt file(s)",
                    file_count="multiple",
                    file_types=[".pdf", ".png", ".jpg", ".jpeg", ".txt"],
                )
                pdf_folder = gr.File(
                    label="Or select an entire folder of PDFs/images/text",
                    file_count="directory",
                )
            gr.Markdown(
                "*Images are processed with OCR (Tesseract) since they have no native text layer — "
                "requires the Tesseract OCR engine installed separately (`!apt-get install -y "
                "tesseract-ocr` on Colab, or see setup_env.ps1 on Windows). `.txt` files skip OCR "
                "but still run through the same cleaning pass.*"
            )

            gr.Markdown("### 2. Cleaning options")
            with gr.Row():
                remove_hf_checkbox = gr.Checkbox(value=True, label="Remove repeated headers/footers")
                remove_pn_checkbox = gr.Checkbox(value=True, label="Remove standalone page numbers")
                norm_unicode_checkbox = gr.Checkbox(value=True, label="Normalize Unicode")

            gr.Markdown(
                "### 2b. Quality filtering\n"
                "Catches garbage — markup, web boilerplate, citation-only fragments, "
                "corrupted extraction text, non-English text, and duplicates — before it "
                "reaches chunking/tokenization."
            )
            with gr.Row():
                english_only_checkbox = gr.Checkbox(value=True, label="English Only")
                remove_markup_checkbox = gr.Checkbox(value=True, label="Remove HTML/SVG/XML")
                remove_web_artifacts_checkbox = gr.Checkbox(value=True, label="Remove obvious web artifacts")
                remove_citations_checkbox = gr.Checkbox(value=True, label="Remove citation/reference-only fragments")
            with gr.Row():
                remove_low_quality_checkbox = gr.Checkbox(value=True, label="Remove corrupted/low-quality text")
                deduplicate_checkbox = gr.Checkbox(value=True, label="Deduplicate")
            with gr.Row():
                english_threshold_slider = gr.Slider(0.0, 1.0, value=0.55, step=0.05, label="English confidence threshold")
                min_quality_score_slider = gr.Slider(0, 100, value=40, step=1, label="Minimum quality score")
                max_unusual_char_ratio_slider = gr.Slider(0.0, 1.0, value=0.15, step=0.01, label="Maximum unusual-character ratio")

            txt_out_dir_box = gr.Textbox(value="cleaned", label="TXT Output Directory")
            extract_button = gr.Button("Extract & Clean", variant="primary")
            extract_status = gr.Textbox(label="Extraction Log", interactive=False, lines=8)

            gr.Markdown("### 3. Preview extracted text")
            with gr.Row():
                with gr.Column(scale=1):
                    doc_selector = gr.Dropdown(label="Processed file", choices=[], interactive=True)
                    doc_stats = gr.Textbox(label="Statistics", interactive=False, lines=15)
                with gr.Column(scale=2):
                    doc_preview = gr.Textbox(label="Cleaned Text Preview (surviving text)", interactive=False, lines=12)
                    quality_examples_display = gr.Textbox(
                        label="Per-paragraph quality decisions (Original / Score / Keep-Remove)",
                        interactive=False, lines=16,
                    )

            gr.Markdown("### 4. Chunking & JSONL export")
            with gr.Row():
                chunk_size_slider = gr.Slider(32, 2048, value=1024, step=16, label="Chunk Size (tokens)")
                chunk_overlap_slider = gr.Slider(0, 256, value=50, step=8, label="Chunk Overlap (tokens)")
                min_chunk_size_slider = gr.Slider(0, 256, value=64, step=8, label="Minimum Chunk Size (tokens)")
            with gr.Row():
                jsonl_out_dir_box = gr.Textbox(value="datasets", label="JSONL Output Directory")
                dataset_filename_box = gr.Textbox(value="dataset.jsonl", label="Dataset Filename")

            build_dataset_button = gr.Button("Build JSONL Dataset", variant="primary")
            dataset_status = gr.Textbox(label="Dataset Build Log", interactive=False, lines=5)
            dataset_sample = gr.Textbox(label="Sample JSONL Examples", interactive=False, lines=10)

            extract_button.click(
                fn=run_extract_and_clean,
                inputs=[
                    pdf_files, pdf_folder, remove_hf_checkbox, remove_pn_checkbox, norm_unicode_checkbox, txt_out_dir_box,
                    remove_markup_checkbox, remove_web_artifacts_checkbox, english_only_checkbox, english_threshold_slider,
                    remove_citations_checkbox, remove_low_quality_checkbox, deduplicate_checkbox,
                    min_quality_score_slider, max_unusual_char_ratio_slider,
                ],
                outputs=[processed_docs_state, doc_selector, extract_status, doc_preview, doc_stats, quality_examples_display],
            )
            doc_selector.change(
                fn=on_select_preview_doc,
                inputs=[doc_selector, processed_docs_state],
                outputs=[doc_preview, doc_stats, quality_examples_display],
            )
            build_dataset_button.click(
                fn=run_build_dataset,
                inputs=[processed_docs_state, chunk_size_slider, chunk_overlap_slider, min_chunk_size_slider, jsonl_out_dir_box, dataset_filename_box],
                outputs=[dataset_status, dataset_sample],
            )

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
                        refresh_datasets_button = gr.Button("Refresh", scale=1)
                    existing_datasets_checkbox = gr.Dropdown(
                        choices=[os.path.relpath(f) for f in list_existing_dataset_files("datasets")],
                        multiselect=True,
                        label="Dataset(s) to train on (already prepared — from Tab 0, an HF download, or a previous upload)",
                    )

                    gr.Markdown("**Or download a dataset from the Hugging Face Hub:**")
                    hf_preset_dropdown = gr.Dropdown(
                        choices=list(DATASET_PRESETS.keys()),
                        value="Custom...",
                        label="Preset dataset (pick one, or leave as Custom... and fill in below)",
                    )
                    with gr.Row():
                        hf_repo_id_box = gr.Textbox(label="Dataset repo id", placeholder="e.g. roneneldan/TinyStories")
                        hf_config_box = gr.Textbox(label="Config name (optional)")
                        hf_split_box = gr.Textbox(value="train", label="Split")
                        hf_text_field_box = gr.Textbox(value="text", label="Text column")
                    with gr.Row():
                        hf_max_examples_box = gr.Number(value=5000, label="Max examples (0 = all)", precision=0)
                        hf_jsonl_out_dir_box = gr.Textbox(value="datasets", label="Output dir")
                        hf_dataset_filename_box = gr.Textbox(label="Filename (optional)")
                    with gr.Row():
                        hf_chunk_size_box = gr.Number(value=1024, label="Chunk size (tokens)", precision=0)
                        hf_chunk_overlap_box = gr.Number(value=50, label="Chunk overlap (tokens)", precision=0)
                        hf_min_chunk_size_box = gr.Number(value=64, label="Min chunk size (tokens)", precision=0)
                    with gr.Row():
                        hf_english_only_checkbox = gr.Checkbox(value=True, label="English Only")
                        hf_remove_markup_checkbox = gr.Checkbox(value=True, label="Remove HTML/SVG/XML")
                        hf_remove_web_artifacts_checkbox = gr.Checkbox(value=True, label="Remove web artifacts")
                        hf_remove_citations_checkbox = gr.Checkbox(value=True, label="Remove citation fragments")
                        hf_remove_low_quality_checkbox = gr.Checkbox(value=True, label="Remove low-quality text")
                        hf_deduplicate_checkbox = gr.Checkbox(value=True, label="Deduplicate")
                    with gr.Row():
                        hf_english_threshold_slider = gr.Slider(0.0, 1.0, value=0.55, step=0.05, label="English confidence threshold")
                        hf_min_quality_score_slider = gr.Slider(0, 100, value=40, step=1, label="Minimum quality score")
                        hf_max_unusual_char_ratio_slider = gr.Slider(0.0, 1.0, value=0.15, step=0.01, label="Max unusual-character ratio")
                    hf_download_button = gr.Button("Download Dataset from Hugging Face", variant="secondary")
                    hf_download_status = gr.Textbox(label="Download Log", interactive=False, lines=6)
                    hf_download_sample = gr.Textbox(label="Sample JSONL Examples", interactive=False, lines=6)

                    hf_preset_dropdown.change(
                        fn=on_select_dataset_preset,
                        inputs=hf_preset_dropdown,
                        outputs=[hf_repo_id_box, hf_config_box, hf_split_box, hf_text_field_box],
                    )

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
                    gradient_checkpointing_checkbox = gr.Checkbox(value=True, label="Gradient Checkpointing (CUDA/DirectML/CPU only — no-op on TPU)")
                    gradient_accumulation_slider = gr.Slider(1, 128, value=8, step=1, label="Gradient Accumulation Steps")
                    use_fp16_checkbox = gr.Checkbox(value=False, label="Train in fp16 (DirectML/CUDA; ignored on TPU)")
                    offload_optimizer_checkbox = gr.Checkbox(value=False, label="Offload Optimizer State to CPU (DirectML only)")
                    model_size_display = gr.Textbox(
                        label="Model Size (estimated from current config)",
                        value=estimate_model_size(1024, 1536, 24, 34),
                        interactive=False,
                        lines=3,
                    )
                    run_name_box = gr.Textbox(
                        value="default",
                        label="Checkpoint name for this model",
                        info="Use a different name per model to train several distinct models without "
                             "overwriting each other (saved under checkpoints/<name>.pt). Leave as "
                             "'default' to use the original TechcodeX_Weights.pt.",
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
                    hf_max_examples_box, hf_jsonl_out_dir_box, hf_dataset_filename_box,
                    hf_chunk_size_box, hf_chunk_overlap_box, hf_min_chunk_size_box,
                    hf_remove_markup_checkbox, hf_remove_web_artifacts_checkbox,
                    hf_english_only_checkbox, hf_english_threshold_slider,
                    hf_remove_citations_checkbox, hf_remove_low_quality_checkbox, hf_deduplicate_checkbox,
                    hf_min_quality_score_slider, hf_max_unusual_char_ratio_slider,
                ],
                outputs=[hf_download_status, hf_download_sample],
            )

            train_button.click(
                fn=start_training,
                inputs=[
                    file_upload, existing_datasets_checkbox, batch_size_slider, block_size_slider,
                    max_steps_slider, lr_slider, resume_checkbox,
                    gradient_checkpointing_checkbox, gradient_accumulation_slider, use_fp16_checkbox,
                    offload_optimizer_checkbox, run_name_box,
                ],
                outputs=[step_display, loss_display, loss_plot, training_status],
            )

        with gr.Tab("Tab 2: Model Testing & Weights"):
            with gr.Row():
                checkpoint_dropdown_tab2 = gr.Dropdown(
                    choices=list_checkpoints(), label="Checkpoint", scale=3,
                    value=(list_checkpoints() or [None])[0],
                )
                refresh_checkpoints_button_tab2 = gr.Button("Refresh", scale=1)
            refresh_checkpoints_button_tab2.click(fn=refresh_checkpoint_list, outputs=checkpoint_dropdown_tab2)

            with gr.Row():
                with gr.Column():
                    check_button = gr.Button("Check Weights File")
                    check_output = gr.Textbox(label="Weights Status", interactive=False)
                    check_button.click(fn=check_weights_exist, inputs=checkpoint_dropdown_tab2, outputs=check_output)

                with gr.Column():
                    export_button = gr.Button("Export Hugging Face-style Bundle")
                    export_output = gr.Textbox(label="Export Status", interactive=False)
                    export_button.click(fn=export_hf_bundle, inputs=checkpoint_dropdown_tab2, outputs=export_output)

                with gr.Column():
                    model_info_button = gr.Button("Show Model Size")
                    model_info_output = gr.Textbox(label="Model Size", interactive=False, lines=5)
                    model_info_button.click(fn=show_model_info, inputs=checkpoint_dropdown_tab2, outputs=model_info_output)

        with gr.Tab("Tab 3: Chat with Eather"):
            with gr.Row():
                checkpoint_dropdown_tab3 = gr.Dropdown(
                    choices=list_checkpoints(), label="Model to chat with", scale=3,
                    value=(list_checkpoints() or [None])[0],
                )
                refresh_checkpoints_button_tab3 = gr.Button("Refresh", scale=1)
            refresh_checkpoints_button_tab3.click(fn=refresh_checkpoint_list, outputs=checkpoint_dropdown_tab3)

            context_file_upload = gr.File(
                label="Attach a file for the model to reference (optional) — .txt/.jsonl/.md/.pdf",
                file_types=[".txt", ".jsonl", ".md", ".pdf"],
            )
            gr.Markdown(
                "*The attached file's text is fed in as context so the model can answer questions "
                "about it — limited by the model's context length (block_size), so only the portion "
                "that fits is used. A from-scratch small model's ability to actually use that context "
                "well depends heavily on how much it's been trained.*"
            )

            chatbot = gr.Chatbot(label="Eather (TechcodeX)", height=400)
            with gr.Row():
                chat_input = gr.Textbox(label="Message", placeholder="Say something...", scale=4)
                send_button = gr.Button("Send", variant="primary", scale=1)
            temperature_slider = gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="Creativity Temperature")

            chat_inputs = [chat_input, temperature_slider, chatbot, checkpoint_dropdown_tab3, context_file_upload]
            send_button.click(fn=chat_with_techcodex, inputs=chat_inputs, outputs=[chatbot, chat_input])
            chat_input.submit(fn=chat_with_techcodex, inputs=chat_inputs, outputs=[chatbot, chat_input])

        with gr.Tab("Tab 4: Upload to Hugging Face"):
            gr.Markdown(
                "Pushes the trained checkpoint to the Hugging Face Hub as a model repo "
                "(config.json + pytorch_model.bin + an auto-generated model card)."
            )
            with gr.Row():
                checkpoint_dropdown_tab4 = gr.Dropdown(
                    choices=list_checkpoints(), label="Checkpoint to upload", scale=3,
                    value=(list_checkpoints() or [None])[0],
                )
                refresh_checkpoints_button_tab4 = gr.Button("Refresh", scale=1)
            refresh_checkpoints_button_tab4.click(fn=refresh_checkpoint_list, outputs=checkpoint_dropdown_tab4)

            with gr.Row():
                hf_upload_repo_id_box = gr.Textbox(label="Repo id", placeholder="e.g. yourname/techcodex-model", scale=2)
                hf_upload_private_checkbox = gr.Checkbox(value=True, label="Private repository", scale=1)
            hf_upload_token_box = gr.Textbox(label="Hugging Face access token (write scope)", placeholder="hf_...", type="password")
            hf_upload_commit_message_box = gr.Textbox(label="Commit message (optional)", placeholder="Upload TechcodeX checkpoint")
            hf_upload_button = gr.Button("Upload to Hugging Face Hub", variant="primary")
            hf_upload_status = gr.Textbox(label="Upload Status", interactive=False, lines=5)

            hf_upload_button.click(
                fn=upload_to_hf_hub,
                inputs=[checkpoint_dropdown_tab4, hf_upload_repo_id_box, hf_upload_token_box, hf_upload_private_checkbox, hf_upload_commit_message_box],
                outputs=hf_upload_status,
            )


if __name__ == "__main__":
    # Colab has no direct access to localhost, so a Gradio public share link
    # is required there; locally, default to a plain local server.
    demo.queue().launch(theme=gr.themes.Soft(primary_hue="violet"), share=_IN_COLAB)
