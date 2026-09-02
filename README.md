# TechcodeX

A from-scratch, raw-PyTorch GPT-style language model — no Hugging Face model classes, no Unsloth, no pre-made wrappers around the architecture itself — with a full Gradio dashboard for preparing data, pre-training, testing, and publishing the model. Runs on **Google Colab (TPU, GPU, or CPU)** or **locally on a Windows PC** (NVIDIA CUDA, AMD via DirectML, or CPU).

## What it is

TechcodeX is a decoder-only transformer (GPT-style) built from first principles: causal self-attention, pre-LayerNorm blocks, a gated SiLU feed-forward network, and tied input/output embeddings. The default configuration is **~1.04B parameters** (`n_embd=1536, n_head=24, n_layer=34, block_size=1024`), but every dimension is adjustable from the UI with a live parameter-count/memory estimate.

Everything — model, training loop, dataset preparation, and the dashboard — lives in one file: [`techcodex_single_file.py`](techcodex_single_file.py). That's deliberate: it's meant to be dropped straight into a single Colab cell.

## Where it runs

Device selection is automatic, in this priority order:

1. **TPU** (Google Colab, via `torch_xla`)
2. **CUDA** (NVIDIA GPU)
3. **DirectML** (AMD GPU, Windows only)
4. **CPU** (fallback everywhere)

The training loop adapts to whichever backend it finds — e.g. TPU training uses `torch_xla`'s lazy-tensor execution (`xm.optimizer_step`, `xm.mark_step`) and skips gradient checkpointing (unsupported on XLA), while DirectML training uses hand-written AdamW variants that avoid PyTorch ops DirectML has no kernel for.

## Features

**Dataset preparation (Tab 0)**
- PDF text extraction (PyMuPDF) and image OCR (Tesseract) for scanned/photographed sources
- Text cleaning: header/footer stripping, page-number removal, markup/web-boilerplate stripping, hyphenation repair, paragraph reflow
- Quality filtering: language detection, citation/reference-fragment removal, corrupted-text detection, near-duplicate removal (simhash + LSH)
- Paragraph-aware chunking into training-ready JSONL

**Training data (Tab 1)**
- Upload your own `.txt`/`.jsonl` files, or pick from anything already prepared via a single multi-select dropdown
- One-click dataset presets from the Hugging Face Hub (TinyStories, WikiText-103, OpenWebText, BookCorpus, C4, Wikipedia) — or any custom repo id — run through the same cleaning/quality pipeline as local files
- Live model-size estimator as you tune `n_embd`/`n_head`/`n_layer`/`block_size`
- Memory controls: gradient checkpointing, gradient accumulation, fp16 with dynamic loss scaling, optimizer-state CPU offload
- Name each training run so multiple distinct models can coexist as separate checkpoints instead of overwriting one file

**Testing & chat (Tabs 2–3)**
- Inspect any saved checkpoint's parameter count, size, and training step count
- Chat with the model, with an optional file attachment (`.txt`/`.jsonl`/`.md`/`.pdf`) fed in as context

**Publishing (Tab 4)**
- Export a Hugging Face-style bundle (`config.json` + `pytorch_model.bin`) or push it straight to the Hugging Face Hub with an auto-generated model card

## Quick start

**Google Colab (TPU)**
```python
!wget -q https://github.com/zarko3/techcodexai/archive/refs/heads/main.zip -O repo.zip
!unzip -q repo.zip
%cd techcodexai-main

!pip install -q torch==2.8.0 "torch_xla[tpu]==2.8.0" -f https://storage.googleapis.com/libtpu-releases/index.html
!pip install -q transformers datasets gradio huggingface_hub

!python techcodex_single_file.py
```
Set the runtime to TPU first (Runtime → Change runtime type → TPU). Check `!pip show torch_xla` to confirm the version and match the `torch==` pin to it — they must be installed together at the same version, since `torch_xla`'s compiled extension is built against one exact torch release.

**Google Colab (GPU/CPU)** — same steps, skip the `torch_xla` install; it'll fall back to CUDA if available, or CPU otherwise.

**Local Windows**
```powershell
pip install torch transformers datasets gradio huggingface_hub pymupdf pytesseract Pillow langdetect
pip install torch-directml   # optional, for AMD GPUs
python techcodex_single_file.py
```

Either way, it opens a Gradio dashboard (a public `*.gradio.live` link on Colab, a local URL on Windows).

## Repo layout

- **`techcodex_single_file.py`** — the whole program, self-contained, for Colab or a quick local run.
- **`modeling_techcodex.py`, `trainer_backend.py`, `app_ui.py`, `dataset_prep/`** — the same functionality split into modules, for local Windows development.
- **`setup_env.ps1`** — installs local Windows dependencies, including `torch-directml` for AMD GPUs.
