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

!pip install -q torch==2.8.0 "torch_xla[tpu]==2.8.0" transformers datasets gradio huggingface_hub -f https://storage.googleapis.com/libtpu-releases/index.html

!python techcodex_single_file.py
```
Set the runtime to TPU first (Runtime → Change runtime type → TPU). All packages are installed in **one** `pip install` call — running torch/torch_xla and the rest as two separate calls is what caused Gradio to silently not install in earlier versions of this guide. If `!pip show torch_xla` shows a version other than 2.8.0 already on the image, match the `torch==` pin to it instead — they must be installed together at the same version, since `torch_xla`'s compiled extension is built against one exact torch release.

**Google Colab (GPU/CPU)** — same steps, skip the `torch_xla` install; it'll fall back to CUDA if available, or CPU otherwise.

**Local Windows**
```powershell
pip install torch transformers datasets gradio huggingface_hub pymupdf pytesseract Pillow langdetect
pip install torch-directml   # optional, for AMD GPUs
python techcodex_single_file.py
```

Either way, it opens a Gradio dashboard (a public `*.gradio.live` link on Colab, a local URL on Windows).

## Troubleshooting (Colab)

**`WARNING: Ignoring invalid distribution ~orch (...)`** during install — harmless leftover from an earlier interrupted install (a partial `torch` package folder got left behind, showing up as `~orch`). It doesn't block the current install. Optional cleanup if it bothers you:
```python
!rm -rf /usr/local/lib/python3.13/dist-packages/~orch*
```

**`ImportError: ... undefined symbol ...` when importing `torch_xla`** — a torch/torch_xla version mismatch (only comes up if you install a different `torch_xla` version than the quick-start command above, or upgrade one without the other). `torch_xla`'s compiled extension is built against one exact torch release and doesn't declare torch as a pip dependency, so both must be installed together, pinned to the same version:
```python
!pip uninstall -y -q torch torch_xla
!pip install -q torch==2.8.0 "torch_xla[tpu]==2.8.0" -f https://storage.googleapis.com/libtpu-releases/index.html
```
Run `!pip show torch_xla` first if unsure which version is already on the image, and match the `torch==` pin to it.

**No public `*.gradio.live` link, only a `127.0.0.1` local URL** — means the app didn't detect it's running in Colab. This only happens with very old pulls of this repo (fixed by checking Colab's environment variables instead of `sys.modules`, since `!python file.py` runs in a subprocess that doesn't inherit the notebook kernel's injected modules) — pull the latest code.

**Training sits at "Current Step: 0" for several minutes on TPU** — expected. The first step compiles the entire computation graph via XLA before anything executes, which can take minutes for a model this size. It's only actually stuck if there's no cell activity at all after 10+ minutes or an actual `ERROR:`/traceback appears.

**Gradio dashboard didn't pick up a code change** — Gradio doesn't hot-reload. Stop the running cell, re-fetch the code, and restart:
```python
%cd /content
!rm -rf techcodexai-main repo.zip
!wget -q https://github.com/zarko3/techcodexai/archive/refs/heads/main.zip -O repo.zip
!unzip -q repo.zip
%cd techcodexai-main
!python techcodex_single_file.py
```
(If you used `git clone` instead of the zip, `!git pull` works too — but the zip method leaves no `.git` folder, so `git pull` there fails with "not a git repository".)

## Repo layout

- **`techcodex_single_file.py`** — the whole program, self-contained, for Colab or a quick local run.
- **`modeling_techcodex.py`, `trainer_backend.py`, `app_ui.py`, `dataset_prep/`** — the same functionality split into modules, for local Windows development.
- **`setup_env.ps1`** — installs local Windows dependencies, including `torch-directml` for AMD GPUs.
