"""
TechcodeX — visual dashboard for local pre-training, weight management,
and chat testing. Pure Gradio, no HF model wrappers.
"""

import json
import os
import shutil

import pandas as pd
import torch
import gradio as gr

from modeling_techcodex import TechcodeXModel, TechcodeXConfig, device, describe_model_size, load_checkpoint
from trainer_backend import run_training_session, WEIGHTS_PATH, get_tokenizer
from dataset_prep import process_sources, build_jsonl_dataset, download_hf_dataset

EXPORT_DIR = "techcodex_hf_export"
PREVIEW_CHAR_LIMIT = 5000

_chat_model = None
_chat_config = None


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


def _format_quality_examples(examples: list, limit: int = 15) -> str:
    if not examples:
        return "(no paragraphs to show)"
    blocks = []
    for ex in examples[:limit]:
        status = "KEEP" if ex["keep"] else f"REMOVE ({ex['reason']})"
        original = ex["original"][:400]
        blocks.append(f"[{status}] quality score={ex['score']}\n{original}")
    return "\n\n---\n\n".join(blocks)


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
        return [], gr.update(choices=[], value=None), "Please upload at least one PDF/image file or a folder of them.", "", "", ""

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
        return "No processed documents available. Run 'Extract & Clean PDFs' first.", ""

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


def list_existing_dataset_files(scan_dir: str = "datasets") -> list:
    """Finds .txt/.jsonl files already on disk (e.g. from a previous PDF/HF download run)."""
    scan_dir = (scan_dir or "datasets").strip() or "datasets"
    search_dirs = [scan_dir, "cleaned", "."]

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


def start_training(
    file_objs, existing_selected, batch_size, block_size, max_steps, learning_rate, resume,
    gradient_checkpointing, gradient_accumulation_steps, use_fp16, offload_optimizer_to_cpu,
):
    """Generator wired directly into the Start Training button — streams live updates."""
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


def format_param_count(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.2f}K"
    return str(n)


def estimate_model_size(block_size):
    """Recomputed live whenever the Context Length slider changes, before any training happens."""
    config = TechcodeXConfig(block_size=int(block_size))
    info = describe_model_size(config)
    return (
        f"Parameters: {format_param_count(info['total_params'])} "
        f"({info['total_params']:,} total, {info['trainable_params']:,} trainable)\n"
        f"Approx size: {info['fp32_size_mb']:.2f} MB (fp32) / {info['fp16_size_mb']:.2f} MB (fp16)\n"
        f"Config: n_embd={config.n_embd}, n_head={config.n_head}, n_layer={config.n_layer}, "
        f"block_size={config.block_size}, vocab_size={config.vocab_size}"
    )


def check_weights_exist():
    if os.path.isfile(WEIGHTS_PATH):
        size_mb = os.path.getsize(WEIGHTS_PATH) / (1024 * 1024)
        return f"Found '{WEIGHTS_PATH}' ({size_mb:.2f} MB)."
    return f"No weights file found at '{WEIGHTS_PATH}'. Train a model first."


def show_model_info():
    """Reports the actual size of the trained checkpoint (not just the estimate)."""
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
    """Re-saves the raw checkpoint plus a Hugging Face-style config.json bundle folder."""
    if not os.path.isfile(WEIGHTS_PATH):
        return f"Cannot export — '{WEIGHTS_PATH}' does not exist yet. Train a model first."

    os.makedirs(EXPORT_DIR, exist_ok=True)

    checkpoint = load_checkpoint(WEIGHTS_PATH, map_location="cpu")
    config_dict = checkpoint.get("config", TechcodeXConfig().__dict__)

    hf_config = {
        "model_type": "techcodex",
        "architectures": ["TechcodeXModel"],
        "vocab_size": config_dict.get("vocab_size", 50257),
        "n_embd": config_dict.get("n_embd", 256),
        "n_head": config_dict.get("n_head", 4),
        "n_layer": config_dict.get("n_layer", 4),
        "block_size": config_dict.get("block_size", 128),
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
        "[TechcodeX](https://github.com) — no Hugging Face model wrapper, weights + config only.\n\n"
        f"- Parameters: {format_param_count(total_params)} ({total_params:,})\n"
        f"- n_embd={config_dict.get('n_embd')}, n_head={config_dict.get('n_head')}, "
        f"n_layer={config_dict.get('n_layer')}, block_size={config_dict.get('block_size')}, "
        f"vocab_size={config_dict.get('vocab_size')}\n"
        f"- Trained for {checkpoint.get('step', '?')} steps\n\n"
        "Load with the `TechcodeXModel` class from this project's `modeling_techcodex.py` "
        "(the architecture is not a standard Hugging Face model class, so `AutoModel` will not load it).\n"
    )
    with open(os.path.join(EXPORT_DIR, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)


def upload_to_hf_hub(repo_id, hf_token, private, commit_message):
    """Generator so the UI can stream progress — uploads can take a while for large checkpoints."""
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

    # map_location="cpu" + load_state_dict onto the already-`.to(device)` model below
    # sidesteps a torch_directml bug where map_location=<device object> raises.
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
    gr.Markdown("# ⚡ TechcodeX\nLocal pre-training & inference dashboard — raw PyTorch, no HF model wrappers.")
    gr.Markdown(f"**Device:** `{device}`")

    with gr.Tabs():
        with gr.Tab("Tab 0: Dataset Preparation (PDF → JSONL)"):
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
                "requires the Tesseract OCR engine installed separately on this machine. "
                "`.txt` files skip OCR but still run through the same cleaning pass "
                "(headers/footers, page numbers, whitespace, line-wrap fixes).*"
            )

            gr.Markdown("### 2. Cleaning options")
            with gr.Row():
                remove_hf_checkbox = gr.Checkbox(value=True, label="Remove repeated headers/footers")
                remove_pn_checkbox = gr.Checkbox(value=True, label="Remove standalone page numbers")
                norm_unicode_checkbox = gr.Checkbox(value=True, label="Normalize Unicode")

            gr.Markdown(
                "### 2b. Quality filtering *(new)*\n"
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
                english_threshold_slider = gr.Slider(
                    0.0, 1.0, value=0.55, step=0.05, label="English confidence threshold"
                )
                min_quality_score_slider = gr.Slider(
                    0, 100, value=40, step=1, label="Minimum quality score"
                )
                max_unusual_char_ratio_slider = gr.Slider(
                    0.0, 1.0, value=0.15, step=0.01, label="Maximum unusual-character ratio"
                )

            txt_out_dir_box = gr.Textbox(value="cleaned", label="TXT Output Directory")
            extract_button = gr.Button("Extract & Clean PDFs", variant="primary")
            extract_status = gr.Textbox(label="Extraction Log", interactive=False, lines=8)

            gr.Markdown("### 3. Preview extracted text")
            gr.Markdown("Original → Cleaned → Quality score → Keep/Remove, per paragraph:")
            with gr.Row():
                with gr.Column(scale=1):
                    doc_selector = gr.Dropdown(label="Processed file", choices=[], interactive=True)
                    doc_stats = gr.Textbox(label="Statistics", interactive=False, lines=15)
                with gr.Column(scale=2):
                    doc_preview = gr.Textbox(label="Cleaned Text Preview (surviving text)", interactive=False, lines=12)
                    quality_examples_display = gr.Textbox(
                        label="Per-paragraph quality decisions (Original / Score / Keep-Remove)",
                        interactive=False,
                        lines=16,
                    )

            gr.Markdown("### 4. Chunking & JSONL export")
            with gr.Row():
                chunk_size_slider = gr.Slider(32, 1024, value=512, step=16, label="Chunk Size (tokens)")
                chunk_overlap_slider = gr.Slider(0, 256, value=50, step=8, label="Chunk Overlap (tokens)")
                min_chunk_size_slider = gr.Slider(0, 256, value=64, step=8, label="Minimum Chunk Size (tokens)")

            with gr.Row():
                jsonl_out_dir_box = gr.Textbox(value="datasets", label="JSONL Output Directory")
                dataset_filename_box = gr.Textbox(value="dataset.jsonl", label="Dataset Filename")

            build_dataset_button = gr.Button("Build JSONL Dataset", variant="primary")
            dataset_status = gr.Textbox(label="Dataset Build Log", interactive=False, lines=5)
            dataset_sample = gr.Textbox(label="Sample JSONL Examples", interactive=False, lines=10)

            gr.Markdown("### 5. Or download a dataset from the Hugging Face Hub")
            with gr.Row():
                hf_repo_id_box = gr.Textbox(
                    label="Dataset repo id",
                    placeholder="e.g. roneneldan/TinyStories",
                )
                hf_config_box = gr.Textbox(label="Config name (optional)", placeholder="e.g. wikitext-2-raw-v1")
                hf_split_box = gr.Textbox(value="train", label="Split")
                hf_text_field_box = gr.Textbox(value="text", label="Text column name")

            with gr.Row():
                hf_max_examples_box = gr.Number(value=5000, label="Max examples (0 = all)", precision=0)
                hf_jsonl_out_dir_box = gr.Textbox(value="datasets", label="JSONL Output Directory")
                hf_dataset_filename_box = gr.Textbox(
                    label="Dataset Filename (optional)",
                    placeholder="defaults to the repo id",
                )

            hf_download_button = gr.Button("Download Dataset from Hugging Face", variant="primary")
            hf_download_status = gr.Textbox(label="Download Log", interactive=False, lines=5)
            hf_download_sample = gr.Textbox(label="Sample JSONL Examples", interactive=False, lines=10)

            hf_download_button.click(
                fn=run_download_hf_dataset,
                inputs=[
                    hf_repo_id_box,
                    hf_config_box,
                    hf_split_box,
                    hf_text_field_box,
                    hf_max_examples_box,
                    hf_jsonl_out_dir_box,
                    hf_dataset_filename_box,
                    chunk_size_slider,
                    chunk_overlap_slider,
                    min_chunk_size_slider,
                    remove_markup_checkbox,
                    remove_web_artifacts_checkbox,
                    english_only_checkbox,
                    english_threshold_slider,
                    remove_citations_checkbox,
                    remove_low_quality_checkbox,
                    deduplicate_checkbox,
                    min_quality_score_slider,
                    max_unusual_char_ratio_slider,
                ],
                outputs=[hf_download_status, hf_download_sample],
            )

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

                    gr.Markdown(
                        "**Or reuse a dataset you already prepared** (from Tab 0's PDF/image/HF "
                        "downloads) — no need to re-upload it every time."
                    )
                    with gr.Row():
                        scan_dir_box = gr.Textbox(
                            value="datasets", label="Folder to scan", scale=2
                        )
                        refresh_datasets_button = gr.Button("Refresh List", scale=1)
                    existing_datasets_checkbox = gr.CheckboxGroup(
                        choices=[os.path.relpath(f) for f in list_existing_dataset_files("datasets")],
                        label="Existing dataset files (check to include in training)",
                    )

                    batch_size_slider = gr.Slider(1, 64, value=8, step=1, label="Batch Size")
                    block_size_slider = gr.Slider(32, 512, value=128, step=32, label="Context Length (block_size)")
                    max_steps_slider = gr.Slider(10, 5000, value=200, step=10, label="Max Steps")
                    lr_slider = gr.Slider(1e-5, 1e-2, value=3e-4, step=1e-5, label="Learning Rate")
                    resume_checkbox = gr.Checkbox(
                        value=True,
                        label="Resume from saved weights (continue previous training instead of starting over)",
                    )
                    gr.Markdown(
                        "**Out of GPU memory?** Enable Gradient Checkpointing and/or lower "
                        "Batch Size while raising Gradient Accumulation Steps by the same "
                        "factor — same effective batch size, far less peak VRAM per step."
                    )
                    gradient_checkpointing_checkbox = gr.Checkbox(
                        value=False,
                        label="Gradient Checkpointing (less VRAM, slower per step)",
                    )
                    gradient_accumulation_slider = gr.Slider(
                        1, 64, value=1, step=1, label="Gradient Accumulation Steps"
                    )
                    use_fp16_checkbox = gr.Checkbox(
                        value=False,
                        label="Train in fp16 (roughly halves weights+gradients+optimizer VRAM; "
                        "uses dynamic loss scaling for stability)",
                    )
                    offload_optimizer_checkbox = gr.Checkbox(
                        value=False,
                        label="Offload Optimizer State to CPU (uses GPU + CPU together — moves "
                        "AdamW's state into RAM, freeing ~50% more GPU memory; adds a CPU<->GPU "
                        "transfer each step, so it's slower)",
                    )
                    model_size_display = gr.Textbox(
                        label="Model Size (estimated from current config)",
                        value=estimate_model_size(128),
                        interactive=False,
                        lines=3,
                    )
                    train_button = gr.Button("Start Training", variant="primary")

                with gr.Column(scale=1):
                    step_display = gr.Number(label="Current Step", value=0, interactive=False)
                    loss_display = gr.Number(label="Current Loss", value=0.0, interactive=False)
                    loss_plot = gr.LinePlot(
                        pd.DataFrame({"step": [], "loss": []}),
                        x="step",
                        y="loss",
                        title="Training Loss",
                    )
                    training_status = gr.Textbox(label="Status", interactive=False)

            block_size_slider.change(
                fn=estimate_model_size,
                inputs=block_size_slider,
                outputs=model_size_display,
            )

            refresh_datasets_button.click(
                fn=refresh_existing_datasets,
                inputs=scan_dir_box,
                outputs=existing_datasets_checkbox,
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

            send_button.click(
                fn=chat_with_techcodex,
                inputs=[chat_input, temperature_slider, chatbot],
                outputs=[chatbot, chat_input],
            )
            chat_input.submit(
                fn=chat_with_techcodex,
                inputs=[chat_input, temperature_slider, chatbot],
                outputs=[chatbot, chat_input],
            )

        with gr.Tab("Tab 4: Upload to Hugging Face"):
            gr.Markdown(
                "Pushes the trained checkpoint to the Hugging Face Hub as a model repo "
                "(re-uses the same export bundle as Tab 2's 'Export Hugging Face-style Bundle' — "
                "config.json + pytorch_model.bin — plus an auto-generated model card)."
            )
            with gr.Row():
                hf_upload_repo_id_box = gr.Textbox(
                    label="Repo id", placeholder="e.g. yourname/techcodex-model", scale=2
                )
                hf_upload_private_checkbox = gr.Checkbox(value=True, label="Private repository", scale=1)
            hf_upload_token_box = gr.Textbox(
                label="Hugging Face access token (write scope)",
                placeholder="hf_...",
                type="password",
            )
            hf_upload_commit_message_box = gr.Textbox(
                label="Commit message (optional)", placeholder="Upload TechcodeX checkpoint"
            )
            hf_upload_button = gr.Button("Upload to Hugging Face Hub", variant="primary")
            hf_upload_status = gr.Textbox(label="Upload Status", interactive=False, lines=5)

            hf_upload_button.click(
                fn=upload_to_hf_hub,
                inputs=[
                    hf_upload_repo_id_box,
                    hf_upload_token_box,
                    hf_upload_private_checkbox,
                    hf_upload_commit_message_box,
                ],
                outputs=hf_upload_status,
            )


if __name__ == "__main__":
    demo.queue().launch(theme=gr.themes.Soft(primary_hue="violet"))
