"""
Optional Hugging Face Hub *dataset* download support.

This only uses the `datasets` library to fetch and iterate an already-published
dataset and re-save it as local JSONL — no Hugging Face model classes or
wrappers are used anywhere in this project.
"""

from __future__ import annotations

import os

from .cleaning import clean_extraction
from .extractors import ExtractedPage, ExtractionResult
from .pipeline import ProcessedDocument, build_jsonl_dataset
from .quality import filter_and_score_text
from .tokenizer_utils import get_tokenizer


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
    """
    Downloads a dataset from the Hugging Face Hub, runs it through the SAME
    cleaning -> quality filtering -> deduplication -> chunking pipeline used
    for local PDF/image/text sources, and writes it out as local JSONL in the
    same {"text": ...} format. Never silently writes an empty or corrupted
    file — raises instead.
    """
    repo_id = (repo_id or "").strip()
    if not repo_id:
        raise ValueError("Hugging Face dataset repo id cannot be empty (e.g. 'roneneldan/TinyStories').")

    split = (split or "train").strip() or "train"
    text_field = (text_field or "text").strip() or "text"
    config_name = (config_name or "").strip() or None

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError(
            "The 'datasets' library is required. Install it with: pip install datasets"
        ) from e

    try:
        hf_dataset = load_dataset(repo_id, config_name, split=split)
    except Exception as e:
        raise RuntimeError(f"Failed to download '{repo_id}' (split='{split}'): {e}") from e

    if text_field not in hf_dataset.column_names:
        raise ValueError(
            f"Column '{text_field}' not found in '{repo_id}'. "
            f"Available columns: {hf_dataset.column_names}"
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
