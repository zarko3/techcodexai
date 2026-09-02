"""
Orchestrates: source discovery -> extract -> clean -> save TXT -> chunk -> save JSONL.

This module never silently produces an empty or corrupted dataset:
 - a file that yields no usable text is excluded (and reported), never written as an
   empty .txt or silently included as empty JSONL rows
 - JSONL is written atomically (temp file + rename) and every line is round-tripped
   through json.loads before being committed
 - building a dataset with zero resulting examples raises instead of writing a file
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from .chunking import chunk_text
from .cleaning import clean_extraction
from .extractors import EXTRACTOR_REGISTRY, get_extractor_for
from .quality import filter_and_score_text
from .tokenizer_utils import get_tokenizer


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
    """Expand a mix of file paths and folder paths into a flat, de-duplicated list of
    files with an extension registered in EXTRACTOR_REGISTRY (currently .pdf, .png, .jpg, .jpeg)."""
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


# kept as an alias — earlier versions of this pipeline only handled PDFs
discover_pdf_files = discover_source_files


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
    file_paths = discover_pdf_files(paths)
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
            "Check that the source PDFs contain extractable text and that chunk settings are reasonable."
        )

    total_tokens = 0
    tmp_path = jsonl_path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            for text in all_examples:
                line = json.dumps({"text": text}, ensure_ascii=False)
                json.loads(line)  # validate before committing — never write a broken line
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
