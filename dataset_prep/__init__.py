"""
TechcodeX dataset preparation toolkit.

Pipeline: source file(s) -> text extraction -> cleaning -> TXT preview -> chunking -> JSONL.

Designed so new source types (.txt, .md, .html, .docx, ...) can be added by writing
a BaseExtractor subclass and registering it — nothing else in the pipeline changes.
"""

from .extractors import (
    BaseExtractor,
    PDFExtractor,
    ImageExtractor,
    TxtExtractor,
    ExtractedPage,
    ExtractionResult,
    register_extractor,
    get_extractor_for,
    EXTRACTOR_REGISTRY,
)
from .cleaning import clean_extraction, CleaningResult
from .chunking import chunk_text, Chunk, split_paragraphs
from .tokenizer_utils import get_tokenizer, estimate_tokens
from .quality import filter_and_score_text, QualityFilterResult
from .hf_source import download_hf_dataset
from .pipeline import (
    discover_source_files,
    discover_pdf_files,
    process_document,
    process_sources,
    build_jsonl_dataset,
    ProcessedDocument,
    DatasetBuildResult,
)

__all__ = [
    "BaseExtractor",
    "PDFExtractor",
    "ImageExtractor",
    "TxtExtractor",
    "ExtractedPage",
    "ExtractionResult",
    "register_extractor",
    "get_extractor_for",
    "EXTRACTOR_REGISTRY",
    "clean_extraction",
    "CleaningResult",
    "chunk_text",
    "Chunk",
    "split_paragraphs",
    "get_tokenizer",
    "estimate_tokens",
    "filter_and_score_text",
    "QualityFilterResult",
    "download_hf_dataset",
    "discover_source_files",
    "discover_pdf_files",
    "process_document",
    "process_sources",
    "build_jsonl_dataset",
    "ProcessedDocument",
    "DatasetBuildResult",
]
