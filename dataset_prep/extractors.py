"""
Source -> raw per-page text extraction.

New source types are added by subclassing BaseExtractor and calling
register_extractor(...) — the rest of the pipeline is source-agnostic.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


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
    """OCR-based extractor for standalone image files that have no native text layer."""

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
                "https://github.com/UB-Mannheim/tesseract/wiki, then add its install folder to PATH."
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
