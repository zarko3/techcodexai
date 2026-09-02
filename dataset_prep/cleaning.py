"""
Conservative text cleaning for extracted PDF text.

Goal: fix extraction artifacts (headers/footers, page numbers, broken line
wraps, stray whitespace, garbage characters) WITHOUT rewriting, summarizing,
or altering the meaning of the source material.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

from .extractors import ExtractionResult

_PAGE_NUMBER_RE = re.compile(
    r"^\s*(?:page\s+)?[\-–—]?\s*\d{1,4}\s*(?:of\s*\d{1,4})?\s*[\-–—]?\s*$",
    re.IGNORECASE,
)
_ROMAN_NUMERAL_RE = re.compile(r"^\s*[ivxlcdm]{1,6}\s*$", re.IGNORECASE)
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_BLANK_LINE_RE = re.compile(r"\n{3,}")
_HYPHEN_BREAK_RE = re.compile(r"(\w)-\n(\w)")

# --- Markup stripping (conservative: only matches actual tag syntax or an
# isolated line that is nothing but a bare markup keyword) ---
_TAG_BLOCK_RE = re.compile(r"<(script|style|svg)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_ANY_TAG_RE = re.compile(r"</?[a-zA-Z][a-zA-Z0-9:_-]*(?:\s+[^<>]*)?/?>")
_XML_DECL_RE = re.compile(r"<\?xml[^>]*\?>", re.IGNORECASE)
_HTML_ENTITY_RE = re.compile(r"&(?:amp|lt|gt|quot|apos|nbsp|#\d+|#x[0-9a-fA-F]+);")
_ISOLATED_MARKUP_WORD_RE = re.compile(r"^\s*(?:svg|html|xml)\s*$", re.IGNORECASE)

# --- Common web boilerplate lines ---
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
                continue  # bare page numbers are handled separately
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

    # strip obvious extraction garbage (control chars, replacement chars)
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
