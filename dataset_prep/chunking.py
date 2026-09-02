"""
Paragraph-aware, token-counted chunking of cleaned text into training examples.

Chunks are packed from whole paragraphs so boundaries land on paragraph breaks
rather than mid-sentence, with token-based overlap carried between chunks and
trailing under-sized fragments merged into the previous chunk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .tokenizer_utils import get_tokenizer

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
            # a single sentence still too big — only case where we hard-split mid-sentence
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

    # merge a trailing too-small chunk into the previous one instead of emitting a tiny fragment
    if len(chunks) >= 2 and chunks[-1].token_count < min_chunk_size:
        last = chunks.pop()
        merged_text = chunks[-1].text + "\n\n" + last.text
        chunks[-1] = Chunk(text=merged_text, token_count=chunks[-1].token_count + last.token_count)

    return chunks
