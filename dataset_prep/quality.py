"""
Paragraph-level quality filtering for the dataset-prep pipeline.

Runs AFTER `clean_extraction` (headers/footers/page numbers/markup already
stripped, lines reflowed into paragraphs) and BEFORE `chunk_text`. Scores
and filters individual paragraphs for: language, citation/reference-only
fragments, corrupted extraction text, and near-duplicates — so this garbage
never reaches the tokenizer.

This is a heuristic, dependency-light filter (script-range + langdetect for
language; regex + ratios for everything else) — it is deliberately
conservative: legitimate English prose containing accents, math notation,
occasional citations, or technical jargon should score highly and survive.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field

from .cleaning import contains_markup
from .chunking import split_paragraphs

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

# A small closed-class set of very common English function words. Real
# English prose of any length uses these constantly; broken/garbled
# extraction text (invented word-salad) typically has none at all — this
# catches corruption that looks superficially word-shaped but isn't a
# dictionary check, so it can't be fooled the other way (technical jargon
# still scores fine as long as ordinary function words surround it).
_ENGLISH_STOPWORDS = frozenset("""
the a an and or but if of to in on for with as by at from is are was were
be been being this that these those it its he she they we you i his her
their our your not no do does did have has had will would can could should
may might must than then so such which who whom what when where why how
also into about between because after before while during each other some
any all most more less much many one two first second new used using use
between over under out up down near own such only just very both same
""".split())

# Unicode script ranges commonly used to write non-English languages outright
# (deliberately excludes Greek/math symbols, which show up in legitimate
# English technical/scientific prose).
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

_langdetect_ready = None  # None=untried, True/False after first attempt


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
        return 1.0  # too short to judge reliably — don't penalize

    if _non_latin_script_ratio(stripped) > 0.25:
        return 0.0

    if _langdetect_ready is False:
        return 1.0  # library unavailable — fall back to the script check only

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
        # langdetect raises LangDetectException on e.g. no-features-in-text input
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
    """A paragraph that IS mostly bibliography/reference noise, not prose
    that merely contains an occasional inline citation."""
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


# --- Deduplication: exact (normalized-text hash) + near-duplicate (simhash + LSH banding) ---

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
    """Returns (kept_paragraphs, exact_removed_count, near_removed_count).

    Exact duplicates are removed by a normalized-text hash. Near-duplicates
    are found via 64-bit simhash of word-4-shingles, bucketed into 4x16-bit
    LSH bands so only paragraphs likely to be similar are ever compared —
    this avoids O(n^2) comparisons across the whole document while still
    catching near-identical paragraphs (hamming distance <= 3 is a standard
    simhash near-dup threshold). Paragraphs that merely share common words
    but differ in most 4-word shingles will not collide.
    """
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
