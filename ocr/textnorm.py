"""Text forms shared by the consensus and the evaluation."""
from __future__ import annotations

# Typographic variants nobody can tell apart in a picture (decided 2026-10-22).
TYPOGRAPHY = str.maketrans({"’": "'", "‘": "'", "′": "'", "“": '"', "”": '"', "„": '"', "″": '"',
                            "–": "-", "—": "-", "−": "-", "‐": "-", "…": "..."})


def alnum_count(text: str) -> int:
    return sum(ch.isalnum() for ch in text)


def has_latin_letter(text: str) -> bool:
    import unicodedata
    return any(ch.isalpha() and unicodedata.name(ch, "").startswith("LATIN") for ch in text)
