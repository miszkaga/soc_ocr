"""Anonymisation of the text read (the text branch of the user's anonymisation script).

The same steps as the script's TXT mode, on each section of out/text.txt:

1. spaCy NER (``[anonymize] model``, en_core_web_trf): entities of the labels in
   ``entity_labels`` are replaced by their label text ("[FULL NAME]", "[ORGANIZATION]", ...);
2. job titles found by the token patterns below are replaced by ``role_label``;
   (both collected first and replaced longest first, every occurrence in the section);
3. regular expressions on the result: e-mails, phone numbers, "Author: ..." and the companies
   listed in ``target_companies`` (with an optional ".com").

The company list and the labels are in config/default.toml. spaCy and the model are an
optional extra (``./setup.sh --anonymize``); without them only the regular expressions run and
a warning says so.
"""
from __future__ import annotations

import re
import sys
from functools import lru_cache

from ocr.config import AnonymizeConfig

TITLE_PATTERNS = [
    [
        {"LOWER": {"IN": ["head", "director", "manager", "vp", "vice", "lead", "chief", "officer", "advisor"]}},
        {"LOWER": "of", "OP": "?"},
        {"POS": "NOUN", "OP": "+"},
        {"POS": "NOUN", "OP": "*"},
    ],
    [
        {"LOWER": {"IN": ["chief", "senior", "junior", "lead", "principal", "global"]}},
        {"POS": "NOUN", "OP": "+"},
        {"LOWER": {"IN": ["officer", "manager", "lead", "specialist", "auditor", "analyst", "director"]}},
    ],
]
EMAIL = r"[\w\.-]+@[\w\.-]+\.\w+"
PHONE = r"\+?\d{1,3}[-.\s]?(?:\d{2,4}[-.\s]?){2,3}\d{2,4}\b"
AUTHOR = r"(?i)\bAuthor:\s*([^\n\r,;]+)"


@lru_cache(maxsize=2)
def _pipeline(model: str):
    """(nlp, matcher), or None when spaCy or the model is not installed."""
    try:
        import spacy
        from spacy.matcher import Matcher
    except ImportError:
        return None
    try:
        nlp = spacy.load(model)
    except OSError:
        return None
    matcher = Matcher(nlp.vocab)
    matcher.add("JOB_TITLE", TITLE_PATTERNS)
    return nlp, matcher


def available(cfg: AnonymizeConfig) -> bool:
    return _pipeline(cfg.model) is not None


def _patterns(cfg: AnonymizeConfig) -> list[tuple[str, str]]:
    out = [(EMAIL, cfg.email_label), (PHONE, cfg.phone_label), (AUTHOR, cfg.person_label)]
    companies = sorted(cfg.target_companies, key=len, reverse=True)
    if companies:
        names = "|".join(re.escape(c) for c in companies)
        out.append((rf"(?i)\b(?:{names})(?:\.com)?\b", cfg.company_label))
    return out


def anonymize(text: str, cfg: AnonymizeConfig, nlp_parts=None) -> str:
    """The text with personal and business data replaced by labels."""
    parts = nlp_parts if nlp_parts is not None else _pipeline(cfg.model)
    replacements: list[tuple[str, str]] = []
    if parts is not None and text.strip():
        nlp, matcher = parts
        doc = nlp(text)
        labels = dict(cfg.entity_labels)
        for ent in doc.ents:
            if ent.label_ in labels and len(ent.text.strip()) > 1:
                replacements.append((ent.text.strip(), labels[ent.label_]))
        for _, start, end in matcher(doc):
            span = doc[start:end]
            if len(span.text.strip()) > 3:
                replacements.append((span.text.strip(), cfg.role_label))
    replacements.sort(key=lambda r: len(r[0]), reverse=True)
    for term, label in replacements:
        text = text.replace(term, label)
    for pattern, label in _patterns(cfg):
        for m in re.finditer(pattern, text):
            target = m.group(1).strip() if m.groups() else m.group()
            if target:
                text = text.replace(target, label)
    return text


_HEADER = re.compile(r"^=== .* ===$")


def anonymize_sections(text: str, cfg: AnonymizeConfig, log=None) -> str:
    """out/text.txt anonymised section by section; the "=== file | page ... ===" headers stay."""
    if not available(cfg):
        print(f"warning: spaCy model {cfg.model!r} not installed (./setup.sh --anonymize): "
              "only e-mails, phone numbers, authors and listed companies are replaced", file=log or sys.stderr)
    out, body = [], []

    def flush():
        if body:
            out.append(anonymize("\n".join(body), cfg))
            body.clear()

    for line in text.split("\n"):
        if _HEADER.match(line):
            flush()
            out.append(line)
        else:
            body.append(line)
    flush()
    return "\n".join(out)
