"""Output contract ``ocr/v1`` (CLAUDE.md) and its validation.

One JSON document per input file::

    {"schema": "ocr/v1", "source": "<path relative to the input folder>", "engine": "...",
     "engine_versions": {...}, "created": "<ISO time>",
     "pages": [{"page": 1, "width_pt": ..., "height_pt": ..., "width_px": ..., "height_px": ...,
                "regions": [{"id": "p1r1", "kind": ..., "bbox_pt": [...] | null,
                             "bbox_px": [...], "render_dpi": 300 | null, "status": ...,
                             "lines": [{"words": [word, ...]}], "text": "..."}]}],
     "review": [{"region": "p1r1", "word": 3, "text": "...", "reason": "...", "crop": "..."}]}

A word: ``text``, ``conf``, ``char_conf``, ``bbox_px``, ``bbox_pt``, ``votes``, ``status``,
``reason``. Words taken from the PDF's own text have reason ``native``, votes ``{"pdf": text}``
and confidence 1. A PDF region also reports ``native_chars`` and ``native_words``.

Coordinates. ``bbox_px`` of a word is in the pixels of its region's image: the region rendered
at ``render_dpi`` for a PDF, the file itself for an image. ``bbox_pt`` is in PDF points on the
page (top-left origin, y down, the page unrotated), null for images. A region's ``bbox_px`` is
its place in the page rendered at ``render_dpi`` (the whole image for an image file).

Kinds: ``embedded_image``, ``scanned_page``, ``vector_diagram``, ``logo`` (PDF regions, see
ocr/pdf_regions.py) and ``image`` (a whole image file). Statuses: ``accepted`` (every word
accepted), ``uncertain`` (some word needs review).

Additive changes keep the version; an incompatible change raises it.
"""
from __future__ import annotations

SCHEMA = "ocr/v1"
KINDS = {"embedded_image", "scanned_page", "vector_diagram", "logo", "image"}
STATUSES = {"accepted", "uncertain", "unreadable"}


def _box(v, nullable: bool = False) -> bool:
    if v is None:
        return nullable
    return (isinstance(v, (list, tuple)) and len(v) == 4 and all(isinstance(x, (int, float)) for x in v)
            and v[0] <= v[2] and v[1] <= v[3])


def validate(doc: dict) -> list[str]:
    """Problems found in a document (empty list: valid)."""
    errors = []

    def need(obj, key, where, kind=None):
        if key not in obj:
            errors.append(f"{where}: missing {key!r}")
            return False
        if kind is not None and not isinstance(obj[key], kind):
            errors.append(f"{where}.{key}: expected {kind}")
            return False
        return True

    if doc.get("schema") != SCHEMA:
        errors.append(f"schema: expected {SCHEMA!r}, got {doc.get('schema')!r}")
    need(doc, "source", "doc", str)
    need(doc, "engine_versions", "doc", dict)
    need(doc, "review", "doc", list)
    region_ids = set()
    if need(doc, "pages", "doc", list):
        for pi, page in enumerate(doc["pages"]):
            where = f"pages[{pi}]"
            need(page, "page", where, int)
            if not need(page, "regions", where, list):
                continue
            for ri, reg in enumerate(page["regions"]):
                rw = f"{where}.regions[{ri}]"
                if need(reg, "id", rw, str):
                    if reg["id"] in region_ids:
                        errors.append(f"{rw}: duplicate id {reg['id']!r}")
                    region_ids.add(reg["id"])
                if need(reg, "kind", rw) and reg["kind"] not in KINDS:
                    errors.append(f"{rw}.kind: unknown {reg['kind']!r}")
                if need(reg, "status", rw) and reg["status"] not in STATUSES:
                    errors.append(f"{rw}.status: unknown {reg['status']!r}")
                if need(reg, "bbox_pt", rw) and not _box(reg["bbox_pt"], nullable=True):
                    errors.append(f"{rw}.bbox_pt: not a box")
                need(reg, "text", rw, str)
                if not need(reg, "lines", rw, list):
                    continue
                for li, line in enumerate(reg["lines"]):
                    if not need(line, "words", f"{rw}.lines[{li}]", list):
                        continue
                    for wi, w in enumerate(line["words"]):
                        ww = f"{rw}.lines[{li}].words[{wi}]"
                        need(w, "text", ww, str)
                        if need(w, "conf", ww, (int, float)) and not 0 <= w["conf"] <= 1:
                            errors.append(f"{ww}.conf: outside 0..1")
                        if need(w, "char_conf", ww, list) and isinstance(w.get("text"), str) \
                                and len(w["char_conf"]) != len(w["text"]):
                            errors.append(f"{ww}.char_conf: {len(w['char_conf'])} values for {len(w['text'])} characters")
                        if need(w, "bbox_px", ww) and not _box(w["bbox_px"]):
                            errors.append(f"{ww}.bbox_px: not a box")
                        if need(w, "bbox_pt", ww) and not _box(w["bbox_pt"], nullable=True):
                            errors.append(f"{ww}.bbox_pt: not a box")
                        if need(w, "status", ww) and w["status"] not in STATUSES:
                            errors.append(f"{ww}.status: unknown {w['status']!r}")
    for ei, item in enumerate(doc.get("review", []) if isinstance(doc.get("review"), list) else []):
        for key in ("region", "reason", "crop"):
            need(item, key, f"review[{ei}]")
        if item.get("region") not in region_ids:
            errors.append(f"review[{ei}].region: unknown region {item.get('region')!r}")
    return errors
