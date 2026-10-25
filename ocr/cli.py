"""Command line: OCR every PDF and image in a folder.

    python -m ocr                      # reads input/ into out/ (both next to the engine)
    python -m ocr extract FOLDER [-o OUT] [--recursive]

Writes into OUT (default ``out/``, kept out of git):

- ``text.txt``: the text read, one section per image file and per PDF region;
- ``boxes/``: every image, and every PDF page with something read, with a box around each word
  read (green: accepted, red: needs review; PDF regions outlined in blue);
- ``json/<file>.json``: the full result, contract ocr/v1 (ocr/schema.py), and
  ``json/<file>.review/``: a crop of every word that needs review.

Everything runs locally; nothing is sent anywhere.

PDF pages are inventoried (ocr/pdf_regions.py); every region (embedded image, scanned page,
vector diagram, logo) is rendered at ``[render] dpi``. Text the PDF keeps as text inside a
region (labels of vector diagrams) is taken as it is (ocr/pdf_text.py: exact, accepted,
reason "native"); OCR reads the rest (outlined text, pictures), and OCR words repeating native
text are dropped. An image file is one region of kind ``image``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

from PIL import Image, ImageOps

from ocr import layout
from ocr.config import Config, load_config
from ocr.engines.base import OcrEngine, OcrWord
from ocr.pdf_text import inside, native_words
from ocr.review import draw_boxes, save_crop
from ocr.schema import SCHEMA, validate

REPO = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO / "input"
DEFAULT_OUT = REPO / "out"
PDF_SUFFIXES = {".pdf"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif"}


def input_files(folder: Path, recursive: bool, skip: Path | None = None) -> list[Path]:
    """PDFs and images in the folder, in a stable order; ``skip`` (the output folder) excluded."""
    pattern = "**/*" if recursive else "*"
    out = []
    for p in sorted(folder.glob(pattern)):
        if not p.is_file() or p.suffix.lower() not in PDF_SUFFIXES | IMAGE_SUFFIXES:
            continue
        if skip is not None and p.resolve().is_relative_to(skip.resolve()):
            continue
        out.append(p)
    return out


def _r(v, n):
    return [round(float(x), n) for x in v]


def _word(w: OcrWord, to_pt) -> dict:
    return {"text": w.text, "conf": round(float(w.conf), 4), "char_conf": [round(float(c), 4) for c in w.char_conf],
            "bbox_px": _r(w.bbox_px, 1), "bbox_pt": None if to_pt is None else _r(to_pt(w.bbox_px), 2),
            "votes": w.votes or {}, "status": w.status or "accepted", "reason": w.reason}


def _region(rid: str, kind: str, image: Image.Image, words: list[OcrWord], bbox_px, bbox_pt, dpi, to_pt,
            review: list, review_dir: Path, json_dir: Path, extra: dict) -> dict:
    grouped = layout.lines(words)
    k = 0
    for line in grouped:
        for w in line:
            if w.status not in (None, "accepted"):
                crop = review_dir / f"{rid}_w{k:04d}.png"
                save_crop(image, w, crop)
                review.append({"region": rid, "word": k, "text": w.text, "reason": w.reason or "uncertain",
                               "votes": w.votes or {}, "crop": str(crop.relative_to(json_dir))})
            k += 1
    status = "uncertain" if any(w.status not in (None, "accepted") for w in words) else "accepted"
    return {"id": rid, "kind": kind, "bbox_pt": bbox_pt, "bbox_px": bbox_px, "render_dpi": dpi,
            "status": status, **extra,
            "lines": [{"words": [_word(w, to_pt) for w in line]} for line in grouped],
            "text": layout.text(grouped)}


def without_native(ocr_words: list[OcrWord], native: list[OcrWord], min_overlap: float) -> list[OcrWord]:
    """OCR words that do not repeat native text: a word overlapping native words by at least
    ``min_overlap`` of its box is the same text read twice."""
    def inter(a, b):
        return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))

    out = []
    for w in ocr_words:
        area = max(1e-6, (w.bbox_px[2] - w.bbox_px[0]) * (w.bbox_px[3] - w.bbox_px[1]))
        if sum(inter(w.bbox_px, n.bbox_px) for n in native) < min_overlap * area:
            out.append(w)
    return out


def extract_pdf(path: Path, engine: OcrEngine, cfg: Config, review_dir: Path, json_dir: Path) -> tuple[list, list, list]:
    """Pages (contract ocr/v1), review items, and (page name, page image with boxes)."""
    import pypdfium2 as pdfium

    from ocr.pdf_regions import crop_region, inventory, render_page

    dpi = cfg.render.dpi
    s = dpi / 72
    inv = inventory(path, cfg.regions)
    pages, review, annotated = [], [], []
    pdf = pdfium.PdfDocument(str(path))
    try:
        for page in inv.pages:
            regions = []
            page_img = render_page(pdf, page.page, dpi) if page.regions else None
            natives = []
            if page.regions and cfg.native_text.enabled:
                pdf_page = pdf[page.page - 1]
                l, _, _, t = pdf_page.get_bbox()
                natives = native_words(pdf_page, (l, t), cfg.native_text.word_gap_em)
            for r in page.regions:
                img = crop_region(page_img, r.bbox_pt, dpi)
                ox, oy = round(r.bbox_pt[0] * s), round(r.bbox_pt[1] * s)
                to_pt = (lambda b, ox=ox, oy=oy: ((ox + b[0]) / s, (oy + b[1]) / s, (ox + b[2]) / s, (oy + b[3]) / s))
                native = [OcrWord(w.text, (w.box[0] * s - ox, w.box[1] * s - oy, w.box[2] * s - ox, w.box[3] * s - oy),
                                  1.0, [1.0] * len(w.text), "accepted", "native", {"pdf": w.text})
                          for w in natives if inside(w, tuple(r.bbox_pt))]
                words = native + without_native(engine.read(img), native, cfg.native_text.ocr_overlap)
                regions.append(_region(r.id, r.kind, img, words, [ox, oy, ox + img.width, oy + img.height],
                                       _r(r.bbox_pt, 2), dpi, to_pt, review, review_dir, json_dir,
                                       {"native_chars": r.native_chars, "native_words": len(native)}))
            pages.append({"page": page.page, "width_pt": round(page.width_pt, 2), "height_pt": round(page.height_pt, 2),
                          "rotation": page.rotation, "native_chars": page.native_chars, "regions": regions})
            if regions:
                boxes = [((reg["bbox_px"][0] + w["bbox_px"][0], reg["bbox_px"][1] + w["bbox_px"][1],
                           reg["bbox_px"][0] + w["bbox_px"][2], reg["bbox_px"][1] + w["bbox_px"][3]), w["status"] == "accepted")
                         for reg in regions for line in reg["lines"] for w in line["words"]]
                annotated.append((f"p{page.page}", draw_boxes(page_img, boxes, [reg["bbox_px"] for reg in regions])))
    finally:
        pdf.close()
    return pages, review, annotated


def extract_image(path: Path, engine: OcrEngine, review_dir: Path, json_dir: Path) -> tuple[list, list, list]:
    with Image.open(path) as im:
        img = ImageOps.exif_transpose(im.convert("RGB"))  # first frame; photos turned upright
    review = []
    region = _region("p1r1", "image", img, engine.read(img), [0, 0, img.width, img.height], None, None, None,
                     review, review_dir, json_dir, {})
    boxes = [(w["bbox_px"], w["status"] == "accepted") for line in region["lines"] for w in line["words"]]
    pages = [{"page": 1, "width_px": img.width, "height_px": img.height, "regions": [region]}]
    return pages, review, [("", draw_boxes(img, boxes))]


def extract_file(path: Path, rel: str, engine: OcrEngine, cfg: Config, out_dir: Path) -> dict:
    """Read one file; write its json (with review crops) and its images with boxes."""
    json_path = out_dir / "json" / f"{rel}.json"
    review_dir = out_dir / "json" / f"{rel}.review"
    if path.suffix.lower() in PDF_SUFFIXES:
        pages, review, annotated = extract_pdf(path, engine, cfg, review_dir, json_path.parent)
    else:
        pages, review, annotated = extract_image(path, engine, review_dir, json_path.parent)
    doc = {"schema": SCHEMA, "source": rel, "engine": engine.name, "engine_versions": engine.versions(),
           "created": dt.datetime.now().isoformat(timespec="seconds"), "pages": pages, "review": review}
    problems = validate(doc)
    if problems:
        raise ValueError(f"output does not match {SCHEMA}: {problems[:5]}")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    for name, image in annotated:
        target = boxes_path(out_dir, rel, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target)
    return doc


def boxes_path(out_dir: Path, rel: str, page: str = "") -> Path:
    """``boxes/a/shot.png`` for an image (``shot.jpg.png`` for other formats: always PNG),
    ``boxes/a/doc.pdf.p1.png`` for page 1 of a PDF."""
    if page:
        return out_dir / "boxes" / f"{rel}.{page}.png"
    return out_dir / "boxes" / (rel if rel.lower().endswith(".png") else f"{rel}.png")


def text_section(doc: dict) -> str:
    """The file's part of text.txt: a header per region read and its text."""
    parts = []
    for page in doc["pages"]:
        for reg in page["regions"]:
            where = doc["source"] if reg["kind"] == "image" else \
                f"{doc['source']} | page {page['page']} | {reg['id']} {reg['kind']}"
            n = sum(w["status"] != "accepted" for line in reg["lines"] for w in line["words"])
            parts.append(f"=== {where}" + (f" | {n} to review" if n else "") + " ===\n"
                         + (reg["text"] or "(no text found)") + "\n")
    if not parts:
        parts.append(f"=== {doc['source']} ===\n(no images or diagrams to read)\n")
    return "\n".join(parts) + "\n"


def run(folder: Path, out_dir: Path, recursive: bool = False, engine: OcrEngine | None = None,
        cfg: Config | None = None, log=None) -> int:
    """OCR every file of the folder; returns the number of files that failed."""
    log = log or sys.stderr
    cfg = cfg or load_config()
    if engine is None:
        from ocr.engines.registry import create
        engine = create("consensus", config=cfg)
    files = input_files(folder, recursive, skip=out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    failed = 0
    with (out_dir / "text.txt").open("w", encoding="utf-8") as text_file:
        for i, path in enumerate(files, 1):
            rel = path.relative_to(folder).as_posix()
            t0 = time.perf_counter()
            try:
                doc = extract_file(path, rel, engine, cfg, out_dir)
            except Exception as e:  # one broken file must not stop the folder
                failed += 1
                text_file.write(f"=== {rel} ===\n(could not be read: {type(e).__name__}: {e})\n\n")
                print(f"[{i}/{len(files)}] {rel}: FAILED {type(e).__name__}: {e}", file=log, flush=True)
                continue
            text_file.write(text_section(doc))
            text_file.flush()
            regions = [r for p in doc["pages"] for r in p["regions"]]
            n_words = sum(len(l["words"]) for r in regions for l in r["lines"])
            print(f"[{i}/{len(files)}] {rel}: {len(doc['pages'])} pages, {len(regions)} regions, {n_words} words, "
                  f"{len(doc['review'])} to review, {time.perf_counter() - t0:.1f}s", file=log, flush=True)
    return failed


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m ocr", description="Local OCR of images and diagrams in PDFs. "
                                 "Without arguments: python -m ocr extract input -o out.")
    sub = ap.add_subparsers(dest="cmd")
    ex = sub.add_parser("extract", help="OCR every PDF and image in a folder")
    ex.add_argument("folder", type=Path, nargs="?", default=DEFAULT_INPUT,
                    help=f"folder to read (default: {DEFAULT_INPUT})")
    ex.add_argument("-o", "--out", type=Path, default=DEFAULT_OUT, help=f"output folder (default: {DEFAULT_OUT})")
    ex.add_argument("-r", "--recursive", action="store_true", help="include subfolders")
    args = ap.parse_args(argv or ["extract"])
    if not args.folder.is_dir():
        print(f"not a folder: {args.folder}", file=sys.stderr)
        return 2
    files = input_files(args.folder, args.recursive, skip=args.out)
    if not files:
        print(f"no PDF or image files in {args.folder}: put the files to read there", file=sys.stderr)
        return 2
    print(f"{len(files)} files -> {args.out}", file=sys.stderr)
    failed = run(args.folder, args.out, args.recursive)
    print(f"done: {len(files) - failed} ok, {failed} failed -> {args.out / 'text.txt'}, "
          f"{args.out / 'boxes'}/, {args.out / 'json'}/", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
