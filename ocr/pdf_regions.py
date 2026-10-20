"""Page inventory and region classification for PDF files.

For every page we list the regions that need OCR:

- ``scanned_page``   -- the page is (almost) entirely covered by raster images,
- ``embedded_image`` -- a raster image placed on the page (screenshot, photo, raster diagram),
- ``logo``           -- a small raster image (may still contain text),
- ``vector_diagram`` -- a cluster of vector paths (boxes, arrows) with its text labels.

Native (selectable) text is not a region; its character count is reported per page and
per region so later stages can decide whether OCR is needed.

Coordinates: ``bbox_pt`` is ``[x0, y0, x1, y1]`` in PDF points with the origin at the
top-left corner of the page's visible box (media box ∩ crop box), y growing downwards,
in the page's unrotated orientation. Rendering uses the same orientation, so
``bbox_px = bbox_pt * dpi / 72``.
"""
from __future__ import annotations

import ctypes
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from PIL import Image

from ocr.config import RegionConfig, load_config

Box = tuple[float, float, float, float]  # x0, y0, x1, y1; top-left origin, y down


@dataclass
class Region:
    id: str
    kind: str
    bbox_pt: list[float]
    native_chars: int
    n_objects: int
    image_px: list[int] | None = None


@dataclass
class PageInventory:
    page: int  # 1-based
    width_pt: float
    height_pt: float
    rotation: int
    native_chars: int
    regions: list[Region] = field(default_factory=list)


@dataclass
class DocumentInventory:
    source: str
    pages: list[PageInventory]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class _Obj:
    box: Box
    kind: int  # FPDF_PAGEOBJ_*
    image_px: tuple[int, int] | None = None
    orthogonal: bool = True  # paths: only horizontal/vertical straight segments
    visible: bool = True  # text: False for render mode "invisible" (OCR layers of scans)


# --- geometry -----------------------------------------------------------------

def _area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _union(a: Box, b: Box) -> Box:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _intersection_area(a: Box, b: Box) -> float:
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def _near(a: Box, b: Box, gap: float) -> bool:
    return a[0] <= b[2] + gap and b[0] <= a[2] + gap and a[1] <= b[3] + gap and b[1] <= a[3] + gap


def _clip(b: Box, w: float, h: float) -> Box:
    return (max(0.0, b[0]), max(0.0, b[1]), min(w, b[2]), min(h, b[3]))


# --- object extraction ----------------------------------------------------------

def _page_space_bounds(obj: pdfium.PdfObject) -> tuple[float, float, float, float]:
    """Bounds (l, b, r, t) in page space.

    PDFium reports bounds of objects nested in Form XObjects in the form's own space,
    so apply the matrices of all enclosing forms.
    """
    l, b, r, t = obj.get_bounds()
    pts = [(l, b), (r, b), (r, t), (l, t)]
    container = obj.container
    while container is not None:
        ma, mb, mc, md, me, mf = container.get_matrix().get()
        pts = [(ma * x + mc * y + me, mb * x + md * y + mf) for x, y in pts]
        container = container.container
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _path_is_orthogonal(obj: pdfium.PdfObject, eps: float = 0.01) -> bool:
    """True if the path has only straight horizontal/vertical segments (rules, rectangles).

    Tables, cell shading and callout boxes are orthogonal; diagrams almost always contain
    an arrowhead, a diagonal connector or a rounded corner.
    """
    ma, mb, mc, md, _, _ = obj.get_matrix().get()
    x, y = ctypes.c_float(), ctypes.c_float()
    start = prev = None
    for i in range(pdfium_c.FPDFPath_CountSegments(obj.raw)):
        seg = pdfium_c.FPDFPath_GetPathSegment(obj.raw, i)
        kind = pdfium_c.FPDFPathSegment_GetType(seg)
        if kind == pdfium_c.FPDF_SEGMENT_BEZIERTO:
            return False
        pdfium_c.FPDFPathSegment_GetPoint(seg, ctypes.byref(x), ctypes.byref(y))
        point = (x.value, y.value)
        if kind == pdfium_c.FPDF_SEGMENT_MOVETO:
            start = point
        else:
            for a, b in ((prev, point), (point, start) if pdfium_c.FPDFPathSegment_GetClose(seg) else (None, None)):
                if a is None or b is None:
                    continue
                dx, dy = b[0] - a[0], b[1] - a[1]
                dx, dy = ma * dx + mc * dy, mb * dx + md * dy  # into page orientation
                if abs(dx) > eps and abs(dy) > eps:
                    return False
        prev = point
    return True


def _extract_objects(page: pdfium.PdfPage, origin: tuple[float, float]) -> list[_Obj]:
    ox, oy = origin  # left and top of the page box in PDF space
    wanted = (pdfium_c.FPDF_PAGEOBJ_IMAGE, pdfium_c.FPDF_PAGEOBJ_PATH, pdfium_c.FPDF_PAGEOBJ_TEXT)
    out = []
    for obj in page.get_objects(filter=wanted):
        l, b, r, t = _page_space_bounds(obj)
        o = _Obj(box=(l - ox, oy - t, r - ox, oy - b), kind=obj.type)
        if obj.type == pdfium_c.FPDF_PAGEOBJ_IMAGE:
            o.image_px = tuple(obj.get_px_size())
        elif obj.type == pdfium_c.FPDF_PAGEOBJ_PATH:
            o.orthogonal = _path_is_orthogonal(obj)
        else:
            mode = pdfium_c.FPDFTextObj_GetTextRenderMode(obj.raw)
            o.visible = mode != pdfium_c.FPDF_TEXTRENDERMODE_INVISIBLE
        out.append(o)
    return out


# --- clustering -----------------------------------------------------------------

def _cluster(boxes: list[Box], gap: float) -> list[list[int]]:
    """Group boxes that are within `gap` of each other (transitively). Sort-and-sweep + union-find."""
    parent = list(range(len(boxes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    order = sorted(range(len(boxes)), key=lambda i: boxes[i][0])
    for a_pos, i in enumerate(order):
        bi = boxes[i]
        for j in order[a_pos + 1:]:
            bj = boxes[j]
            if bj[0] > bi[2] + gap:
                break
            if _near(bi, bj, gap):
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(boxes)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def _merge_overlapping(boxes: list[tuple[Box, int]]) -> list[tuple[Box, int]]:
    """Merge (box, object_count) pairs whose boxes overlap, until stable."""
    changed = True
    while changed:
        changed = False
        out: list[tuple[Box, int]] = []
        for box, n in boxes:
            for k, (ob, on) in enumerate(out):
                if _intersection_area(box, ob) > 0:
                    out[k] = (_union(box, ob), n + on)
                    changed = True
                    break
            else:
                out.append((box, n))
        boxes = out
    return boxes


# --- classification -------------------------------------------------------------

def _vector_diagrams(paths: list[_Obj], texts: list[Box], w: float, h: float,
                     cfg: RegionConfig) -> list[tuple[Box, int]]:
    page_area = w * h
    kept, shaped = [], []
    for p in paths:
        b = p.box
        bw, bh = b[2] - b[0], b[3] - b[1]
        if _area(b) > cfg.path_max_page_fraction * page_area:
            continue  # background or page frame
        if min(bw, bh) <= cfg.rule_max_thickness_pt and max(bw, bh) >= cfg.separator_min_page_width_fraction * w:
            continue  # page-wide separator line
        kept.append(b)
        shaped.append(not p.orthogonal)
    if not kept:
        return []
    if len(kept) > cfg.max_paths_per_page:
        groups = [list(range(len(kept)))]
    else:
        groups = _cluster(kept, cfg.cluster_gap_pt)

    diagrams = []
    for g in groups:
        if len(g) < cfg.diagram_min_paths:
            continue
        if sum(shaped[i] for i in g) < cfg.diagram_min_shaped_paths:
            continue  # only horizontal/vertical lines and rectangles: a table or a shaded box
        box = kept[g[0]]
        for i in g[1:]:
            box = _union(box, kept[i])
        if _area(box) < cfg.diagram_min_page_fraction * page_area:
            continue
        diagrams.append((box, len(g)))

    # Attach text labels: text within the margin of the diagram that lies above/below it
    # (centre inside its x-range) or beside it (centre inside its y-range). When unsure we
    # include the text: a crop that is too large costs time, a crop that cuts text loses it.
    m = cfg.label_margin_pt
    labelled = []
    for box, n in diagrams:
        grown = box
        for t in texts:
            cx, cy = (t[0] + t[2]) / 2, (t[1] + t[3]) / 2
            aligned = box[0] - m <= cx <= box[2] + m or box[1] - m <= cy <= box[3] + m
            if aligned and _near(t, box, m):
                grown = _union(grown, t)
        labelled.append((grown, n))
    return _merge_overlapping(labelled)


def _classify_page(objs: list[_Obj], w: float, h: float, cfg: RegionConfig) -> list[tuple[str, Box, int, tuple | None]]:
    page_area = w * h
    images = [o for o in objs if o.kind == pdfium_c.FPDF_PAGEOBJ_IMAGE]
    images = [o for o in images
              if min(o.box[2] - o.box[0], o.box[3] - o.box[1]) >= cfg.min_image_side_pt]
    paths = [o for o in objs if o.kind == pdfium_c.FPDF_PAGEOBJ_PATH]
    texts = [o.box for o in objs if o.kind == pdfium_c.FPDF_PAGEOBJ_TEXT]
    visible_texts = sum(1 for o in objs if o.kind == pdfium_c.FPDF_PAGEOBJ_TEXT and o.visible)

    coverage = sum(_area(_clip(o.box, w, h)) for o in images) / page_area if images else 0.0
    # A page-sized image under visible native text is a designed background (cover, section
    # divider), not a scan; it stays an embedded_image. An invisible OCR layer still means scan.
    if coverage >= cfg.scan_min_coverage and visible_texts <= cfg.scan_max_visible_text_objects:
        px = images[0].image_px if len(images) == 1 else None
        return [("scanned_page", (0.0, 0.0, w, h), len(images), px)]

    regions: list[tuple[str, Box, int, tuple | None]] = []
    diagrams = _vector_diagrams(paths, texts, w, h, cfg)
    for img in images:
        absorbed = False
        for k, (dbox, n) in enumerate(diagrams):
            if _intersection_area(img.box, dbox) >= cfg.absorb_min_overlap * _area(img.box):
                diagrams[k] = (_union(dbox, img.box), n + 1)
                absorbed = True
                break
        if absorbed:
            continue
        kind = "logo" if _area(img.box) < cfg.logo_max_page_fraction * page_area else "embedded_image"
        regions.append((kind, img.box, 1, img.image_px))
    regions += [("vector_diagram", box, n, None) for box, n in _merge_overlapping(diagrams)]
    return [(k, _clip(b, w, h), n, px) for k, b, n, px in regions]


def _native_chars(textpage: pdfium.PdfTextPage, box: Box, origin: tuple[float, float]) -> int:
    ox, oy = origin
    text = textpage.get_text_bounded(left=box[0] + ox, bottom=oy - box[3], right=box[2] + ox, top=oy - box[1])
    return sum(1 for ch in text if not ch.isspace())


def _page_box(page: pdfium.PdfPage) -> tuple[float, float, float, float]:
    return page.get_bbox()  # media box ∩ crop box, PDF space (l, b, r, t)


def inventory_page(page: pdfium.PdfPage, page_no: int, cfg: RegionConfig) -> PageInventory:
    l, b, r, t = _page_box(page)
    w, h = r - l, t - b
    origin = (l, t)
    textpage = page.get_textpage()
    native = sum(1 for ch in textpage.get_text_range() if not ch.isspace())
    classified = _classify_page(_extract_objects(page, origin), w, h, cfg)
    classified.sort(key=lambda c: (round(c[1][1], 1), round(c[1][0], 1)))  # reading order: top, then left
    regions = [
        Region(id=f"p{page_no}r{i}", kind=kind, bbox_pt=[round(v, 2) for v in box],
               native_chars=_native_chars(textpage, box, origin), n_objects=n,
               image_px=list(px) if px else None)
        for i, (kind, box, n, px) in enumerate(classified, start=1)
    ]
    return PageInventory(page=page_no, width_pt=round(w, 2), height_pt=round(h, 2),
                         rotation=page.get_rotation(), native_chars=native, regions=regions)


def inventory(pdf_path: str | Path, cfg: RegionConfig | None = None) -> DocumentInventory:
    cfg = cfg or load_config().regions
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        pages = [inventory_page(pdf[i], i + 1, cfg) for i in range(len(pdf))]
    finally:
        pdf.close()
    return DocumentInventory(source=Path(pdf_path).name, pages=pages)


# --- rendering ------------------------------------------------------------------

def render_page(pdf: pdfium.PdfDocument, page_no: int, dpi: int) -> Image.Image:
    """Render a page (1-based) in its unrotated orientation, matching ``bbox_pt`` coordinates."""
    page = pdf[page_no - 1]
    # PDFium applies the page's /Rotate when rendering; undo it so pixels match bbox_pt.
    undo = (360 - page.get_rotation()) % 360
    return page.render(scale=dpi / 72, rotation=undo).to_pil().convert("RGB")


def crop_region(page_image: Image.Image, bbox_pt: list[float], dpi: int) -> Image.Image:
    s = dpi / 72
    x0, y0, x1, y1 = (round(v * s) for v in bbox_pt)
    return page_image.crop((x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)))


def render_region(pdf_path: str | Path, page_no: int, bbox_pt: list[float], dpi: int) -> Image.Image:
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        return crop_region(render_page(pdf, page_no, dpi), bbox_pt, dpi)
    finally:
        pdf.close()


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python -m ocr.pdf_regions file.pdf", file=sys.stderr)
        return 2
    print(json.dumps(inventory(argv[0]).to_dict(), indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
