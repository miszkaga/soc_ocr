"""Builds small, deterministic PDFs with known content for region tests."""
from __future__ import annotations

from pathlib import Path

from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

A4 = (595.0, 842.0)


def solid_image(color: str, size=(200, 100)) -> ImageReader:
    return ImageReader(Image.new("RGB", size, color))


def to_pdf_y(top: float, height: float, page_h: float = A4[1]) -> float:
    """Convert a top-left y (as used in bbox_pt) to reportlab's bottom-left y."""
    return page_h - top - height


def new_canvas(path: Path, pagesize=A4) -> canvas.Canvas:
    c = canvas.Canvas(str(path), pagesize=pagesize, invariant=1)
    return c


def body_text(c: canvas.Canvas, lines=20, top=60.0):
    c.setFont("Helvetica", 11)
    for i in range(lines):
        c.drawString(50, to_pdf_y(top + i * 14, 0), f"Line {i}: plain native text that needs no OCR.")


def diagram(c: canvas.Canvas, x: float, top: float, labels=("Client", "API", "DB")):
    """Three labelled boxes joined by arrows, plus a caption below the boxes. Returns bbox_pt."""
    c.setFont("Helvetica", 9)
    bw, bh, gap = 90.0, 40.0, 50.0
    for i, label in enumerate(labels):
        bx = x + i * (bw + gap)
        c.rect(bx, to_pdf_y(top, bh), bw, bh)
        c.drawString(bx + 10, to_pdf_y(top + 25, 0), label)
        if i < len(labels) - 1:
            ax0, ax1, ay = bx + bw, bx + bw + gap, to_pdf_y(top + bh / 2, 0)
            c.line(ax0, ay, ax1, ay)
            p = c.beginPath()
            p.moveTo(ax1, ay); p.lineTo(ax1 - 6, ay + 4); p.lineTo(ax1 - 6, ay - 4); p.close()
            c.drawPath(p, fill=1)
    caption_top = top + bh + 4
    c.drawString(x, to_pdf_y(caption_top + 8, 0), "edge label")
    right = x + len(labels) * bw + (len(labels) - 1) * gap
    return [x, top, right, caption_top + 8]


def table(c: canvas.Canvas, x: float, top: float, rows=5, cols=3, cw=100.0, rh=18.0):
    c.setLineWidth(0.5)
    for r in range(rows + 1):
        y = to_pdf_y(top + r * rh, 0)
        c.line(x, y, x + cols * cw, y)
    for k in range(cols + 1):
        c.line(x + k * cw, to_pdf_y(top, 0), x + k * cw, to_pdf_y(top + rows * rh, 0))
    c.setFont("Helvetica", 9)
    for r in range(rows):
        for k in range(cols):
            c.drawString(x + k * cw + 4, to_pdf_y(top + r * rh + 13, 0), f"r{r}c{k}")
