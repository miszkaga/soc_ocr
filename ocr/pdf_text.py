"""Native (selectable) text of a PDF page as words with boxes.

A vector diagram usually keeps its labels as text: PDFium gives every character and its box,
exactly, with no OCR. Characters are joined into words; a word ends at whitespace (including the
spaces and line breaks PDFium generates between text runs), at a gap wider than ``word_gap_em``
of the font size, at a jump to another line or at a change of direction.

Not taken: text in render mode "invisible" (the OCR layer of a scanned page: the picture is what
the reader sees, so OCR reads it), and characters without a usable Unicode mapping (the PDF
shows a glyph whose meaning is unknown; OCR reads it instead).

Boxes are the glyphs' tight boxes in PDF points, top-left origin, y down, the page unrotated
(as in ocr/pdf_regions.py).
"""
from __future__ import annotations

import ctypes
from dataclasses import dataclass

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

Box = tuple[float, float, float, float]


@dataclass
class NativeWord:
    text: str
    box: Box  # points, top-left origin


def _char_box(textpage, i: int) -> tuple[float, float, float, float] | None:
    l, r, b, t = ctypes.c_double(), ctypes.c_double(), ctypes.c_double(), ctypes.c_double()
    if not pdfium_c.FPDFText_GetCharBox(textpage.raw, i, l, r, b, t):
        return None
    return l.value, b.value, r.value, t.value


def native_words(page: pdfium.PdfPage, origin: tuple[float, float], word_gap_em: float = 0.3) -> list[NativeWord]:
    """Visible native words of a page. ``origin``: left and top of the page box in PDF space."""
    ox, oy = origin
    textpage = page.get_textpage()
    try:
        n = pdfium_c.FPDFText_CountChars(textpage.raw)
        visible: dict[int, bool] = {}
        words: list[NativeWord] = []
        cur_text, cur_box, prev = "", None, None  # prev: (box, font size, angle)

        def flush():
            nonlocal cur_text, cur_box, prev
            if cur_text.strip() and cur_box is not None:
                words.append(NativeWord(cur_text, cur_box))
            cur_text, cur_box, prev = "", None, None

        for i in range(n):
            ch = chr(pdfium_c.FPDFText_GetUnicode(textpage.raw, i))
            if ch.isspace() or ch == "\x00" or pdfium_c.FPDFText_IsGenerated(textpage.raw, i) == 1:
                flush()
                continue
            obj = pdfium_c.FPDFText_GetTextObject(textpage.raw, i)
            key = ctypes.cast(obj, ctypes.c_void_p).value or 0
            if key not in visible:
                visible[key] = bool(obj) and \
                    pdfium_c.FPDFTextObj_GetTextRenderMode(obj) != pdfium_c.FPDF_TEXTRENDERMODE_INVISIBLE
            if not visible[key] or pdfium_c.FPDFText_HasUnicodeMapError(textpage.raw, i) == 1:
                flush()
                continue
            raw = _char_box(textpage, i)
            if raw is None or raw[2] <= raw[0] or raw[3] <= raw[1]:
                flush()  # a glyph without extent
                continue
            l, b, r, t = raw
            box = (l - ox, oy - t, r - ox, oy - b)
            size = max(1.0, pdfium_c.FPDFText_GetFontSize(textpage.raw, i))
            angle = round(pdfium_c.FPDFText_GetCharAngle(textpage.raw, i), 2)
            if prev is not None:
                pbox, psize, pangle = prev
                em = max(size, psize)
                if angle != pangle:
                    flush()
                elif abs(angle) < 0.1 or abs(angle - 3.1416) < 0.1:  # horizontal text
                    same_line = min(box[3], pbox[3]) - max(box[1], pbox[1]) > -0.2 * em
                    gap = max(box[0] - pbox[2], pbox[0] - box[2])
                    if not same_line or gap > word_gap_em * em:
                        flush()
                else:  # vertical or slanted: gap along the larger extent
                    gap = max(box[0] - pbox[2], pbox[0] - box[2], box[1] - pbox[3], pbox[1] - box[3])
                    if gap > word_gap_em * em:
                        flush()
            cur_text += ch
            cur_box = box if cur_box is None else (min(cur_box[0], box[0]), min(cur_box[1], box[1]),
                                                   max(cur_box[2], box[2]), max(cur_box[3], box[3]))
            prev = (box, size, angle)
        flush()
        return words
    finally:
        textpage.close()


def inside(word: NativeWord, region: Box) -> bool:
    """The word's centre lies in the region."""
    cx, cy = (word.box[0] + word.box[2]) / 2, (word.box[1] + word.box[3]) / 2
    return region[0] <= cx <= region[2] and region[1] <= cy <= region[3]
