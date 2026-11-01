"""Erase words from an image and write replacement text in their place (prototype).

For each word box (the engine's box, snapped to ink):

1. The background is the commonest colour of a ring around the box; the word's ink is every
   pixel in the box (widened by ``margin_px``) far enough from it, grown by ``dilate_px`` so
   antialiased edges go too.
2. The word is removed:
   - ``box_inpaint`` (default): the whole box rebuilt from its surroundings - nothing of the
     word can stay inside the box;
   - ``box``: the whole box painted (``fill_colour`` or the background colour);
   - ``fill``: only the ink painted with the background colour - right on a plain background;
   - ``inpaint``: rebuilt from the surroundings (OpenCV Telea) - for gradients, textures,
     highlights; chosen automatically (``auto``) when the ring around the box is not plain.
   The ink-only methods keep lines and icons crossing the box, but on gradients a faint outline
   of the letters can stay (eval/redact_eval.py measures it).
3. Replacement text (optional) is written into the box in the word's own colour (the median of
   its strongest ink), as tall as the box, shrunk to fit its width; a word taller than wide is
   written turned.

Lines or icons crossing the box go with the word in the box methods, and with the ink methods
when they are as dark as the text; eval/redact_eval.py reports that damage. ``preview`` paints
the boxes black and writes the text read in red, for testing.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ocr.config import RedactConfig


Box = tuple[float, float, float, float]


def _clip(box: Box, w: int, h: int, m: int) -> tuple[int, int, int, int]:
    return (max(0, int(np.floor(box[0])) - m), max(0, int(np.floor(box[1])) - m),
            min(w, int(np.ceil(box[2])) + m), min(h, int(np.ceil(box[3])) + m))


def _background(a: np.ndarray, x0, y0, x1, y1, ring: int) -> tuple[np.ndarray, float]:
    """The commonest colour of a ring of ``ring`` px around the box, and the share of ring
    pixels close to it (1: plain)."""
    H, W = a.shape[:2]
    X0, Y0, X1, Y1 = max(0, x0 - ring), max(0, y0 - ring), min(W, x1 + ring), min(H, y1 + ring)
    outer = a[Y0:Y1, X0:X1].reshape(-1, 3)
    inner = np.zeros((Y1 - Y0, X1 - X0), bool)
    inner[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = True
    px = outer[~inner.reshape(-1)] if (~inner).any() else outer
    q = (px >> 4).astype(np.int32)
    key = (q[:, 0] << 8) | (q[:, 1] << 4) | q[:, 2]
    bg = np.median(px[key == np.bincount(key).argmax()], axis=0)
    return bg, float(np.mean(np.abs(px.astype(float) - bg).max(axis=1) <= 12.0))


def word_mask(a: np.ndarray, box: Box, cfg: RedactConfig) -> tuple[np.ndarray, np.ndarray, float, tuple]:
    """Ink of one word as a full-image mask, plus its background colour, the ring's plainness
    and the widened box."""
    H, W = a.shape[:2]
    x0, y0, x1, y1 = _clip(box, W, H, cfg.margin_px)
    mask = np.zeros((H, W), bool)
    if x1 <= x0 or y1 <= y0:
        return mask, np.array([255.0, 255.0, 255.0]), 1.0, (x0, y0, x1, y1)
    bg, plain = _background(a, x0, y0, x1, y1, ring=max(2, cfg.margin_px + 1))
    dist = np.abs(a[y0:y1, x0:x1].astype(float) - bg).max(axis=2)
    thr = max(cfg.min_contrast, cfg.rel_contrast * float(dist.max()))
    ink = (dist >= thr).astype(np.uint8)
    if cfg.dilate_px > 0:
        k = 2 * cfg.dilate_px + 1
        ink = cv2.dilate(ink, np.ones((k, k), np.uint8))
    mask[y0:y1, x0:x1] = ink.astype(bool)
    return mask, bg, plain, (x0, y0, x1, y1)


def text_colour(a: np.ndarray, mask: np.ndarray, bg: np.ndarray) -> tuple[int, int, int]:
    px = a[mask].astype(float)
    if len(px) == 0:
        return (0, 0, 0)
    d = np.abs(px - bg).max(axis=1)
    strong = px[d >= 0.7 * d.max()]
    return tuple(int(v) for v in np.median(strong, axis=0))


def erase(image: Image.Image, boxes: list[Box], cfg: RedactConfig | None = None) -> tuple[Image.Image, list[dict]]:
    """The image with the words in ``boxes`` erased, and per word: mask, background, text
    colour and the method used."""
    cfg = cfg or RedactConfig()
    a = np.asarray(image.convert("RGB")).copy()
    info, to_inpaint = [], np.zeros(a.shape[:2], np.uint8)
    for box in boxes:
        mask, bg, plain, wide = word_mask(a, box, cfg)
        colour = text_colour(a, mask, bg)
        method = cfg.method
        if method == "auto":
            method = "fill" if plain >= cfg.plain_share else "inpaint"
        fill = np.array(cfg.fill_colour if cfg.fill_colour else bg, dtype=np.uint8)
        x0, y0, x1, y1 = wide
        if method == "box":
            a[y0:y1, x0:x1] = fill
        elif method == "box_inpaint":
            to_inpaint[y0:y1, x0:x1] = 1
        elif method == "fill":
            a[mask] = fill
        elif method == "inpaint":
            to_inpaint |= mask.astype(np.uint8)
        else:
            raise ValueError(f"unknown method {method!r}")
        if method in ("box", "box_inpaint"):
            mask = np.zeros_like(mask)
            mask[y0:y1, x0:x1] = True
        info.append({"box": box, "mask": mask, "background": tuple(int(v) for v in bg), "colour": colour,
                     "method": method})
    if to_inpaint.any():
        a = cv2.inpaint(a, to_inpaint, cfg.inpaint_radius, cv2.INPAINT_TELEA)
    return Image.fromarray(a), info


def _font_for(text: str, w: float, h: float):
    """A font whose ink for ``text`` is as tall as ``h`` (as the erased word was), shrunk to fit
    the width ``w`` if the replacement is longer."""
    from ocr.review import _font

    size = max(6, int(round(h)))
    for _ in range(3):  # the ink of a font is shorter than its size: scale until it matches
        font = _font(size)
        x0, y0, x1, y1 = font.getbbox(text)
        if y1 - y0 <= 0:
            break
        new = max(6, int(round(size * h / (y1 - y0))))
        if new == size:
            break
        size = new
    font = _font(size)
    tw = font.getlength(text)
    if tw > w and size > 6:
        font = _font(max(6, int(size * w / tw)))
    return font


def write(image: Image.Image, box: Box, text: str, colour=(0, 0, 0)) -> Image.Image:
    """``text`` written into ``box`` (in place): its ink as tall as the box, shrunk to fit the
    box's width; a box taller than wide gets the text turned (read bottom to top)."""
    x0, y0, x1, y1 = box
    w, h = max(1.0, x1 - x0), max(1.0, y1 - y0)
    vertical = h > 1.5 * w and len(text) > 1
    if vertical:
        w, h = h, w
    font = _font_for(text, w, h)
    bx0, by0, bx1, by1 = font.getbbox(text)
    tile = Image.new("RGBA", (int(np.ceil(w)) + 2, int(np.ceil(h)) + 2), (0, 0, 0, 0))
    ImageDraw.Draw(tile).text((1 - bx0, 1 - by0 + (h - (by1 - by0)) / 2), text, font=font, fill=tuple(colour) + (255,))
    if vertical:
        tile = tile.rotate(90, expand=True)
    image.paste(tile, (int(round(x0)) - 1, int(round(y0)) - 1), tile)
    return image


def redact(image: Image.Image, words: list[tuple[Box, str | None]], cfg: RedactConfig | None = None) -> Image.Image:
    """Erase every word, then write each replacement (None: leave the place empty)."""
    cfg = cfg or RedactConfig()
    out, info = erase(image, [b for b, _ in words], cfg)
    for (box, new), i in zip(words, info):
        if new:
            write(out, box, new, tuple(cfg.text_colour) if cfg.text_colour else i["colour"])
    return out


def preview(image: Image.Image, words: list[tuple[Box, str]], cfg: RedactConfig | None = None) -> Image.Image:
    """For testing the redaction path: every word box (widened by margin_px) filled with
    ``preview_fill`` and the text read written into it unchanged in ``preview_text``."""
    import dataclasses

    cfg = cfg or RedactConfig()
    return redact(image, words, dataclasses.replace(cfg, method="box", fill_colour=tuple(cfg.preview_fill),
                                                    text_colour=tuple(cfg.preview_text)))


def _similar(a: str, b: str) -> float:
    from rapidfuzz.distance import Levenshtein

    a, b = "".join(a.split()), "".join(b.split())
    return Levenshtein.normalized_similarity(a, b) if a and b else 0.0


def erase_verified(image: Image.Image, words: list[tuple[Box, str]], reader, cfg: RedactConfig | None = None
                   ) -> tuple[Image.Image, list[dict]]:
    """``erase``, then the recogniser (``reader``, any engine) reads every erased place again.
    Where it still reads the word - a faint outline left on a gradient or a textured fill - the
    whole box (widened by margin_px) is rebuilt from its surroundings; if the word still shows,
    the box is painted with the background colour. Each word's info gets ``step``: "ink",
    "box_inpaint" or "box_fill"."""
    cfg = cfg or RedactConfig()
    out, info = erase(image, [b for b, _ in words], cfg)
    for i in info:
        i["step"] = "ink"

    def leaks(img, k):
        box, text = words[k]
        got = " ".join(w.text for w in reader.recognize(img, [box]))
        return _similar(got, text) >= cfg.leak_similarity

    for k in [k for k in range(len(words)) if words[k][1] and leaks(out, k)]:
        a = np.asarray(out).copy()
        H, W = a.shape[:2]
        x0, y0, x1, y1 = _clip(words[k][0], W, H, cfg.margin_px + 1)
        m = np.zeros((H, W), np.uint8)
        m[y0:y1, x0:x1] = 1
        out = Image.fromarray(cv2.inpaint(a, m, max(cfg.inpaint_radius, 5), cv2.INPAINT_TELEA))
        info[k]["step"] = "box_inpaint"
        if leaks(out, k):
            a = np.asarray(out).copy()
            a[y0:y1, x0:x1] = np.array(info[k]["background"], np.uint8)
            out = Image.fromarray(a)
            info[k]["step"] = "box_fill"
    return out, info
