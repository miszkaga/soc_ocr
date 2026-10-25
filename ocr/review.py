"""Review queue: a PNG crop for every uncertain word, with the word's box outlined."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from ocr.engines.base import OcrWord


def save_crop(image: Image.Image, word: OcrWord, path: Path, margin: float = 1.5) -> None:
    """The word with ``margin`` word heights of context around it; the box outlined in red."""
    x0, y0, x1, y1 = word.bbox_px
    m = margin * max(4.0, y1 - y0)
    cx0, cy0 = max(0, int(x0 - m)), max(0, int(y0 - m))
    cx1, cy1 = min(image.width, int(x1 + m) + 1), min(image.height, int(y1 + m) + 1)
    crop = image.crop((cx0, cy0, max(cx1, cx0 + 1), max(cy1, cy0 + 1))).convert("RGB")
    ImageDraw.Draw(crop).rectangle([x0 - cx0 - 1, y0 - cy0 - 1, x1 - cx0, y1 - cy0], outline=(220, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(path)


ACCEPTED = (0, 170, 0)
UNCERTAIN = (230, 0, 0)
REGION = (40, 110, 230)


def draw_boxes(image: Image.Image, words: list[tuple[tuple, bool]], regions: list[tuple] = ()) -> Image.Image:
    """A copy of the image with a box around every word: green when accepted, red when it
    needs review; ``regions`` (PDF regions read on a page) outlined in blue."""
    out = image.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    width = max(1, round(max(out.size) / 1000))
    for box in regions:
        d.rectangle([box[0] - 2 * width, box[1] - 2 * width, box[2] + 2 * width, box[3] + 2 * width],
                    outline=REGION, width=width)
    for box, accepted in words:
        d.rectangle([box[0] - width, box[1] - width, box[2] + width, box[3] + width],
                    outline=ACCEPTED if accepted else UNCERTAIN, width=width)
    return out


_FONT_PATHS = ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/TTF/DejaVuSans.ttf",
               "/Library/Fonts/Arial Unicode.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf")


def _font(size: int):
    from PIL import ImageFont

    for p in _FONT_PATHS:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size)


def _fitted(text: str, w: float, h: float):
    """The largest font that fits the text into a w x h box (at least 6 px)."""
    size = max(6, int(h * 0.85))
    font = _font(size)
    tw = font.getlength(text) if text else 1
    if tw > w and size > 6:
        size = max(6, int(size * w / tw))
        font = _font(size)
    return font


def draw_text_overlay(image: Image.Image, words: list[tuple[tuple, str, bool]], regions: list[tuple] = ()) -> Image.Image:
    """For checking a result by eye: the image with boxes (left) next to the text read, written
    into the same boxes over a faded copy of the image (right). Green: accepted, red: needs
    review. A word taller than wide (vertical text) is written turned."""
    boxed = draw_boxes(image, [(b, ok) for b, _, ok in words], regions)
    faded = Image.blend(image.convert("RGB"), Image.new("RGB", image.size, "white"), 0.85)
    d = ImageDraw.Draw(faded)
    for box, text, ok in words:
        x0, y0, x1, y1 = box
        w, h = max(1.0, x1 - x0), max(1.0, y1 - y0)
        colour = (0, 120, 0) if ok else (200, 0, 0)
        d.rectangle([x0, y0, x1, y1], outline=(200, 200, 200))
        if h > 1.5 * w and len(text) > 1:
            font = _fitted(text, h, w)
            tile = Image.new("RGBA", (int(h) + 4, int(w) + 4), (0, 0, 0, 0))
            ImageDraw.Draw(tile).text((2, 2), text, font=font, fill=colour + (255,))
            tile = tile.rotate(90, expand=True)
            faded.paste(tile, (int(x0) - 2, int(y0) - 2), tile)
        else:
            font = _fitted(text, w, h)
            d.text((x0, y0 + (h - font.size) / 2), text, font=font, fill=colour)
    out = Image.new("RGB", (image.width * 2 + 10, image.height), (160, 160, 160))
    out.paste(boxed, (0, 0))
    out.paste(faded, (image.width + 10, 0))
    return out
