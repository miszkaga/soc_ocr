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
