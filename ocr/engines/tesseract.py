"""Tesseract 5 through its hOCR output, which carries per-character confidences."""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import pytesseract
from PIL import Image

from ocr.config import TesseractConfig, load_config
from ocr.engines.base import Box, OcrEngine, OcrWord

_BBOX = re.compile(r"bbox (\d+) (\d+) (\d+) (\d+)")
_WCONF = re.compile(r"x_wconf (\d+(?:\.\d+)?)")
_CCONF = re.compile(r"x_conf (\d+(?:\.\d+)?)")


def parse_hocr(hocr: str | bytes, offset: tuple[float, float] = (0.0, 0.0)) -> list[OcrWord]:
    """Words from Tesseract hOCR made with ``hocr_char_boxes=1``.

    Character confidences come from the ``ocrx_cinfo`` spans; the word text is their
    concatenation, so ``len(char_conf) == len(text)`` always holds.
    """
    root = ET.fromstring(hocr)
    ox, oy = offset
    words = []
    for el in root.iter():
        if el.get("class") != "ocrx_word":
            continue
        title = el.get("title", "")
        bbox = _BBOX.search(title)
        wconf = _WCONF.search(title)
        if not bbox or not wconf:
            continue
        chars = [(c.text or "", float(_CCONF.search(c.get("title", "")).group(1)) / 100)
                 for c in el.iter() if c.get("class") == "ocrx_cinfo" and _CCONF.search(c.get("title", ""))]
        if chars:
            text = "".join(t for t, _ in chars)
            char_conf = [conf for t, conf in chars for _ in t]
        else:
            text = "".join(el.itertext())
            char_conf = [float(wconf.group(1)) / 100] * len(text)
        if not text.strip():
            continue
        x0, y0, x1, y1 = (float(v) for v in bbox.groups())
        words.append(OcrWord(text=text, bbox_px=(x0 + ox, y0 + oy, x1 + ox, y1 + oy),
                             conf=float(wconf.group(1)) / 100, char_conf=char_conf))
    return words


class TesseractEngine(OcrEngine):
    name = "tesseract"

    def __init__(self, cfg: TesseractConfig | None = None):
        self.cfg = cfg or load_config().tesseract

    def _config(self, psm: int) -> str:
        parts = [f"--oem {self.cfg.oem}", f"--psm {psm}", "-c hocr_char_boxes=1"]
        if self.cfg.tessdata_dir:
            parts.append(f'--tessdata-dir "{self.cfg.tessdata_dir}"')
        return " ".join(parts)

    def _hocr(self, img: Image.Image, psm: int) -> bytes:
        return pytesseract.image_to_pdf_or_hocr(img.convert("RGB"), extension="hocr", lang=self.cfg.lang,
                                                config=self._config(psm), timeout=self.cfg.timeout_s)

    def versions(self) -> dict[str, str]:
        return {"tesseract": str(pytesseract.get_tesseract_version()), "lang": self.cfg.lang,
                "oem": str(self.cfg.oem), "psm": str(self.cfg.psm),
                "tessdata": self.cfg.tessdata_dir or "system"}

    def read(self, img: Image.Image) -> list[OcrWord]:
        return parse_hocr(self._hocr(img, self.cfg.psm))

    def detect(self, img: Image.Image) -> list[Box]:
        # Tesseract finds words only while recognizing them.
        return [w.bbox_px for w in self.read(img)]

    def _variants(self, img: Image.Image, box: Box):
        """Crops of one box to read: (image, rotation) pairs, plus the scale, the padding and the
        crop origin needed to map positions back.

        The crop keeps ``recognize_pad_px`` of real pixels around the box. Small crops are
        enlarged to ``recognize_min_height_px``; tall, narrow crops (vertical text) are also
        tried rotated both ways.
        """
        c = self.cfg
        pad = c.recognize_pad_px
        x0, y0 = max(0, int(box[0]) - pad), max(0, int(box[1]) - pad)
        x1, y1 = min(img.width, int(box[2]) + pad + 1), min(img.height, int(box[3]) + pad + 1)
        crop = img.crop((x0, y0, x1, y1)).convert("RGB")
        short = min(crop.size)
        scale = 1.0
        if c.recognize_min_height_px and 0 < short < c.recognize_min_height_px:
            scale = min(c.recognize_max_scale, c.recognize_min_height_px / short)
            crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.LANCZOS)
        variants = [(crop, 0)]
        if c.rotate_min_aspect and (box[3] - box[1]) >= c.rotate_min_aspect * (box[2] - box[0]):
            variants += [(crop.rotate(90, expand=True), 90), (crop.rotate(270, expand=True), 270)]
        return variants, scale, 0, (x0, y0)

    def _read_box(self, img: Image.Image, box: Box) -> list[OcrWord]:
        variants, scale, pad, (ox, oy) = self._variants(img, box)
        best: list[OcrWord] = []
        for crop, angle in variants:
            words = parse_hocr(self._hocr(crop, self.cfg.recognize_psm))
            if not words:
                continue
            if angle:
                # Positions inside a rotated crop are not mapped back; the box stands for them.
                words = [OcrWord(w.text, tuple(box), w.conf, w.char_conf) for w in words]
            else:
                words = [OcrWord(w.text, ((w.bbox_px[0] - pad) / scale + ox, (w.bbox_px[1] - pad) / scale + oy,
                                          (w.bbox_px[2] - pad) / scale + ox, (w.bbox_px[3] - pad) / scale + oy),
                                 w.conf, w.char_conf) for w in words]
            if not best or min(w.conf for w in words) > min(w.conf for w in best):
                best = words
        return best

    def recognize_each(self, img: Image.Image, boxes: list[Box]) -> list[list[OcrWord]]:
        # Every call starts a tesseract process; run them in parallel.
        if self.cfg.workers <= 1 or len(boxes) <= 1:
            return [self._read_box(img, b) for b in boxes]
        with ThreadPoolExecutor(max_workers=self.cfg.workers) as pool:
            return list(pool.map(lambda b: self._read_box(img, b), boxes))

    def recognize(self, img: Image.Image, boxes: list[Box]) -> list[OcrWord]:
        """One word per box that was read; several words in a box are joined, as the caller
        asked for the box as a whole."""
        out = []
        for words in self.recognize_each(img, boxes):
            if len(words) == 1:
                out.append(words[0])
            elif words:
                out.append(OcrWord(text=" ".join(w.text for w in words),
                                   bbox_px=(min(w.bbox_px[0] for w in words), min(w.bbox_px[1] for w in words),
                                            max(w.bbox_px[2] for w in words), max(w.bbox_px[3] for w in words)),
                                   conf=min(w.conf for w in words),
                                   char_conf=[c for i, w in enumerate(words)
                                              for c in ([] if i == 0 else [min(w.conf, words[i - 1].conf)]) + w.char_conf]))
        return out
