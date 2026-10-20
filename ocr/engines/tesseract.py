"""Tesseract 5 through its hOCR output, which carries per-character confidences."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET

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

    def recognize(self, img: Image.Image, boxes: list[Box]) -> list[OcrWord]:
        pad = self.cfg.recognize_pad_px
        out = []
        for x0, y0, x1, y1 in boxes:
            cx0, cy0 = max(0, int(x0) - pad), max(0, int(y0) - pad)
            cx1, cy1 = min(img.width, int(x1) + pad + 1), min(img.height, int(y1) + pad + 1)
            if cx1 <= cx0 or cy1 <= cy0:
                continue
            crop = img.crop((cx0, cy0, cx1, cy1))
            words = parse_hocr(self._hocr(crop, self.cfg.recognize_psm), offset=(cx0, cy0))
            if not words:
                continue
            if len(words) == 1:
                out.append(words[0])
                continue
            # One box read as several words: keep it as one word, as the caller asked for a box.
            out.append(OcrWord(text=" ".join(w.text for w in words),
                               bbox_px=(min(w.bbox_px[0] for w in words), min(w.bbox_px[1] for w in words),
                                        max(w.bbox_px[2] for w in words), max(w.bbox_px[3] for w in words)),
                               conf=min(w.conf for w in words),
                               char_conf=[c for i, w in enumerate(words)
                                          for c in ([] if i == 0 else [min(w.conf, words[i - 1].conf)]) + w.char_conf]))
        return out
