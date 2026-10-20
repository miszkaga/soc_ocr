"""PaddleOCR models (detection + recognition) through RapidOCR on ONNX Runtime.

PaddleOCR reads text lines. We ask RapidOCR for per-character boxes and scores and split
each line into words at its spaces, so words carry character confidences like Tesseract's.
A word's confidence is its weakest character: one doubtful character makes the word doubtful.
"""
from __future__ import annotations

import hashlib
from functools import cached_property
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from ocr.config import PaddleConfig, load_config
from ocr.engines.base import Box, OcrEngine, OcrWord

REPO = Path(__file__).resolve().parents[2]


def _quad_box(quad) -> Box:
    pts = np.asarray(quad, dtype=float).reshape(-1, 2)
    return (float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max()))


def _union(boxes: list[Box]) -> Box:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def line_to_words(text: str, chars: list[tuple[str, float, Box]], offset=(0.0, 0.0)) -> list[OcrWord]:
    """Split one recognized line into words.

    `chars` are the line's non-space characters in order, each with score and box. When they
    line up with the non-space characters of `text`, spaces in `text` mark word breaks;
    otherwise the line is kept as a single word (no guessing at boundaries).
    """
    ox, oy = offset
    chars = [(c, s, (b[0] + ox, b[1] + oy, b[2] + ox, b[3] + oy)) for c, s, b in chars]
    if not chars:
        return []
    groups: list[list[tuple[str, float, Box]]] = []
    if "".join(c for c, _, _ in chars) == text.replace(" ", ""):
        it = iter(chars)
        current: list = []
        for ch in text:
            if ch == " ":
                if current:
                    groups.append(current)
                current = []
            else:
                current.append(next(it))
        if current:
            groups.append(current)
    else:
        groups = [chars]
    words = []
    for g in groups:
        conf = [s for c, s, _ in g for _ in c]
        words.append(OcrWord(text="".join(c for c, _, _ in g), bbox_px=_union([b for _, _, b in g]),
                             conf=min(conf), char_conf=conf))
    return words


class PaddleOnnxEngine(OcrEngine):
    name = "paddle_onnx"
    backend = "onnxruntime"

    def __init__(self, cfg: PaddleConfig | None = None):
        self.cfg = cfg or load_config().paddle

    def _model_root(self) -> Path:
        p = Path(self.cfg.model_root_dir)
        return p if p.is_absolute() else REPO / p

    @cached_property
    def _ocr(self):
        from rapidocr import RapidOCR
        from rapidocr.utils.typings import EngineType, LangDet, LangRec, ModelType, OCRVersion

        c = self.cfg
        self._model_root().mkdir(parents=True, exist_ok=True)
        engine = EngineType(self.backend)
        params = {
            "Global.model_root_dir": str(self._model_root()),
            "Global.log_level": "warning",
            "Global.text_score": c.text_score,
            "Global.max_side_len": c.max_side_len,
            "Global.use_cls": c.use_cls,
            "Det.engine_type": engine, "Cls.engine_type": engine, "Rec.engine_type": engine,
            "Det.ocr_version": OCRVersion(c.ocr_version), "Det.model_type": ModelType(c.det_model_type),
            "Det.lang_type": LangDet(c.det_lang),
            "Det.limit_side_len": c.det_limit_side_len, "Det.limit_type": c.det_limit_type,
            "Det.thresh": c.det_thresh, "Det.box_thresh": c.det_box_thresh,
            "Det.unclip_ratio": c.det_unclip_ratio,
            "Rec.ocr_version": OCRVersion(c.ocr_version), "Rec.model_type": ModelType(c.rec_model_type),
            "Rec.lang_type": LangRec(c.rec_lang),
        }
        if self.backend == "onnxruntime":
            params["EngineConfig.onnxruntime.intra_op_num_threads"] = c.threads
        else:
            params["EngineConfig.openvino.inference_num_threads"] = c.threads
        return RapidOCR(params=params)

    def _model_files(self) -> dict[str, str]:
        """task -> model file name, resolved the way RapidOCR does from its model list."""
        import rapidocr

        routes_file = Path(rapidocr.__file__).parent / "default_models.yaml"
        d = yaml.safe_load(routes_file.read_text(encoding="utf-8"))
        c = self.cfg
        out = {}
        for task, mtype, lang in (("det", c.det_model_type, c.det_lang), ("rec", c.rec_model_type, c.rec_lang)):
            routes = d["model_routes"][c.ocr_version][task][mtype]
            route = next((r for r in routes if r["lang"] == lang or lang in r.get("supported_langs", [])), routes[0])
            entry = d["onnxruntime"][c.ocr_version][task][route["model_key"]]
            out[task] = Path(entry["model_dir"]).name
        return out

    def versions(self) -> dict[str, str]:
        import rapidocr

        v = {"rapidocr": getattr(rapidocr, "__version__", None) or _pkg_version("rapidocr"),
             "backend": self.backend, "runtime": _pkg_version("onnxruntime" if self.backend == "onnxruntime" else "openvino"),
             "ocr_version": self.cfg.ocr_version}
        for task, name in self._model_files().items():
            path = self._model_root() / name
            v[f"{task}_model"] = name
            if path.exists():
                v[f"{task}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        return v

    def _run(self, img: Image.Image):
        # Pass every switch explicitly: RapidOCR keeps per-call switches for later calls.
        return self._ocr(img.convert("RGB"), use_det=True, use_cls=self.cfg.use_cls, use_rec=True,
                         return_word_box=True, return_single_char_box=True)

    @staticmethod
    def _words(out, offset=(0.0, 0.0)) -> list[OcrWord]:
        words = []
        if out is None or out.txts is None:
            return words
        for text, chars in zip(out.txts, out.word_results or [()] * len(out.txts)):
            entries = [(c, float(s), _quad_box(q)) for c, s, q in (chars or []) if q is not None]
            words += line_to_words(text, entries, offset)
        return words

    def read(self, img: Image.Image) -> list[OcrWord]:
        return self._words(self._run(img))

    def detect(self, img: Image.Image) -> list[Box]:
        out = self._ocr(img.convert("RGB"), use_det=True, use_cls=False, use_rec=False)
        return [] if out.boxes is None else [_quad_box(q) for q in out.boxes]

    def recognize(self, img: Image.Image, boxes: list[Box]) -> list[OcrWord]:
        """Recognition only, on each given box (a detector inside a small crop misfires).

        Character positions come from the recognizer's CTC columns, mapped linearly onto the
        box width, so character and word boxes are approximate along the line.
        """
        pad = self.cfg.recognize_pad_px
        words = []
        for x0, y0, x1, y1 in boxes:
            cx0, cy0 = max(0, int(x0) - pad), max(0, int(y0) - pad)
            cx1, cy1 = min(img.width, int(x1) + pad + 1), min(img.height, int(y1) + pad + 1)
            if cx1 <= cx0 or cy1 <= cy0:
                continue
            out = self._ocr(img.crop((cx0, cy0, cx1, cy1)).convert("RGB"), use_det=False, use_cls=False,
                            use_rec=True, return_word_box=True, return_single_char_box=True)
            if not out.txts or not out.word_results or not out.word_results[0]:
                continue
            info = out.word_results[0]
            chars = [c for w in info.words for c in w]
            cols = [c for w in info.word_cols for c in w]
            step = (cx1 - cx0) / max(info.line_txt_len, 1.0)
            entries = [(ch, float(conf), (cx0 + col * step, cy0, cx0 + (col + 1) * step, cy1))
                       for ch, col, conf in zip(chars, cols, info.confs)]
            words += line_to_words(out.txts[0], entries)
        return words

def _pkg_version(name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return "missing"
