"""PaddleOCR models (detection + recognition) through RapidOCR on ONNX Runtime.

PaddleOCR reads text lines. We ask RapidOCR for per-character boxes and scores and split
each line into words at its spaces, so words carry character confidences like Tesseract's.
A word's confidence is its weakest character: one doubtful character makes the word doubtful.

RapidOCR's character boxes are estimated from CTC columns, not measured; with refine_boxes the
word boxes are snapped to the ink of each word inside its detected line (ocr/box_refine.py).
"""
from __future__ import annotations

import dataclasses
import hashlib
from functools import cached_property
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from ocr.config import BoxRefineConfig, PaddleConfig, load_config
from ocr.engines.base import Box, OcrEngine, OcrWord

REPO = Path(__file__).resolve().parents[2]


def _quad_box(quad) -> Box:
    pts = np.asarray(quad, dtype=float).reshape(-1, 2)
    return (float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max()))


def _union(boxes: list[Box]) -> Box:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def char_confs(text: str, confs) -> list[float] | None:
    """Confidences of the non-space characters of `text`. The recogniser gives one per
    character of the line, spaces included; None when they do not line up."""
    if confs is None or len(confs) != len(text):
        return None
    return [float(c) for ch, c in zip(text, confs) if not ch.isspace()]


def mirror_along(quad, points) -> np.ndarray:
    """Points reflected across the middle of a line quad, along its long side.

    RapidOCR places character boxes in the line crop as it was before the direction classifier
    turned it by 180 degrees, while the CTC columns come from the turned crop: for those lines
    (text upside down, or vertical text read bottom to top) the boxes come out mirrored along
    the line."""
    q = np.asarray(quad, dtype=np.float64).reshape(-1, 2)
    top, left = q[1] - q[0], q[3] - q[0]
    e = left if np.linalg.norm(left) > np.linalg.norm(top) else top
    e = e / max(np.linalg.norm(e), 1e-9)
    c = q.mean(axis=0)
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return p - 2.0 * ((p - c) @ e)[:, None] * e


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


def settle_turns(ocr, crops, rec_res, flipped: list[bool]) -> int:
    """Second opinion on lines the direction classifier turned by 180 degrees.

    The classifier sometimes turns an upright line ("client app", score 0.99), and the
    recogniser then still reads the upside-down crop in the right order - but its character
    positions are spatially reversed, so mirroring the boxes back would swap the words. Each
    turned line is read again as detected (not turned); if that reading is more confident, it
    replaces the turned one and the line counts as not turned. Changes rec_res and `flipped` in
    place; returns how many turns were undone."""
    turned = [i for i, f in enumerate(flipped) if f and i < len(crops)]
    if not turned or rec_res.txts is None:
        return 0
    try:
        again = ocr.recognize_txt([crops[i] for i in turned])
    except Exception:  # RapidOCRError when nothing is read: keep the classifier's decision
        return 0
    txts, scores = list(rec_res.txts), list(rec_res.scores)
    infos = list(rec_res.word_results or ())
    undone = 0
    for k, i in enumerate(turned):
        if k >= len(again.txts or ()) or i >= len(scores):
            continue
        if again.txts[k].strip() and again.scores[k] > scores[i]:
            txts[i], scores[i] = again.txts[k], again.scores[k]
            if again.word_results and k < len(again.word_results) and i < len(infos):
                infos[i] = again.word_results[k]
            flipped[i] = False
            undone += 1
    rec_res.txts, rec_res.scores, rec_res.word_results = tuple(txts), scores, tuple(infos)
    return undone


class PaddleOnnxEngine(OcrEngine):
    name = "paddle_onnx"
    backend = "onnxruntime"

    def __init__(self, cfg: PaddleConfig | None = None, box_cfg: BoxRefineConfig | None = None):
        self.cfg = cfg or load_config().paddle
        self.box_cfg = box_cfg or load_config().box_refine

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
             "ocr_version": self.cfg.ocr_version, "refine_boxes": str(self.cfg.refine_boxes)}
        for task, name in self._model_files().items():
            path = self._model_root() / name
            v[f"{task}_model"] = name
            if path.exists():
                v[f"{task}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        return v

    def _run(self, img: Image.Image):
        """RapidOCR.__call__ (3.10) step by step, to keep what its final output loses: which
        lines the direction classifier turned by 180 degrees, and the character confidences
        before they are paired with the wrong characters (attached as ``line_flipped`` and
        ``line_confs``, one per output line)."""
        ocr = self._ocr
        # Pass every switch explicitly: RapidOCR keeps per-call switches for later calls.
        ocr.update_params(use_det=True, use_cls=self.cfg.use_cls, use_rec=True,
                          return_word_box=True, return_single_char_box=True)
        ori = ocr.load_img(img.convert("RGB"))
        pre, op_record = ocr.preprocess_img(ori)
        det_res, cls_res, rec_res, crops = ocr.run_ocr_steps(pre, op_record)
        txts = list(rec_res.txts or ())
        n = len(txts)
        flipped = [False] * n
        if cls_res.cls_res is not None and len(cls_res.cls_res) == n:
            thresh = ocr.text_cls.cls_thresh
            flipped = [("180" in label and score > thresh) for label, score in cls_res.cls_res]
        settle_turns(ocr, crops, rec_res, flipped)
        txts = list(rec_res.txts or ())
        infos = list(rec_res.word_results or ())
        confs = [infos[i].confs if i < len(infos) and hasattr(infos[i], "confs") else None for i in range(n)]
        # build_final_output keeps the lines with text, in order.
        valid = [i for i, t in enumerate(txts) if t.strip()]
        out = ocr.build_final_output(ori, det_res, cls_res, rec_res, crops, op_record)
        if out is not None and out.txts is not None and list(out.txts) == [txts[i] for i in valid]:
            out.line_flipped = [flipped[i] for i in valid]
            out.line_confs = [confs[i] for i in valid]
        return out

    @staticmethod
    def _lines(out, offset=(0.0, 0.0)) -> list[tuple[np.ndarray | None, list[OcrWord]]]:
        """(line quad, words) per recognised line."""
        lines = []
        if out is None or out.txts is None:
            return lines
        n = len(out.txts)
        boxes = out.boxes if out.boxes is not None and len(out.boxes) == n else [None] * n
        word_results = list(out.word_results or ()) + [()] * max(0, n - len(out.word_results or ()))
        flipped = getattr(out, "line_flipped", None) or [False] * n
        line_confs = getattr(out, "line_confs", None) or [None] * n
        for text, chars, quad, flip, confs in zip(out.txts, word_results, boxes, flipped, line_confs):
            chars = [(c, s, q) for c, s, q in (chars or []) if q is not None]
            if flip and quad is not None:
                chars = [(c, s, mirror_along(quad, q)) for c, s, q in chars]
            scores = char_confs(text, confs)
            if scores is None or len(scores) != len(chars):
                scores = [s for _, s, _ in chars]
            entries = [(c, float(s), _quad_box(q)) for (c, _, q), s in zip(chars, scores)]
            q = None if quad is None else np.asarray(quad, dtype=np.float64) + np.asarray(offset, dtype=np.float64)
            lines.append((q, line_to_words(text, entries, offset)))
        return lines

    @staticmethod
    def _words(out, offset=(0.0, 0.0)) -> list[OcrWord]:
        return [w for _, ws in PaddleOnnxEngine._lines(out, offset) for w in ws]

    def _refine(self, img: Image.Image, lines) -> list[OcrWord]:
        """Word boxes snapped to ink, line by line (see ocr/box_refine.py)."""
        from ocr.box_refine import refine_line

        arr = np.asarray(img.convert("RGB"))
        quads = [q for q, _ in lines if q is not None]
        words = []
        for q, ws in lines:
            if q is None or not ws:
                words += ws
                continue
            others = [o for o in quads if o is not q]
            refined = refine_line(arr, q, [(w.text, w.bbox_px) for w in ws], self.box_cfg, others)
            words += [dataclasses.replace(w, bbox_px=r.bbox) for w, r in zip(ws, refined)]
        return words

    def read(self, img: Image.Image) -> list[OcrWord]:
        out = self._run(img)
        if not self.cfg.refine_boxes:
            return self._words(out)
        return self._refine(img, self._lines(out))

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
            confs = char_confs(out.txts[0], info.confs)
            if confs is None or len(confs) != len(chars):
                confs = list(info.confs)
            entries = [(ch, float(conf), (cx0 + col * step, cy0, cx0 + (col + 1) * step, cy1))
                       for ch, col, conf in zip(chars, cols, confs)]
            words += line_to_words(out.txts[0], entries)
        return words

def _pkg_version(name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return "missing"
