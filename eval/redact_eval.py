"""How well erasing words removes them (ocr/redact.py), measured on the SVG ground truth.

Every word the engine found (the boxes of a saved run) is erased, as if all of them were to be
anonymised. The SVG renders without text (``*.notext.png``) show what the picture looks like
with no text at all, so per ground-truth word:

- ``residual``: share of its ink still there - pixels whose difference to the text-free render
  is at least half of what it was and at least ``ink_level``;
- ``leak``: a recogniser, run again on the erased place, still reads the word (similarity of
  the reading to the word at least ``leak_similarity``); reported for PaddleOCR (which
  ``verified`` itself uses to check its work) and for Tesseract (independent);
- ``damage``: pixels outside every word's ink changed by more than ``ink_level`` (frames,
  lines, icons erased with the words), per erased word;
- ``restore_error``: mean difference to the text-free render over the erased pixels (how well
  the background was rebuilt).

Words the engine did not find are not erased and are reported apart (``not_found``).

Usage:
    python -m eval.redact_eval --from-run <run> --images eval/dev_sets/svg_dev_v1.txt [--method auto ...]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from rapidfuzz.distance import Levenshtein

from eval.box_ink import text_ink, word_inks
from eval.metrics import overlap, scored_text
from ocr.config import load_config
from ocr.config import RedactConfig
from ocr.redact import erase, erase_verified


def measure_image(img: Image.Image, notext: np.ndarray, gt, preds, cfg: RedactConfig, eval_cfg, readers=None,
                  ink_level: int = 30, leak_similarity: float = 0.6, verifier=None) -> dict:
    """readers: {name: engine} that re-read the erased places (leak check); verifier: an engine
    for erase_verified (None: plain erase)."""
    a = np.asarray(img.convert("RGB")).astype(np.int16)
    b = notext.astype(np.int16)
    diff0 = np.abs(a - b).max(axis=2)
    inks = word_inks(diff0.astype(np.uint8), gt)
    if verifier is not None:
        erased, info = erase_verified(img, [(p, t) for t, p in preds], verifier, cfg)
    else:
        erased, info = erase(img, [p for _, p in preds], cfg)
    e = np.asarray(erased).astype(np.int16)
    diff1 = np.abs(e - b).max(axis=2)
    all_ink = np.zeros(diff0.shape, bool)
    for wi in inks:
        if wi is not None:
            all_ink[wi.ys, wi.xs] = True
    out = {"words": 0, "not_found": 0, "residual": [], "leaks": {}, "checked": 0, "erased": len(preds),
           "damage_px": int(((diff1 > ink_level) & ~all_ink & (diff0 <= ink_level)).sum()),
           "restore_error": [], "methods": {}}
    for i in info:
        key = i["method"] + ("/" + i["step"] if i.get("step", "ink") != "ink" else "")
        out["methods"][key] = out["methods"].get(key, 0) + 1
        x0, y0, x1, y1 = i["wide"]
        m = i["mask"] & ~all_ink[y0:y1, x0:x1]
        if m.any():
            out["restore_error"].append(float(diff1[y0:y1, x0:x1][m].mean()))
    for wi in inks:
        if wi is None or wi.gt.ignore or len(scored_text(wi.gt.text, eval_cfg)) < eval_cfg.min_scored_chars:
            continue
        out["words"] += 1
        if not any(overlap(wi.gt.bbox, p) >= eval_cfg.match_min_overlap for _, p in preds):
            out["not_found"] += 1
            continue
        d0, d1 = diff0[wi.ys, wi.xs], diff1[wi.ys, wi.xs]
        still = (d1 >= ink_level) & (d1 >= 0.5 * d0)
        out["residual"].append(float(still.mean()))
        if readers:
            out["checked"] += 1
            for name, reader in readers.items():
                got = " ".join(w.text for w in reader.recognize(erased, [wi.gt.bbox]))
                a_, b_ = scored_text(got, eval_cfg), scored_text(wi.gt.text, eval_cfg)
                if a_ and Levenshtein.normalized_similarity(a_, b_) >= leak_similarity:
                    out["leaks"][name] = out["leaks"].get(name, 0) + 1
    return out


def summarize(parts: list[dict]) -> dict:
    res = np.array([r for p in parts for r in p["residual"]])
    words = sum(p["words"] for p in parts)
    erased = sum(p["erased"] for p in parts)
    checked = sum(p["checked"] for p in parts)
    methods: dict[str, int] = {}
    for p in parts:
        for k, v in p["methods"].items():
            methods[k] = methods.get(k, 0) + v
    restore = [r for p in parts for r in p["restore_error"]]
    return {
        "words": words, "not_found": sum(p["not_found"] for p in parts), "erased_boxes": erased,
        "residual_zero": float(np.mean(res == 0)) if len(res) else None,
        "residual_le_1pct": float(np.mean(res <= 0.01)) if len(res) else None,
        "residual_mean": float(res.mean()) if len(res) else None,
        "residual_p99": float(np.percentile(res, 99)) if len(res) else None,
        "leak_rate": {n: sum(p["leaks"].get(n, 0) for p in parts) / checked
                      for n in sorted({n for p in parts for n in p["leaks"]} | {"paddle", "tesseract"})} if checked else None,
        "damage_px_per_erased_word": sum(p["damage_px"] for p in parts) / max(1, erased),
        "restore_error_mean": float(np.mean(restore)) if restore else None,
        "methods": methods,
    }


def main(argv: list[str]) -> int:
    from eval.run_eval import DEFAULT_GT, REPORTS, RUNS, ReplayEngine, images_from, load_gt, load_index

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from-run", required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--gt", type=Path, default=DEFAULT_GT)
    ap.add_argument("--method", action="append",
                    help="box_inpaint, box, auto, fill, inpaint, verified (auto + re-read and escalate); default: all")
    ap.add_argument("--no-reread", action="store_true", help="skip the leak check by the recogniser")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    cfg = load_config()
    from ocr.engines.registry import create
    paddle = create(cfg.consensus.primary, config=cfg)
    readers = None if args.no_reread else {"paddle": paddle, "tesseract": create(cfg.consensus.secondary, config=cfg)}
    engine = ReplayEngine(RUNS / args.from_run, REPORTS, False)
    rows = [r for r in images_from(args.images, load_index(args.gt)) if r["json"] in engine.preds]
    methods = args.method or ["box_inpaint", "box", "fill", "inpaint", "auto", "verified"]
    parts = {m: [] for m in methods}
    for k, row in enumerate(rows, 1):
        img, gt, _ = load_gt(args.gt, row)
        png = (args.gt / row["json"]).with_name(json.loads((args.gt / row["json"]).read_text())["image"])
        notext_png = png.with_name(png.stem + ".notext.png")
        if not notext_png.exists():
            continue
        notext = np.asarray(Image.open(notext_png).convert("RGB"))
        words, _ = engine.read_row(row)
        preds = [(w.text, tuple(w.bbox_px)) for w in words]
        for m in methods:
            rc = dataclasses.replace(RedactConfig(), method="auto" if m == "verified" else m)
            parts[m].append(measure_image(img, notext, gt, preds, rc, cfg.eval, readers,
                                          verifier=paddle if m == "verified" else None))
        if k % 20 == 0:
            print(f"{k}/{len(rows)} images", file=sys.stderr, flush=True)
    summary = {m: summarize(p) for m, p in parts.items()}
    print(json.dumps(summary, indent=1))
    if args.out:
        args.out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
