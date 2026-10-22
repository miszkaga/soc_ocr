"""Word boxes against the exact ink of each ground-truth word.

The SVG ground truth is rendered twice (eval/svg_ground_truth.py): normally and with the text
made transparent. |image - notext| is the visible ink of the text and nothing else, so every
ground-truth word gets its own ink pixels: pixels at least half as dark as the word's darkest
ones, inside its box widened by a tenth of its height (the typographic box already holds accents and
descenders; ink of text dropped from the ground truth must not be counted), each pixel owned by
the nearest word
box. Words without such ink (manual labels of outlined text, text hidden under shapes) are not
measured.

For each measured word and the predicted boxes that overlap it (the evaluation's match rule):
- full: every ink pixel lies inside some of the boxes; need_px: how far the boxes would have
  to grow for that (the margin a redaction would need);
- per side: box edge minus ink edge (+ cuts ink, - slack), in px and in ink heights;
- excess: ink of *other* words inside the boxes, as a share of the word's own ink (a box that
  only grows looks better on coverage and worse here).

Usage:
    python -m eval.box_ink --from-run <run> --images eval/dev_sets/svg_dev_v1.txt [--refine]
--refine snaps the replayed words to ink first (ocr/box_refine.py, lines rebuilt from boxes).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from eval.metrics import GtWord, overlap, scored_text
from ocr.config import EvalConfig, load_config

Box = tuple[float, float, float, float]


@dataclass
class WordInk:
    gt: GtWord
    xs: np.ndarray  # ink pixel columns
    ys: np.ndarray  # ink pixel rows

    @property
    def box(self) -> Box:
        return (float(self.xs.min()), float(self.ys.min()), float(self.xs.max() + 1), float(self.ys.max() + 1))


def text_ink(png: Path) -> np.ndarray | None:
    """|image - image without text| (max over RGB), or None if the second render is missing."""
    notext = png.with_name(png.stem + ".notext.png")
    if not notext.exists():
        return None
    a = np.asarray(Image.open(png).convert("RGB")).astype(np.int16)
    b = np.asarray(Image.open(notext).convert("RGB")).astype(np.int16)
    return np.abs(a - b).max(axis=2).astype(np.uint8)


def word_inks(diff: np.ndarray, words: list[GtWord], min_peak: int = 40, pad: float = 0.1) -> list[WordInk | None]:
    """Ink pixels of each ground-truth word (None: no measurable ink)."""
    H, W = diff.shape
    owner = np.full((H, W), -1, np.int32)
    best = np.full((H, W), np.inf, np.float32)
    windows = []
    for k, g in enumerate(words):
        x0, y0, x1, y1 = g.bbox
        pad_px = max(2.0, pad * (min(y1 - y0, x1 - x0) if g.rotated else (y1 - y0)))
        X0, Y0 = max(0, int(x0 - pad_px)), max(0, int(y0 - pad_px))
        X1, Y1 = min(W, int(np.ceil(x1 + pad_px))), min(H, int(np.ceil(y1 + pad_px)))
        windows.append((X0, Y0, X1, Y1))
        if X1 <= X0 or Y1 <= Y0:
            continue
        yy, xx = np.mgrid[Y0:Y1, X0:X1]
        cx, cy = xx + 0.5, yy + 0.5
        dist = np.maximum(np.maximum(x0 - cx, cx - x1), np.maximum(y0 - cy, cy - y1)).clip(min=0)
        sub = best[Y0:Y1, X0:X1]
        upd = dist < sub
        sub[upd] = dist[upd]
        owner[Y0:Y1, X0:X1][upd] = k
    out: list[WordInk | None] = []
    for k, (g, (X0, Y0, X1, Y1)) in enumerate(zip(words, windows)):
        if X1 <= X0 or Y1 <= Y0:
            out.append(None)
            continue
        sub = diff[Y0:Y1, X0:X1]
        mine = owner[Y0:Y1, X0:X1] == k
        if not mine.any():
            out.append(None)
            continue
        peak = float(np.percentile(sub[mine], 99))
        ink = mine & (sub >= 0.5 * peak)
        if peak < min_peak or ink.sum() < 8:
            out.append(None)
            continue
        ys, xs = np.nonzero(ink)
        out.append(WordInk(g, xs + X0, ys + Y0))
    return out


def measure(inks: list[WordInk | None], preds: list[tuple[str, Box]], cfg: EvalConfig) -> list[dict]:
    """One record per measured, scored ground-truth word; preds are (text, box)."""
    recs = []
    for wi in inks:
        if wi is None or wi.gt.ignore or len(scored_text(wi.gt.text, cfg)) < cfg.min_scored_chars:
            continue
        readers = [(t, b) for t, b in preds if overlap(wi.gt.bbox, b) >= cfg.match_min_overlap]
        rec = {"text": wi.gt.text, "rotated": wi.gt.rotated, "detected": bool(readers)}
        if readers:
            cx, cy = wi.xs + 0.5, wi.ys + 0.5
            need = np.full(len(cx), np.inf)
            for _, b in readers:
                d = np.maximum(np.maximum(b[0] - cx, cx - b[2]), np.maximum(b[1] - cy, cy - b[3])).clip(min=0)
                need = np.minimum(need, d)
            ix0, iy0, ix1, iy1 = wi.box
            ux0, uy0 = min(b[0] for _, b in readers), min(b[1] for _, b in readers)
            ux1, uy1 = max(b[2] for _, b in readers), max(b[3] for _, b in readers)
            excess = 0
            for o in inks:
                if o is None or o is wi:
                    continue
                if all(overlap(o.box, b) == 0 for _, b in readers):
                    continue
                ox, oy = o.xs + 0.5, o.ys + 0.5
                inside = np.zeros(len(ox), bool)
                for _, b in readers:
                    inside |= (ox >= b[0]) & (ox <= b[2]) & (oy >= b[1]) & (oy <= b[3])
                excess += int(inside.sum())
            rec.update({"recall": float((need == 0).mean()), "need_px": float(need.max()),
                        "ink_h": float(iy1 - iy0), "sides_px": [ux0 - ix0, ix1 - ux1, uy0 - iy0, iy1 - uy1],
                        "excess": excess / len(cx), "n_readers": len(readers),
                        "read_right": len(readers) == 1 and scored_text(readers[0][0], cfg) == scored_text(wi.gt.text, cfg)})
        recs.append(rec)
    return recs


def summarize(recs: list[dict]) -> dict:
    """Aggregate numbers, horizontal and rotated text apart."""
    out = {"words": len(recs), "detected": sum(r["detected"] for r in recs)}
    for name, sel in (("horizontal", [r for r in recs if r["detected"] and not r["rotated"]]),
                      ("rotated", [r for r in recs if r["detected"] and r["rotated"]])):
        if not sel:
            continue
        need = np.array([r["need_px"] for r in sel])
        sides = np.array([r["sides_px"] for r in sel])
        sides_h = sides / np.array([max(1.0, r["ink_h"]) for r in sel])[:, None]
        exc = np.array([r["excess"] for r in sel])
        right = [r for r in sel if r["read_right"]]
        out[name] = {
            "n": len(sel),
            "full": float(np.mean(need == 0)),
            "full_read_right": float(np.mean([r["need_px"] == 0 for r in right])) if right else None,
            "within_1px": float(np.mean(need <= 1)), "within_2px": float(np.mean(need <= 2)),
            "mean_recall": float(np.mean([r["recall"] for r in sel])),
            "need_px_p99": float(np.percentile(need, 99)), "need_px_p999": float(np.percentile(need, 99.9)),
            "sides": {s: {"cuts": float(np.mean(sides[:, k] > 0.5)),
                          "p5_p50_p95_ink_heights": [float(np.percentile(sides_h[:, k], q)) for q in (5, 50, 95)]}
                      for k, s in enumerate(("left", "right", "top", "bottom"))},
            "excess_5pct": float(np.mean(exc >= 0.05)),
            "excess_5pct_read_right": float(np.mean([r["excess"] >= 0.05 for r in right])) if right else None,
        }
    return out


def main(argv: list[str]) -> int:
    from eval.run_eval import DEFAULT_GT, REPORTS, RUNS, ReplayEngine, images_from, load_gt, load_index

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from-run", required=True, help="artifacts/eval_runs/<name>")
    ap.add_argument("--images", type=Path, required=True, help="list of ground-truth images (dev set)")
    ap.add_argument("--gt", type=Path, default=DEFAULT_GT)
    ap.add_argument("--refine", action="store_true", help="snap the replayed word boxes to ink first")
    ap.add_argument("--out", type=Path, help="write the summary as JSON")
    args = ap.parse_args(argv)
    cfg = load_config()
    engine = ReplayEngine(RUNS / args.from_run, REPORTS, False)
    rows = images_from(args.images, load_index(args.gt))
    recs, missing = [], 0
    for row in rows:
        if row["json"] not in engine.preds:
            continue
        img, gt, _ = load_gt(args.gt, row)
        diff = text_ink((args.gt / row["json"]).with_name(json.loads((args.gt / row["json"]).read_text())["image"]))
        if diff is None:
            missing += 1
            continue
        words, _ = engine.read_row(row)
        preds = [(w.text, tuple(w.bbox_px)) for w in words]
        if args.refine:
            from ocr.box_refine import refine_words
            refined = refine_words(np.asarray(img), preds, cfg.box_refine)
            preds = [(t, r.bbox) for (t, _), r in zip(preds, refined)]
        recs += measure(word_inks(diff, gt), preds, cfg.eval)
    if missing:
        print(f"{missing} images have no .notext.png: rebuild the ground truth (python -m eval.svg_ground_truth)",
              file=sys.stderr)
    summary = summarize(recs)
    print(json.dumps(summary, indent=1))
    if args.out:
        args.out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
