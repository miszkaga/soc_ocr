"""Where do errors concentrate? Break down a run's outcomes by properties of the words.

Every scored ground-truth word gets one outcome:
- missed           no prediction covers it: it would get no flag at all (worst);
- accepted_wrong   matched, accepted by consensus (or above a confidence threshold), but wrong:
                   a silent error;
- review           matched but sent to a human (uncertain / below threshold): costs work;
- accepted_ok      matched, accepted and right.

Outcomes are counted per property value: source, string category, font size, rotation,
background brightness, word length. "Silent" = missed + accepted_wrong.

Usage:  python -m eval.error_analysis --from-run RUN [--threshold 0.99] [--reapply-flags]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from eval.metrics import GtWord, PredWord, categories, evaluate_image
from eval.run_eval import DEFAULT_GT, REPORTS, RUNS, ReplayEngine, load_gt
from ocr.config import load_config

OUTCOMES = ("missed", "accepted_wrong", "review", "accepted_ok")


def font_bucket(px: float | None) -> str:
    if px is None:
        return "manual label"
    for limit, name in ((12, "<12 px"), (16, "12-16 px"), (24, "16-24 px"), (40, "24-40 px")):
        if px < limit:
            return name
    return ">=40 px"


def background(img: np.ndarray, box) -> str:
    """Brightness of the word's surroundings, from the border pixels of its box."""
    h, w = img.shape[:2]
    x0, y0 = max(0, int(box[0]) - 2), max(0, int(box[1]) - 2)
    x1, y1 = min(w, int(box[2]) + 2), min(h, int(box[3]) + 2)
    if x1 <= x0 or y1 <= y0:
        return "unknown"
    crop = img[y0:y1, x0:x1]
    border = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]])
    lum = float(np.median(border @ np.array([0.299, 0.587, 0.114])))
    return "dark" if lum < 100 else "mid" if lum < 200 else "light"


def length_bucket(text: str) -> str:
    n = len(text)
    return "1 char" if n == 1 else "2-3 chars" if n <= 3 else "4-10 chars" if n <= 10 else ">10 chars"


def word_outcomes(engine: ReplayEngine, gt_dir: Path, threshold: float | None, cfg):
    """Yield (properties, outcome, gt text, predicted text) for every scored word, as the
    metrics score it (eval.metrics.evaluate_image)."""
    for image_json in engine.preds:
        if not (gt_dir / image_json).exists():
            continue  # no longer in the ground truth (e.g. all its text turned out hidden)
        img, gt, ignore = load_gt(gt_dir, {"json": image_json})
        pixels = np.asarray(img, dtype=float)
        raw = json.loads((gt_dir / image_json).read_text(encoding="utf-8"))
        fonts = {tuple(w["bbox_px"]): w["font_px"] for w in raw["words"]}
        words, _ = engine.read_row({"json": image_json})
        pred = [PredWord(w.text, w.bbox_px, w.conf, w.status) for w in words]
        res = evaluate_image(gt, pred, cfg, ignore)
        source = image_json.split("/")[0]
        for w in res.words:
            g = gt[w["gt"]]
            if w["outcome"] == "missed" or w["accepted"] is not None:
                outcome = w["outcome"]  # consensus decided
            elif threshold is not None and w["conf"] >= threshold:
                outcome = "accepted_ok" if w["exact"] else "accepted_wrong"
            else:
                outcome = "review"  # single engine below (or without) a threshold
            props = {
                "source": source,
                "category": sorted(categories(GtWord(w["text"], g.bbox)) - {"all", "hard", "rotated"}) or ["plain"],
                "font": font_bucket(fonts.get(tuple(g.bbox))),
                "rotated": "rotated" if g.rotated else "horizontal",
                "background": background(pixels, g.bbox),
                "length": length_bucket(w["text"]),
            }
            text = " ".join(words[i].text for i in w["readers"]) or None
            yield props, outcome, w["text"], text


def analyse(engine: ReplayEngine, gt_dir: Path, threshold: float | None = None) -> dict:
    cfg = load_config().eval
    table: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    totals = Counter()
    examples: dict[str, list] = defaultdict(list)
    for props, outcome, gt_text, pred_text in word_outcomes(engine, gt_dir, threshold, cfg):
        totals[outcome] += 1
        for prop, value in props.items():
            for v in value if isinstance(value, list) else [value]:
                table[prop][v][outcome] += 1
        if outcome in ("missed", "accepted_wrong") and len(examples[outcome]) < 60:
            examples[outcome].append([gt_text, pred_text])
    silent_total = totals["missed"] + totals["accepted_wrong"]
    rows = {}
    for prop, values in table.items():
        rows[prop] = []
        for value, c in values.items():
            n = sum(c.values())
            silent = c["missed"] + c["accepted_wrong"]
            rows[prop].append({"value": value, "words": n, **{o: c[o] for o in OUTCOMES},
                               "silent_rate": silent / n, "review_rate": c["review"] / n,
                               "share_of_silent": silent / silent_total if silent_total else 0.0})
        rows[prop].sort(key=lambda r: -r["share_of_silent"])
    return {"totals": dict(totals), "by": rows, "examples": examples}


def print_report(res: dict):
    t = res["totals"]
    n = sum(t.values())
    print(f"{n} words: " + ", ".join(f"{o} {t.get(o, 0)} ({t.get(o, 0) / n:.1%})" for o in OUTCOMES))
    for prop, rows in res["by"].items():
        print(f"\n{prop:12} {'words':>6} {'missed':>7} {'acc.wrong':>9} {'silent%':>8} {'review%':>8} {'share of silent':>15}")
        for r in rows:
            print(f"  {str(r['value']):22} {r['words']:6} {r['missed']:7} {r['accepted_wrong']:9} "
                  f"{r['silent_rate']:8.1%} {r['review_rate']:8.1%} {r['share_of_silent']:15.1%}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--from-run", required=True)
    ap.add_argument("--gt", type=Path, default=DEFAULT_GT)
    ap.add_argument("--threshold", type=float, help="confidence that counts as accepted (single engines)")
    ap.add_argument("--reapply-flags", action="store_true")
    ap.add_argument("--json", type=Path, help="also write the breakdown here")
    args = ap.parse_args(argv)
    res = analyse(ReplayEngine(RUNS / args.from_run, REPORTS, args.reapply_flags), args.gt, args.threshold)
    print_report(res)
    if args.json:
        args.json.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
