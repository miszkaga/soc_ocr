"""Run an OCR engine on a ground-truth set and write a report.

Report: eval/reports/<date>_<machine>_<commit>_<engine>[_<tag>].json, with the commit,
machine, power state, engine versions and configuration, the sampled images, timing and
metrics overall and per source. Raw predictions go to artifacts/eval_runs/<report name>/.

Usage:
    python -m eval.run_eval --engine tesseract --sample 200
    python -m eval.run_eval --engine tesseract --sample 200 --override psm=3 --tag psm3
"""
from __future__ import annotations

import argparse
import dataclasses
import functools
import datetime as dt
import hashlib
import json
import platform
import random
import socket
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from eval.metrics import GtWord, PredWord, aggregate, evaluate_image
from ocr.config import load_config
from ocr.engines.base import OcrEngine, OcrWord
from ocr.engines.registry import create

REPO = Path(__file__).resolve().parent.parent
DEFAULT_GT = REPO / "artifacts" / "datasets" / "gt" / "svg"
REPORTS = REPO / "eval" / "reports"
RUNS = REPO / "artifacts" / "eval_runs"


def load_index(gt_dir: Path) -> list[dict]:
    with (gt_dir / "index.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def sample_index(rows: list[dict], n: int | None, seed: int) -> list[dict]:
    """Deterministic sample, proportional to each source's share (at least one per source)."""
    if n is None or n >= len(rows):
        return rows
    by_source = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)
    rng = random.Random(seed)
    picked = []
    for source in sorted(by_source):
        group = sorted(by_source[source], key=lambda r: r["json"])
        k = max(1, round(n * len(group) / len(rows)))
        picked += rng.sample(group, min(k, len(group)))
    return sorted(picked, key=lambda r: r["json"])


def manual_path(gt_dir: Path) -> Path:
    """Human decisions on text the SVG lacks (see eval/label_regions.py); kept outside the
    generated ground truth so that regenerating it does not lose them."""
    return gt_dir.parent / "manual" / f"{gt_dir.name}_regions.jsonl"


@functools.lru_cache(maxsize=8)
def _manual(path: Path, mtime: float) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            out[rec["json"]].append(rec)
    return out


def manual_records(gt_dir: Path, image_json: str) -> list[dict]:
    path = manual_path(gt_dir)
    return _manual(path, path.stat().st_mtime).get(image_json, []) if path.exists() else []


def _split_box(box, words: list[str]) -> list[tuple]:
    """Approximate word boxes: the region box split along x by character counts."""
    x0, y0, x1, y1 = box
    total = sum(len(w) + 1 for w in words) - 1
    out, x = [], x0
    for w in words:
        width = (x1 - x0) * len(w) / total
        out.append((x, y0, x + width, y1))
        x += width + (x1 - x0) / total
    return out


def load_gt(gt_dir: Path, row: dict) -> tuple[Image.Image, list[GtWord], list[tuple]]:
    """Image, ground-truth words and regions without ground truth (embedded raster images,
    regions a human marked unreadable). Manually labelled words are merged in."""
    data = json.loads((gt_dir / row["json"]).read_text(encoding="utf-8"))
    img = Image.open((gt_dir / row["json"]).with_name(data["image"])).convert("RGB")
    words = [GtWord(text=w["text"], bbox=tuple(w["bbox_px"]), ignore=not w["legible"] or w["clipped"],
                    rotated=w["rotated"]) for w in data["words"]]
    regions = [tuple(b) for b in data.get("ignore_regions_px", [])]
    for rec in manual_records(gt_dir, row["json"]):
        if rec["status"] == "text" and rec["text"]:
            parts = rec["text"].split()
            words += [GtWord(text=t, bbox=b) for t, b in zip(parts, _split_box(rec["box"], parts))]
        elif rec["status"] == "unreadable":
            regions.append(tuple(rec["box"]))
    return img, words, regions


class ReplayEngine:
    """Predictions saved by an earlier run, to rescore them against updated ground truth or
    with changed consensus flag rules, without running the engines again."""

    def __init__(self, run_dir: Path, reports: Path, reapply_flags: bool):
        self.run_dir = run_dir
        self.preds = {}
        with (run_dir / "predictions.jsonl").open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                self.preds[rec["json"]] = rec
        original = reports / f"{run_dir.name}.json"
        self.original = json.loads(original.read_text(encoding="utf-8")) if original.exists() else {}
        self.name = self.original.get("engine", "replay")
        self.cfg = None
        self.reapply_flags = reapply_flags

    def versions(self) -> dict:
        return {**self.original.get("engine_versions", {}), "replayed_from": self.run_dir.name}

    def read_row(self, row: dict) -> tuple[list[OcrWord], float]:
        rec = self.preds[row["json"]]
        words = [OcrWord(**{**w, "bbox_px": tuple(w["bbox_px"])}) for w in rec["words"]]
        if self.reapply_flags:
            from ocr.consensus import apply_flags
            cfg = load_config().consensus
            words = [apply_flags(_unflag(w), cfg) for w in words]
        return words, rec["seconds"]


def _unflag(w: OcrWord) -> OcrWord:
    """Undo an earlier flag rule, so the current rules decide from the vote alone."""
    if w.reason in ("ambiguous_glyphs", "edge_dash"):
        return OcrWord(w.text, w.bbox_px, w.conf, w.char_conf, "accepted", "agree", w.votes)
    return w


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def machine_info() -> dict:
    cpu = ""
    try:
        cpu = next(l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines()
                   if l.startswith("model name"))
    except (OSError, StopIteration):
        cpu = platform.processor()
    ac = None  # unknown (no Linux power supply info, e.g. on the desktop)
    for p in Path("/sys/class/power_supply").glob("*/type"):
        if p.read_text().strip() == "Mains" and (p.parent / "online").exists():
            ac = (p.parent / "online").read_text().strip() == "1"
    return {"hostname": socket.gethostname(), "platform": platform.platform(), "cpu": cpu,
            "python": platform.python_version(), "on_ac_power": ac}


def run(engine: OcrEngine, gt_dir: Path, rows: list[dict], cfg) -> tuple[dict, list[dict]]:
    results_all, by_source, timings, preds_log = [], defaultdict(list), [], []
    for i, row in enumerate(rows, 1):
        img, gt, regions = load_gt(gt_dir, row)
        if isinstance(engine, ReplayEngine):
            words, seconds = engine.read_row(row)
            timings.append(seconds)
        else:
            t0 = time.perf_counter()
            words = engine.read(img)
            timings.append(time.perf_counter() - t0)
        pred = [PredWord(text=w.text, bbox=w.bbox_px, conf=w.conf, status=w.status) for w in words]
        res = evaluate_image(gt, pred, cfg, regions)
        for e in res.errors:
            e["image"] = row["json"]
        results_all.append(res)
        by_source[row["source"]].append(res)
        preds_log.append({"json": row["json"], "seconds": round(timings[-1], 3),
                          "words": [dataclasses.asdict(w) for w in words]})
        if i % 20 == 0:
            print(f"{i}/{len(rows)} images", file=sys.stderr, flush=True)
    summary = {
        "overall": aggregate(results_all, cfg),
        "by_source": {s: {k: v for k, v in aggregate(r, cfg).items() if k != "errors"}
                      for s, r in sorted(by_source.items())},
        "timing_s": {"total": round(sum(timings), 2), "mean": round(statistics.mean(timings), 3),
                     "median": round(statistics.median(timings), 3),
                     "p95": round(sorted(timings)[int(0.95 * (len(timings) - 1))], 3)},
    }
    return summary, preds_log


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Evaluate an OCR engine on a ground-truth set.")
    ap.add_argument("--engine", default="tesseract")
    ap.add_argument("--gt", type=Path, default=DEFAULT_GT)
    ap.add_argument("--sample", type=int, help="number of images (proportional per source)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                    help="override an engine config value, e.g. psm=3")
    ap.add_argument("--tag", default="")
    ap.add_argument("--reports", type=Path, default=REPORTS)
    ap.add_argument("--runs", type=Path, default=RUNS)
    ap.add_argument("--from-run", metavar="NAME",
                    help="rescore saved predictions of an earlier run (artifacts/eval_runs/NAME) "
                         "instead of running an engine; uses that run's images")
    ap.add_argument("--reapply-flags", action="store_true",
                    help="with --from-run: apply the current consensus flag rules")
    args = ap.parse_args(argv)

    cfg = load_config().eval
    index = load_index(args.gt)
    if args.from_run:
        engine = ReplayEngine(args.runs / args.from_run, args.reports, args.reapply_flags)
        by_json = {r["json"]: r for r in index}
        rows = [by_json[j] for j in engine.preds if j in by_json]
    else:
        overrides = dict(o.split("=", 1) for o in args.override)
        engine = create(args.engine, overrides)
        rows = sample_index(index, args.sample, args.seed)
    # Record the code state before running: edits made while it runs must not mark it dirty.
    commit = _git("rev-parse", "--short", "HEAD") or "nogit"
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))

    summary, preds_log = run(engine, args.gt, rows, cfg)

    host = socket.gethostname().lower()
    name = "_".join(p for p in (dt.date.today().isoformat(), host, commit + ("-dirty" if dirty else ""),
                                engine.name, args.tag) if p)
    report = {
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "commit": commit, "dirty": dirty, "machine": machine_info(),
        "engine": engine.name, "engine_versions": engine.versions(),
        "engine_config": (engine.original.get("engine_config", {}) if isinstance(engine, ReplayEngine)
                          else dataclasses.asdict(engine.cfg) if hasattr(engine, "cfg") else {}),
        "eval_config": dataclasses.asdict(cfg),
        "dataset": {"gt_dir": str(args.gt.relative_to(REPO) if args.gt.is_relative_to(REPO) else args.gt),
                    "index_sha256": hashlib.sha256((args.gt / "index.jsonl").read_bytes()).hexdigest()[:16],
                    "images": len(rows), "of": len(index), "seed": args.seed,
                    "sources": dict(sorted(Counter(r["source"] for r in rows).items()))},
        **summary,
        "images": [r["json"] for r in rows],
    }
    args.reports.mkdir(parents=True, exist_ok=True)
    out = args.reports / f"{name}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    run_dir = args.runs / name
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for p in preds_log:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    o = report["overall"]["categories"]["all"]
    print(f"{len(rows)} images  recall={o['detection_recall']:.3f}  CER={o['cer']:.3f}  WER={o['wer']:.3f}  "
          f"ECE={report['overall']['ece']:.3f}  {report['timing_s']['mean']:.2f}s/img  -> {out.relative_to(REPO) if out.is_relative_to(REPO) else out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
