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
from ocr.engines.base import OcrEngine

REPO = Path(__file__).resolve().parent.parent
DEFAULT_GT = REPO / "artifacts" / "datasets" / "gt" / "svg"
REPORTS = REPO / "eval" / "reports"
RUNS = REPO / "artifacts" / "eval_runs"


def make_engine(name: str, overrides: dict[str, str]) -> OcrEngine:
    cfg = load_config()
    if name == "tesseract":
        from ocr.engines.tesseract import TesseractEngine
        return TesseractEngine(_override(cfg.tesseract, overrides))
    if name == "paddle_onnx":
        from ocr.engines.paddle_onnx import PaddleOnnxEngine
        return PaddleOnnxEngine(_override(cfg.paddle, overrides))
    if name == "paddle_openvino":
        from ocr.engines.paddle_openvino import PaddleOpenVinoEngine
        return PaddleOpenVinoEngine(_override(cfg.paddle, overrides))
    raise ValueError(f"unknown engine {name!r}")


def _override(cfg, overrides: dict[str, str]):
    fields = {f.name: f.type for f in dataclasses.fields(cfg)}
    typed = {}
    for k, v in overrides.items():
        if k not in fields:
            raise ValueError(f"unknown option {k!r} for {type(cfg).__name__}")
        cur = getattr(cfg, k)
        if isinstance(cur, bool):
            if v.lower() not in ("true", "false"):
                raise ValueError(f"{k} must be true or false, got {v!r}")
            typed[k] = v.lower() == "true"
        else:
            typed[k] = type(cur)(v)
    return dataclasses.replace(cfg, **typed)


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


def load_gt(gt_dir: Path, row: dict) -> tuple[Image.Image, list[GtWord]]:
    data = json.loads((gt_dir / row["json"]).read_text(encoding="utf-8"))
    img = Image.open((gt_dir / row["json"]).with_name(data["image"])).convert("RGB")
    words = [GtWord(text=w["text"], bbox=tuple(w["bbox_px"]), ignore=not w["legible"] or w["clipped"],
                    rotated=w["rotated"]) for w in data["words"]]
    return img, words


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
        img, gt = load_gt(gt_dir, row)
        t0 = time.perf_counter()
        words = engine.read(img)
        timings.append(time.perf_counter() - t0)
        pred = [PredWord(text=w.text, bbox=w.bbox_px, conf=w.conf) for w in words]
        res = evaluate_image(gt, pred, cfg)
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
    args = ap.parse_args(argv)

    cfg = load_config().eval
    overrides = dict(o.split("=", 1) for o in args.override)
    engine = make_engine(args.engine, overrides)
    index = load_index(args.gt)
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
        "engine_config": dataclasses.asdict(engine.cfg) if hasattr(engine, "cfg") else {},
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
