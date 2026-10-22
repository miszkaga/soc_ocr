import json

import numpy as np
from PIL import Image

from eval.error_analysis import analyse, background, font_bucket, length_bucket
from eval.run_eval import ReplayEngine


def test_buckets():
    assert font_bucket(None) == "manual label" and font_bucket(10) == "<12 px" and font_bucket(50) == ">=40 px"
    assert length_bucket("1") == "1 char" and length_bucket("Azure") == "4-10 chars"
    img = np.zeros((40, 100, 3)); img[:, 50:] = 255
    assert background(img, (5, 5, 30, 30)) == "dark" and background(img, (60, 5, 90, 30)) == "light"


def test_outcomes_are_counted_per_property(tmp_path):
    gt_dir = tmp_path / "gt"
    (gt_dir / "src").mkdir(parents=True)
    Image.new("RGB", (400, 60), "white").save(gt_dir / "src" / "a.png")
    words = [("Azure", [10, 10, 60, 30], 20), ("10.0.0.1", [70, 10, 150, 30], 20),
             ("DNS", [160, 10, 200, 30], 10), ("42", [210, 10, 230, 30], 20)]
    (gt_dir / "src" / "a.json").write_text(json.dumps({"image": "a.png", "words": [
        {"text": t, "bbox_px": b, "font_px": f, "legible": True, "clipped": False, "rotated": False}
        for t, b, f in words]}))
    run = tmp_path / "runs" / "r"
    run.mkdir(parents=True)
    preds = [("Azure", [10, 12, 60, 28], "accepted"), ("l0.0.0.1", [70, 12, 150, 28], "accepted"),
             ("DN5", [160, 12, 200, 28], "uncertain")]  # "42" is missed
    (run / "predictions.jsonl").write_text(json.dumps({"json": "src/a.json", "seconds": 1, "words": [
        {"text": t, "bbox_px": b, "conf": 0.9, "char_conf": [], "status": s} for t, b, s in preds]}) + "\n")

    res = analyse(ReplayEngine(run, tmp_path / "reports", False), gt_dir)
    assert res["totals"] == {"accepted_ok": 1, "accepted_wrong": 1, "review": 1, "missed": 1}
    cat = {r["value"]: r for r in res["by"]["category"]}
    assert cat["ip"]["accepted_wrong"] == 1 and cat["number"]["missed"] == 1
    assert {r["value"]: r["review"] for r in res["by"]["font"]}["<12 px"] == 1
    assert sum(r["share_of_silent"] for r in res["by"]["source"]) == 1.0
    assert res["examples"]["accepted_wrong"] == [["10.0.0.1", "l0.0.0.1"]]
