import json
import shutil

import pytest

from eval.run_eval import images_from, main, sample_index
from ocr.engines.registry import override as _override
from ocr.config import load_config
from tests.test_tesseract_engine import text_image


def test_sample_is_deterministic_and_proportional():
    rows = [{"json": f"a/{i}.json", "source": "a"} for i in range(80)] + \
           [{"json": f"b/{i}.json", "source": "b"} for i in range(20)]
    s1, s2 = sample_index(rows, 10, seed=1), sample_index(rows, 10, seed=1)
    assert s1 == s2
    assert sum(r["source"] == "a" for r in s1) == 8 and sum(r["source"] == "b" for r in s1) == 2
    assert sample_index(rows, 10, seed=2) != s1
    assert sample_index(rows, None, seed=0) == rows


def test_override_keeps_types_and_rejects_unknown_keys():
    cfg = _override(load_config().tesseract, {"psm": "3", "lang": "eng"})
    assert cfg.psm == 3 and cfg.lang == "eng"
    with pytest.raises(ValueError):
        _override(cfg, {"nope": "1"})
    paddle = load_config().paddle
    assert _override(paddle, {"use_cls": "false"}).use_cls is False
    assert _override(paddle, {"use_cls": "True"}).use_cls is True
    with pytest.raises(ValueError):
        _override(paddle, {"use_cls": "0"})


def test_section_overrides_reach_the_engines_inside_the_consensus():
    from ocr.engines.registry import create

    e = create("consensus", {"paddle.refine_boxes": "false", "reread_scale": "3"})
    assert e.primary.cfg.refine_boxes is False and e.cfg.reread_scale == 3.0
    assert create("consensus", {"paddle.refine_boxes": "true"}).primary.cfg.refine_boxes is True
    with pytest.raises(ValueError):
        create("consensus", {"nope.x": "1"})


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
def test_end_to_end_report(tmp_path):
    gt_dir = tmp_path / "gt"
    (gt_dir / "src").mkdir(parents=True)
    words = ["Private", "endpoint", "10.1.2.4", "Łódź"]
    img, boxes = text_image(words)
    img.save(gt_dir / "src" / "one.png")
    (gt_dir / "src" / "one.json").write_text(json.dumps({"image": "one.png", "words": [
        {"text": w, "bbox_px": list(b), "legible": True, "clipped": False, "rotated": False}
        for w, b in zip(words, boxes)]}), encoding="utf-8")
    (gt_dir / "index.jsonl").write_text(json.dumps({"json": "src/one.json", "source": "src"}) + "\n")

    assert main(["--gt", str(gt_dir), "--reports", str(tmp_path / "reports"),
                 "--runs", str(tmp_path / "runs"), "--tag", "test"]) == 0
    [report_path] = (tmp_path / "reports").glob("*_tesseract_test.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["engine_versions"]["tesseract"].startswith("5.")
    assert report["machine"]["hostname"] and report["commit"]
    a = report["overall"]["categories"]["all"]
    assert a["words"] == 4 and a["detection_recall"] == 1.0 and a["cer"] == 0.0
    assert report["by_source"]["src"]["categories"]["polish"]["words"] == 1
    assert (tmp_path / "runs" / report_path.stem / "predictions.jsonl").exists()


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
def test_rescore_saved_run_with_ignore_regions(tmp_path):
    gt_dir = tmp_path / "gt"
    (gt_dir / "src").mkdir(parents=True)
    words = ["Azure", "Firewall"]
    img, boxes = text_image(words)
    img.save(gt_dir / "src" / "one.png")
    gt = {"image": "one.png", "words": [
        {"text": "Azure", "bbox_px": list(boxes[0]), "legible": True, "clipped": False, "rotated": False}]}
    (gt_dir / "src" / "one.json").write_text(json.dumps(gt), encoding="utf-8")
    (gt_dir / "index.jsonl").write_text(json.dumps({"json": "src/one.json", "source": "src"}) + "\n")
    common = ["--gt", str(gt_dir), "--reports", str(tmp_path / "reports"), "--runs", str(tmp_path / "runs")]

    assert main(common + ["--tag", "first"]) == 0
    [first] = (tmp_path / "reports").glob("*_first.json")
    assert json.loads(first.read_text())["overall"]["false_positives"] == 1  # "Firewall" has no ground truth

    # Ground truth gains an ignore region over "Firewall" (as if it were an embedded image).
    gt["ignore_regions_px"] = [list(boxes[1])]
    (gt_dir / "src" / "one.json").write_text(json.dumps(gt), encoding="utf-8")
    assert main(common + ["--from-run", first.stem, "--tag", "rescored"]) == 0
    [again] = (tmp_path / "reports").glob("*_rescored.json")
    report = json.loads(again.read_text())
    assert report["overall"]["false_positives"] == 0
    assert report["engine"] == "tesseract" and report["engine_versions"]["replayed_from"] == first.stem


def test_images_from_file_keeps_order_and_skips_missing(tmp_path, capsys):
    index = [{"json": f"a/{i}.json", "source": "a"} for i in range(5)]
    listing = tmp_path / "dev.txt"
    listing.write_text("# fixed set\na/3.json\na/1.json\n\na/9.json\n")
    assert [r["json"] for r in images_from(listing, index)] == ["a/3.json", "a/1.json"]
    assert "a/9.json" in capsys.readouterr().err
