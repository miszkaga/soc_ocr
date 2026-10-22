import json

from PIL import Image

from eval.adjudicate import apply_decisions, collect, prepare
from eval.run_eval import ReplayEngine, corrections_path, load_gt


def setup(tmp_path):
    gt_dir = tmp_path / "gt" / "svg"
    (gt_dir / "src").mkdir(parents=True)
    Image.new("RGB", (400, 60), "white").save(gt_dir / "src" / "a.png")
    gt_words = [("essage", [10, 10, 60, 30]), ("Azure", [70, 10, 120, 30]), ("ghost", [130, 10, 180, 30]),
                ("DNS", [190, 10, 220, 30])]
    (gt_dir / "src" / "a.json").write_text(json.dumps({"image": "a.png", "words": [
        {"text": t, "bbox_px": b, "font_px": 20, "legible": True, "clipped": False, "rotated": False}
        for t, b in gt_words]}))
    run = tmp_path / "runs" / "r"
    run.mkdir(parents=True)
    preds = [("message", [8, 12, 60, 28], "accepted"),   # mismatch: the label is wrong
             ("Azure", [70, 12, 120, 28], "accepted"),   # agreement: not a case
             ("DN5", [190, 12, 220, 28], "accepted"),    # mismatch: the engine is wrong
             ("SQL", [300, 12, 340, 28], "accepted"),    # extra: text the label lacks
             ("zz", [350, 12, 380, 28], "uncertain")]    # uncertain: not a case
    (run / "predictions.jsonl").write_text(json.dumps({"json": "src/a.json", "seconds": 1, "words": [
        {"text": t, "bbox_px": b, "conf": 0.9, "char_conf": [], "status": s, "votes": {"p": t}} for t, b, s in preds]}) + "\n")
    return gt_dir, ReplayEngine(run, tmp_path / "reports", False)


def test_cases_are_only_disagreements(tmp_path):
    gt_dir, engine = setup(tmp_path)
    cases = collect(engine, gt_dir, ["src/a.json"])
    assert [(c["kind"], c["gt"], c["pred"]) for c in cases] == [
        ("mismatch", "essage", "message"), ("missed", "ghost", None), ("mismatch", "DNS", "DN5"), ("extra", None, "SQL")]


def test_decisions_fix_the_ground_truth(tmp_path):
    gt_dir, engine = setup(tmp_path)
    work = tmp_path / "work"
    cases = {(c["kind"], c["gt"] or c["pred"]): c["id"] for c in prepare(engine, gt_dir, ["src/a.json"], work)}
    assert (work / "crops" / "c0000.png").exists()
    out = apply_decisions(work, {
        cases[("mismatch", "essage")]: {"decision": "gt", "text": "message"},
        cases[("missed", "ghost")]: {"decision": "gt", "text": ""},          # there is no text there
        cases[("mismatch", "DNS")]: {"decision": "engine"},
        cases[("extra", "SQL")]: {"decision": "gt", "text": "SQL"}})
    assert out == corrections_path(gt_dir)
    _, words, _ = load_gt(gt_dir, {"json": "src/a.json"})
    assert [w.text for w in words] == ["message", "Azure", "DNS", "SQL"]
    # Everything is decided: nothing is asked again (the engine's own error stays an error).
    assert collect(engine, gt_dir, ["src/a.json"]) == []


def test_unreadable_excludes_the_word(tmp_path):
    gt_dir, engine = setup(tmp_path)
    work = tmp_path / "work"
    [first] = [c for c in prepare(engine, gt_dir, ["src/a.json"], work) if c["gt"] == "essage"]
    apply_decisions(work, {first["id"]: {"decision": "unreadable"}})
    _, words, regions = load_gt(gt_dir, {"json": "src/a.json"})
    assert [w.ignore for w in words if w.text == "essage"] == [True]
    assert tuple(first["box"]) in regions
