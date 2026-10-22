import json

from PIL import Image

from eval.adjudicate import apply_decisions, collect, prepare
from eval.run_eval import ReplayEngine, corrections_path, load_gt


def setup(tmp_path):
    gt_dir = tmp_path / "gt" / "svg"
    (gt_dir / "src").mkdir(parents=True)
    Image.new("RGB", (700, 60), "white").save(gt_dir / "src" / "a.png")
    # Words far apart (more than a line height), so each is its own group.
    gt_words = [("essage", [10, 10, 60, 30]), ("Azure", [120, 10, 170, 30]), ("ghost", [230, 10, 280, 30]),
                ("DNS", [340, 10, 370, 30])]
    (gt_dir / "src" / "a.json").write_text(json.dumps({"image": "a.png", "words": [
        {"text": t, "bbox_px": b, "font_px": 20, "legible": True, "clipped": False, "rotated": False}
        for t, b in gt_words]}))
    run = tmp_path / "runs" / "r"
    run.mkdir(parents=True)
    preds = [("message", [8, 12, 60, 28], "accepted"),   # mismatch: the label is wrong
             ("Azure", [120, 12, 170, 28], "accepted"),  # agreement: not a case
             ("DN5", [340, 12, 370, 28], "accepted"),    # mismatch: the engine is wrong
             ("SQL", [450, 12, 490, 28], "accepted"),    # extra: text the label lacks
             ("zz", [560, 12, 590, 28], "uncertain")]    # uncertain: not a case
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


def test_dev_and_test_adjudication_keep_each_others_records(tmp_path):
    gt_dir, engine = setup(tmp_path)
    dev, test = tmp_path / "adjudicate", tmp_path / "adjudicate_test"
    [c] = [c for c in prepare(engine, gt_dir, ["src/a.json"], dev) if c["gt"] == "essage"]
    apply_decisions(dev, {c["id"]: {"decision": "gt", "text": "message"}})
    [d] = [c for c in prepare(engine, gt_dir, ["src/a.json"], test) if c["gt"] == "DNS"]
    apply_decisions(test, {d["id"]: {"decision": "engine"}})
    records = [json.loads(l) for l in corrections_path(gt_dir).read_text().splitlines()]
    assert sorted((r["gt"], r["set"]) for r in records) == [("DNS", "adjudicate_test"), ("essage", "adjudicate")]


def _setup_group(tmp_path, gt_words, preds):
    gt_dir = tmp_path / "gt" / "svg"
    (gt_dir / "src").mkdir(parents=True)
    Image.new("RGB", (400, 60), "white").save(gt_dir / "src" / "a.png")
    (gt_dir / "src" / "a.json").write_text(json.dumps({"image": "a.png", "words": [
        {"text": t, "bbox_px": b, "font_px": 20, "legible": True, "clipped": False, "rotated": False} for t, b in gt_words]}))
    run = tmp_path / "runs" / "r"
    run.mkdir(parents=True)
    (run / "predictions.jsonl").write_text(json.dumps({"json": "src/a.json", "seconds": 1, "words": [
        {"text": t, "bbox_px": b, "conf": 0.9, "char_conf": [], "status": "accepted", "votes": {"p": t}} for t, b in preds]}) + "\n")
    return gt_dir, ReplayEngine(run, tmp_path / "reports", False)


def test_joined_words_are_not_a_case(tmp_path):
    gt_dir, engine = _setup_group(tmp_path, [("(name,", [10, 10, 60, 30]), ("s", [66, 10, 74, 30])],
                                  [("(name, s", [12, 12, 75, 28])])
    assert collect(engine, gt_dir, ["src/a.json"]) == []


def test_label_split_into_pieces_is_not_a_case(tmp_path):
    # "m" + "essage" in the label, "message" read: the characters agree.
    gt_dir, engine = _setup_group(tmp_path, [("m", [10, 10, 20, 30]), ("essage", [20, 10, 70, 30])],
                                  [("message", [10, 12, 70, 28])])
    assert collect(engine, gt_dir, ["src/a.json"]) == []


def test_group_case_and_its_correction(tmp_path):
    # The label lacks a word the image shows; the engine reads it.
    gt_dir, engine = _setup_group(tmp_path, [("Customer", [10, 10, 90, 30])],
                                  [("Customer subscription", [10, 12, 200, 28])])
    [c] = collect(engine, gt_dir, ["src/a.json"])
    assert (c["kind"], c["gt"], c["pred"]) == ("mismatch", "Customer", "Customer subscription")
    work = tmp_path / "work"
    [c] = prepare(engine, gt_dir, ["src/a.json"], work)
    apply_decisions(work, {c["id"]: {"decision": "gt", "text": "Customer subscription"}})
    _, words, _ = load_gt(gt_dir, {"json": "src/a.json"})
    assert [w.text for w in words] == ["Customer", "subscription"]
    assert collect(engine, gt_dir, ["src/a.json"]) == []


def test_group_correction_keeps_word_boxes_when_the_count_matches(tmp_path):
    gt_dir, engine = _setup_group(tmp_path, [("Azure", [10, 10, 60, 30]), ("Fi rewall", [66, 10, 140, 30])],
                                  [("Azure Firewall", [10, 12, 140, 28])])
    work = tmp_path / "work"
    [c] = prepare(engine, gt_dir, ["src/a.json"], work)
    apply_decisions(work, {c["id"]: {"decision": "gt", "text": "Azure Firewall"}})
    _, words, _ = load_gt(gt_dir, {"json": "src/a.json"})
    assert [(w.text, w.bbox) for w in words] == [("Azure", (10, 10, 60, 30)), ("Firewall", (66, 10, 140, 30))]
