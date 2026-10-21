import json
import threading
import urllib.request

from PIL import Image

from eval.label_regions import apply_decisions, collect_regions, prepare, serve, word_like
from eval.run_eval import load_gt, manual_path


def make_gt(tmp_path):
    gt_dir = tmp_path / "gt" / "svg"
    (gt_dir / "src").mkdir(parents=True)
    for name in ("a", "b"):
        Image.new("RGB", (400, 100), "white").save(gt_dir / "src" / f"{name}.png")
        (gt_dir / "src" / f"{name}.json").write_text(json.dumps({"image": f"{name}.png", "words": [
            {"text": "Azure", "bbox_px": [10, 10, 60, 30], "legible": True, "clipped": False, "rotated": False}]}))
    return gt_dir


def make_run(tmp_path, name, readings):
    run = tmp_path / "runs" / name
    run.mkdir(parents=True)
    with (run / "predictions.jsonl").open("w") as f:
        for image, words in readings.items():
            f.write(json.dumps({"json": image, "seconds": 1, "words": [
                {"text": t, "bbox_px": b, "conf": 0.9, "char_conf": []} for t, b in words]}) + "\n")
    return run


def test_word_like():
    assert word_like("SQL") and word_like("10.0.0.1") and word_like("Łódź")
    assert not word_like("<>") and not word_like("a") and not word_like("春网")


def test_candidates_need_two_engines_and_word_like_text(tmp_path):
    gt_dir = make_gt(tmp_path)
    icon = [100, 10, 140, 30]
    r1 = make_run(tmp_path, "paddle", {"src/a.json": [("Azure", [10, 12, 60, 28]), ("SQL", icon), ("<>", [200, 10, 230, 30])],
                                       "src/b.json": [("SQL", icon)]})
    r2 = make_run(tmp_path, "tess", {"src/a.json": [("SQL", [101, 11, 139, 29]), ("<>", [200, 10, 230, 30]),
                                                     ("III", [300, 10, 330, 30])]})
    regions = collect_regions(gt_dir, [r1, r2])
    # "Azure" has ground truth, "<>" is not word-like, "III" and b.json's "SQL" were read by one engine only.
    assert [(r["json"], r["proposal"]) for r in regions] == [("src/a.json", "SQL")]


def test_decisions_become_ground_truth(tmp_path):
    gt_dir = make_gt(tmp_path)
    icon, logo, blur = [100, 10, 140, 30], [200, 10, 300, 30], [320, 10, 380, 30]
    runs = [make_run(tmp_path, n, {"src/a.json": [("SQL", icon), ("Microsoft Azure", logo), ("xx yy", blur)],
                                   "src/b.json": [("SQL", icon)]}) for n in ("p", "t")]
    work = tmp_path / "work"
    groups = prepare(gt_dir, runs, work)
    by_text = {g["proposal"]: g["id"] for g in groups}
    assert (work / groups[0]["crop"]).exists()
    out = apply_decisions(work, {by_text["SQL"]: {"status": "text", "text": "SQL"},
                                 by_text["Microsoft Azure"]: {"status": "no_text"},
                                 by_text["xx yy"]: {"status": "unreadable"}})
    assert out == manual_path(gt_dir)

    _, words, regions = load_gt(gt_dir, {"json": "src/a.json"})
    assert [w.text for w in words] == ["Azure", "SQL"]
    assert regions == [tuple(blur)]
    _, words, _ = load_gt(gt_dir, {"json": "src/b.json"})
    assert [w.text for w in words] == ["Azure", "SQL"]  # the same icon in another image (same group)
    assert collect_regions(gt_dir, runs) == []  # decided regions are not offered again


def test_multi_word_label_gets_one_box_per_word(tmp_path):
    gt_dir = make_gt(tmp_path)
    logo = [200, 10, 300, 30]
    runs = [make_run(tmp_path, n, {"src/a.json": [("Microsoft Azure", logo)]}) for n in ("p", "t")]
    work = tmp_path / "work"
    [g] = prepare(gt_dir, runs, work)
    apply_decisions(work, {g["id"]: {"status": "text", "text": "Microsoft Azure"}})
    _, words, _ = load_gt(gt_dir, {"json": "src/a.json"})
    ms, az = words[1], words[2]
    assert (ms.text, az.text) == ("Microsoft", "Azure")
    assert ms.bbox[0] == 200 and az.bbox[2] == 300 and ms.bbox[2] < az.bbox[0]


def test_page_saves_decisions(tmp_path):
    gt_dir = make_gt(tmp_path)
    runs = [make_run(tmp_path, n, {"src/a.json": [("SQL", [100, 10, 140, 30])]}) for n in ("p", "t")]
    work = tmp_path / "work"
    [g] = prepare(gt_dir, runs, work)
    port = 18765
    threading.Thread(target=serve, args=(work, port), daemon=True).start()
    import time
    for _ in range(50):
        try:
            page = urllib.request.urlopen(f"http://127.0.0.1:{port}/").read().decode()
            break
        except OSError:
            time.sleep(0.1)
    assert "Zapisz" in page
    req = urllib.request.Request(f"http://127.0.0.1:{port}/save", method="POST",
                                 data=json.dumps({g["id"]: {"status": "text", "text": "SQL"}}).encode(),
                                 headers={"Content-Type": "application/json"})
    assert urllib.request.urlopen(req).status == 204
    assert json.loads((work / "decisions.json").read_text())[g["id"]]["text"] == "SQL"
    assert manual_path(gt_dir).exists()
