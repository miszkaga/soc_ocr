import json
import shutil

import numpy as np
import pytest
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from ocr.cli import input_files, main, run
from ocr.config import load_config
from ocr.engines.base import OcrEngine, OcrWord
from ocr.layout import lines
from ocr.schema import SCHEMA, validate
from tests.test_paddle_engine import V6, _cfg, _models_present
from tests.test_tesseract_engine import text_image


class FakeEngine(OcrEngine):
    """Two words on one line at fixed pixels of whatever it reads; the second needs review."""
    name = "fake"

    def versions(self):
        return {"fake": "1"}

    def detect(self, img):
        return []

    def recognize(self, img, boxes):
        return []

    def read(self, img):
        return [OcrWord("Azure", (10.0, 20.0, 60.0, 40.0), 0.99, [0.99] * 5, "accepted", "agree", {"p": "Azure"}),
                OcrWord("10.0.0.1", (70.0, 20.0, 150.0, 40.0), 0.6, [0.6] * 8, "uncertain", "engines_disagree",
                        {"p": "10.0.0.1", "s": "l0.0.0.1"})]


def _folder(tmp_path):
    src = tmp_path / "in"
    (src / "sub").mkdir(parents=True)
    Image.new("RGB", (300, 120), "white").save(src / "shot.png")
    Image.new("RGB", (300, 120), "white").save(src / "sub" / "deep.jpg")
    (src / "notes.txt").write_text("not an input")
    c = canvas.Canvas(str(src / "doc.pdf"), pagesize=A4)
    # a 600x240 px image drawn 300x120 pt at (100 pt, 200 pt from the top)
    Image.new("RGB", (600, 240), "white").save(tmp_path / "embedded.png")
    c.drawImage(str(tmp_path / "embedded.png"), 100, A4[1] - 200 - 120, 300, 120)
    c.showPage()
    c.save()
    return src


def test_inputs_are_pdfs_and_images_and_the_output_folder_is_skipped(tmp_path):
    src = _folder(tmp_path)
    assert [p.name for p in input_files(src, recursive=False)] == ["doc.pdf", "shot.png"]
    assert [p.name for p in input_files(src, recursive=True)] == ["doc.pdf", "shot.png", "deep.jpg"]
    (src / "out").mkdir()
    Image.new("RGB", (5, 5)).save(src / "out" / "old.png")
    assert "old.png" not in [p.name for p in input_files(src, True, skip=src / "out")]


def test_folder_is_read_into_text_boxes_and_valid_documents(tmp_path):
    src, out = _folder(tmp_path), tmp_path / "out"
    assert run(src, out, recursive=True, engine=FakeEngine()) == 0
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*.json")) == [
        "json/doc.pdf.json", "json/shot.png.json", "json/sub/deep.jpg.json"]
    img = json.loads((out / "json" / "shot.png.json").read_text())
    assert validate(img) == [] and img["schema"] == SCHEMA and img["source"] == "shot.png"
    [region] = img["pages"][0]["regions"]
    assert (region["kind"], region["status"], region["text"]) == ("image", "uncertain", "Azure 10.0.0.1")
    assert region["lines"][0]["words"][0]["bbox_pt"] is None
    [item] = img["review"]
    assert (item["region"], item["word"], item["text"], item["reason"]) == ("p1r1", 1, "10.0.0.1", "engines_disagree")
    assert (out / "json" / item["crop"]).exists()
    deep = json.loads((out / "json" / "sub" / "deep.jpg.json").read_text())
    assert (out / "json" / "sub" / deep["review"][0]["crop"]).exists()  # crop paths are relative to the json

    text = (out / "text.txt").read_text()
    assert text.index("=== doc.pdf | page 1 | p1r1 embedded_image | 1 to review ===") < text.index("=== shot.png")
    assert "=== shot.png | 1 to review ===\nAzure 10.0.0.1\n" in text and "=== sub/deep.jpg" in text

    assert sorted(p.relative_to(out).as_posix() for p in (out / "boxes").rglob("*.png")) == [
        "boxes/doc.pdf.p1.png", "boxes/shot.png", "boxes/sub/deep.jpg.png"]
    boxed = Image.open(out / "boxes" / "shot.png").convert("RGB")
    assert boxed.size == (300, 120)
    assert boxed.getpixel((9, 30)) == (0, 170, 0) and boxed.getpixel((69, 30)) == (230, 0, 0)  # word box edges
    assert sorted(p.relative_to(out).as_posix() for p in (out / "overlay").rglob("*.png")) == [
        "overlay/doc.pdf.p1.png", "overlay/shot.png", "overlay/sub/deep.jpg.png"]
    over = Image.open(out / "overlay" / "shot.png").convert("RGB")
    assert over.size == (300 * 2 + 10, 120)  # image with boxes | text written into the boxes
    right = over.crop((310, 0, 610, 120))
    a = np.asarray(right).astype(int)
    green, red = a[20:40, 10:60], a[20:40, 70:150]
    assert ((green[..., 1] > 100) & (green[..., 0] < 60)).any()  # "Azure" written in green
    assert ((red[..., 0] > 150) & (red[..., 1] < 60)).any()      # "10.0.0.1" in red: to review
    page = Image.open(out / "boxes" / "doc.pdf.p1.png").convert("RGB")
    assert page.getpixel((round(100 * 300 / 72) + 9, round(200 * 300 / 72) + 30)) != (255, 255, 255)


def test_each_run_replaces_the_previous_output_but_keeps_other_files(tmp_path):
    src, out = _folder(tmp_path), tmp_path / "out"
    run(src, out, engine=FakeEngine())
    (out / "notes.md").write_text("mine")
    (src / "shot.png").unlink()
    run(src, out, engine=FakeEngine())
    assert not (out / "json" / "shot.png.json").exists() and not (out / "boxes" / "shot.png").exists()
    assert not (out / "overlay" / "shot.png").exists() and "shot.png" not in (out / "text.txt").read_text()
    assert (out / "json" / "doc.pdf.json").exists() and (out / "notes.md").read_text() == "mine"


def test_pdf_words_get_page_coordinates(tmp_path):
    src, out = _folder(tmp_path), tmp_path / "out"
    run(src, out, engine=FakeEngine())
    doc = json.loads((out / "json" / "doc.pdf.json").read_text())
    assert validate(doc) == []
    page = doc["pages"][0]
    [region] = page["regions"]
    assert region["kind"] == "embedded_image" and region["render_dpi"] == 300
    assert region["bbox_pt"] == pytest.approx([100, 200, 400, 320], abs=0.5)
    word = region["lines"][0]["words"][0]
    # pixel (10, 20) of the region rendered at 300 dpi lies 2.4 / 4.8 pt into it
    assert word["bbox_pt"] == pytest.approx([102.4, 204.8, 114.4, 209.6], abs=0.3)


def test_a_broken_file_is_reported_and_the_rest_still_read(tmp_path, capsys):
    src, out = _folder(tmp_path), tmp_path / "out"
    (src / "broken.pdf").write_bytes(b"%PDF-1.4 nonsense")
    assert run(src, out, engine=FakeEngine()) == 1
    assert "broken.pdf: FAILED" in capsys.readouterr().err
    assert (out / "json" / "shot.png.json").exists() and not (out / "json" / "broken.pdf.json").exists()
    assert "=== broken.pdf ===\n(could not be read" in (out / "text.txt").read_text()


def test_native_pdf_text_is_taken_and_ocr_repeating_it_dropped(tmp_path):
    from tests.pdf_factory import diagram, new_canvas

    src = tmp_path / "in"
    src.mkdir()
    c = new_canvas(src / "diagram.pdf")
    diagram(c, 60, 300, labels=("Client", "API", "DB"))
    c.save()

    class OverlappingEngine(FakeEngine):
        def read(self, img):
            # "Client" read again where the PDF has it (region origin: 60 pt, ~300 pt), and a
            # word in a picture where the PDF has no text
            s = 300 / 72
            return [OcrWord("Cl1ent", ((70 - 60) * s, (318 - 300) * s, (93 - 60) * s, (325 - 300) * s), 0.9,
                            [0.9] * 6, "accepted", "agree", {}),
                    OcrWord("outlined", (10.0, 150.0, 90.0, 170.0), 0.9, [0.9] * 8, "uncertain", "engines_disagree", {})]

    assert run(src, tmp_path / "out", engine=OverlappingEngine()) == 0
    doc = json.loads((tmp_path / "out" / "json" / "diagram.pdf.json").read_text())
    assert validate(doc) == []
    [region] = doc["pages"][0]["regions"]
    assert region["kind"] == "vector_diagram" and region["native_words"] == 5
    words = [w for line in region["lines"] for w in line["words"]]
    by_text = {w["text"]: w for w in words}
    assert "Cl1ent" not in by_text and "outlined" in by_text
    assert by_text["Client"]["reason"] == "native" and by_text["Client"]["votes"] == {"pdf": "Client"}
    assert by_text["Client"]["bbox_pt"][0] == pytest.approx(70.5, abs=1.0)
    assert [r["text"] for r in doc["review"]] == ["outlined"]


def test_without_arguments_the_input_folder_is_read_into_out(tmp_path, monkeypatch):
    import ocr.cli as cli

    (tmp_path / "input").mkdir()
    Image.new("RGB", (50, 20), "white").save(tmp_path / "input" / "a.png")
    monkeypatch.setattr(cli, "DEFAULT_INPUT", tmp_path / "input")
    monkeypatch.setattr(cli, "DEFAULT_OUT", tmp_path / "out")
    calls = []
    monkeypatch.setattr(cli, "run", lambda folder, out, recursive: calls.append((folder, out, recursive)) or 0)
    assert cli.main([]) == 0
    assert calls == [(tmp_path / "input", tmp_path / "out", False)]


def test_main_rejects_a_missing_or_empty_folder(tmp_path):
    assert main(["extract", str(tmp_path / "nope")]) == 2
    (tmp_path / "empty").mkdir()
    assert main(["extract", str(tmp_path / "empty")]) == 2


def test_schema_validation_finds_problems():
    good = {"schema": SCHEMA, "source": "a.png", "engine_versions": {}, "review": [], "pages": [
        {"page": 1, "regions": [{"id": "p1r1", "kind": "image", "bbox_pt": None, "status": "accepted", "text": "a",
                                 "lines": [{"words": [{"text": "ab", "conf": 0.9, "char_conf": [0.9, 0.9],
                                                       "bbox_px": [0, 0, 1, 1], "bbox_pt": None,
                                                       "status": "accepted"}]}]}]}]}
    assert validate(good) == []
    bad = json.loads(json.dumps(good))
    w = bad["pages"][0]["regions"][0]["lines"][0]["words"][0]
    w["char_conf"], w["conf"], w["bbox_px"] = [0.9], 1.5, [5, 0, 1, 1]
    bad["review"] = [{"region": "p9r9", "reason": "x", "crop": "c.png"}]
    bad["schema"] = "ocr/v0"
    problems = " ".join(validate(bad))
    for part in ("schema", "char_conf", "conf: outside", "bbox_px", "unknown region"):
        assert part in problems


def test_layout_keeps_lines_and_separate_labels():
    w = lambda t, x0, y0, x1, y1: OcrWord(t, (x0, y0, x1, y1), 0.9, [0.9] * len(t))  # noqa: E731
    got = lines([w("network", 70, 0, 130, 20), w("Virtual", 0, 0, 60, 20), w("Hub", 0, 30, 30, 50),
                 w("Label", 400, 2, 450, 20)])
    assert [[x.text for x in ln] for ln in got] == [["Virtual", "network"], ["Label"], ["Hub"]]


@pytest.mark.skipif(shutil.which("tesseract") is None or not _models_present(_cfg(**V6)),
                    reason="needs tesseract and the PP-OCRv6 models")
def test_real_engines_read_an_image_folder(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    words = ["Azure", "Firewall", "10.0.0.1"]
    img, _ = text_image(words)
    img.save(src / "a.png")
    assert run(src, tmp_path / "out") == 0
    doc = json.loads((tmp_path / "out" / "json" / "a.png.json").read_text())
    assert validate(doc) == [] and doc["pages"][0]["regions"][0]["text"] == " ".join(words)
    assert load_config().consensus.primary in json.dumps(doc["engine_versions"])
