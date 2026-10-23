import dataclasses
from types import SimpleNamespace

import numpy as np

import pytest

from eval.metrics import overlap
from ocr.config import load_config
from ocr.engines.paddle_onnx import REPO, PaddleOnnxEngine, char_confs, line_to_words, mirror_along
from ocr.engines.paddle_openvino import PaddleOpenVinoEngine
from tests.test_tesseract_engine import text_image

V6 = dict(ocr_version="PP-OCRv6", det_model_type="medium", rec_model_type="medium", det_lang="ch", rec_lang="en")
V5 = dict(ocr_version="PP-OCRv5", det_model_type="mobile", rec_model_type="mobile", det_lang="multi", rec_lang="latin")


def _cfg(**kw):
    return dataclasses.replace(load_config().paddle, **kw)


def _models_present(cfg) -> bool:
    root = REPO / cfg.model_root_dir
    return all((root / name).exists() for name in PaddleOnnxEngine(cfg)._model_files().values())


# Tests that run the models need them downloaded once (see artifacts/MANIFEST.json); never fetch in tests.
needs_v6 = pytest.mark.skipif(not _models_present(_cfg(**V6)), reason="PP-OCRv6 medium models not downloaded")


def test_line_is_split_into_words_at_spaces():
    chars = [("a", 0.9, (0, 0, 10, 20)), ("b", 0.8, (10, 0, 20, 20)), ("1", 0.99, (40, 0, 50, 20))]
    words = line_to_words("ab 1", chars, offset=(100, 5))
    assert [(w.text, w.conf, w.char_conf, w.bbox_px) for w in words] == [
        ("ab", 0.8, [0.9, 0.8], (100, 5, 120, 25)), ("1", 0.99, [0.99], (140, 5, 150, 25))]


def test_line_kept_whole_when_characters_do_not_line_up():
    chars = [("a", 0.9, (0, 0, 10, 20)), ("b", 0.8, (10, 0, 20, 20))]
    [w] = line_to_words("a b c", chars)
    assert w.text == "ab" and len(w.char_conf) == 2


def test_model_files_resolve_like_rapidocr():
    assert PaddleOnnxEngine(_cfg(**V6))._model_files() == {
        "det": "PP-OCRv6_det_medium.onnx", "rec": "PP-OCRv6_rec_medium.onnx"}
    assert PaddleOnnxEngine(_cfg(**V5))._model_files()["rec"] == "latin_PP-OCRv5_rec_mobile.onnx"


@needs_v6
@pytest.mark.parametrize("engine_cls", [PaddleOnnxEngine, PaddleOpenVinoEngine])
def test_read_polish_and_hard_strings(engine_cls):
    words = ["Zażółć", "gęślą", "jaźń", "j.kowalski@example.com", "10.0.0.1"]
    img, boxes = text_image(words)
    got = engine_cls(_cfg(**V6)).read(img)
    assert [w.text for w in got] == words
    for w, box in zip(got, boxes):
        assert 0 < w.conf <= 1 and len(w.char_conf) == len(w.text)
        assert overlap(w.bbox_px, box) > 0.8


@needs_v6
def test_detect_and_recognize_given_boxes():
    words = ["Private", "endpoint", "10.1.2.4"]
    img, boxes = text_image(words)
    e = PaddleOnnxEngine(_cfg(**V6))
    assert len(e.detect(img)) >= 1
    got = e.recognize(img, boxes)
    assert [w.text for w in got] == words
    assert all(overlap(w.bbox_px, b) > 0.5 for w, b in zip(got, boxes))


@needs_v6
def test_versions_include_model_hashes():
    v = PaddleOnnxEngine(_cfg(**V6)).versions()
    assert v["rec_model"] == "PP-OCRv6_rec_medium.onnx" and len(v["rec_sha256"]) == 16
    assert v["backend"] == "onnxruntime"


def test_char_confidences_skip_the_spaces_of_the_line():
    assert char_confs("ab c", [0.9, 0.8, 0.99, 0.1]) == [0.9, 0.8, 0.1]
    assert char_confs("ab c", [0.9, 0.8]) is None


def _cell(x0, x1, y0=0.0, y1=20.0):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def test_low_confidence_of_the_last_letter_reaches_its_word():
    # RapidOCR pairs the non-space characters with confidences that include the space, so the
    # second word's are shifted and the last letter's is lost; the engine reassigns them.
    text = "Azure Kubernetes"
    confs = [0.99] * (len(text) - 1) + [0.2]
    chars = [(c, 0.99, _cell(10 * i, 10 * i + 9)) for i, c in enumerate(text.replace(" ", ""))]
    out = SimpleNamespace(txts=(text,), boxes=None, word_results=(tuple(chars),),
                          line_flipped=[False], line_confs=[confs])
    azure, kube = PaddleOnnxEngine._words(out)
    assert kube.text == "Kubernetes" and kube.conf == 0.2 and kube.char_conf[-1] == 0.2
    assert azure.conf == 0.99


def test_turned_line_has_its_character_boxes_mirrored_along_the_line():
    quad = np.array(_cell(0, 100))
    assert np.allclose(mirror_along(quad, [[10, 5], [30, 15]]), [[90, 5], [70, 15]])
    tall = np.array(_cell(0, 20, 0, 100))
    assert np.allclose(mirror_along(tall, [[5, 10]]), [[5, 90]])
    chars = [("a", 0.9, _cell(0, 10)), ("b", 0.9, _cell(10, 20)), ("c", 0.9, _cell(40, 50))]
    out = SimpleNamespace(txts=("ab c",), boxes=np.array([quad]), word_results=(tuple(chars),),
                          line_flipped=[True], line_confs=[None])
    ab, c = PaddleOnnxEngine._words(out)
    assert ab.bbox_px == (80.0, 0.0, 100.0, 20.0) and c.bbox_px == (50.0, 0.0, 60.0, 20.0)


@needs_v6
@pytest.mark.parametrize("angle", [90, -90, 180])
def test_word_boxes_of_turned_text_lie_on_their_words(angle):
    # 90: vertical, read bottom to top (the classifier turns the crop by 180 degrees);
    # -90: read top to bottom; 180: upside down.
    words = ["Fabric", "domains"]
    img, boxes = text_image(words)
    W, H = img.size
    turned = img.rotate(angle, expand=True)

    def turn(b):
        x0, y0, x1, y1 = b
        pts = {90: [(y0, W - x0), (y1, W - x1)], -90: [(H - y0, x0), (H - y1, x1)],
               180: [(W - x0, H - y0), (W - x1, H - y1)]}[angle]
        (ax, ay), (bx, by) = pts
        return (min(ax, bx), min(ay, by), max(ax, bx), max(ay, by))

    got = PaddleOnnxEngine(_cfg(**V6, refine_boxes=False)).read(turned)
    assert [w.text for w in got] == words
    for w, b in zip(got, boxes):
        assert overlap(w.bbox_px, turn(b)) > 0.8, (w.text, w.bbox_px, turn(b))


def test_turn_the_classifier_got_wrong_is_undone_by_a_more_confident_reading():
    from ocr.engines.paddle_onnx import settle_turns

    class FakeOcr:
        def recognize_txt(self, crops):
            # read as detected: line 0 better than its turned reading, line 1 worse
            return SimpleNamespace(txts=("client app", "sdde"), scores=[0.98, 0.40],
                                   word_results=("info0", "info1"))

    rec = SimpleNamespace(txts=("cdlient app", "apps", "x"), scores=[0.90, 0.95, 0.99],
                          word_results=("t0", "t1", "t2"))
    flipped = [True, True, False]
    assert settle_turns(FakeOcr(), ["c0", "c1", "c2"], rec, flipped) == 1
    assert flipped == [False, True, False]
    assert rec.txts == ("client app", "apps", "x") and rec.word_results == ("info0", "t1", "t2")
