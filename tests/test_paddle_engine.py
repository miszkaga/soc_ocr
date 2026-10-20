import dataclasses

import pytest

from eval.metrics import overlap
from ocr.config import load_config
from ocr.engines.paddle_onnx import REPO, PaddleOnnxEngine, line_to_words
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
