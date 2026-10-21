import dataclasses

import pytest
from PIL import Image

from ocr.config import load_config
from ocr.consensus import ConsensusEngine, flag_reason, secondary_as
from ocr.engines.base import OcrEngine, OcrWord
from tests.test_paddle_engine import V6, _cfg, _models_present
from tests.test_tesseract_engine import text_image


class Scripted(OcrEngine):
    """Fake engine: answers come from a script keyed by (scale, word index); boxes are fixed."""

    def __init__(self, name, readings, conf=0.95, extra=()):
        self.name, self.readings, self.conf, self.extra = name, readings, conf, list(extra)
        self.boxes = [(10.0 + 60 * i, 10.0, 60.0 + 60 * i, 30.0) for i in range(len(readings[1]))]

    def versions(self):
        return {"fake": "1"}

    def detect(self, img):
        return self.boxes

    def _word(self, text, box):
        return OcrWord(text, box, self.conf, [self.conf] * len(text))

    def read(self, img):
        return [self._word(t, b) for t, b in zip(self.readings[1], self.boxes)] + self.extra

    def recognize(self, img, boxes):
        out = []
        for box in boxes:
            if box[0] == 0.0 and box[1] == 0.0:  # a whole enlarged crop from the re-read
                scale, idx = 2, self._crop_index(img)
            else:
                scale, idx = 1, self.boxes.index(tuple(box))
            text = self.readings[scale][idx]
            if text:
                out.append(self._word(text, box))
        return out

    def _crop_index(self, img):
        return img.getpixel((0, 0))[0]  # the test image encodes the word index in red


def image_with_index_stripes(n):
    """Each word's box area has red == its index, so an enlarged crop reveals which word it is."""
    img = Image.new("RGB", (60 * n + 20, 40), (255, 255, 255))
    for i in range(n):
        img.paste((i, 0, 0), (0 + 60 * i, 0, 66 + 60 * i, 40))
    return img


CFG = load_config().consensus


def consensus(primary, secondary, **kw):
    return ConsensusEngine(dataclasses.replace(CFG, **kw), primary=primary, secondary=secondary)


def test_agreement_with_confidence_is_accepted():
    p = Scripted("p", {1: ["Azure", "10.0.0.1"], 2: ["?", "?"]})
    s = Scripted("s", {1: ["Azure", "10.0.0.1"], 2: ["!", "!"]})
    words = consensus(p, s).read(image_with_index_stripes(2))
    assert [(w.text, w.status, w.reason) for w in words] == [
        ("Azure", "accepted", "agree"), ("10.0.0.1", "accepted", "agree")]
    assert words[0].votes == {"p": "Azure", "s": "Azure"}


def test_disagreement_resolved_by_reread_majority():
    p = Scripted("p", {1: ["l0.0.0.1"], 2: ["10.0.0.1"]})
    s = Scripted("s", {1: ["10.0.0.1"], 2: ["10.0.0.1"]})
    [w] = consensus(p, s).read(image_with_index_stripes(1))
    assert (w.text, w.status, w.reason) == ("10.0.0.1", "accepted", "reread_agree")
    assert w.votes == {"p": "l0.0.0.1", "s": "10.0.0.1", "p@x2": "10.0.0.1", "s@x2": "10.0.0.1"}
    assert len(w.char_conf) == len(w.text)


def test_persistent_disagreement_goes_to_review_with_primary_reading():
    p = Scripted("p", {1: ["AI"], 2: ["AI"]})
    s = Scripted("s", {1: ["Al"], 2: ["Al"]})
    [w] = consensus(p, s).read(image_with_index_stripes(1))
    assert (w.text, w.status, w.reason) == ("AI", "uncertain", "engines_disagree")


def test_unsure_agreement_is_reread_and_accepted_only_if_it_holds():
    p = Scripted("p", {1: ["DNS"], 2: ["DNS"]}, conf=0.3)
    s = Scripted("s", {1: ["DNS"], 2: ["ONS"]}, conf=0.3)
    [w] = consensus(p, s).read(image_with_index_stripes(1))
    assert (w.status, w.reason) == ("accepted", "reread_agree")  # 3 of 4 readings say DNS
    p = Scripted("p", {1: ["DNS"], 2: ["ONS"]}, conf=0.3)
    s = Scripted("s", {1: ["DNS"], 2: ["DN5"]}, conf=0.3)
    [w] = consensus(p, s).read(image_with_index_stripes(1))
    assert (w.status, w.reason) == ("uncertain", "low_confidence")


def test_secondary_did_not_read_anything():
    p = Scripted("p", {1: ["Azure"], 2: ["Azure"]})
    s = Scripted("s", {1: [""], 2: [""]})
    [w] = consensus(p, s).read(image_with_index_stripes(1))
    assert (w.status, w.reason) == ("uncertain", "engines_disagree")


def test_secondary_only_text_is_added_for_review_when_enabled():
    extra = OcrWord("hidden", (500.0, 10.0, 560.0, 30.0), 0.9, [0.9] * 6)
    on_icon = OcrWord("Azure", (12.0, 12.0, 58.0, 28.0), 0.9, [0.9] * 5)  # overlaps a primary word
    p = Scripted("p", {1: ["Azure"], 2: ["Azure"]})
    s = Scripted("s", {1: ["Azure"], 2: ["Azure"]}, extra=[extra, on_icon])
    img = image_with_index_stripes(1)
    assert len(consensus(p, s).read(img)) == 1
    words = consensus(p, s, secondary_detection=True).read(img)
    assert [(w.text, w.status, w.reason) for w in words] == [
        ("Azure", "accepted", "agree"), ("hidden", "uncertain", "secondary_only")]


@pytest.mark.skipif(not _models_present(_cfg(**V6)), reason="PP-OCRv6 medium models not downloaded")
def test_real_engines_accept_clear_text():
    words = ["Private", "endpoint", "10.1.2.4", "Łódź"]
    img, _ = text_image(words)
    got = ConsensusEngine()
    out = got.read(img)
    assert [w.text for w in out] == words
    assert all(w.status == "accepted" for w in out)
    assert set(out[0].votes) == {"paddle_openvino", "tesseract"}
    assert "paddle_openvino.rec_model" in got.versions()


@pytest.mark.parametrize("text,reason", [
    ("Al", "ambiguous_glyphs"), ("OpenAl", "ambiguous_glyphs"), ("Al,", "ambiguous_glyphs"),
    ("loT", "ambiguous_glyphs"), ("MDT0", "ambiguous_glyphs"), ("l0.0.0.1", "ambiguous_glyphs"),
    ("-SAP", "edge_dash"), ("network—", "edge_dash"), ("plane-", "edge_dash"),
    ("AI", None), ("IoT", None), ("ID", None), ("BlobStorage", None), ("DevOps", None),
    ("T1055", None), ("10.0.0.8", None), ("IPv4", None), ("Azure", None), ("-", None),
])
def test_flag_rules(text, reason):
    assert flag_reason(text, CFG) == reason


def test_flags_turn_agreed_words_into_review_and_can_be_disabled():
    p = Scripted("p", {1: ["OpenAl"], 2: ["OpenAl"]})
    s = Scripted("s", {1: ["OpenAl"], 2: ["OpenAl"]})
    [w] = consensus(p, s).read(image_with_index_stripes(1))
    assert (w.status, w.reason) == ("uncertain", "ambiguous_glyphs")
    [w] = consensus(p, s, flag_ambiguous_glyphs=False).read(image_with_index_stripes(1))
    assert (w.status, w.reason) == ("accepted", "agree")


@pytest.mark.parametrize("secondary,primary,counts_as", [
    ("Azure |", "Azure", "Azure"), ("Private.", "Private", "Private"), ("‘Instance", "Instance", "Instance"),
    ("Azure", "Azure", "Azure"),
    ("Azure", "Azure.", "Azure"),          # the primary's own edge "." must be confirmed exactly
    ("network.", "network.", "network."),
    ("ClusterlP", "ClusterIP", "ClusterlP"),  # inside the word: a real disagreement
    ("| soul", "SQL", "| soul"),
])
def test_secondary_edge_junk(secondary, primary, counts_as):
    assert secondary_as(secondary, primary, CFG) == counts_as


def test_edge_junk_does_not_cause_review():
    p = Scripted("p", {1: ["Azure"], 2: ["Azure"]})
    s = Scripted("s", {1: ["Azure |"], 2: ["Azure."]})
    [w] = consensus(p, s).read(image_with_index_stripes(1))
    assert (w.text, w.status, w.reason) == ("Azure", "accepted", "agree")
    assert w.votes["s"] == "Azure |"  # the raw reading is kept
    [w] = consensus(p, s, secondary_edge_junk="").read(image_with_index_stripes(1))
    assert (w.status, w.reason) == ("uncertain", "engines_disagree")
