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


def test_majority_of_another_length_does_not_overrule_the_primary_reading():
    # The box holds only the end of "EncryptResponse>",": the re-reads agree on a fragment.
    p = Scripted("p", {1: ["'\"i"], 2: ['se>",']})
    s = Scripted("s", {1: ['se>",'], 2: ['se>",']})
    [w] = consensus(p, s).read(image_with_index_stripes(1))
    assert (w.text, w.status, w.reason) == ("'\"i", "uncertain", "reread_length")
    assert w.votes["s@x2"] == 'se>",'
    [w] = consensus(p, s, reread_max_length_change=-1).read(image_with_index_stripes(1))
    assert (w.text, w.status) == ('se>",', "accepted")
    p = Scripted("p", {1: ["ploud"], 2: ["Cloud"]})  # same length: a misread letter is fixed
    s = Scripted("s", {1: ["Cloud"], 2: ["Cloud"]})
    [w] = consensus(p, s).read(image_with_index_stripes(1))
    assert (w.text, w.status) == ("Cloud", "accepted")


def test_majority_may_not_drop_letters_from_the_primary_reading_edges():
    # A box cutting the first letter: every re-read agrees on "inux".
    p = Scripted("p", {1: ["Linux"], 2: ["inux"]})
    s = Scripted("s", {1: ["inux"], 2: ["inux"]})
    [w] = consensus(p, s).read(image_with_index_stripes(1))
    assert (w.text, w.status, w.reason) == ("Linux", "uncertain", "reread_edge_letter")
    p = Scripted("p", {1: ["-Begin"], 2: ["Begin"]})  # edge punctuation may still go
    s = Scripted("s", {1: ["Begin"], 2: ["Begin"]})
    [w] = consensus(p, s, flag_edge_dashes=False).read(image_with_index_stripes(1))
    assert (w.text, w.status) == ("Begin", "accepted")


def test_letter_o_next_to_digits_in_the_same_image_goes_to_review():
    from ocr.consensus import apply_sibling_flags

    words = [OcrWord(t, (0, 0, 1, 1), 0.9, [0.9] * len(t), "accepted", "agree") for t in ("MDTO", "MDT1", "ATO", "TO")]
    got = {w.text: w.reason for w in apply_sibling_flags(words, CFG)}
    assert got == {"MDTO": "ambiguous_glyphs", "MDT1": "agree", "ATO": "agree", "TO": "agree"}


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
    ("tenantld", "ambiguous_glyphs"), ("<Appld>,", "ambiguous_glyphs"), ("Image", None), ("Igła", None),
    ("build", None), ("field", None), ("world", None), ("Is", None), ("In", None), ("If", None), ("Id", None),
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
    ("Understand 1", "Understand", "Understand"), ("| agents i", "agents", "agents"),  # slivers apart
    ("i my |", "my", "my"), ("cache i", "cache", "cache"),
    ("Azure A", "Azure A", "Azure A"), ("Azure Al", "Azure", "Azure Al"),  # two-letter token counts
    ("ab", "a", "ab"),
    ("Gateway”", 'Gateway"', 'Gateway"'), ("AgentCore”", 'AgentCore"', 'AgentCore"'),  # typography
    ("“Deploy:", "“Deploy", "“Deploy"), ("“Make |", '"Make', '"Make'),  # each edge on its own
    ("(12)", "12", "12"), ("12)", "12", "12"),  # a number in a circle
    ("Claude", "(Claude", "Claude"),  # the primary's own bracket must be confirmed
    ("compromis(", "compromis", "compromis("),  # "(" after the word is not a circle
    ("'(name, :", "(name,", "(name,"), ("_(T1497)", "(T1497)", "(T1497)"),
])
def test_secondary_edge_junk(secondary, primary, counts_as):
    assert secondary_as(secondary, primary, CFG) == counts_as


def test_edge_junk_does_not_cause_review():
    p = Scripted("p", {1: ["Azure"], 2: ["Azure"]})
    s = Scripted("s", {1: ["Azure |"], 2: ["Azure."]})
    [w] = consensus(p, s).read(image_with_index_stripes(1))
    assert (w.text, w.status, w.reason) == ("Azure", "accepted", "agree")
    assert w.votes["s"] == "Azure |"  # the raw reading is kept
    [w] = consensus(p, s, secondary_edge_junk="", secondary_edge_tokens=False).read(image_with_index_stripes(1))
    assert (w.status, w.reason) == ("uncertain", "engines_disagree")


def test_short_words_are_not_sent_to_review_unless_a_reading_saw_more():
    from ocr.consensus import settle_short

    w = lambda t, votes: OcrWord(t, (0, 0, 1, 1), 0.5, [0.5] * len(t), "uncertain", "engines_disagree", votes)  # noqa: E731
    assert settle_short(w("1", {"p": "1", "s": "O"}), CFG).reason == "short"
    assert settle_short(w("个", {"p": "个", "s": ">", "p2": "7", "s2": "><"}), CFG).status == "accepted"
    assert settle_short(w("'\"i", {"p": "'\"i", "s": 'se>",'}), CFG).status == "uncertain"  # a misread long word
    assert settle_short(w("10", {"p": "10", "s": ""}), CFG).status == "uncertain"  # two digits: reviewed
    assert settle_short(w("D", {"p": "D", "s": "a"}), CFG).status == "uncertain"  # may be left of "ID"
