import numpy as np
from PIL import Image

from eval.box_ink import measure, summarize, text_ink, word_inks
from eval.metrics import GtWord
from ocr.config import load_config

CFG = load_config().eval


def _diff_with_two_words():
    """Text ink of two words: 'alpha' at x 10..50, 'beta' at x 60..90, rows 20..40."""
    d = np.zeros((60, 120), np.uint8)
    d[20:40, 10:50] = 200
    d[20:40, 60:90] = 200
    d[25, 10:50] = 90  # antialiasing lighter than half of the peak is not counted
    words = [GtWord("alpha", (9.0, 16.0, 51.0, 44.0)), GtWord("beta", (59.0, 16.0, 91.0, 44.0))]
    return d, words


def test_word_ink_is_owned_by_the_nearest_word():
    d, words = _diff_with_two_words()
    a, b = word_inks(d, words)
    assert a.box == (10.0, 20.0, 50.0, 40.0) and b.box == (60.0, 20.0, 90.0, 40.0)
    assert len(a.xs) == 19 * 40


def test_full_need_sides_and_excess():
    d, words = _diff_with_two_words()
    inks = word_inks(d, words)
    cut = measure(inks, [("alpha", (13.0, 16.0, 51.0, 44.0)), ("beta", (59.0, 16.0, 91.0, 44.0))], CFG)
    assert cut[0]["need_px"] == 2.5 and cut[0]["sides_px"][0] == 3.0 and cut[0]["recall"] < 1
    assert cut[1]["need_px"] == 0 and cut[1]["excess"] == 0
    wide = measure(inks, [("alpha", (5.0, 10.0, 70.0, 50.0))], CFG)
    assert wide[0]["need_px"] == 0 and wide[0]["excess"] == 200 / 760  # a third of 'beta' inside
    s = summarize(cut)
    assert s["horizontal"]["n"] == 2 and s["horizontal"]["full"] == 0.5


def test_text_ink_needs_the_second_render(tmp_path):
    img = np.full((10, 10, 3), 255, np.uint8)
    Image.fromarray(img).save(tmp_path / "a.png")
    assert text_ink(tmp_path / "a.png") is None
    img2 = img.copy()
    img2[2:4, 2:4] = 0
    Image.fromarray(img2).save(tmp_path / "b.png")
    Image.fromarray(img).save(tmp_path / "b.notext.png")
    d = text_ink(tmp_path / "b.png")
    assert d[2:4, 2:4].min() == 255 and d.sum() == 4 * 255
