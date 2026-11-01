import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from ocr.config import RedactConfig
from ocr.redact import erase, redact, text_colour, word_mask
from tests.test_box_refine import FONT, ink_box, render_line


def _line(fg="black", bg="white", words=("Jan", "Kowalski", "10.0.0.1")):
    img, inks = render_line(list(words), fg=fg, bg=bg)
    return img, [ink_box(i) for i in inks], inks


def _ink_left(before: Image.Image, after: Image.Image, ink, bg) -> int:
    ys, xs = ink
    a = np.asarray(after).astype(int)[ys, xs]
    return int((np.abs(a - np.array(bg)).max(axis=1) > 40).sum())


@pytest.mark.parametrize("method", ["fill", "inpaint", "box", "box_inpaint", "auto"])
def test_erased_words_leave_no_ink_and_other_words_stay(method):
    img, boxes, inks = _line()
    out, info = erase(img, boxes[:2], RedactConfig(method=method))
    for ink in inks[:2]:
        assert _ink_left(img, out, ink, (255, 255, 255)) == 0
    # the third word is untouched
    ys, xs = inks[2]
    assert np.array_equal(np.asarray(out)[ys, xs], np.asarray(img)[ys, xs])


def test_coloured_label_keeps_its_fill_and_gets_its_colour_back():
    img, boxes, inks = _line(fg="white", bg=(0, 120, 212))
    out, info = erase(img, boxes[:1], RedactConfig(method="fill"))
    assert info[0]["background"] == (0, 120, 212) and info[0]["colour"] == (255, 255, 255)
    assert _ink_left(img, out, inks[0], (0, 120, 212)) == 0


def test_text_on_a_gradient_is_inpainted_without_a_flat_patch():
    w, h = 600, 160
    grad = np.tile(np.linspace(80, 230, w).astype(np.uint8)[None, :, None], (h, 1, 3))
    img = Image.fromarray(grad)
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT, 32)
    d.text((200, 60), "secret", font=font, fill="black")
    box = d.textbbox((200, 60), "secret", font=font)
    out, info = erase(img, [box])  # default: the whole box rebuilt from its surroundings
    assert info[0]["method"] == "box_inpaint"
    a = np.asarray(out).astype(int)[box[1]:box[3], box[0]:box[2], 0]
    expect = grad[box[1]:box[3], box[0]:box[2], 0].astype(int)
    assert np.abs(a - expect).max() < 25  # the gradient goes on under the erased word


def test_replacement_is_written_in_the_words_colour():
    img, boxes, inks = _line(fg=(200, 0, 0))
    out = redact(img, [(boxes[1], "[NAME]")])
    a = np.asarray(out).astype(int)
    x0, y0, x1, y1 = (int(v) for v in boxes[1])
    region = a[y0:y1, x0:x1]
    reddish = (region[..., 0] > 150) & (region[..., 1] < 80)
    assert reddish.sum() > 20  # new text in red
    ys, xs = inks[0]
    assert np.array_equal(a[ys, xs], np.asarray(img).astype(int)[ys, xs])  # the other words stay


def test_mask_covers_the_word_and_its_antialiased_edge():
    img, boxes, inks = _line()
    mask, bg, plain, _ = word_mask(np.asarray(img), boxes[1], RedactConfig())
    ys, xs = inks[1]
    assert mask[ys, xs].all() and plain == 1.0 and tuple(int(v) for v in bg) == (255, 255, 255)
    assert text_colour(np.asarray(img), mask, bg) == (0, 0, 0)


def test_replacement_is_as_tall_as_the_erased_word():
    img, boxes, inks = _line(words=("Kowalski",))
    out = redact(img, [(boxes[0], "Nowak")])
    a = np.asarray(out).astype(int)
    ys, xs = np.nonzero((a < 128).all(axis=2))
    assert abs((ys.max() - ys.min() + 1) - (boxes[0][3] - boxes[0][1])) <= 2  # same ink height
    assert xs.min() >= boxes[0][0] - 1 and xs.max() <= boxes[0][2] + 1


def test_verified_erase_escalates_while_the_word_still_reads():
    from ocr.engines.base import OcrWord
    from ocr.redact import erase_verified

    class Reader:
        """Still reads the word for the first ``n`` re-readings (a faint outline left behind)."""
        def __init__(self, n):
            self.n = n

        def recognize(self, img, boxes):
            self.n -= 1
            return [OcrWord("Kowalski", boxes[0], 0.9, [0.9] * 8)] if self.n >= 0 else []

    img, boxes, inks = _line(words=("Kowalski", "Nowak"))
    words = [(boxes[0], "Kowalski"), (boxes[1], "")]  # no text: not checked
    for n, step in ((0, "ink"), (1, "box_inpaint"), (2, "box_fill")):
        out, info = erase_verified(img, words, Reader(n), RedactConfig())
        assert [i["step"] for i in info] == [step, "ink"]
    x0, y0, x1, y1 = (int(v) for v in boxes[0])
    assert (np.asarray(out)[y0:y1, x0:x1] == 255).all()  # box_fill: the box in the background colour
