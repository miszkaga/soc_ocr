import shutil

import pytest
from PIL import Image, ImageDraw, ImageFont

from eval.metrics import overlap
from ocr.engines.tesseract import TesseractEngine, parse_hocr

needs_tesseract = pytest.mark.skipif(shutil.which("tesseract") is None, reason="tesseract not installed")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def text_image(words, size=32):
    """White image with the words on one line; returns the image and each word's box."""
    font = ImageFont.truetype(FONT, size)
    img = Image.new("RGB", (60 + sum(int(font.getlength(w + " ")) for w in words), size * 3), "white")
    d = ImageDraw.Draw(img)
    x, boxes = 30, []
    for w in words:
        d.text((x, size), w, font=font, fill="black")
        boxes.append(d.textbbox((x, size), w, font=font))
        x += int(font.getlength(w + " "))
    return img, boxes


def test_parse_hocr_char_confidences_and_offset():
    hocr = """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><body>
 <span class='ocrx_word' id='word_1_1' title='bbox 10 20 50 40; x_wconf 91'>
  <span class='ocrx_cinfo' title='x_bboxes 10 20 20 40; x_conf 99.5'>A</span><span class='ocrx_cinfo' title='x_bboxes 20 20 30 40; x_conf 80'>&amp;</span><span class='ocrx_cinfo' title='x_bboxes 30 20 50 40; x_conf 90'>B</span>
 </span>
 <span class='ocrx_word' id='word_1_2' title='bbox 60 20 70 40; x_wconf 50'> </span>
</body></html>"""
    [w] = parse_hocr(hocr, offset=(100, 5))
    assert w.text == "A&B" and w.conf == 0.91
    assert w.char_conf == pytest.approx([0.995, 0.8, 0.9])
    assert w.bbox_px == (110, 25, 150, 45)


@needs_tesseract
def test_read_hard_strings_with_char_confidences():
    words = ["Zażółć", "j.kowalski@example.com", "10.0.0.1"]
    img, boxes = text_image(words)
    got = TesseractEngine().read(img)
    assert [w.text for w in got] == words
    for w, box in zip(got, boxes):
        assert 0.5 < w.conf <= 1.0
        assert len(w.char_conf) == len(w.text) and all(0 <= c <= 1 for c in w.char_conf)
        assert overlap(w.bbox_px, box) > 0.9


@needs_tesseract
def test_recognize_given_boxes():
    words = ["Azure", "Firewall", "10.100.0.132"]
    img, boxes = text_image(words)
    got = TesseractEngine().recognize(img, boxes)
    assert [w.text for w in got] == words
    assert abs(got[2].bbox_px[0] - boxes[2][0]) <= 4  # coordinates mapped back from the crop


@needs_tesseract
def test_versions_are_reported():
    v = TesseractEngine().versions()
    assert v["tesseract"].startswith("5.") and v["lang"] == "pol+eng"
