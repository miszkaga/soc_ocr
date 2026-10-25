import pypdfium2 as pdfium

from ocr.pdf_text import inside, native_words
from tests.pdf_factory import diagram, new_canvas, to_pdf_y


def _words(path):
    pdf = pdfium.PdfDocument(str(path))
    try:
        page = pdf[0]
        l, _, _, t = page.get_bbox()
        return native_words(page, (l, t))
    finally:
        pdf.close()


def test_diagram_labels_are_words_with_exact_boxes(tmp_path):
    c = new_canvas(tmp_path / "d.pdf")
    region = diagram(c, 60, 300, labels=("Client app", "API 10.0.0.1", "DB"))
    c.save()
    words = _words(tmp_path / "d.pdf")
    assert [w.text for w in words] == ["Client", "app", "API", "10.0.0.1", "DB", "edge", "label"]
    client = words[0]
    # drawn at x 70 pt, baseline 325 pt from the top, 9 pt Helvetica: the glyphs sit just above it
    assert 69.5 < client.box[0] < 71.5 and 317 < client.box[1] < 320 and 324 < client.box[3] < 326.5
    assert all(inside(w, tuple(region)) for w in words)


def test_invisible_text_is_left_to_ocr_and_vertical_text_is_kept(tmp_path):
    c = new_canvas(tmp_path / "t.pdf")
    c.setFont("Helvetica", 10)
    t = c.beginText(60, to_pdf_y(100, 0))
    t.setTextRenderMode(3)  # the OCR layer of a scan
    t.textLine("hidden layer")
    t.setTextRenderMode(0)
    c.drawText(t)
    c.saveState()
    c.translate(400, 300)
    c.rotate(90)
    c.drawString(0, 0, "vertical label")
    c.restoreState()
    c.save()
    words = _words(tmp_path / "t.pdf")
    assert [w.text for w in words] == ["vertical", "label"]
    v = words[0].box
    assert v[3] - v[1] > v[2] - v[0]  # taller than wide
