from dataclasses import replace
from pathlib import Path

import pypdfium2 as pdfium
import pytest

from ocr.config import load_config
from ocr.pdf_regions import inventory, render_region
from tests.pdf_factory import (A4, body_text, diagram, new_canvas, solid_image, table,
                               to_pdf_y)

CFG = load_config().regions
TOL = 1.5  # pt; path bounds include half the stroke width


def approx_box(actual, expected, tol=TOL):
    return all(abs(a - e) <= tol for a, e in zip(actual, expected))


def kinds(inv, page=0):
    return [r.kind for r in inv.pages[page].regions]


def test_native_text_only_page_has_no_regions(tmp_path):
    c = new_canvas(tmp_path / "t.pdf"); body_text(c); c.save()
    inv = inventory(tmp_path / "t.pdf", CFG)
    p = inv.pages[0]
    assert p.regions == []
    assert p.native_chars > 500
    assert (p.width_pt, p.height_pt) == A4


def test_embedded_image_bbox_in_top_left_coordinates(tmp_path):
    c = new_canvas(tmp_path / "i.pdf")
    c.drawImage(solid_image("red"), 100, to_pdf_y(200, 150), 300, 150)
    c.save()
    [r] = inventory(tmp_path / "i.pdf", CFG).pages[0].regions
    assert r.kind == "embedded_image"
    assert approx_box(r.bbox_pt, [100, 200, 400, 350], tol=0.01)
    assert r.image_px == [200, 100]
    assert r.id == "p1r1"


def test_full_page_image_is_scanned_page(tmp_path):
    c = new_canvas(tmp_path / "s.pdf")
    c.drawImage(solid_image("white", (1240, 1754)), 0, 0, *A4)
    c.save()
    [r] = inventory(tmp_path / "s.pdf", CFG).pages[0].regions
    assert r.kind == "scanned_page"
    assert r.bbox_pt == [0, 0, A4[0], A4[1]]
    assert r.native_chars == 0


def test_scan_split_into_strips_is_still_scanned_page(tmp_path):
    c = new_canvas(tmp_path / "strips.pdf")
    for k in range(4):
        c.drawImage(solid_image("white"), 0, k * A4[1] / 4, A4[0], A4[1] / 4)
    c.save()
    assert kinds(inventory(tmp_path / "strips.pdf", CFG)) == ["scanned_page"]


def test_small_image_is_logo_and_tiny_image_is_ignored(tmp_path):
    c = new_canvas(tmp_path / "l.pdf")
    c.drawImage(solid_image("blue"), 40, to_pdf_y(30, 30), 60, 30)  # logo
    c.drawImage(solid_image("blue"), 300, to_pdf_y(300, 4), 4, 4)  # bullet, below min side
    c.save()
    assert kinds(inventory(tmp_path / "l.pdf", CFG)) == ["logo"]


def test_vector_diagram_includes_labels_outside_boxes(tmp_path):
    c = new_canvas(tmp_path / "d.pdf")
    expected = diagram(c, 60, 300)
    c.save()
    [r] = inventory(tmp_path / "d.pdf", CFG).pages[0].regions
    assert r.kind == "vector_diagram"
    # The caption under the boxes must be inside the region, otherwise OCR would miss it.
    assert r.bbox_pt[3] >= expected[3]
    assert approx_box(r.bbox_pt[:3], expected[:3])
    assert r.native_chars > 0


def test_table_of_rules_is_not_a_diagram(tmp_path):
    c = new_canvas(tmp_path / "tab.pdf")
    table(c, 60, 100)
    c.line(20, to_pdf_y(800, 0), A4[0] - 20, to_pdf_y(800, 0))  # page separator
    c.save()
    assert kinds(inventory(tmp_path / "tab.pdf", CFG)) == []


def test_shaded_table_and_callout_box_are_not_diagrams(tmp_path):
    # NIST SP 800-82r3 style: grey header row + rules, and a grey "recommendations" box.
    c = new_canvas(tmp_path / "shaded.pdf")
    c.setFillColorRGB(0.8, 0.8, 0.8); c.rect(60, to_pdf_y(100, 18), 300, 18, fill=1, stroke=0)
    c.setFillColorRGB(0, 0, 0); table(c, 60, 100)
    c.setFillColorRGB(0.9, 0.9, 0.9); c.rect(60, to_pdf_y(300, 120), 400, 120, fill=1, stroke=1)
    c.setFillColorRGB(0, 0, 0); body_text(c, lines=6, top=320)
    c.save()
    assert kinds(inventory(tmp_path / "shaded.pdf", CFG)) == []


def test_rounded_boxes_without_arrows_are_a_diagram(tmp_path):
    c = new_canvas(tmp_path / "round.pdf")
    for i in range(4):
        c.roundRect(60 + i * 100, to_pdf_y(300, 50), 90, 50, radius=8)  # 10 pt apart
    c.save()
    assert kinds(inventory(tmp_path / "round.pdf", CFG)) == ["vector_diagram"]


def test_page_image_under_visible_title_is_not_a_scan(tmp_path):
    # CERT Polska report style: section divider = full-bleed artwork + native title.
    c = new_canvas(tmp_path / "divider.pdf")
    c.drawImage(solid_image("black", (620, 877)), 0, 0, *A4)
    c.setFillColorRGB(1, 1, 1); c.setFont("Helvetica", 24); c.drawString(50, 750, "Incydenty")
    c.save()
    [r] = inventory(tmp_path / "divider.pdf", CFG).pages[0].regions
    assert r.kind == "embedded_image"
    assert r.bbox_pt == [0, 0, A4[0], A4[1]]


def test_scan_with_invisible_ocr_layer_is_still_scanned_page(tmp_path):
    c = new_canvas(tmp_path / "ocrlayer.pdf")
    c.drawImage(solid_image("white", (1240, 1754)), 0, 0, *A4)
    t = c.beginText(50, 750); t.setTextRenderMode(3); t.textLine("hidden OCR text layer"); c.drawText(t)
    c.save()
    [r] = inventory(tmp_path / "ocrlayer.pdf", CFG).pages[0].regions
    assert r.kind == "scanned_page"
    assert r.native_chars > 0  # the text layer is reported, so later stages can compare


def test_page_background_does_not_swallow_page(tmp_path):
    c = new_canvas(tmp_path / "bg.pdf")
    c.setFillColorRGB(0.95, 0.95, 0.95); c.rect(0, 0, *A4, fill=1, stroke=0); c.setFillColorRGB(0, 0, 0)
    diagram(c, 60, 300)
    c.save()
    [r] = inventory(tmp_path / "bg.pdf", CFG).pages[0].regions
    assert r.kind == "vector_diagram"
    assert r.bbox_pt[2] - r.bbox_pt[0] < A4[0] * 0.9


def test_image_inside_diagram_is_absorbed(tmp_path):
    c = new_canvas(tmp_path / "abs.pdf")
    expected = diagram(c, 60, 300)
    c.drawImage(solid_image("green", (32, 32)), 70, to_pdf_y(305, 12), 12, 12)  # icon in first box
    c.save()
    [r] = inventory(tmp_path / "abs.pdf", CFG).pages[0].regions
    assert r.kind == "vector_diagram"
    assert approx_box(r.bbox_pt[:3], expected[:3])


def test_image_in_transformed_form_xobject(tmp_path):
    c = new_canvas(tmp_path / "f.pdf")
    c.beginForm("F1"); c.drawImage(solid_image("red"), 10, 20, 100, 50); c.endForm()
    c.saveState(); c.translate(200, 300); c.scale(2, 2); c.doForm("F1"); c.restoreState()
    c.save()
    [r] = inventory(tmp_path / "f.pdf", CFG).pages[0].regions
    # Form space (10, 20)-(110, 70), scaled x2 and moved by (200, 300) -> PDF (220, 340)-(420, 440).
    assert r.kind == "embedded_image"
    assert approx_box(r.bbox_pt, [220, A4[1] - 440, 420, A4[1] - 340], tol=0.01)


def test_mediabox_offset_is_normalised(tmp_path):
    c = new_canvas(tmp_path / "m.pdf")
    c.drawImage(solid_image("red"), 150, 500, 100, 50)
    c.save()
    pdf = pdfium.PdfDocument(str(tmp_path / "m.pdf"))
    pdf[0].set_mediabox(100, 100, 500, 700)
    pdf.save(str(tmp_path / "m2.pdf")); pdf.close()
    p = inventory(tmp_path / "m2.pdf", CFG).pages[0]
    assert (p.width_pt, p.height_pt) == (400, 600)
    [r] = p.regions
    # PDF (150, 500)-(250, 550) relative to the box (100, 100)-(500, 700), top-left origin.
    assert approx_box(r.bbox_pt, [50, 150, 150, 200], tol=0.01)


def test_reading_order_of_region_ids(tmp_path):
    c = new_canvas(tmp_path / "o.pdf")
    c.drawImage(solid_image("red"), 320, to_pdf_y(400, 100), 200, 100)
    c.drawImage(solid_image("red"), 50, to_pdf_y(100, 100), 200, 100)
    c.drawImage(solid_image("red"), 50, to_pdf_y(400, 100), 200, 100)
    c.save()
    regions = inventory(tmp_path / "o.pdf", CFG).pages[0].regions
    assert [r.id for r in regions] == ["p1r1", "p1r2", "p1r3"]
    assert [r.bbox_pt[:2] for r in regions] == [[50, 100], [50, 400], [320, 400]]


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_render_region_matches_bbox_on_rotated_pages(tmp_path, rotation):
    c = new_canvas(tmp_path / "r.pdf")
    c.drawImage(solid_image("red"), 100, to_pdf_y(200, 150), 300, 150)
    c.save()
    pdf = pdfium.PdfDocument(str(tmp_path / "r.pdf"))
    pdf[0].set_rotation(rotation)
    pdf.save(str(tmp_path / "rot.pdf")); pdf.close()

    p = inventory(tmp_path / "rot.pdf", CFG).pages[0]
    assert p.rotation == rotation
    [r] = p.regions
    img = render_region(tmp_path / "rot.pdf", 1, r.bbox_pt, dpi=144)
    assert img.size == (600, 300)
    # Sample inside the margin; the crop must be the red image, not white page.
    px = img.getpixel((img.width // 2, img.height // 2))
    assert px[0] > 200 and px[1] < 60 and px[2] < 60


def test_inventory_is_deterministic(tmp_path):
    c = new_canvas(tmp_path / "det.pdf")
    diagram(c, 60, 300); c.drawImage(solid_image("red"), 50, 50, 200, 100); body_text(c, lines=5)
    c.save()
    assert inventory(tmp_path / "det.pdf", CFG) == inventory(tmp_path / "det.pdf", CFG)


def test_thresholds_come_from_config(tmp_path):
    c = new_canvas(tmp_path / "cfg.pdf")
    c.drawImage(solid_image("blue"), 40, to_pdf_y(30, 30), 60, 30)
    c.save()
    loose = replace(CFG, logo_max_page_fraction=0.0)
    assert kinds(inventory(tmp_path / "cfg.pdf", loose)) == ["embedded_image"]


def test_config_rejects_unknown_and_missing_keys(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text((Path(__file__).parent.parent / "config" / "default.toml").read_text()
                   .replace("cluster_gap_pt", "cluster_gap_typo"))
    with pytest.raises(ValueError, match="cluster_gap"):
        load_config(bad)


ARTIFACTS = Path(__file__).parent.parent / "artifacts" / "datasets" / "raw"


@pytest.mark.skipif(not (ARTIFACTS / "nist" / "NIST.SP.800-207.pdf").exists(),
                    reason="public sample PDFs not downloaded (artifacts/ is outside git)")
def test_public_pdf_smoke():
    inv = inventory(ARTIFACTS / "nist" / "NIST.SP.800-207.pdf", CFG)
    regions = [r for p in inv.pages for r in p.regions]
    assert len(inv.pages) == 59
    assert any(r.kind in ("embedded_image", "vector_diagram") for r in regions)
    for p in inv.pages:
        for r in p.regions:
            x0, y0, x1, y1 = r.bbox_pt
            assert 0 <= x0 < x1 <= p.width_pt and 0 <= y0 < y1 <= p.height_pt
