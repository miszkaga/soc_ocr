import json
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from eval.svg_ground_truth import build, iter_sources, latin_fraction, launch_browser, render_svg
from ocr.config import load_config

CFG = load_config().svg_gt


@pytest.fixture(scope="module")
def browser():
    sync_playwright = pytest.importorskip("playwright.sync_api").sync_playwright
    with sync_playwright() as p:
        try:
            b = launch_browser(p)
        except Exception as e:  # browser binaries not installed
            pytest.skip(f"chromium not available: {e}")
        yield b
        b.close()


def svg(tmp_path, body, attrs='width="400" height="200" viewBox="0 0 200 100"', name="t.svg"):
    p = tmp_path / name
    p.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" {attrs}>{body}</svg>', encoding="utf-8")
    return p


def render(browser, path, cfg=CFG):
    return render_svg(browser, path, path.with_suffix(".png"), cfg)


def test_words_polish_text_and_tspans(browser, tmp_path):
    p = svg(tmp_path, '<text x="10" y="20" font-size="10">Zażółć gęślą <tspan font-weight="bold">jaźń</tspan></text>'
                      '<text x="10" y="50" font-size="10">j.kowalski@example.com 10.0.0.1</text>')
    gt = render(browser, p)
    assert [w.text for w in gt.words] == ["Zażółć", "gęślą", "jaźń", "j.kowalski@example.com", "10.0.0.1"]
    assert [w.group for w in gt.words] == [0, 0, 0, 1, 1]
    # 400x200 CSS px, scale clamped to max_scale (4) -> 1600x800 image.
    assert (gt.width_px, gt.height_px, gt.scale) == (1600, 800, 4.0)
    assert Image.open(p.with_suffix(".png")).size == (1600, 800)


def test_boxes_match_rendered_pixels(browser, tmp_path):
    p = svg(tmp_path, '<rect width="200" height="100" fill="white"/>'
                      '<text x="20" y="40" font-size="12" fill="black">ALPHA</text>'
                      '<text x="120" y="80" font-size="12" fill="black">omega</text>')
    gt = render(browser, p)
    img = np.asarray(Image.open(p.with_suffix(".png")).convert("L"))
    ink = img < 128
    for w in gt.words:
        x0, y0, x1, y1 = (round(v) for v in w.bbox_px)
        assert ink[y0:y1, x0:x1].sum() > 50, w.text  # glyphs are drawn inside the box
    # Every dark pixel lies inside some word box: nothing drawn is missing from the ground truth.
    mask = np.zeros_like(ink)
    for w in gt.words:
        x0, y0, x1, y1 = (round(v) for v in w.bbox_px)
        mask[max(0, y0 - 2):y1 + 2, max(0, x0 - 2):x1 + 2] = True
    assert not (ink & ~mask).any()


def test_hidden_text_is_skipped(browser, tmp_path):
    p = svg(tmp_path, '<title>title text</title><desc>desc text</desc>'
                      '<defs><text id="d" x="0" y="10">in defs</text></defs>'
                      '<text x="10" y="20" display="none">display none</text>'
                      '<text x="10" y="30" visibility="hidden">visibility hidden</text>'
                      '<text x="10" y="40" opacity="0">opacity zero</text>'
                      '<g opacity="0"><text x="10" y="50">group opacity</text></g>'
                      '<text x="10" y="60" fill="none" stroke="none">no paint</text>'
                      '<text x="10" y="70">shown</text>'
                      '<text x="500" y="70">off canvas</text>')
    assert [w.text for w in render(browser, p).words] == ["shown"]


def test_rotated_text_is_flagged(browser, tmp_path):
    p = svg(tmp_path, '<text x="50" y="50" font-size="8" transform="rotate(-30 50 50)">tilted</text>'
                      '<text x="10" y="90" font-size="8">level</text>')
    words = {w.text: w for w in render(browser, p).words}
    assert words["tilted"].rotated and not words["level"].rotated


def test_html_in_foreign_object(browser, tmp_path):
    p = svg(tmp_path, '<foreignObject x="10" y="10" width="180" height="40">'
                      '<div xmlns="http://www.w3.org/1999/xhtml" style="font-size:8px">api.example.com '
                      '<b>Łódź</b></div></foreignObject>')
    words = render(browser, p).words
    assert [w.text for w in words] == ["api.example.com", "Łódź"]
    assert words[0].font_px == pytest.approx(8 * 2 * 4, rel=0.01)  # 8 user px x viewBox 2 x scale 4


def test_size_from_viewbox_when_width_missing(browser, tmp_path):
    p = svg(tmp_path, '<text x="10" y="50" font-size="10">only viewBox</text>', attrs='viewBox="0 0 1000 500"')
    gt = render(browser, p)
    assert (gt.width_px, gt.height_px) == (2000, 1000)  # long side scaled to target 2000
    assert len(gt.words) == 2


def test_tiny_text_is_marked_illegible(browser, tmp_path):
    p = svg(tmp_path, '<text x="10" y="20" font-size="10">big</text><text x="10" y="40" font-size="0.5">tiny</text>')
    words = {w.text: w for w in render(browser, p).words}
    assert words["big"].legible and not words["tiny"].legible


def test_external_resources_are_not_fetched(browser, tmp_path):
    p = svg(tmp_path, '<image href="http://192.0.2.1/x.png" width="10" height="10"/>'
                      '<style>@import url("http://192.0.2.1/f.css");</style><text x="10" y="50">offline</text>')
    assert [w.text for w in render(browser, p).words] == ["offline"]


def test_rendering_is_deterministic(browser, tmp_path):
    p = svg(tmp_path, '<text x="10" y="20" font-size="10">same every time</text>')
    a = render(browser, p)
    png_a = p.with_suffix(".png").read_bytes()
    b = render(browser, p)
    assert a.words == b.words and png_a == p.with_suffix(".png").read_bytes()


def test_build_indexes_sources_and_skips_textless_svgs(browser, tmp_path):
    raw = tmp_path / "raw"
    (raw / "azure-architecture-center" / "docs").mkdir(parents=True)
    (raw / "kubernetes-website").mkdir()
    (raw / "wikimedia-commons" / "files").mkdir(parents=True)
    svg(raw / "azure-architecture-center" / "docs", '<text x="10" y="20">Azure label</text>', name="a.svg")
    svg(raw / "kubernetes-website", '<rect width="10" height="10"/>', name="no_text.svg")
    svg(raw / "wikimedia-commons" / "files", '<text x="10" y="20">Sieć lokalna</text>', name="c.svg")
    (raw / "wikimedia-commons" / "metadata.jsonl").write_text(json.dumps(
        {"file": "files/c.svg", "mime": "image/svg+xml", "license": "CC BY-SA 4.0"}) + "\n")

    assert [(s, p.name, lic) for s, p, lic in iter_sources(raw)] == [
        ("azure-architecture-center", "a.svg", "CC BY 4.0"), ("wikimedia-commons", "c.svg", "CC BY-SA 4.0")]
    index = build(tmp_path / "out", raw=raw, browser=browser)
    assert [(r["source"], r["words"], r["polish"]) for r in index] == [
        ("azure-architecture-center", 2, False), ("wikimedia-commons", 2, True)]
    gt = json.loads((tmp_path / "out" / index[1]["json"]).read_text(encoding="utf-8"))
    assert gt["license"] == "CC BY-SA 4.0" and gt["words"][0]["text"] == "Sieć"


def test_html_cut_off_by_its_foreign_object_is_marked_clipped(browser, tmp_path):
    p = svg(tmp_path, '<foreignObject x="10" y="10" width="40" height="20">'
                      '<div xmlns="http://www.w3.org/1999/xhtml" style="font-size:8px;white-space:nowrap">'
                      'fits overflowingword</div></foreignObject>')
    words = {w.text: w for w in render(browser, p).words}
    assert not words["fits"].clipped and words["overflowingword"].clipped


def test_missing_windows_fonts_render_with_substitutes(browser, tmp_path):
    # Azure diagrams ask for Segoe UI; it must render as Selawik, not the default serif.
    p = svg(tmp_path, '<text x="10" y="20" font-size="10" font-family="Segoe UI">Firewall subnet</text>'
                      '<text x="10" y="50" font-size="10" font-family="Selawik">Firewall subnet</text>'
                      '<text x="10" y="80" font-size="10" font-family="serif">Firewall subnet</text>')
    gt = render(browser, p)
    img = np.asarray(Image.open(p.with_suffix(".png")).convert("L"))
    bands = [img[int(w.bbox_px[1]):int(w.bbox_px[1]) + 60, 0:900] for w in gt.words[0::2]]
    assert np.array_equal(bands[0], bands[1])  # Segoe UI is drawn exactly as Selawik
    assert not np.array_equal(bands[1], bands[2])  # ... and Selawik is not the serif fallback


def test_latin_fraction():
    assert latin_fraction("Zażółć gęślą jaźń 10.0.0.1") == 1.0
    assert latin_fraction("クラスター Pod") < 0.5
    assert latin_fraction("123 -> 456") == 1.0


def test_build_skips_non_latin_diagrams(browser, tmp_path):
    raw = tmp_path / "raw"
    (raw / "kubernetes-website").mkdir(parents=True)
    svg(raw / "kubernetes-website", '<text x="10" y="20">クラスター クライアント Pod</text>', name="ja.svg")
    svg(raw / "kubernetes-website", '<text x="10" y="20">Cluster client Pod</text>', name="en.svg")
    index = build(tmp_path / "out", raw=raw, browser=browser)
    assert [r["words"] for r in index] == [3]
    assert not list((tmp_path / "out" / "kubernetes-website").glob("ja_*"))


def test_build_skips_duplicate_diagrams(browser, tmp_path):
    raw = tmp_path / "raw"
    for lang in ("en", "bn"):
        d = raw / "kubernetes-website" / "content" / lang
        d.mkdir(parents=True)
        svg(d, '<text x="10" y="20">Ingress service1:80</text>', name="ingress.svg")
    assert len(build(tmp_path / "out", raw=raw, browser=browser)) == 1


def test_word_split_across_tspans_is_joined(browser, tmp_path):
    # Visio-style export: first letter in its own tspan, no whitespace in between.
    p = svg(tmp_path, '<text x="10" y="20" font-size="10"><tspan>S</tspan><tspan>ubnet</tspan> '
                      '<tspan>a</tspan><tspan>ccount</tspan></text>')
    assert [w.text for w in render(browser, p).words] == ["Subnet", "account"]


def test_label_lines_without_whitespace_are_not_joined(browser, tmp_path):
    p = svg(tmp_path, '<text font-size="10"><tspan x="10" y="20">Workload</tspan>'
                      '<tspan x="10" y="32">subnet</tspan></text>')
    assert [w.text for w in render(browser, p).words] == ["Workload", "subnet"]


def _placed(x, word, width):
    # textLength fixes the rendered width, so the test does not depend on the font.
    return (f'<text x="{x}" y="20" font-size="10" textLength="{width}" '
            f'lengthAdjust="spacingAndGlyphs">{word}</text>')


def test_touching_text_elements_are_one_word(browser, tmp_path):
    # PowerPoint export: one <text> per piece, positioned to touch.
    p = svg(tmp_path, _placed(10, "On", 12) + _placed(22, "-", 4) + _placed(26, "premises", 40))
    assert [w.text for w in render(browser, p).words] == ["On-premises"]


def test_text_elements_a_space_apart_stay_separate(browser, tmp_path):
    p = svg(tmp_path, _placed(10, "Key", 17) + _placed(30.5, "Vault", 25))  # 3.5 units = 0.35 em
    assert [w.text for w in render(browser, p).words] == ["Key", "Vault"]


def test_text_in_background_colour_is_dropped_and_separates_words(browser, tmp_path):
    # Table "columns" padded with white dashes on a white background.
    p = svg(tmp_path, '<rect width="200" height="100" fill="white"/>' + _placed(10, "Stgworkload00", 60)
            + '<text x="70" y="20" font-size="10" fill="white" textLength="20" '
              'lengthAdjust="spacingAndGlyphs">-----</text>' + _placed(90, "A", 6))
    assert [w.text for w in render(browser, p).words] == ["Stgworkload00", "A"]


def test_icon_font_characters_are_removed(browser, tmp_path):
    p = svg(tmp_path, '<text x="10" y="20" font-size="10">\uf0e0 SD-WAN \uf0e0</text>')
    assert [w.text for w in render(browser, p).words] == ["SD-WAN"]

def test_background_coloured_text_is_dropped_even_when_a_line_crosses_it(browser, tmp_path):
    p = svg(tmp_path, '<rect width="200" height="100" fill="white"/>' + _placed(10, "Name", 25)
            + '<text x="35" y="20" font-size="10" fill="#ffffff" textLength="30" '
              'lengthAdjust="spacingAndGlyphs">------</text>' + _placed(65, "Type", 22)
            + '<line x1="5" y1="22" x2="195" y2="22" stroke="#4472c4" stroke-width="1"/>')
    assert [w.text for w in render(browser, p).words] == ["Name", "Type"]


def test_large_bold_text_is_visible(browser, tmp_path):
    # The glyphs fill most of the box; visibility must not be judged by the box's median colour.
    p = svg(tmp_path, '<text x="5" y="60" font-size="60" font-weight="900" fill="#000">WM</text>')
    assert [w.text for w in render(browser, p).words] == ["WM"]
