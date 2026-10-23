import dataclasses
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ocr.box_refine import group_lines, refine_line, refine_words
from ocr.config import load_config
from ocr.engines.paddle_onnx import PaddleOnnxEngine

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
CFG = load_config().box_refine


def render_line(words, size=28, fg="black", bg="white", origin=(40, 40), canvas=(900, 160), img=None):
    """Draw the words on one line. Returns the image and, per word, the ink pixels (ys, xs)."""
    font = ImageFont.truetype(FONT, size)
    img = img or Image.new("RGB", canvas, bg)
    d = ImageDraw.Draw(img)
    x, y = origin
    inks = []
    for w in words:
        alone = Image.new("L", img.size, 0)
        ImageDraw.Draw(alone).text((x, y), w, font=font, fill=255)
        d.text((x, y), w, font=font, fill=fg)
        ys, xs = np.nonzero(np.asarray(alone) >= 128)
        inks.append((ys, xs))
        x += int(font.getlength(w + " "))
    return img, inks


def ink_box(ink):
    ys, xs = ink
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def line_quad(inks, pad_y=0.25, pad_x=0.2):
    """A detector-like line box: the union of the ink, widened (unclip)."""
    boxes = [ink_box(i) for i in inks]
    x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
    x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
    h = y1 - y0
    x0, x1, y0, y1 = x0 - pad_x * h, x1 + pad_x * h, y0 - pad_y * h, y1 + pad_y * h
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)


def ctc_like_priors(words, inks, quad, shift=0.35, cut=0.3):
    """RapidOCR-like word boxes: full line height, edges off by a fraction of a character."""
    out = []
    for w, ink in zip(words, inks):
        x0, _, x1, _ = ink_box(ink)
        cw = (x1 - x0) / len(w)
        out.append((w, (x0 + shift * cw, quad[0, 1], x1 - cut * cw, quad[2, 1])))
    return out


def covers(box, ink, tol=0.0):
    ys, xs = ink
    return bool(np.all((xs + 0.5 >= box[0] - tol) & (xs + 0.5 <= box[2] + tol)
                       & (ys + 0.5 >= box[1] - tol) & (ys + 0.5 <= box[3] + tol)))


def holds_any(box, ink):
    ys, xs = ink
    return bool(np.any((xs + 0.5 > box[0]) & (xs + 0.5 < box[2]) & (ys + 0.5 > box[1]) & (ys + 0.5 < box[3])))


def assert_snapped(img, words, inks, quad, priors=None, others=(), slack=2.0):
    priors = priors or ctc_like_priors(words, inks, quad)
    out = refine_line(np.asarray(img), quad, priors, CFG, others)
    for k, (r, ink) in enumerate(zip(out, inks)):
        assert r.source == "ink", (words[k], r.source)
        assert covers(r.bbox, ink), (words[k], r.bbox, ink_box(ink))
        ib = ink_box(ink)
        assert all(abs(a - b) <= slack for a, b in zip(r.bbox, ib)), (words[k], r.bbox, ib)
        for j, other in enumerate(inks):
            if j != k:
                assert not holds_any(r.bbox, other), (words[k], words[j])
    return out


def test_cut_ctc_boxes_are_snapped_to_the_whole_word():
    words = ["Hub", "virtual", "network", "10.0.0.1", "j.kowalski@example.com"]
    img, inks = render_line(words, canvas=(1300, 140))
    quad = line_quad(inks)
    priors = ctc_like_priors(words, inks, quad)
    assert not all(covers(b, ink) for (_, b), ink in zip(priors, inks))  # the priors do cut letters
    assert_snapped(img, words, inks, quad, priors)


def test_polish_diacritics_and_descenders_are_inside():
    words = ["Żółć", "gęślą", "jaźń", "ŹRÓDŁO"]
    img, inks = render_line(words, size=30)
    assert_snapped(img, words, inks, line_quad(inks, pad_y=0.1))


def test_tightly_stacked_lines_do_not_leak_into_each_other():
    font = ImageFont.truetype(FONT, 22)
    img = Image.new("RGB", (500, 140), "white")
    _, upper = render_line(["typography", "pipeline"], size=22, origin=(30, 30), img=img)
    _, lower = render_line(["deployment", "ŻÓŁTY"], size=22, origin=(30, 30 + font.getbbox("Ay")[3] + 1), img=img)
    q_up, q_low = line_quad(upper), line_quad(lower)
    assert q_low[0, 1] < q_up[2, 1]  # the detector-like boxes overlap
    out = assert_snapped(img, ["deployment", "ŻÓŁTY"], lower, q_low, others=[q_up])
    assert all(not holds_any(r.bbox, ink) for r in out for ink in upper)


def test_frames_rules_and_icons_are_not_part_of_words():
    img, inks = render_line(["Private", "endpoint"], canvas=(600, 160), origin=(80, 50))
    d = ImageDraw.Draw(img)
    boxes = [ink_box(i) for i in inks]
    x0, y0, x1, y1 = boxes[0][0], min(b[1] for b in boxes), boxes[1][2], max(b[3] for b in boxes)
    d.rectangle([x0 - 4, y0 - 4, x1 + 4, y1 + 4], outline="black")       # a frame close around
    d.line([(10, y1 + 7), (590, y1 + 7)], fill="black", width=1)          # a connector below
    d.rectangle([x0 - 34, y0, x0 - 12, y0 + 22], fill="black")            # an icon left of it
    d.line([(x1 + 4, (y0 + y1) / 2), (x1 + 60, (y0 + y1) / 2)], fill=(230, 120, 0), width=2)  # orange arrow
    assert_snapped(img, ["Private", "endpoint"], inks, line_quad(inks))


def test_light_text_on_dark_background():
    words = ["vm-weu-prod-001", "rg_shared_services"]
    img, inks = render_line(words, fg=(230, 230, 230), bg=(30, 30, 30))
    assert_snapped(img, words, inks, line_quad(inks))


def test_coloured_label_on_coloured_fill():
    words = ["Azure", "Firewall"]
    img, inks = render_line(words, fg="white", bg=(0, 120, 212))
    assert_snapped(img, words, inks, line_quad(inks))


def test_low_contrast_line_keeps_widened_prior():
    words = ["faint"]
    img, inks = render_line(words, fg=(250, 250, 250), bg="white")
    quad = line_quad([(np.array([40, 60]), np.array([40, 100]))])
    [r] = refine_line(np.asarray(img), quad, [("faint", (45.0, 40.0, 95.0, 60.0))], CFG)
    assert r.source.startswith("prior") and r.bbox[0] < 45.0 and r.bbox[2] > 95.0


def test_letter_dropped_by_the_recogniser_stays_in_the_box():
    img, [ink] = render_line(["Firewall"])
    quad = line_quad([ink])
    x0, _, x1, _ = ink_box(ink)
    cw = (x1 - x0) / 8
    prior = ("Firewal", (x0 + 0.3 * cw, quad[0, 1], x1 - 1.2 * cw, quad[2, 1]))  # last "l" not read
    [r] = refine_line(np.asarray(img), quad, [prior], CFG)
    assert r.source == "ink" and covers(r.bbox, ink)


def test_vertical_line():
    words = ["Capacities", "(1..n)"]
    img, inks = render_line(words, canvas=(500, 120))
    quad = line_quad(inks)
    priors = ctc_like_priors(words, inks, quad)
    W = img.width
    rot = img.transpose(Image.Transpose.ROTATE_90)  # text now runs bottom to top
    to_rot = lambda x, y: (y, W - x)  # noqa: E731
    rinks = [(W - 1 - xs, ys) for ys, xs in inks]  # (rows, cols) after the turn
    rquad = np.array([to_rot(*p) for p in quad])
    rquad = np.array([rquad[1], rquad[2], rquad[3], rquad[0]])  # tl, tr, br, bl of the tall box
    rpriors = [(w, (b[1], W - b[2], b[3], W - b[0])) for w, b in priors]
    out = refine_line(np.asarray(rot), rquad, rpriors, CFG)
    for r, ink in zip(out, rinks):
        assert r.source == "ink" and covers(r.bbox, ink)


def test_replayed_words_are_grouped_into_lines_and_snapped():
    font = ImageFont.truetype(FONT, 24)
    img = Image.new("RGB", (700, 200), "white")
    _, l1 = render_line(["Application", "Gateway"], size=24, origin=(30, 30), img=img)
    _, l2 = render_line(["10.1.2.4", "subnet"], size=24, origin=(30, 100), img=img)
    words = ctc_like_priors(["Application", "Gateway"], l1, line_quad(l1)) + \
        ctc_like_priors(["10.1.2.4", "subnet"], l2, line_quad(l2))
    assert sorted(map(sorted, group_lines([b for _, b in words]))) == [[0, 1], [2, 3]]
    for r, ink in zip(refine_words(np.asarray(img), words, CFG), l1 + l2):
        assert r.source == "ink" and covers(r.bbox, ink)


def test_paddle_engine_snaps_words_of_each_line(monkeypatch):
    words = ["Key", "Vault"]
    img, inks = render_line(words)
    quad = line_quad(inks)
    chars = []
    for w, ink in zip(words, inks):
        x0, _, x1, _ = ink_box(ink)
        cw = (x1 - x0) / len(w)
        for i, c in enumerate(w):  # RapidOCR-like character cells, shifted right
            cx0, cx1 = x0 + (i + 0.35) * cw, x0 + (i + 1.25) * cw
            chars.append((c, 0.99, [[cx0, quad[0, 1]], [cx1, quad[0, 1]], [cx1, quad[2, 1]], [cx0, quad[2, 1]]]))
    out = SimpleNamespace(txts=("Key Vault",), boxes=np.array([quad]), word_results=(tuple(chars),))
    cfg = load_config()
    e = PaddleOnnxEngine(dataclasses.replace(cfg.paddle, refine_boxes=True), cfg.box_refine)
    monkeypatch.setattr(e, "_run", lambda _img: out)
    got = e.read(img)
    assert [w.text for w in got] == words
    assert all(covers(w.bbox_px, ink) for w, ink in zip(got, inks))
    e_off = PaddleOnnxEngine(dataclasses.replace(cfg.paddle, refine_boxes=False), cfg.box_refine)
    monkeypatch.setattr(e_off, "_run", lambda _img: out)
    assert not all(covers(w.bbox_px, ink) for w, ink in zip(e_off.read(img), inks))


def test_other_lines_far_away_or_at_the_window_edge_are_harmless():
    words = ["Hub", "network"]
    img, inks = render_line(words)
    quad = line_quad(inks)
    h = quad[2, 1] - quad[0, 1]
    far = quad + [0, 3 * h]           # a line just below the search window
    edge = quad + [0, 1.9 * h]        # a line whose boundary falls at the window's edge
    huge = np.array([[0, 0], [900, 0], [900, 160], [0, 160]], dtype=np.float64)  # a block around it
    assert_snapped(img, words, inks, quad, others=[far, edge, huge])


def test_subpixel_rendered_text_keeps_its_thin_letters():
    # ClearType: orange and blue fringes left and right of each stroke. Without them the stem of
    # "L" fell off its foot and was taken for the side of a frame ("Linux" -> box from "i").
    words = ["Linux", "Iliad"]
    img, inks = render_line(words, size=13, canvas=(300, 60), origin=(20, 20))
    a = np.asarray(img).astype(int)
    mask = (a[..., 0] < 128).astype(int)
    out = np.full_like(a, 255)
    out[..., 0] -= 255 * np.roll(mask, 1, axis=1)   # red channel dark one pixel to the right
    out[..., 1] -= 255 * mask
    out[..., 2] -= 255 * np.roll(mask, -1, axis=1)  # blue channel dark one pixel to the left
    img = Image.fromarray(out.clip(0, 255).astype(np.uint8))
    inks = [(ys, xs) for ys, xs in inks]
    quad = line_quad(inks)
    for r, ink in zip(refine_line(np.asarray(img), quad, ctc_like_priors(words, inks, quad), CFG), inks):
        assert r.source == "ink" and covers(r.bbox, ink, tol=1.0), (r.bbox, ink_box(ink))


def test_glyph_missing_from_the_ink_does_not_shift_the_other_words():
    # A red "#" in black text is not text-coloured ink; the words after it keep their own glyphs.
    font = ImageFont.truetype(FONT, 28)
    img = Image.new("RGB", (500, 120), "white")
    d = ImageDraw.Draw(img)
    words, inks, x = ["#", "-", "Optional"], [], 40
    for w in words:
        alone = Image.new("L", img.size, 0)
        ImageDraw.Draw(alone).text((x, 40), w, font=font, fill=255)
        d.text((x, 40), w, font=font, fill="red" if w == "#" else "black")
        ys, xs = np.nonzero(np.asarray(alone) >= 128)
        inks.append((ys, xs))
        x += int(font.getlength(w + " "))
    quad = line_quad(inks)
    out = refine_line(np.asarray(img), quad, ctc_like_priors(words, inks, quad), CFG)
    for r, ink in zip(out, inks):
        assert covers(r.bbox, ink, tol=1.0), (r.source, r.bbox, ink_box(ink))
    assert out[2].source == "ink" and not holds_any(out[2].bbox, inks[1])
