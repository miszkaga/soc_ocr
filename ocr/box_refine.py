"""Snap word boxes to the ink of the word.

PaddleOCR finds text *lines*. A word's box inside a line is an estimate from the recogniser's
CTC columns (RapidOCR CalRecBoxes): every character gets the line's mean pitch, centred on the
column where it was emitted, and the full height of the line box. No pixel is looked at, so the
boxes cut into the first or last letters of words and carry the line's slack above and below
(docs/boxes.md).

Here the engine's word boxes are only priors. Within each line:

1. take a strip around the detector's line quad, widened so that cut letters and diacritics
   fit, rectified so that the text runs along it (vertical lines are turned);
2. find the ink: the background is the commonest colour of the line box; a pixel is ink when it
   is a blend of the background and the text colour (antialiasing), far enough from the
   background; strong pixels seed glyphs, weak ones extend them. Long horizontal and vertical
   rules (frames, connectors, underlines) are removed; where a detected line above or below
   touches this one, the rows beyond the thinnest ink between them are blanked;
3. connected ink becomes "atoms" (a glyph or touching glyphs) when its centre lies in the middle
   of the line box; small marks above (accents, dots, the dot of Ż) or inside the box (_ , ;)
   join the glyph they belong to; a glyph merged with a shape is split by its strong pixels;
4. a small dynamic programme splits the atoms into the line's words: boundaries go where gaps
   are wide (spaces), near where the priors put them; atoms outside every prior (icons, frame
   edges at the line's ends) may be left out; unassigned atoms touching a word join it (letters
   the recogniser dropped still belong to the word's ink);
5. a word's box is the union of its atoms, grown into ink touching it (antialiasing, glyph parts
   removed with a rule), mapped back to the image.

A word keeps its prior box widened by a safety margin ("prior:<why>") when the line has too
little contrast, its atoms cannot be assigned, or the snapped box lands implausibly far from the
prior. Parameters: [box_refine] in config/default.toml.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ocr.config import BoxRefineConfig, load_config

Box = tuple[float, float, float, float]


@dataclass
class Refined:
    bbox: Box
    poly: np.ndarray  # 4x2 polygon in image coordinates
    source: str       # "ink" | "prior"


# ----------------------------------------------------------------------------- geometry

def _frame(quad: np.ndarray, cfg: BoxRefineConfig):
    """Affine map image -> strip for a line quad (tl, tr, br, bl): u runs along the text, v
    across it; the strip holds the quad widened by expand_along / expand_across line heights.
    Returns (A, strip_w, strip_h, line_length, line_height, du, dv), where (du, dv) is where
    the line box starts in the strip."""
    q = np.asarray(quad, dtype=np.float64)
    top = q[1] - q[0]
    left = q[3] - q[0]
    w, h = np.linalg.norm(top), np.linalg.norm(left)
    if h > 1.5 * w:  # vertical line: text runs along the quad's long side
        ex, ey = left / max(h, 1e-6), top / max(w, 1e-6)
    else:
        ex, ey = top / max(w, 1e-6), left / max(h, 1e-6)
    # Text in documents is horizontal or vertical far more often than the detector's slightly
    # tilted rectangles suggest: frames within axis_snap_deg of an axis are snapped to it, so
    # the strip is an exact copy of the pixels (no resampling).
    cos_snap = np.cos(np.radians(cfg.axis_snap_deg))
    for e in (ex, ey):
        k = int(np.argmax(np.abs(e)))
        if abs(e[k]) >= cos_snap:
            sign = 1.0 if e[k] >= 0 else -1.0
            e[:] = 0.0
            e[k] = sign
    us, vs = q @ ex, q @ ey
    length, height = us.max() - us.min(), vs.max() - vs.min()
    u_min = us.min() - cfg.expand_along * height
    v_min = vs.min() - cfg.expand_across * height
    if np.all(np.isin(np.abs(np.r_[ex, ey]), (0.0, 1.0))):
        u_min, v_min = np.floor(u_min), np.floor(v_min)  # exact pixel copy, no resampling
    A = np.array([[ex[0], ex[1], -u_min], [ey[0], ey[1], -v_min]])
    sw = int(np.ceil(us.max() + cfg.expand_along * height - u_min))
    sh = int(np.ceil(vs.max() + cfg.expand_across * height - v_min))
    return A, sw, sh, length, height, us.min() - u_min, vs.min() - v_min


def _to_strip(A, box: Box) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    pts = np.array([[x0, y0, 1], [x1, y0, 1], [x1, y1, 1], [x0, y1, 1]], dtype=np.float64)
    uv = pts @ A.T
    return uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()


def _from_strip(A, u0, v0, u1, v1) -> np.ndarray:
    Ainv = cv2.invertAffineTransform(A)
    pts = np.array([[u0, v0, 1], [u1, v0, 1], [u1, v1, 1], [u0, v1, 1]], dtype=np.float64)
    return pts @ Ainv.T


def _poly_box(poly: np.ndarray) -> Box:
    return (float(poly[:, 0].min()), float(poly[:, 1].min()), float(poly[:, 0].max()), float(poly[:, 1].max()))


def _warp(img: np.ndarray, A, sw, sh) -> np.ndarray:
    exact = np.all(np.isclose(np.abs(A[:, :2]), np.round(np.abs(A[:, :2])))) and \
        np.allclose(A[:, 2], np.round(A[:, 2]))
    flags = cv2.INTER_NEAREST if exact else cv2.INTER_LINEAR
    return cv2.warpAffine(img, A, (sw, sh), flags=flags, borderMode=cv2.BORDER_REPLICATE)


# ----------------------------------------------------------------------------- ink

def _mode_colour(pixels: np.ndarray) -> np.ndarray:
    q = (pixels >> 4).astype(np.int32)
    key = (q[:, 0] << 8) | (q[:, 1] << 4) | q[:, 2]
    k = np.bincount(key).argmax()
    return np.median(pixels[key == k], axis=0)


def _ink(strip: np.ndarray, core: tuple[int, int, int, int], cfg: BoxRefineConfig):
    """Weak and strong ink masks of the strip, or None when the line has too little contrast.
    core = (u0, v0, u1, v1) of the line box in the strip."""
    u0, v0, u1, v1 = core
    box_px = strip[v0:v1, u0:u1].reshape(-1, 3)
    if len(box_px) == 0:
        return None
    bg = _mode_colour(box_px)
    dist = np.abs(strip.astype(np.float32) - bg.astype(np.float32)).max(axis=2)
    h = v1 - v0
    # Contrast of the text: a high percentile of the distance over the ink in the middle of the
    # line box, after removing rules (a black frame line must not set the bar for grey text).
    provisional = _remove_rules(dist >= 30, 0.7 * h, cfg)
    b0, b1 = v0 + int(0.15 * h), max(v0 + int(0.15 * h) + 1, v1 - int(0.15 * h))
    cand = dist[b0:b1, u0:u1][provisional[b0:b1, u0:u1]]
    if cand.size < 5:
        return None
    contrast = float(np.percentile(cand, 90))
    if contrast < cfg.min_contrast:
        return None
    if not cfg.text_colour_model:
        strong = dist >= cfg.strong_ink * contrast
        weak = dist >= max(cfg.weak_ink * contrast, 12.0)
        return weak, strong
    # Text colour: the commonest colour among the darkest ink of the band. A pixel is ink when
    # it lies near the segment background -> text colour (antialiased text is a blend of the
    # two); orange connectors or blue icons next to black text are not.
    band_rgb = strip[b0:b1, u0:u1].reshape(-1, 3)
    core_ink = (provisional[b0:b1, u0:u1] & (dist[b0:b1, u0:u1] >= 0.8 * contrast)).reshape(-1)
    if core_ink.sum() < 3:
        return None
    text = _mode_colour(band_rgb[core_ink]).astype(np.float32)
    v = text - bg.astype(np.float32)
    vv = float(v @ v)
    if vv < 1.0:
        return None
    d = strip.astype(np.float32) - bg.astype(np.float32)
    alpha = (d @ v) / vv
    resid = np.linalg.norm(d - alpha[..., None] * v, axis=2) / np.sqrt(vv)
    on_line = resid <= cfg.colour_tolerance
    strong = on_line & (alpha >= cfg.strong_ink)
    weak = on_line & (alpha >= cfg.weak_ink) & (dist >= 12.0)
    if cfg.fringe_px > 0:
        # Subpixel-rendered text (ClearType) has orange and blue fringes off the background ->
        # text line; without them thin strokes fall apart ("L" into a stem and a foot, the stem
        # then taken for a frame). Pixels far enough from the background next to text ink count.
        k = 2 * cfg.fringe_px + 1
        near = cv2.dilate(weak.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
        weak = weak | (near & (dist >= max(cfg.weak_ink * contrast, 12.0)))
    return weak, strong


def _cut_other_lines(ink, A, core, others):
    """Separate this line from detected lines above and below it: the detector's boxes of
    stacked lines overlap and their glyphs may touch. A one-pixel barrier is cleared at the row
    with the least ink between the two lines (near the middle between their centres); pieces of
    the other line's glyphs then touch the barrier and are discarded, while accents of this line
    that float free of it are kept. Returns (weak, strong, barriers) with barrier rows or None."""
    weak, strong = ink
    u0c, v0c, u1c, v1c = core
    h = v1c - v0c
    mid = (v0c + v1c) / 2.0
    H = weak.shape[0]
    top = bottom = None
    for q in others:
        q = np.asarray(q, dtype=np.float64).reshape(-1, 2)
        uv = np.c_[q, np.ones(len(q))] @ A.T
        ou0, ov0, ou1, ov1 = uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()
        c0, c1 = int(max(ou0, u0c)), int(min(ou1, u1c))
        if c1 - c0 < 2 or ov1 - ov0 > 2.5 * h or ov0 <= mid <= ov1:
            continue
        oc = (ov0 + ov1) / 2.0
        if abs(oc - mid) < 0.5 * h or ov1 < 0 or ov0 > H:
            continue
        m = (oc + mid) / 2.0
        # search the half towards the other line: a cut through this line's own letters is worse
        # than leaving a descender tip of the other line
        lo_r, hi_r = (m - 0.3 * h, m + 0.05 * h) if oc < mid else (m - 0.05 * h, m + 0.3 * h)
        r0, r1 = int(max(0, lo_r)), int(min(H - 1, hi_r)) + 1
        if r1 - r0 < 1:
            continue  # the space between the lines lies outside the window
        prof = weak[r0:r1, c0:c1].sum(axis=1)
        best = np.flatnonzero(prof == prof.min())
        cut = r0 + int(best[np.argmin(np.abs(best + r0 - m))])
        if oc < mid:
            top = cut if top is None else max(top, cut)
        else:
            bottom = cut if bottom is None else min(bottom, cut)
    if top is None and bottom is None:
        return weak, strong, (None, None)
    weak, strong = weak.copy(), strong.copy()
    for r in (top, bottom):
        if r is not None:
            weak[r] = False
            strong[r] = False
    return weak, strong, (top, bottom)


def _remove_rules(mask: np.ndarray, th: float, cfg: BoxRefineConfig) -> np.ndarray:
    m = mask.astype(np.uint8)
    lh = max(3, int(round(cfg.rule_h_len * th)))
    lv = max(3, int(round(cfg.rule_v_len * th)))
    rules = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((1, lh), np.uint8)) | \
        cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((lv, 1), np.uint8))
    return mask & ~rules.astype(bool)


# ----------------------------------------------------------------------------- atoms

def _atoms(weak, strong, barriers, core, cfg: BoxRefineConfig):
    """Glyph atoms in strip coordinates: list of [u0, v0, u1, v1]; and the text height.

    A component is a glyph of this line when its vertical centre lies in the middle part of the
    detector's line box (glyphs of the lines above and below, which reach into the widened
    window or even into the box, are centred elsewhere). Small components above the glyphs
    (accents, dots, the dot of Ż) or inside the box (underscores, comma tails) are marks and join
    the glyph they belong to; small components below the box are left alone (they are usually
    accents of the next line).
    """
    u0c, v0c, u1c, v1c = core
    h = v1c - v0c
    mid = (v0c + v1c) / 2.0
    lo, hi = mid - cfg.glyph_centre_band * h, mid + cfg.glyph_centre_band * h
    n, lab, st, _ = cv2.connectedComponentsWithStats(weak.astype(np.uint8), connectivity=8)
    has_strong = np.zeros(n, bool)
    has_strong[np.unique(lab[strong])] = True

    def centred(i):
        return lo <= st[i, 1] + st[i, 3] / 2.0 <= hi

    hs = [st[i, 3] for i in range(1, n) if has_strong[i] and centred(i) and 0.25 * h <= st[i, 3] <= 1.2 * h
          and st[i, 2] <= 1.5 * h]
    th = float(np.percentile(hs, 75)) if hs else 0.7 * h
    # Frame sides are told from thin letters (l, I, the stem of L) by being taller than the
    # tallest letters, not than the typical (often lowercase) one.
    side_h = cfg.min_rule_height * max(th, float(np.percentile(hs, 95)) if hs else th)
    clean = _remove_rules(weak, th, cfg)
    n, lab, st, _ = cv2.connectedComponentsWithStats(clean.astype(np.uint8), connectivity=8)
    has_strong = np.zeros(n, bool)
    has_strong[np.unique(lab[strong & clean])] = True
    glyphs, small = [], []

    top_bar, bottom_bar = barriers

    def classify(x, y, w, hh):
        b = [float(x), float(y), float(x + w), float(y + hh)]
        beyond = (top_bar is not None and y + hh <= top_bar + 1) or (bottom_bar is not None and y >= bottom_bar)
        if beyond:  # the other line's side of a barrier: at most an accent of this line
            if hh <= cfg.max_mark_height * th and w <= 1.0 * th:
                small.append(b)
            return True
        if hh > cfg.max_glyph_height * th:
            return False  # icon, shape, frame corner
        if w <= max(2.0, cfg.max_rule_width * th) and hh >= side_h:
            return True   # the side of a frame drawn close around the text
        if lo <= y + hh / 2.0 <= hi:
            glyphs.append(b)
        elif hh <= cfg.max_mark_height * th and w <= 1.0 * th:
            small.append(b)
        return True

    for i in range(1, n):
        if not has_strong[i]:
            continue
        x, y, w, hh = st[i, :4]
        if classify(x, y, w, hh):
            continue
        # Too tall: a glyph touching a shape (a letter against an icon's rim, an arrow head)
        # joins it through antialiased pixels. Split it by its strong pixels and keep the parts
        # that look like glyphs of this line.
        part = (lab[y:y + hh, x:x + w] == i) & strong[y:y + hh, x:x + w]
        m2, _, st2, _ = cv2.connectedComponentsWithStats(part.astype(np.uint8), connectivity=8)
        for j in range(1, m2):
            xx, yy, ww, hh2 = st2[j, :4]
            classify(x + xx, y + yy, ww, hh2)
    sh = weak.shape[0]
    for m in small:
        if m[1] <= 0 or m[3] >= sh:
            continue  # cut by the window: part of a glyph of another line
        if (top_bar is not None and m[3] in (top_bar, top_bar + 1)) or \
                (bottom_bar is not None and m[1] in (bottom_bar, bottom_bar + 1)):
            continue  # touches a barrier: a piece cut from the other line's glyph
        above = None
        if m[1] >= v0c and m[3] <= v1c + 0.1 * th:
            above = False                                   # inside the line box: _ , ; tails
        best, best_score = None, None
        for g in glyphs:
            ov = min(m[2], g[2]) - max(m[0], g[0])
            if above is False:
                ok = ov > -0.3 * th
            else:                                           # accent or dot sitting on g
                gap = g[1] - m[3]
                ok = -0.15 * th <= gap <= cfg.max_mark_gap * th and ov >= 0.3 * (m[2] - m[0])
            if ok and (best_score is None or ov > best_score):
                best, best_score = g, ov
        if best is not None:
            best[0], best[1] = min(best[0], m[0]), min(best[1], m[1])
            best[2], best[3] = max(best[2], m[2]), max(best[3], m[3])
    glyphs.sort(key=lambda b: (b[0], b[2]))
    # merge glyphs stacked on each other (mostly the same columns)
    atoms: list[list[float]] = []
    for g in glyphs:
        if atoms:
            a = atoms[-1]
            ov = min(a[2], g[2]) - max(a[0], g[0])
            if ov > 0.5 * min(a[2] - a[0], g[2] - g[0]):
                a[0], a[1], a[2], a[3] = min(a[0], g[0]), min(a[1], g[1]), max(a[2], g[2]), max(a[3], g[3])
                continue
        atoms.append(list(g))
    # a frame side split by a connector crossing it comes back together above: drop it again
    atoms = [a for a in atoms if not (a[2] - a[0] <= max(2.0, cfg.max_rule_width * th)
                                      and a[3] - a[1] >= side_h)]
    return atoms, th


# ----------------------------------------------------------------------------- assignment

def _assign(atoms, priors, n_chars, th, cfg: BoxRefineConfig):
    """Contiguous atom ranges for the words (sorted along the line), or None.
    Returns a list of (s, e) per word, e inclusive."""
    n, m = len(atoms), len(priors)
    if n < m or m == 0:
        return None
    pitch = max(1.0, sum(p[1] - p[0] for p in priors) / max(1, sum(n_chars)))
    x0 = np.array([a[0] for a in atoms])
    x1 = np.array([a[2] for a in atoms])
    run_x1 = np.maximum.accumulate(x1)
    # gap before atom i (to the farthest right edge so far)
    gap_before = np.empty(n)
    gap_before[0] = np.inf
    gap_before[1:] = x0[1:] - run_x1[:-1]
    # skipping an atom is free outside every (slightly widened) prior, costly inside one
    slack = cfg.prior_slack_pitch * pitch
    skip = np.zeros(n)
    for i in range(n):
        w = max(1.0, x1[i] - x0[i])
        ov = sum(max(0.0, min(x1[i], p[1] + slack) - max(x0[i], p[0] - slack)) for p in priors)
        skip[i] = 10.0 * min(1.0, ov / w)
    skip_cum = np.concatenate([[0.0], np.cumsum(skip)])

    def skip_cost(a, b):  # atoms a..b-1
        return skip_cum[b] - skip_cum[a] if b > a else 0.0

    space = 0.35 * th
    inner_pen = np.maximum(0.0, gap_before - space) / th
    inner_pen[0] = 0.0
    inner_cum = np.concatenate([[0.0], np.cumsum(inner_pen)])

    def word_cost(k, s, e):
        p0, p1 = priors[k]
        c = cfg.w_prior * (_huber((x0[s] - p0) / pitch) + _huber((x1[s:e + 1].max() - p1) / pitch))
        c += cfg.w_inner_gap * (inner_cum[e + 1] - inner_cum[s + 1])  # gaps inside the word
        return c

    INF = 1e18
    # A word spans at most 2 atoms per character (+ a few for split glyphs and marks); at most
    # cfg.max_skip_between atoms are left out between two words. Both keep long lines fast.
    span = [2 * c + 4 for c in n_chars]
    K = cfg.max_skip_between
    f = np.full((m, n), INF)
    arg = np.full((m, n, 2), -1, dtype=np.int64)
    for e in range(n):
        for s in range(max(0, e - span[0] + 1), e + 1):
            c = skip_cost(0, s) + word_cost(0, s, e)
            if c < f[0, e]:
                f[0, e], arg[0, e] = c, (s, -1)
    for k in range(1, m):
        for e in range(k, n):
            best, best_arg = INF, (-1, -1)
            for s in range(max(k, e - span[k] + 1), e + 1):
                wc = word_cost(k, s, e)
                for ep in range(max(k - 1, s - 1 - K), s):
                    if f[k - 1, ep] >= INF:
                        continue
                    g = x0[s] - x1[ep]
                    c = f[k - 1, ep] + skip_cost(ep + 1, s) + wc - cfg.w_boundary_gap * min(max(g, 0.0), 1.5 * space) / th
                    if c < best:
                        best, best_arg = c, (s, ep)
            f[k, e], arg[k, e] = best, best_arg
    total = [f[m - 1, e] + skip_cost(e + 1, n) for e in range(n)]
    e = int(np.argmin(total))
    if total[e] >= INF:
        return None
    out = []
    for k in range(m - 1, -1, -1):
        s, ep = arg[k, e]
        out.append((int(s), int(e)))
        e = int(ep)
    return out[::-1]


def _shifted(atoms, groups, priors, pitch, cfg: BoxRefineConfig) -> bool:
    """Some word's glyphs lie outside its prior: the assignment slipped by a glyph (a glyph
    missing from the ink, like a red "#" in black text, moves every word after it)."""
    slack = cfg.shift_check_pitch * pitch
    for g, (p0, p1) in zip(groups, priors):
        c = (min(atoms[i][0] for i in g) + max(atoms[i][2] for i in g)) / 2
        if c < p0 - slack or c > p1 + slack:
            return True
    return False


def _by_prior(atoms, priors, pitch, th, cfg: BoxRefineConfig) -> list[list[int]]:
    """Each glyph to the word whose prior (widened a little) holds its centre, the nearest one
    when several do; glyphs outside every prior join the nearest word if they touch it.
    A word may end up with no glyph."""
    slack = cfg.prior_slack_pitch * pitch
    groups: list[list[int]] = [[] for _ in priors]
    for i, a in enumerate(atoms):
        c = (a[0] + a[2]) / 2
        holds = [j for j, (p0, p1) in enumerate(priors) if p0 - slack <= c <= p1 + slack]
        if holds:
            j = min(holds, key=lambda j: abs(c - (priors[j][0] + priors[j][1]) / 2))
        else:
            j = min(range(len(priors)), key=lambda j: max(priors[j][0] - a[2], a[0] - priors[j][1]))
            if max(priors[j][0] - a[2], a[0] - priors[j][1]) > cfg.letter_gap * th:
                continue
        groups[j].append(i)
    return groups


def _complete_words(atoms, ranges, th, cfg: BoxRefineConfig):
    """A word takes the unassigned atoms that touch it (closer than a space): letters the
    recogniser dropped ("Firewal" for Firewall) still belong to the word's ink."""
    used = set()
    for s, e in ranges:
        used.update(range(s, e + 1))
    out = []
    lim = cfg.letter_gap * th
    inter = [atoms[ranges[k + 1][0]][0] - max(a[2] for a in atoms[ranges[k][0]:ranges[k][1] + 1])
             for k in range(len(ranges) - 1)]
    inter = [g for g in inter if g > 0]
    if inter:
        lim = min(lim, 0.6 * min(inter))
    for s, e in ranges:
        while s - 1 >= 0 and (s - 1) not in used and atoms[s][0] - atoms[s - 1][2] < lim:
            s -= 1
            used.add(s)
        while e + 1 < len(atoms) and (e + 1) not in used and atoms[e + 1][0] - atoms[e][2] < lim:
            e += 1
            used.add(e)
        out.append((s, e))
    return out


def _grow(u0, v0, u1, v1, mask, r):
    """Grow a box by up to r px on each side while the next row/column holds ink: antialiased
    edges, and glyph parts that were removed together with a rule touching them."""
    H, W = mask.shape
    u0, v0, u1, v1 = int(np.floor(u0)), int(np.floor(v0)), int(np.ceil(u1)), int(np.ceil(v1))
    for _ in range(r):
        changed = False
        if v0 > 0 and mask[v0 - 1, u0:u1].any():
            v0 -= 1; changed = True
        if v1 < H and mask[v1, u0:u1].any():
            v1 += 1; changed = True
        if u0 > 0 and mask[v0:v1, u0 - 1].any():
            u0 -= 1; changed = True
        if u1 < W and mask[v0:v1, u1].any():
            u1 += 1; changed = True
        if not changed:
            break
    return float(u0), float(v0), float(u1), float(v1)


def _is_exact(A) -> bool:
    return bool(np.all(np.isclose(np.abs(A[:, :2]), np.round(np.abs(A[:, :2])))))


def _huber(x, d=0.5):
    a = abs(x)
    return 0.5 * a * a / d if a <= d else a - 0.5 * d


# ----------------------------------------------------------------------------- main

def refine_line(img: np.ndarray, quad, words: list[tuple[str, Box]], cfg: BoxRefineConfig | None = None,
                others=()) -> list[Refined]:
    """Snap the boxes of one line's words to their ink.

    img: H x W x 3 uint8 (RGB); quad: the detector's line quad (tl, tr, br, bl), or None to use
    the union of the word boxes; words: (text, prior box) in reading order; others: the quads of
    the image's other lines (glyphs of tightly stacked lines touch; the line between them is
    cut where the ink is thinnest).
    """
    if not words:
        return []
    cfg = cfg or load_config().box_refine
    if quad is None:
        x0 = min(b[0] for _, b in words); y0 = min(b[1] for _, b in words)
        x1 = max(b[2] for _, b in words); y1 = max(b[3] for _, b in words)
        quad = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)
    A, sw, sh, length, height, du, dv = _frame(np.asarray(quad, dtype=np.float64), cfg)

    def fallback(box: Box, pitch: float, why: str = "") -> Refined:
        mu = cfg.fallback_margin_pitch * pitch
        mv = cfg.fallback_margin_height * height
        u0, v0, u1, v1 = _to_strip(A, box)
        poly = _from_strip(A, u0 - mu, v0 - mv, u1 + mu, v1 + mv)
        b = _poly_box(poly)
        b = (max(0.0, b[0]), max(0.0, b[1]), min(float(img.shape[1]), b[2]), min(float(img.shape[0]), b[3]))
        return Refined(b, poly, "prior" + (":" + why if why else ""))

    n_chars = [max(1, len(t)) for t, _ in words]
    priors_uv = [_to_strip(A, b) for _, b in words]
    pitch = max(1.0, sum(p[2] - p[0] for p in priors_uv) / sum(n_chars))
    if sw <= 2 or sh <= 2 or sw * sh > 40_000_000:
        return [fallback(b, pitch, "size") for _, b in words]
    strip = _warp(img, A, sw, sh)
    core = (int(max(0, round(du))), int(max(0, round(dv))), int(min(sw, round(du + length))),
            int(min(sh, round(dv + height))))
    ink = _ink(strip, core, cfg)
    if ink is None:
        return [fallback(b, pitch, "contrast") for _, b in words]
    weak, strong, barriers = _cut_other_lines(ink, A, core, others)
    atoms, th = _atoms(weak, strong, barriers, core, cfg)
    order = sorted(range(len(words)), key=lambda k: (priors_uv[k][0] + priors_uv[k][2]) / 2)
    pri = [(priors_uv[k][0], priors_uv[k][2]) for k in order]
    nch = [n_chars[k] for k in order]
    ranges = _assign(atoms, pri, nch, th, cfg)
    groups = None
    if ranges is not None:
        ranges = _complete_words(atoms, ranges, th, cfg)
        groups = [list(range(s, e + 1)) for s, e in ranges]
        if _shifted(atoms, groups, pri, pitch, cfg):
            groups = _by_prior(atoms, pri, pitch, th, cfg)
    exact = _is_exact(A)
    extra = 0.0 if exact else cfg.rotated_margin * th
    grow = max(1, int(round(cfg.grow_into_ink * th)))
    out: list[Refined | None] = [None] * len(words)
    for j, k in enumerate(order):
        if groups is None:
            out[k] = fallback(words[k][1], pitch, "assign")
            continue
        if not groups[j]:
            out[k] = fallback(words[k][1], pitch, "noink")
            continue
        sel = [atoms[i] for i in groups[j]]
        u0, v0 = min(a[0] for a in sel), min(a[1] for a in sel)
        u1, v1 = max(a[2] for a in sel), max(a[3] for a in sel)
        u0, v0, u1, v1 = _grow(u0, v0, u1, v1, weak, grow)
        p = priors_uv[k]
        limit = cfg.max_shift_pitch * pitch + 0.5 * th
        if abs(u0 - p[0]) > limit or abs(u1 - p[2]) > limit:
            out[k] = fallback(words[k][1], pitch, "shift")
            continue
        m = cfg.margin_px + extra
        poly = _from_strip(A, u0 - m, v0 - m, u1 + m, v1 + m)
        b = _poly_box(poly)
        b = (max(0.0, b[0]), max(0.0, b[1]), min(float(img.shape[1]), b[2]), min(float(img.shape[0]), b[3]))
        out[k] = Refined(b, poly, "ink")
    return out  # type: ignore[return-value]


def group_lines(boxes: list[Box], min_v_overlap: float = 0.8) -> list[list[int]]:
    """Indices of word boxes grouped into lines, for words without their line quad (replayed
    runs). RapidOCR gives all words of a horizontal line the top and bottom of the line box, so
    words of one line overlap vertically almost entirely; a line is cut where the horizontal gap
    is wider than twice the line height."""
    order = sorted(range(len(boxes)), key=lambda i: (boxes[i][1], boxes[i][0]))
    lines: list[list[int]] = []
    for i in order:
        b = boxes[i]
        for line in lines:
            ref = boxes[line[-1]]
            ov = min(b[3], ref[3]) - max(b[1], ref[1])
            h = min(b[3] - b[1], ref[3] - ref[1])
            gap = b[0] - max(boxes[j][2] for j in line)
            if h > 0 and ov >= min_v_overlap * h and -h <= gap <= 2 * max(ref[3] - ref[1], 1.0):
                line.append(i)
                break
        else:
            lines.append([i])
    return [sorted(line, key=lambda i: boxes[i][0]) for line in lines]


def refine_words(img: np.ndarray, words: list[tuple[str, Box]], cfg: BoxRefineConfig | None = None) -> list[Refined]:
    """refine_line for words whose line quads are unknown: lines are rebuilt with group_lines
    and each line's box (the union of its words) stands for the quad."""
    cfg = cfg or load_config().box_refine
    lines = group_lines([b for _, b in words])
    union = [(min(words[i][1][0] for i in ln), min(words[i][1][1] for i in ln),
              max(words[i][1][2] for i in ln), max(words[i][1][3] for i in ln)) for ln in lines]
    quads = [np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64) for x0, y0, x1, y1 in union]
    out: list[Refined | None] = [None] * len(words)
    for k, ln in enumerate(lines):
        others = quads[:k] + quads[k + 1:]
        for i, r in zip(ln, refine_line(img, quads[k], [words[i] for i in ln], cfg, others)):
            out[i] = r
    return out  # type: ignore[return-value]
