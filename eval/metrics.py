"""OCR quality metrics (CLAUDE.md, "Miary i cel").

Per image we match predicted words to ground-truth words by box overlap, then count:

- detection completeness: ground-truth words covered by some prediction. A missed word
  never gets a flag, so this matters most;
- CER / WER over ground-truth words: a matched word costs its edit distance, a missed word
  costs all its characters; reported overall and per category of hard strings;
- calibration (ECE) of word confidence against "the word is exactly right", where
  predictions that match nothing (hallucinated text) count as wrong;
- the accepted/review trade-off: for a confidence threshold, the CER of accepted words and
  the share of ground-truth words left for a human.

Ground-truth words marked ``ignore`` (illegible, clipped) are neither scored nor penalise
predictions that land on them. Counts are additive, so results from many images (or a
subset, e.g. one source) are summed with :func:`aggregate`.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from rapidfuzz.distance import Levenshtein

from ocr.config import EvalConfig

Box = tuple[float, float, float, float]

CATEGORIES = ("all", "hard", "email", "host", "ip", "number", "id", "polish", "rotated")
_EMAIL = re.compile(r"^[\w.+-]+@[\w-]+(\.[\w-]+)+$")
_IPV4 = re.compile(r"^\d{1,3}(\.\d{1,3}){3}(/\d{1,2})?(:\d+)?$")
_IPV6 = re.compile(r"^[0-9a-fA-F]{0,4}(:[0-9a-fA-F]{0,4}){2,7}(/\d{1,3})?$")
_HOST = re.compile(r"^([a-z0-9-]+\.)+[a-z]{2,}(:\d+)?(/\S*)?$|^\w+://\S+$", re.I)
_NUMBER = re.compile(r"^[+-]?\d[\d.,:/%x-]*$")
_POLISH = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")
_STRIP = ",;:.!?()[]{}\"'"


@dataclass
class GtWord:
    text: str
    bbox: Box
    ignore: bool = False
    rotated: bool = False


@dataclass
class PredWord:
    text: str
    bbox: Box
    conf: float
    status: str | None = None  # "accepted" / "uncertain" from consensus; None for single engines


def categories(word: GtWord) -> set[str]:
    """Categories of a ground-truth word; "hard" = any of the difficult string types."""
    t = word.text.strip(_STRIP) or word.text
    cats = {"all"}
    if _EMAIL.match(t):
        cats.add("email")
    elif _IPV4.match(t) or (t.count(":") >= 2 and _IPV6.match(t)):
        cats.add("ip")
    elif _HOST.match(t) and not t.replace(".", "").isdigit():
        cats.add("host")
    elif _NUMBER.match(t):
        cats.add("number")
    elif any(c.isalpha() for c in t) and any(c.isdigit() for c in t):
        cats.add("id")
    if any(c in _POLISH for c in t):
        cats.add("polish")
    if cats & {"email", "ip", "host", "number", "id", "polish"}:
        cats.add("hard")
    if word.rotated:
        cats.add("rotated")
    return cats


# --- geometry -------------------------------------------------------------------

def _area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a: Box, b: Box) -> float:
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def overlap(a: Box, b: Box) -> float:
    """Intersection over the smaller box: 1 when a tight box lies inside a line-height box."""
    small = min(_area(a), _area(b))
    return _inter(a, b) / small if small > 0 else 0.0


def iou(a: Box, b: Box) -> float:
    i = _inter(a, b)
    u = _area(a) + _area(b) - i
    return i / u if u > 0 else 0.0


def coverage(gt: Box, preds: list[Box]) -> float:
    """Fraction of the ground-truth word's length (its long axis) covered by predictions
    that overlap it across the short axis by at least half of the smaller extent."""
    horizontal = (gt[2] - gt[0]) >= (gt[3] - gt[1])
    a0, a1, b0, b1 = (0, 2, 1, 3) if horizontal else (1, 3, 0, 2)
    length = gt[a1] - gt[a0]
    if length <= 0:
        return 0.0
    spans = []
    for p in preds:
        cross = min(gt[b1], p[b1]) - max(gt[b0], p[b0])
        if cross < 0.5 * min(gt[b1] - gt[b0], p[b1] - p[b0]):
            continue
        s, e = max(gt[a0], p[a0]), min(gt[a1], p[a1])
        if e > s:
            spans.append((s, e))
    covered, end = 0.0, float("-inf")
    for s, e in sorted(spans):
        if e > end:
            covered += e - max(s, end)
            end = e
    return covered / length


# --- per-image evaluation -----------------------------------------------------------

@dataclass
class Counts:
    words: int = 0
    detected: int = 0
    chars: int = 0
    edits: int = 0
    exact: int = 0
    # Box accuracy (detected words with visible ink): how much of the word's ink lies inside
    # the boxes of the readings it was matched to.
    box_words: int = 0
    box_full: int = 0  # ink coverage >= box_full_coverage
    box_coverage: float = 0.0  # sum of coverages

    def add(self, other: "Counts"):
        for k in ("words", "detected", "chars", "edits", "exact", "box_words", "box_full", "box_coverage"):
            setattr(self, k, getattr(self, k) + getattr(other, k))


@dataclass
class ImageResult:
    counts: dict[str, Counts] = field(default_factory=lambda: defaultdict(Counts))
    # One entry per scored prediction: (confidence, exactly right, edits, gt chars, accepted).
    # Hallucinated predictions have gt chars 0 and edits = len(prediction); accepted is the
    # consensus status (None for single engines).
    scored: list[tuple[float, bool, int, int, bool | None]] = field(default_factory=list)
    false_positives: int = 0
    errors: list[dict] = field(default_factory=list)
    # Per scored ground-truth word: dict(gt index, outcome, ...) - used by error analysis.
    words: list[dict] = field(default_factory=list)
    # Groups where the readings disagree with the ground truth - used by adjudication.
    disagreements: list[dict] = field(default_factory=list)


def match(gt: list[GtWord], pred: list[PredWord], cfg: EvalConfig) -> list[tuple[int, int]]:
    """One-to-one pairs (gt index, pred index), greedily by IoU among overlapping boxes."""
    cands = []
    for gi, g in enumerate(gt):
        for pi, p in enumerate(pred):
            if overlap(g.bbox, p.bbox) >= cfg.match_min_overlap:
                cands.append((iou(g.bbox, p.bbox), gi, pi))
    pairs, used_g, used_p = [], set(), set()
    for _, gi, pi in sorted(cands, reverse=True):
        if gi not in used_g and pi not in used_p:
            pairs.append((gi, pi))
            used_g.add(gi)
            used_p.add(pi)
    return sorted(pairs)


_TYPOGRAPHY = str.maketrans({"’": "'", "‘": "'", "′": "'", "“": '"', "”": '"', "„": '"', "″": '"',
                             "–": "-", "—": "-", "−": "-", "‐": "-", "…": "..."})


def scored_text(text: str, cfg: EvalConfig) -> str:
    """The form in which a word is scored: unified typography, no punctuation at its edges,
    nothing at all for a word without letters and digits (with require_alnum)."""
    if cfg.normalize_typography:
        text = text.translate(_TYPOGRAPHY)
    text = text.strip(cfg.strip_edge_chars + " ")
    if cfg.require_alnum and not any(ch.isalnum() for ch in text):
        return ""  # symbols only ("<>", "++", "—"): never anonymised
    return text


def _scored(gt: list[GtWord], pred: list[PredWord], cfg: EvalConfig) -> tuple[list[GtWord], list[PredWord]]:
    """Ground truth and readings in scored form. Ground-truth words that become too short are
    ignored (never anonymised: single characters, punctuation); such readings are dropped."""
    g2 = []
    for g in gt:
        t = scored_text(g.text, cfg)
        g2.append(GtWord(text=t, bbox=g.bbox, ignore=g.ignore or len(t) < cfg.min_scored_chars, rotated=g.rotated))
    p2 = []
    for p in pred:
        t = scored_text(p.text, cfg)
        if len(t) >= cfg.min_scored_chars:
            p2.append(PredWord(text=t, bbox=p.bbox, conf=p.conf, status=p.status))
    return g2, p2


def _accepted(p: PredWord) -> bool | None:
    return None if p.status is None else p.status == "accepted"


def evaluate_image(gt: list[GtWord], pred: list[PredWord], cfg: EvalConfig,
                   ignore_regions: list[Box] = (), image: np.ndarray | None = None) -> ImageResult:
    """Score one image. Predictions lying mostly inside an ``ignore_regions`` box (e.g. a raster
    image embedded in an SVG, whose text has no ground truth) are not counted as invented text.

    Words are compared in scored form (see :func:`scored_text`); single characters and pure
    punctuation are not scored. With ``cfg.matching == "align"`` (default) text is compared by
    character alignment within groups of overlapping boxes, so how the engine splits or joins
    words and where exactly it puts word boxes does not count as a reading error; with
    "word" words are matched one to one by box overlap. Given the image, box accuracy is
    measured too."""
    if cfg.matching == "align":
        return _evaluate_aligned(gt, pred, cfg, ignore_regions, image)
    return _evaluate_by_words(gt, pred, cfg, ignore_regions)


def _evaluate_by_words(gt, pred, cfg, ignore_regions) -> ImageResult:
    gt, pred = _scored(gt, pred, cfg)
    res = ImageResult()
    pairs = dict(match(gt, pred, cfg))
    matched_preds = set(pairs.values())
    pred_boxes = [p.bbox for p in pred]

    for gi, g in enumerate(gt):
        if g.ignore:
            continue
        pi = pairs.get(gi)
        detected = coverage(g.bbox, pred_boxes) >= cfg.detect_min_coverage
        text = pred[pi].text if pi is not None else ""
        edits = Levenshtein.distance(g.text, text)
        exact = pi is not None and text == g.text
        c = Counts(words=1, detected=int(detected), chars=len(g.text), edits=edits, exact=int(exact))
        for cat in categories(g):
            res.counts[cat].add(c)
        if pi is not None:
            res.scored.append((pred[pi].conf, exact, edits, len(g.text), _accepted(pred[pi])))
        if not exact:
            res.errors.append({"gt": g.text, "pred": text if pi is not None else None,
                               "conf": round(pred[pi].conf, 3) if pi is not None else None,
                               "detected": detected, "categories": sorted(categories(g) - {"all"})})

    ignored = [g.bbox for g in gt if g.ignore]
    for pi, p in enumerate(pred):
        if pi in matched_preds:
            continue
        if any(overlap(p.bbox, b) >= cfg.match_min_overlap for b in ignored):
            continue  # text we chose not to score (illegible, clipped)
        if any(_inter(p.bbox, r) >= cfg.match_min_overlap * _area(p.bbox) for r in ignore_regions):
            continue  # inside an area without ground truth
        if any(overlap(p.bbox, gt[gi].bbox) >= cfg.match_min_overlap for gi in pairs):
            continue  # a second piece of a word that is already scored (split word)
        res.false_positives += 1
        res.scored.append((p.conf, False, len(p.text), 0, _accepted(p)))
    return res


# --- character alignment --------------------------------------------------------------

def _linked(a: Box, b: Box, min_overlap: float) -> bool:
    small = min(_area(a), _area(b))
    return small > 0 and _inter(a, b) >= min_overlap * small


def _same_line(a: Box, b: Box, max_gap: float) -> bool:
    """Adjacent words of one line: overlapping across the line by half the smaller extent and
    at most ``max_gap`` times that extent apart along it. Horizontal lines for words wider than
    tall, vertical lines only for words taller than wide (rotated text) - otherwise the rows of
    a table would chain into one "vertical line"."""
    tall = lambda r: (r[3] - r[1]) > (r[2] - r[0])
    for along, across in ((0, 1), (1, 0)):
        if along == 0 and (tall(a) and tall(b)):
            continue
        if along == 1 and not (tall(a) and tall(b)):
            continue
        ext = min(a[across + 2] - a[across], b[across + 2] - b[across])
        if ext <= 0:
            continue
        cross = min(a[across + 2], b[across + 2]) - max(a[across], b[across])
        gap = max(a[along], b[along]) - min(a[along + 2], b[along + 2])
        if cross >= 0.5 * ext and gap <= max_gap * ext:
            return True
    return False


def _groups(gt: list[GtWord], pred: list[PredWord], cfg: EvalConfig) -> list[tuple[list[int], list[int]]]:
    """Connected groups of ground-truth words and readings whose boxes overlap."""
    n = len(gt) + len(pred)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for gi, g in enumerate(gt):
        for pi, p in enumerate(pred):
            if _linked(g.bbox, p.bbox, cfg.group_min_overlap):
                parent[find(gi)] = find(len(gt) + pi)
    # Neighbouring ground-truth words of one line (one label) form one group, so that readings
    # whose boxes are shifted or swapped along the line are still compared with the whole line.
    for a in range(len(gt)):
        for b in range(a + 1, len(gt)):
            if _same_line(gt[a].bbox, gt[b].bbox, cfg.group_line_gap):
                parent[find(a)] = find(b)
    groups: dict[int, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
    for gi in range(len(gt)):
        groups[find(gi)][0].append(gi)
    for pi in range(len(pred)):
        groups[find(len(gt) + pi)][1].append(len(gt) + pi - len(gt))
    return list(groups.values())


def _reading_order(boxes: list[Box], vertical: bool, reverse: bool, flat: bool = False) -> list[int]:
    """Indices in reading order: lines top to bottom, left to right (vertical text: columns,
    bottom to top or top to bottom). ``flat`` treats everything as one line."""
    if not boxes:
        return []
    if flat:
        a = 0 if not vertical else 1
        return sorted(range(len(boxes)), key=lambda i: boxes[i][a], reverse=reverse)
    a, b = (0, 1) if not vertical else (1, 0)  # along the line, across lines
    sizes = sorted(bx[b + 2] - bx[b] for bx in boxes)
    tol = 0.5 * sizes[len(sizes) // 2]
    order = sorted(range(len(boxes)), key=lambda i: (boxes[i][b] + boxes[i][b + 2]) / 2)
    lines, current, last = [], [], None
    for i in order:
        centre = (boxes[i][b] + boxes[i][b + 2]) / 2
        if last is not None and centre - last > tol:
            lines.append(current)
            current = []
        current.append(i)
        last = centre
    lines.append(current)
    out = []
    for line in lines:
        out += sorted(line, key=lambda i: boxes[i][a], reverse=reverse)
    return out


def _chars(texts: list[str], order: list[int]) -> tuple[str, list[int]]:
    s, owner = [], []
    for i in order:
        s.append(texts[i])
        owner += [i] * len(texts[i])
    return "".join(s), owner


def _ink_coverage(image: np.ndarray, box: Box, readings: list[Box], threshold: int) -> float | None:
    """Fraction of the word's ink (pixels differing from the box border's colour) inside the
    union of the reading boxes; None when the box shows no ink."""
    h, w = image.shape[:2]
    x0, y0 = max(0, int(box[0])), max(0, int(box[1]))
    x1, y1 = min(w, int(np.ceil(box[2]))), min(h, int(np.ceil(box[3])))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    crop = image[y0:y1, x0:x1].astype(int)
    border = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]])
    ink = np.abs(crop - np.median(border, axis=0)).max(axis=2) > threshold
    total = int(ink.sum())
    if total == 0:
        return None
    inside = np.zeros_like(ink)
    for r in readings:
        rx0, ry0 = max(0, int(np.floor(r[0])) - x0), max(0, int(np.floor(r[1])) - y0)
        rx1, ry1 = min(x1 - x0, int(np.ceil(r[2])) - x0), min(y1 - y0, int(np.ceil(r[3])) - y0)
        if rx1 > rx0 and ry1 > ry0:
            inside[ry0:ry1, rx0:rx1] = True
    return float((ink & inside).sum()) / total


def _evaluate_aligned(gt, pred, cfg, ignore_regions, image) -> ImageResult:
    res = ImageResult()
    # Scored form; every ground-truth word takes part in the alignment (short or ignored ones
    # absorb what the engine reads there) but only scored ones count. Readings keep all their
    # space-separated tokens, each in scored form.
    gtext = [scored_text(g.text, cfg) for g in gt]
    gscored = [not g.ignore and len(t) >= cfg.min_scored_chars for g, t in zip(gt, gtext)]
    # Readings are aligned whole (only typography unified, spaces removed): a split reading
    # "10.0.0." + "1" must still give "10.0.0.1". Punctuation the engine adds at the edge of a
    # ground-truth word ("Resource:", "-SAP", "(name,") is free, see below.
    typo = (lambda t: t.translate(_TYPOGRAPHY)) if cfg.normalize_typography else (lambda t: t)
    scored_boxes = [g.bbox for g, ok in zip(gt, gscored) if ok]

    def reading(p: PredWord) -> str:
        tokens = typo(p.text).split()
        if len(tokens) == 1 and len(scored_text(tokens[0], cfg)) < cfg.min_scored_chars:
            # A short piece counts only as part of a scored word ("10.0.0." + "1"); elsewhere it
            # is a stray character, which is not scored either way.
            inside = any(_inter(p.bbox, b) >= 0.5 * _area(p.bbox) for b in scored_boxes)
            return tokens[0] if inside and scored_text(tokens[0], cfg) else ""
        # Within a joined reading, stray single characters ("Private Link s service") drop out.
        return "".join(t for t in tokens if len(scored_text(t, cfg)) >= cfg.min_scored_chars)

    ptext = [reading(p) for p in pred]

    for gis, pis in _groups(gt, pred, cfg):
        gis = [i for i in gis]
        pis = [i for i in pis if ptext[i]]
        scored_here = [i for i in gis if gscored[i]]
        if not scored_here:
            # No scored ground truth: readings here are invented text, unless the group holds
            # ground truth we chose not to score or lies inside an area without ground truth.
            if gis:
                continue
            for pi in pis:
                p = pred[pi]
                n = len(scored_text(ptext[pi], cfg))
                if n < cfg.min_scored_chars:
                    continue
                if any(_inter(p.bbox, r) >= cfg.match_min_overlap * _area(p.bbox) for r in ignore_regions):
                    continue
                res.false_positives += 1
                res.scored.append((p.conf, False, n, 0, _accepted(p)))
                if p.status is None or p.status == "accepted":
                    res.disagreements.append({"kind": "extra", "gt": [], "pred": [pi], "wrong": []})
            continue

        # Reading order is not known for sure (rotated labels, labels on several lines), and the
        # engine's word boxes may be in a different order than its words (PaddleOCR swaps word
        # boxes along rotated lines). Try rows, columns read bottom-up or top-down, and single
        # lines, for the ground truth and the readings independently; keep the best agreement.
        gboxes = [gt[i].bbox for i in gis]
        pboxes = [pred[i].bbox for i in pis]
        orders = ((False, False, False), (True, True, False), (True, False, False),
                  (False, False, True), (True, True, True), (True, False, True))
        g_options = {tuple(gis[k] for k in _reading_order(gboxes, *o)) for o in orders}
        p_options = {tuple(pis[k] for k in _reading_order(pboxes, *o)) for o in orders}
        best = None
        for go in sorted(g_options):
            gs, gown = _chars(gtext, list(go))
            for po in sorted(p_options):
                ps, pown = _chars(ptext, list(po))
                d = Levenshtein.distance(gs, ps)
                if best is None or d < best[0]:
                    best = (d, gs, gown, ps, pown, list(go))
        _, gs, gown, ps, pown, gorder = best

        # Positions in the ground-truth string where a word starts or ends.
        bounds = {0, len(gs)} | {i for i in range(1, len(gs)) if gown[i] != gown[i - 1]}
        edge = set(cfg.strip_edge_chars)
        edits = defaultdict(int)          # per ground-truth word
        aligned = defaultdict(int)        # characters with a counterpart
        contributors = defaultdict(set)   # readings behind each word
        for tag, i1, i2, j1, j2 in Levenshtein.opcodes(gs, ps):
            if tag in ("equal", "replace"):
                for k in range(i2 - i1):
                    gi = gown[i1 + k]
                    aligned[gi] += 1
                    if k < j2 - j1:
                        contributors[gi].add(pown[j1 + k])
                if tag == "replace":
                    for k in range(max(i2 - i1, j2 - j1)):
                        gi = gown[min(i1 + k, i2 - 1)]
                        edits[gi] += 1
                        if j1 + k < j2:
                            contributors[gi].add(pown[j1 + k])
            elif tag == "delete":
                for k in range(i1, i2):
                    edits[gown[k]] += 1
            elif tag == "insert":
                # Extra characters belong to the word they follow (or precede, at the start).
                # Edge punctuation inserted at a word boundary is not an error.
                gi = gown[i1 - 1] if i1 > 0 else gown[0]
                extra = ps[j1:j2]
                if not (i1 in bounds and all(ch in edge for ch in extra)):
                    edits[gi] += j2 - j1
                for k in range(j1, j2):
                    contributors[gi].add(pown[k])

        wrong = []
        for gi in scored_here:
            g, text = gt[gi], gtext[gi]
            n = len(text)
            detected = aligned[gi] >= cfg.detect_min_coverage * n
            e = min(edits[gi], max(n, edits[gi])) if detected else n
            exact = detected and edits[gi] == 0
            readers = sorted(contributors[gi])
            conf = min((pred[i].conf for i in readers), default=0.0)
            accepted = None
            if readers and any(pred[i].status is not None for i in readers):
                accepted = all(pred[i].status == "accepted" for i in readers)
            c = Counts(words=1, detected=int(detected), chars=n, edits=e, exact=int(exact))
            cov = None
            if image is not None and detected and readers:
                cov = _ink_coverage(image, g.bbox, [pred[i].bbox for i in readers], cfg.box_ink_threshold)
                if cov is not None:
                    c.box_words, c.box_coverage = 1, cov
                    c.box_full = int(cov >= cfg.box_full_coverage)
            for cat in categories(GtWord(text=text, bbox=g.bbox, rotated=g.rotated)):
                res.counts[cat].add(c)
            if readers:
                res.scored.append((conf, exact, e, n, accepted))
            outcome = ("missed" if not detected else
                       "review" if accepted is False else
                       ("accepted_ok" if exact else "accepted_wrong"))
            res.words.append({"gt": gi, "text": text, "detected": detected, "exact": exact, "edits": e,
                              "accepted": accepted, "conf": conf, "readers": readers, "box_coverage": cov,
                              "outcome": outcome})
            if not exact:
                res.errors.append({"gt": text, "pred": " ".join(pred[i].text for i in readers) or None,
                                   "conf": round(conf, 3) if readers else None, "detected": detected,
                                   "categories": sorted(categories(GtWord(text=text, bbox=g.bbox)) - {"all"})})
                if not detected or accepted is not False:
                    wrong.append(gi)
        if wrong:
            res.disagreements.append({"kind": "missed" if not pis else "mismatch", "gt": gorder,
                                      "pred": pis, "wrong": wrong})
    return res


# --- aggregation ----------------------------------------------------------------------

def _rates(c: Counts) -> dict:
    return {"words": c.words, "chars": c.chars,
            "detection_recall": c.detected / c.words if c.words else None,
            "cer": c.edits / c.chars if c.chars else None,
            "wer": 1 - c.exact / c.words if c.words else None,
            "box_words": c.box_words,
            "box_full_coverage": c.box_full / c.box_words if c.box_words else None,
            "box_mean_coverage": c.box_coverage / c.box_words if c.box_words else None}


def expected_calibration_error(scored: list[tuple], bins: int) -> float | None:
    if not scored:
        return None
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        items = [(s[0], s[1]) for s in scored if lo <= s[0] < hi or (b == bins - 1 and s[0] == 1.0)]
        if items:
            acc = sum(ok for _, ok in items) / len(items)
            mean_conf = sum(conf for conf, _ in items) / len(items)
            total += abs(acc - mean_conf) * len(items)
    return total / len(scored)


def _accepted_row(acc: list[tuple], gt_words: int) -> dict:
    acc_gt = [s for s in acc if s[3] > 0]
    correct = sum(1 for s in acc_gt if s[1])
    chars = sum(s[3] for s in acc_gt)
    edits = sum(s[2] for s in acc)  # includes accepted hallucinations
    return {"accepted_words": len(acc_gt), "accepted_hallucinations": len(acc) - len(acc_gt),
            "accepted_cer": edits / chars if chars else None,
            "accepted_word_error": 1 - correct / len(acc_gt) if acc_gt else None,
            "review_rate": 1 - len(acc_gt) / gt_words if gt_words else None}


def accept_table(scored: list[tuple], gt_words: int, thresholds: tuple[float, ...]) -> list[dict]:
    """For each confidence threshold: how good accepted words are and how much goes to review."""
    return [{"threshold": t, **_accepted_row([s for s in scored if s[0] >= t], gt_words)} for t in thresholds]


def status_row(scored: list[tuple], gt_words: int) -> dict | None:
    """The same for words a consensus marked "accepted" (None for single engines)."""
    if not any(len(s) > 4 and s[4] is not None for s in scored):
        return None
    return _accepted_row([s for s in scored if len(s) > 4 and s[4]], gt_words)


def aggregate(results: list[ImageResult], cfg: EvalConfig) -> dict:
    counts: dict[str, Counts] = defaultdict(Counts)
    scored, fps, errors = [], 0, []
    for r in results:
        for cat, c in r.counts.items():
            counts[cat].add(c)
        scored += r.scored
        fps += r.false_positives
        errors += r.errors
    words = counts["all"].words
    return {
        "categories": {cat: _rates(counts[cat]) for cat in CATEGORIES if counts[cat].words},
        "false_positives": fps,
        "ece": expected_calibration_error(scored, cfg.calibration_bins),
        "accept": accept_table(scored, words, cfg.accept_thresholds),
        "consensus_accepted": status_row(scored, words),
        "errors": _spread(errors, cfg.max_error_examples),
    }


def _spread(items: list, n: int) -> list:
    """Up to n items taken evenly across the list, so examples come from all images."""
    if len(items) <= n:
        return items
    return [items[int(i * len(items) / n)] for i in range(n)]
