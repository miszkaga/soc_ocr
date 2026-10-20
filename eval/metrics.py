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

    def add(self, other: "Counts"):
        for k in ("words", "detected", "chars", "edits", "exact"):
            setattr(self, k, getattr(self, k) + getattr(other, k))


@dataclass
class ImageResult:
    counts: dict[str, Counts] = field(default_factory=lambda: defaultdict(Counts))
    # One entry per scored prediction: (confidence, exactly right, edits, gt chars).
    # Hallucinated predictions have gt chars 0 and edits = len(prediction).
    scored: list[tuple[float, bool, int, int]] = field(default_factory=list)
    false_positives: int = 0
    errors: list[dict] = field(default_factory=list)


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


def evaluate_image(gt: list[GtWord], pred: list[PredWord], cfg: EvalConfig) -> ImageResult:
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
            res.scored.append((pred[pi].conf, exact, edits, len(g.text)))
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
        if any(overlap(p.bbox, gt[gi].bbox) >= cfg.match_min_overlap for gi in pairs):
            continue  # a second piece of a word that is already scored (split word)
        res.false_positives += 1
        res.scored.append((p.conf, False, len(p.text), 0))
    return res


# --- aggregation ----------------------------------------------------------------------

def _rates(c: Counts) -> dict:
    return {"words": c.words, "chars": c.chars,
            "detection_recall": c.detected / c.words if c.words else None,
            "cer": c.edits / c.chars if c.chars else None,
            "wer": 1 - c.exact / c.words if c.words else None}


def expected_calibration_error(scored: list[tuple[float, bool, int, int]], bins: int) -> float | None:
    if not scored:
        return None
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        items = [(conf, ok) for conf, ok, _, _ in scored if lo <= conf < hi or (b == bins - 1 and conf == 1.0)]
        if items:
            acc = sum(ok for _, ok in items) / len(items)
            mean_conf = sum(conf for conf, _ in items) / len(items)
            total += abs(acc - mean_conf) * len(items)
    return total / len(scored)


def accept_table(scored: list[tuple[float, bool, int, int]], gt_words: int,
                 thresholds: tuple[float, ...]) -> list[dict]:
    """For each threshold: how good accepted words are and how much goes to review."""
    rows = []
    for t in thresholds:
        acc = [s for s in scored if s[0] >= t]
        acc_gt = [s for s in acc if s[3] > 0]
        correct = sum(ok for _, ok, _, n in acc_gt)
        chars = sum(n for _, _, _, n in acc_gt)
        edits = sum(e for _, _, e, _ in acc)  # includes accepted hallucinations
        rows.append({"threshold": t, "accepted_words": len(acc_gt), "accepted_hallucinations": len(acc) - len(acc_gt),
                     "accepted_cer": edits / chars if chars else None,
                     "accepted_word_error": 1 - correct / len(acc_gt) if acc_gt else None,
                     "review_rate": 1 - len(acc_gt) / gt_words if gt_words else None})
    return rows


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
        "errors": _spread(errors, cfg.max_error_examples),
    }


def _spread(items: list, n: int) -> list:
    """Up to n items taken evenly across the list, so examples come from all images."""
    if len(items) <= n:
        return items
    return [items[int(i * len(items) / n)] for i in range(n)]
