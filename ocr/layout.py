"""Words -> lines in reading order. In diagrams labels stay apart: words join a line only when
they share it (vertical overlap) and are close along it."""
from __future__ import annotations

from ocr.engines.base import OcrWord


def _height(w: OcrWord) -> float:
    return max(1.0, w.bbox_px[3] - w.bbox_px[1])


def lines(words: list[OcrWord], min_overlap: float = 0.5, max_gap: float = 1.5) -> list[list[OcrWord]]:
    """Words grouped into lines, top to bottom, each line left to right.

    A word joins a line when it overlaps the line's last word vertically by at least
    ``min_overlap`` of the smaller height and starts at most ``max_gap`` heights after it.
    """
    out: list[list[OcrWord]] = []
    for w in sorted(words, key=lambda w: (w.bbox_px[0], w.bbox_px[1])):
        best = None
        for line in out:
            last = line[-1]
            ov = min(w.bbox_px[3], last.bbox_px[3]) - max(w.bbox_px[1], last.bbox_px[1])
            h = min(_height(w), _height(last))
            gap = w.bbox_px[0] - last.bbox_px[2]
            if ov >= min_overlap * h and -0.5 * h <= gap <= max_gap * max(_height(w), _height(last)):
                if best is None or gap < best[0]:
                    best = (gap, line)
        if best is None:
            out.append([w])
        else:
            best[1].append(w)
    out.sort(key=lambda ln: (min(w.bbox_px[1] for w in ln), min(w.bbox_px[0] for w in ln)))
    return out


def text(grouped: list[list[OcrWord]]) -> str:
    return "\n".join(" ".join(w.text for w in line) for line in grouped)
