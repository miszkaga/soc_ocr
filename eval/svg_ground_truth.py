"""Ground truth from SVG diagrams: render to PNG and read every word's box from the browser.

The SVG source holds the exact text, and Chromium's layout tells us where each word is
drawn, so a rendered diagram comes with word-level ground truth and no manual labelling.

Words are whitespace-separated runs of text from <text>/<tspan> and from HTML inside
<foreignObject> (draw.io style). Hidden text (display/visibility/opacity, fill and
stroke "none", <defs>, <title>, ...) is skipped. Boxes are axis-aligned and cover the
line box (ascender to descender), so they are slightly taller than the glyphs.

Known gaps: text inside <symbol>s instantiated by <use> lives in shadow trees and is not
read; text occluded by shapes drawn on top of it is still reported.

Usage:  python -m eval.svg_ground_truth [--limit N] [--out DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

from playwright.sync_api import Browser, sync_playwright

from ocr.config import SvgGtConfig, load_config

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "artifacts" / "datasets" / "raw"
DEFAULT_OUT = REPO / "artifacts" / "datasets" / "gt" / "svg"
FONTS_CONF = REPO / "config" / "fonts.conf"
POLISH = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")

_MEASURE_JS = r"""
() => {
  const s = document.documentElement;
  if (s.localName !== 'svg') return null;
  const vb = s.viewBox && s.viewBox.baseVal && s.viewBox.baseVal.width ? s.viewBox.baseVal : null;
  const abs = v => v && !v.trim().endsWith('%');
  const r = s.getBoundingClientRect();
  if (abs(s.getAttribute('width')) && abs(s.getAttribute('height'))) return [r.width, r.height];
  if (vb) return [vb.width, vb.height];
  const b = s.getBBox();
  return [b.x + b.width, b.y + b.height];
}
"""

_WORDS_JS = r"""
() => {
  const skipTags = new Set(['title', 'desc', 'style', 'script', 'metadata']);
  const hidden = el => {
    if (el.closest('defs, clipPath, mask, symbol, pattern, marker')) return true;
    for (let e = el; e && e.nodeType === 1; e = e.parentElement) {
      const cs = getComputedStyle(e);
      if (cs.display === 'none' || cs.visibility !== 'visible' || parseFloat(cs.opacity) === 0) return true;
    }
    const cs = getComputedStyle(el);
    if (el.closest('text')) return cs.fill === 'none' && cs.stroke === 'none';
    return cs.color === 'rgba(0, 0, 0, 0)' || cs.color === 'transparent';
  };
  const groups = new Map();
  const out = [];
  const walker = document.createTreeWalker(document.documentElement, NodeFilter.SHOW_TEXT);
  let n;
  while ((n = walker.nextNode())) {
    const el = n.parentElement;
    if (!el || skipTags.has(el.localName) || !/\S/.test(n.data) || hidden(el)) continue;
    const owner = el.closest('text') || el.closest('foreignObject');
    const ctm = owner && owner.getScreenCTM ? owner.getScreenCTM() : null;
    const userScale = ctm ? Math.sqrt(Math.abs(ctm.a * ctm.d - ctm.b * ctm.c)) : 1;
    const rotated = ctm ? (Math.abs(ctm.b) > 1e-3 || Math.abs(ctm.c) > 1e-3) : false;
    const groupEl = el.closest('text') || el;
    if (!groups.has(groupEl)) groups.set(groupEl, groups.size);
    const fontPx = parseFloat(getComputedStyle(el).fontSize) * userScale;
    // HTML in a foreignObject is clipped to the foreignObject's box (draw.io labels that
    // overflow their shape show "Poc" for "Pod").
    const fo = el.closest('foreignObject');
    const clip = fo ? fo.getBoundingClientRect() : null;
    const re = /\S+/g;
    let m;
    while ((m = re.exec(n.data))) {
      const r = document.createRange();
      r.setStart(n, m.index);
      r.setEnd(n, m.index + m[0].length);
      const b = r.getBoundingClientRect();
      if (b.width <= 0 || b.height <= 0) continue;
      const clipped = clip ? (b.left < clip.left - 0.5 || b.top < clip.top - 0.5 ||
                              b.right > clip.right + 0.5 || b.bottom > clip.bottom + 0.5) : false;
      out.push({text: m[0], box: [b.left, b.top, b.right, b.bottom], font: fontPx,
                rotated: rotated, clipped: clipped, group: groups.get(groupEl)});
    }
  }
  return out;
}
"""


@dataclass
class Word:
    text: str
    bbox_px: list[float]
    font_px: float
    rotated: bool
    legible: bool
    clipped: bool  # partly cut off by its container: the image does not show the full text
    group: int  # words with the same group come from one <text> element / HTML block


@dataclass
class SvgGroundTruth:
    schema: str
    source: str
    source_file: str
    license: str
    image: str
    width_px: int
    height_px: int
    scale: float
    renderer: dict
    words: list[Word] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


def _scale_for(w: float, h: float, cfg: SvgGtConfig) -> float:
    return min(cfg.max_scale, max(cfg.min_scale, cfg.target_long_side_px / max(w, h, 1.0)))


def _block_network(context):
    # Determinism and privacy: the SVG may reference web fonts or images; never fetch them.
    context.route("**/*", lambda route: route.continue_() if route.request.url.startswith("file:")
                  else route.abort())


def render_svg(browser: Browser, svg_path: Path, png_path: Path, cfg: SvgGtConfig,
               source: str = "", license: str = "") -> SvgGroundTruth | None:
    """Render `svg_path` to `png_path` and return its word-level ground truth.

    `browser` must come from :func:`launch_browser` so that fonts match config/fonts.conf.

    Returns None if the file is not a standalone SVG document.
    """
    uri = svg_path.resolve().as_uri()
    ctx = browser.new_context(viewport={"width": 4000, "height": 4000})
    try:
        _block_network(ctx)
        page = ctx.new_page()
        page.goto(uri, wait_until="load")
        size = page.evaluate(_MEASURE_JS)
    finally:
        ctx.close()
    if not size or size[0] <= 0 or size[1] <= 0:
        return None

    w, h = size
    scale = _scale_for(w, h, cfg)
    vw, vh = math.ceil(w), math.ceil(h)
    ctx = browser.new_context(viewport={"width": vw, "height": vh}, device_scale_factor=scale)
    try:
        _block_network(ctx)
        page = ctx.new_page()
        page.goto(uri, wait_until="load")
        page.evaluate("([w, h]) => { const s = document.documentElement;"
                      " s.setAttribute('width', w); s.setAttribute('height', h); }", [w, h])
        page.evaluate("() => document.fonts.ready.then(() => true)")
        raw_words = page.evaluate(_WORDS_JS)
        png_path.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(png_path), clip={"x": 0, "y": 0, "width": vw, "height": vh})
    finally:
        ctx.close()

    width_px, height_px = round(vw * scale), round(vh * scale)
    words = []
    for rw in raw_words:
        x0, y0, x1, y1 = (v * scale for v in rw["box"])
        cx0, cy0, cx1, cy1 = max(0.0, x0), max(0.0, y0), min(width_px, x1), min(height_px, y1)
        inside = max(0.0, cx1 - cx0) * max(0.0, cy1 - cy0) / ((x1 - x0) * (y1 - y0))
        if inside < cfg.min_visible_fraction:
            continue
        font_px = rw["font"] * scale
        words.append(Word(text=rw["text"], bbox_px=[round(v, 1) for v in (cx0, cy0, cx1, cy1)],
                          font_px=round(font_px, 1), rotated=rw["rotated"],
                          legible=font_px >= cfg.min_legible_font_px,
                          clipped=rw["clipped"] or inside < 1.0, group=rw["group"]))
    return SvgGroundTruth(
        schema="gt-svg/v1", source=source, source_file=str(svg_path), license=license,
        image=png_path.name, width_px=width_px, height_px=height_px, scale=round(scale, 4),
        renderer={"chromium": browser.version, "platform": platform.platform(),
                  "fonts_conf_sha256": hashlib.sha256(FONTS_CONF.read_bytes()).hexdigest()[:16]},
        words=words)


# --- dataset builder --------------------------------------------------------------

def _has_text(path: Path) -> bool:
    data = path.read_bytes()
    return b"<text" in data or b"foreignObject" in data


def iter_sources(raw: Path = RAW):
    """Yield (source, svg_path, license) for every public SVG that may contain text."""
    for source, root in (("azure-architecture-center", raw / "azure-architecture-center"),
                         ("kubernetes-website", raw / "kubernetes-website")):
        for p in sorted(root.rglob("*.svg")):
            if ".git" not in p.parts and _has_text(p):
                yield source, p, "CC BY 4.0"
    meta = raw / "wikimedia-commons" / "metadata.jsonl"
    if meta.exists():
        for line in meta.open(encoding="utf-8"):
            rec = json.loads(line)
            p = raw / "wikimedia-commons" / rec["file"]
            if rec["mime"] == "image/svg+xml" and p.exists() and _has_text(p):
                yield "wikimedia-commons", p, rec["license"]


def _stem(source: str, path: Path, raw: Path) -> str:
    rel = path.relative_to(raw / source) if path.is_relative_to(raw / source) else path
    digest = hashlib.sha256(str(rel).encode()).hexdigest()[:8]
    name = re.sub(r"[^\w.-]+", "_", path.stem)[:60]
    return f"{name}_{digest}"


def launch_browser(playwright) -> Browser:
    """Chromium with the repo's font substitutions (config/fonts.conf), so that missing
    Windows fonts render as similar sans fonts instead of the default serif."""
    env = {**os.environ, "FONTCONFIG_FILE": str(FONTS_CONF)}
    return playwright.chromium.launch(env=env)


def latin_fraction(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 1.0
    return sum(1 for ch in letters if unicodedata.name(ch, "").startswith("LATIN")) / len(letters)


def build(out: Path, limit: int | None = None, raw: Path = RAW, browser: Browser | None = None) -> list[dict]:
    if browser is None:
        with sync_playwright() as p:
            b = launch_browser(p)
            try:
                return build(out, limit, raw, b)
            finally:
                b.close()

    cfg = load_config().svg_gt
    index = []
    seen_texts: set[str] = set()  # the same diagram is copied into many doc translations
    for k, (source, svg, lic) in enumerate(iter_sources(raw)):
        if limit is not None and k >= limit:
            break
        stem = _stem(source, svg, raw)
        try:
            gt = render_svg(browser, svg, out / source / f"{stem}.png", cfg, source, lic)
        except Exception as e:  # a broken SVG must not stop the batch
            print(f"skip {svg}: {e}", file=sys.stderr)
            continue
        # Skip empty diagrams and those in other scripts (e.g. Japanese translations of the
        # Kubernetes docs): the engines are measured for Polish and English.
        if (gt is None or not gt.words or latin_fraction(gt.text) < cfg.min_latin_letter_fraction
                or gt.text in seen_texts):
            (out / source / f"{stem}.png").unlink(missing_ok=True)
            continue
        seen_texts.add(gt.text)
        gt.source_file = str(svg.relative_to(REPO)) if svg.is_relative_to(REPO) else str(svg)
        (out / source / f"{stem}.json").write_text(
            json.dumps(asdict(gt), ensure_ascii=False, indent=1), encoding="utf-8")
        index.append({"json": f"{source}/{stem}.json", "source": source, "license": lic,
                      "words": len(gt.words), "legible": sum(w.legible for w in gt.words),
                      "polish": any(ch in POLISH for ch in gt.text),
                      "width_px": gt.width_px, "height_px": gt.height_px})
    out.mkdir(parents=True, exist_ok=True)
    with (out / "index.jsonl").open("w", encoding="utf-8") as f:
        for row in index:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return index


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args(argv)
    index = build(args.out, args.limit)
    print(f"{len(index)} SVGs with text, {sum(r['words'] for r in index)} words -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
