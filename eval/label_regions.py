"""Manual ground truth for text the SVG does not contain as text.

Some text in the rendered diagrams has no characters in the SVG: labels converted to outlines,
lettering inside icons ("SQL", "DNS", "Microsoft Azure"). Readings there would count as
invented text although they may be right. A human decides for each such region:

- the text it shows      -> added to the ground truth (scored like any other word),
- "no text"              -> nothing added; readings there stay invented text,
- "unreadable"           -> the region is excluded from scoring.

Candidates are regions where at least two engines read something word-like that matches no
ground truth. Regions read by only one engine (almost always Tesseract reading an icon) are
left as invented text. Visually identical regions are grouped, so each look is decided once.

Usage:
    python -m eval.label_regions prepare --runs RUN [RUN ...]   # crops + candidates.json
    python -m eval.label_regions serve                          # labelling page on 127.0.0.1
Decisions are written to artifacts/datasets/gt/manual/<gt dir name>_regions.jsonl, which
eval.run_eval merges into the ground truth.
"""
from __future__ import annotations

import argparse
import datetime as dt
import http.server
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from eval.metrics import _area, _inter, overlap
from eval.run_eval import DEFAULT_GT, RUNS, load_gt, manual_path, manual_records

REPO = Path(__file__).resolve().parent.parent
WORK = REPO / "artifacts" / "labeling" / "svg_regions"
PAD = 6


def word_like(text: str) -> bool:
    """At least two Latin letters or digits, and no other script."""
    alnum = [c for c in text if c.isalnum()]
    return len(alnum) >= 2 and all(unicodedata.name(c, "").startswith(("LATIN", "DIGIT")) for c in alnum)


def _merge(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def collect_regions(gt_dir: Path, run_dirs: list[Path], min_engines: int = 2) -> list[dict]:
    """Regions without ground truth that ``min_engines`` runs read as word-like text."""
    regions: dict[str, list[dict]] = defaultdict(list)
    for run in run_dirs:
        for line in (run / "predictions.jsonl").open(encoding="utf-8"):
            rec = json.loads(line)
            _, gt, ignore = load_gt(gt_dir, {"json": rec["json"]})
            decided = [tuple(m["box"]) for m in manual_records(gt_dir, rec["json"])]  # incl. "no text"
            for w in rec["words"]:
                b = tuple(w["bbox_px"])
                if (any(overlap(b, g.bbox) >= 0.5 for g in gt) or any(_inter(b, r) >= 0.5 * _area(b) for r in ignore)
                        or any(overlap(b, d) >= 0.5 for d in decided)):
                    continue
                for r in regions[rec["json"]]:
                    if any(overlap(b, rb) > 0.3 for _, _, rb in r["_reads"]):
                        r["_reads"].append((run.name, w["text"], b))
                        break
                else:
                    regions[rec["json"]].append({"json": rec["json"], "_reads": [(run.name, w["text"], b)]})
    out = []
    for rs in regions.values():
        for r in rs:
            reads = r.pop("_reads")
            if len({e for e, _, _ in reads}) < min_engines or sum(word_like(t) for _, t, _ in reads) < 2:
                continue
            r["proposal"] = Counter(t for _, t, _ in reads if word_like(t)).most_common(1)[0][0]
            # The region is where the proposed text was read, not the union of all readings
            # (a wide box from another engine would pull in neighbouring labels).
            box = None
            for _, t, b in reads:
                if t == r["proposal"]:
                    box = b if box is None else _merge(box, b)
            r["box"] = box
            r["reads"] = [[e, t] for e, t, _ in reads]
            out.append(r)
    return sorted(out, key=lambda r: (r["json"], r["box"]))


def _ahash(img: Image.Image) -> np.ndarray:
    a = np.asarray(img.convert("L").resize((16, 16), Image.BILINEAR), dtype=float)
    return a > a.mean()


def group_regions(gt_dir: Path, regions: list[dict], max_hash_diff: int = 40) -> list[dict]:
    """Group regions with the same proposal that look alike (average hash, similar width)."""
    groups: list[dict] = []
    for r in regions:
        x0, y0, x1, y1 = r["box"]
        img = Image.open(gt_dir / Path(r["json"]).with_suffix(".png"))
        h = _ahash(img.crop((max(0, x0 - PAD), max(0, y0 - PAD), x1 + PAD, y1 + PAD)))
        for g in groups:
            if (g["proposal"] == r["proposal"] and np.count_nonzero(g["_hash"] != h) <= max_hash_diff
                    and abs(g["_width"] - (x1 - x0)) < 0.35 * max(g["_width"], x1 - x0)):
                g["members"].append(r)
                break
        else:
            groups.append({"id": f"g{len(groups):04d}", "proposal": r["proposal"], "_hash": h,
                           "_width": x1 - x0, "members": [r]})
    for g in groups:
        del g["_hash"], g["_width"]
    return groups


def prepare(gt_dir: Path, run_dirs: list[Path], work: Path = WORK) -> list[dict]:
    groups = group_regions(gt_dir, collect_regions(gt_dir, run_dirs))
    (work / "crops").mkdir(parents=True, exist_ok=True)
    for g in groups:
        r = g["members"][0]
        x0, y0, x1, y1 = r["box"]
        img = Image.open(gt_dir / Path(r["json"]).with_suffix(".png")).convert("RGB")
        cx0, cy0 = max(0, x0 - 3 * PAD), max(0, y0 - 3 * PAD)
        crop = img.crop((cx0, cy0, x1 + 3 * PAD, y1 + 3 * PAD))
        # Outline the region to decide on; the context around it is for orientation only.
        ImageDraw.Draw(crop).rectangle((x0 - cx0 - 3, y0 - cy0 - 3, x1 - cx0 + 3, y1 - cy0 + 3), outline=(220, 0, 0), width=2)
        crop.save(work / "crops" / f"{g['id']}.png")
        g["crop"] = f"crops/{g['id']}.png"
        g["reads"] = sorted({t for m in g["members"] for _, t in m["reads"]})
    (work / "candidates.json").write_text(json.dumps({"gt_dir": str(gt_dir), "runs": [d.name for d in run_dirs],
                                                     "groups": groups}, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    return groups


def apply_decisions(work: Path, decisions: dict[str, dict], who: str = "user") -> Path:
    """Write one manual ground-truth record per region from per-group decisions.

    A decision is {"status": "text" | "no_text" | "unreadable", "text": "..."}.
    """
    cand = json.loads((work / "candidates.json").read_text(encoding="utf-8"))
    gt_dir = Path(cand["gt_dir"])
    out = manual_path(gt_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    today = dt.date.today().isoformat()
    with out.open("w", encoding="utf-8") as f:
        for g in cand["groups"]:
            d = decisions.get(g["id"])
            if not d or d.get("status") not in ("text", "no_text", "unreadable"):
                continue
            for m in g["members"]:
                f.write(json.dumps({"json": m["json"], "box": [round(v, 1) for v in m["box"]], "status": d["status"],
                                    "text": d.get("text", "").strip() if d["status"] == "text" else "",
                                    "group": g["id"], "by": who, "date": today}, ensure_ascii=False) + "\n")
    return out


_PAGE = """<!doctype html><html lang="pl"><head><meta charset="utf-8"><title>Tekst bez wzorca</title>
<style>
body{font:14px system-ui,sans-serif;margin:16px;background:#fafafa;color:#222}
.card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:10px;margin:0 0 10px;display:grid;
grid-template-columns:minmax(120px,320px) 1fr;gap:12px;align-items:center}
.card img{max-width:320px;max-height:160px;image-rendering:pixelated;border:1px solid #eee;background:#fff}
.card.text{border-left:6px solid #2e7d32}.card.no_text{border-left:6px solid #c62828}.card.unreadable{border-left:6px solid #888}
input{font:15px monospace;width:100%;max-width:420px;padding:4px}
button{margin:4px 4px 0 0;padding:4px 10px;cursor:pointer}
.meta{color:#666;font-size:12px}
#bar{position:sticky;top:0;background:#fafafa;padding:8px 0;z-index:1;border-bottom:1px solid #ddd;margin-bottom:10px}
</style></head><body>
<div id="bar"><b>Tekst widoczny na obrazie, którego nie ma w SVG.</b> Oceniaj tylko to, co jest w <span style="color:#c00">czerwonej
ramce</span>. Wpisz dokładnie to, co widać (wielkość liter, znaki), albo wybierz „brak tekstu” (ikona, kształt bez napisu)
/ „nieczytelne” (napis jest, ale nie da się go przeczytać).
<br><span id="count"></span> <button onclick="save()">Zapisz</button> <span id="msg"></span></div>
<div id="list"></div>
<script>
let cand = null, dec = {};
function render(){
  const list = document.getElementById('list'); list.innerHTML = '';
  for (const g of cand.groups){
    const d = dec[g.id] || {}; const div = document.createElement('div');
    div.className = 'card ' + (d.status || '');
    div.innerHTML = `<img src="${g.crop}"><div>
      <input value="${(d.text ?? g.proposal).replace(/"/g,'&quot;')}" data-id="${g.id}">
      <div><button data-s="text">OK</button><button data-s="no_text">brak tekstu</button>
      <button data-s="unreadable">nieczytelne</button></div>
      <div class="meta">${g.members.length}× · odczyty: ${g.reads.map(r=>r.replace(/</g,'&lt;')).join(' | ')}</div></div>`;
    div.querySelectorAll('button').forEach(b => b.onclick = () => {
      dec[g.id] = {status: b.dataset.s, text: div.querySelector('input').value};
      div.className = 'card ' + b.dataset.s; upd();
    });
    div.querySelector('input').oninput = e => { if (dec[g.id]) dec[g.id].text = e.target.value; };
    list.appendChild(div);
  }
  upd();
}
function upd(){ document.getElementById('count').textContent =
  `Ocenione: ${Object.keys(dec).length} z ${cand.groups.length}`; }
async function save(){
  const r = await fetch('save', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(dec)});
  document.getElementById('msg').textContent = r.ok ? 'Zapisano ' + new Date().toLocaleTimeString() : 'Błąd zapisu';
}
window.getDecisions = () => dec;
Promise.all([fetch('candidates.json').then(r=>r.json()), fetch('decisions.json').then(r=>r.ok?r.json():{})])
  .then(([c, d]) => { cand = c; dec = d; render(); });
</script></body></html>"""


def serve(work: Path = WORK, port: int = 8765):
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(work), **kw)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = _PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            super().do_GET()

        def do_POST(self):
            if self.path != "/save":
                self.send_error(404)
                return
            decisions = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            (work / "decisions.json").write_text(json.dumps(decisions, ensure_ascii=False, indent=1), encoding="utf-8")
            apply_decisions(work, decisions)
            self.send_response(204)
            self.end_headers()

        def log_message(self, *a):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"http://127.0.0.1:{port}/", flush=True)
    httpd.serve_forever()


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Label text that exists in the image but not in the SVG.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--gt", type=Path, default=DEFAULT_GT)
    p.add_argument("--runs", nargs="+", required=True, help="run names under artifacts/eval_runs")
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8765)
    args = ap.parse_args(argv)
    if args.cmd == "prepare":
        groups = prepare(args.gt, [RUNS / r for r in args.runs])
        print(f"{len(groups)} groups, {sum(len(g['members']) for g in groups)} regions -> {WORK}")
    else:
        serve(port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
