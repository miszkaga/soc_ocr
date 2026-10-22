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
import json
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from eval.metrics import _area, _inter, overlap
from eval.labeling_server import serve_page
from eval.run_eval import DEFAULT_GT, RUNS, images_from, load_gt, load_index, manual_path, manual_records

REPO = Path(__file__).resolve().parent.parent
WORK = REPO / "artifacts" / "labeling" / "svg_regions"
PAD = 6


def word_like(text: str) -> bool:
    """At least two Latin letters or digits, and no other script."""
    alnum = [c for c in text if c.isalnum()]
    return len(alnum) >= 2 and all(unicodedata.name(c, "").startswith(("LATIN", "DIGIT")) for c in alnum)


def _merge(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _readers(run_name: str, word: dict) -> list[tuple[str, str]]:
    """(engine, text) pairs behind one predicted word: every engine that voted in a consensus
    run (re-reads count as the same engine), otherwise the run itself."""
    votes = word.get("votes")
    if not votes:
        return [(run_name, word["text"])]
    return [(k.split("@")[0], v) for k, v in votes.items() if v]


def collect_regions(gt_dir: Path, run_dirs: list[Path], min_engines: int = 2,
                    images: set[str] | None = None) -> list[dict]:
    """Regions without ground truth that ``min_engines`` engines read as word-like text."""
    regions: dict[str, list[dict]] = defaultdict(list)
    for run in run_dirs:
        for line in (run / "predictions.jsonl").open(encoding="utf-8"):
            rec = json.loads(line)
            if images is not None and rec["json"] not in images:
                continue
            _, gt, ignore = load_gt(gt_dir, {"json": rec["json"]})
            decided = [tuple(m["box"]) for m in manual_records(gt_dir, rec["json"])]  # incl. "no text"
            for w in rec["words"]:
                b = tuple(w["bbox_px"])
                if (any(overlap(b, g.bbox) >= 0.5 for g in gt) or any(_inter(b, r) >= 0.5 * _area(b) for r in ignore)
                        or any(overlap(b, d) >= 0.5 for d in decided)):
                    continue
                reads = [(engine, text, b) for engine, text in _readers(run.name, w)]
                for r in regions[rec["json"]]:
                    if any(overlap(b, rb) > 0.3 for _, _, rb in r["_reads"]):
                        r["_reads"].extend(reads)
                        break
                else:
                    regions[rec["json"]].append({"json": rec["json"], "_reads": reads})
    out = []
    for rs in regions.values():
        for r in rs:
            reads = r.pop("_reads")
            if len({e for e, t, _ in reads if word_like(t)}) < min_engines:
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
    return groups


def _same_look(a: dict, b: dict, max_hash_diff: int = 40) -> bool:
    return (a["proposal"] == b["proposal"] and np.count_nonzero(a["_hash"] != b["_hash"]) <= max_hash_diff
            and abs(a["_width"] - b["_width"]) < 0.35 * max(a["_width"], b["_width"]))


def suggestions_from(earlier_work: Path, groups: list[dict]) -> int:
    """Attach to each group the decision made earlier for a group that looks the same."""
    cand = json.loads((earlier_work / "candidates.json").read_text(encoding="utf-8"))
    decisions = json.loads((earlier_work / "decisions.json").read_text(encoding="utf-8"))
    earlier = [g for g in group_regions(Path(cand["gt_dir"]), [g["members"][0] for g in cand["groups"]])]
    by_member = {(g["members"][0]["json"], tuple(g["members"][0]["box"])): g["id"] for g in cand["groups"]}
    # Fallback: the same proposed text was always decided the same way (icons are drawn at
    # different scales, so their look does not always match).
    by_text: dict[str, set] = defaultdict(set)
    for e in earlier:
        d = decisions.get(by_member.get((e["members"][0]["json"], tuple(e["members"][0]["box"]))))
        if d:
            by_text[e["proposal"]].add((d["status"], d.get("text", "") if d["status"] == "text" else ""))
    n = 0
    for g in groups:
        for e in earlier:
            d = decisions.get(by_member.get((e["members"][0]["json"], tuple(e["members"][0]["box"]))))
            if d and _same_look(g, e):
                g["suggestion"] = {"status": d["status"], "text": d.get("text", ""), "from": earlier_work.name}
                break
        else:
            same = by_text.get(g["proposal"], set())
            if len(same) == 1:
                status, text = next(iter(same))
                g["suggestion"] = {"status": status, "text": text, "from": earlier_work.name, "by": "text"}
        n += "suggestion" in g
    return n


def prepare(gt_dir: Path, run_dirs: list[Path], work: Path = WORK, images: set[str] | None = None,
            suggest_from: Path | None = None) -> list[dict]:
    groups = group_regions(gt_dir, collect_regions(gt_dir, run_dirs, images=images))
    if suggest_from is not None:
        suggestions_from(suggest_from, groups)
    for g in groups:
        del g["_hash"], g["_width"]
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
    (work / "candidates.json").write_text(json.dumps({"gt_dir": str(gt_dir), "set": work.name,
                                                     "runs": [d.name for d in run_dirs],
                                                     "groups": groups}, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    return groups


def apply_decisions(work: Path, decisions: dict[str, dict], who: str = "user") -> Path:
    """Write one manual ground-truth record per region from per-group decisions.

    A decision is {"status": "text" | "no_text" | "unreadable", "text": "..."}.
    """
    cand = json.loads((work / "candidates.json").read_text(encoding="utf-8"))
    gt_dir = Path(cand["gt_dir"])
    labelling_set = cand.get("set", work.name)
    out = manual_path(gt_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    today = dt.date.today().isoformat()
    # Several labelling sets share the file (dev, test); replace only this set's records.
    kept = [l for l in (out.read_text(encoding="utf-8").splitlines() if out.exists() else [])
            if l.strip() and json.loads(l).get("set", "svg_regions") != labelling_set]
    with out.open("w", encoding="utf-8") as f:
        for l in kept:
            f.write(l + "\n")
        for g in cand["groups"]:
            d = decisions.get(g["id"])
            if not d or d.get("status") not in ("text", "no_text", "unreadable"):
                continue
            for m in g["members"]:
                f.write(json.dumps({"json": m["json"], "box": [round(v, 1) for v in m["box"]], "status": d["status"],
                                    "text": d.get("text", "").strip() if d["status"] == "text" else "",
                                    "group": g["id"], "set": labelling_set, "by": who, "date": today},
                                   ensure_ascii=False) + "\n")
    return out


_PAGE = """<!doctype html><html lang="pl"><head><meta charset="utf-8"><title>Tekst bez wzorca</title>
<style>
body{font:14px system-ui,sans-serif;margin:16px;background:#fafafa;color:#222}
.card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:10px;margin:0 0 10px;display:flex;
flex-wrap:wrap;gap:12px;align-items:center}.card>div{flex:1 1 280px;min-width:0}
.card img{max-width:min(320px,100%);max-height:160px;image-rendering:pixelated;border:1px solid #eee;background:#fff}
.card.text{border-left:6px solid #2e7d32}.card.no_text{border-left:6px solid #c62828}.card.unreadable{border-left:6px solid #888}
.card.sugg{border-left:6px dashed #999;background:#fffdf3}.sugg-note{font-size:12px;color:#8a6d00}
input{font:15px monospace;width:100%;max-width:420px;padding:4px}
button{margin:4px 4px 0 0;padding:4px 10px;cursor:pointer}
.meta{color:#666;font-size:12px}
#bar{position:sticky;top:0;background:#fafafa;padding:8px 0;z-index:1;border-bottom:1px solid #ddd;margin-bottom:10px}
</style></head><body>
<div id="bar"><b>Tekst widoczny na obrazie, którego nie ma w SVG.</b> Oceniaj tylko to, co jest w <span style="color:#c00">czerwonej
ramce</span>. Wpisz dokładnie to, co widać (wielkość liter, znaki), albo wybierz „brak tekstu” (ikona, kształt bez napisu)
/ „nieczytelne” (napis jest, ale nie da się go przeczytać).
<br><span id="count"></span> <button onclick="save()">Zapisz</button>
<button onclick="acceptSuggestions()">Zatwierdź wszystkie podpowiedzi</button> <span id="msg"></span></div>
<div id="list"></div>
<script>
let cand = null, dec = {};
function render(){
  const list = document.getElementById('list'); list.innerHTML = '';
  for (const g of cand.groups){
    const d = dec[g.id] || {}; const div = document.createElement('div');
    const sg = g.suggestion;
    div.className = 'card ' + (d.status || (sg ? 'sugg' : ''));
    const SG = {text: 'tekst', no_text: 'brak tekstu', unreadable: 'nieczytelne'};
    div.innerHTML = `<img src="${g.crop}"><div>
      ${sg && !d.status ? `<div class="sugg-note">podpowiedź z wcześniejszej oceny: ${SG[sg.status]}${sg.text ? ' „' + sg.text.replace(/</g,'&lt;') + '”' : ''}</div>` : ''}
      <input value="${(d.text ?? (sg && sg.status === 'text' ? sg.text : g.proposal)).replace(/"/g,'&quot;')}" data-id="${g.id}">
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
function acceptSuggestions(){
  for (const g of cand.groups) if (g.suggestion && !dec[g.id]) dec[g.id] = {status: g.suggestion.status, text: g.suggestion.text};
  render();
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
    serve_page(work, port, _PAGE, lambda decisions: apply_decisions(work, decisions))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Label text that exists in the image but not in the SVG.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--gt", type=Path, default=DEFAULT_GT)
    p.add_argument("--runs", nargs="+", required=True, help="run names under artifacts/eval_runs")
    p.add_argument("--images", type=Path, help="limit to the images listed in this file")
    p.add_argument("--work", type=Path, default=WORK, help="working directory (one per labelled set)")
    p.add_argument("--suggest-from", type=Path, help="working directory of an earlier set whose decisions "
                   "are offered for regions that look the same")
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--work", type=Path, default=WORK)
    args = ap.parse_args(argv)
    if args.cmd == "prepare":
        images = ({r["json"] for r in images_from(args.images, load_index(args.gt))} if args.images else None)
        groups = prepare(args.gt, [RUNS / r for r in args.runs], args.work, images, args.suggest_from)
        print(f"{len(groups)} groups, {sum(len(g['members']) for g in groups)} regions, "
              f"{sum('suggestion' in g for g in groups)} with a suggestion -> {args.work}")
    else:
        serve(args.work, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
