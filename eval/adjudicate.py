"""Adjudicate disagreements between a run and the ground truth.

A wrong label only distorts the metrics where the engine's reading disagrees with it (if
both say the same, both are right or both are wrong in the very same way, which is rare).
So a human looks only at disagreements and decides who is right:

- mismatch  an accepted reading differs from the ground-truth word,
- missed    a ground-truth word no prediction covers,
- extra     an accepted reading where there is no ground truth at all.

Decisions: "engine" (the ground truth is right; the engine erred), "gt" (the ground truth is
wrong; the correct text is given, empty = there is no text), "unreadable" (excluded from
scoring). They are stored in artifacts/datasets/gt/manual/<gt dir>_corrections.jsonl and
applied by eval.run_eval.load_gt, so they survive regenerating the ground truth. Fixing labels
is fair on a test set too: it changes the yardstick, not the system.

Usage:
    python -m eval.adjudicate prepare --from-run RUN [--images eval/dev_sets/svg_dev_v1.txt]
    python -m eval.adjudicate serve
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

from eval.labeling_server import serve_page
from eval.metrics import PredWord, _area, _inter, coverage, match, overlap, scored_text
from eval.run_eval import (DEFAULT_GT, REPORTS, RUNS, ReplayEngine, correction_records, corrections_path,
                           images_from, load_gt, load_index, manual_records)
from ocr.config import load_config

REPO = Path(__file__).resolve().parent.parent
WORK = REPO / "artifacts" / "labeling" / "adjudicate"
KINDS = ("mismatch", "missed", "extra")


def collect(engine: ReplayEngine, gt_dir: Path, images: list[str]) -> list[dict]:
    cfg = load_config().eval
    cases = []
    for image_json in images:
        if image_json not in engine.preds:
            continue
        _, gt, ignore = load_gt(gt_dir, {"json": image_json})
        words, _ = engine.read_row({"json": image_json})
        pred = [PredWord(w.text, w.bbox_px, w.conf, w.status) for w in words]
        pairs = dict(match(gt, pred, cfg))
        matched = set(pairs.values())
        decided = [tuple(m["box"]) for m in manual_records(gt_dir, image_json)]
        base = {"json": image_json}
        for gi, g in enumerate(gt):
            if g.ignore:
                continue
            pi = pairs.get(gi)
            if pi is None:
                if coverage(g.bbox, [p.bbox for p in pred]) < cfg.detect_min_coverage:
                    cases.append({**base, "kind": "missed", "box": list(g.bbox), "gt": g.text, "pred": None,
                                  "votes": None})
            elif pred[pi].status == "accepted" and pred[pi].text != g.text:
                cases.append({**base, "kind": "mismatch", "box": list(g.bbox), "gt": g.text, "pred": pred[pi].text,
                              "votes": words[pi].votes})
        for pi, p in enumerate(pred):
            if p.status != "accepted" or pi in matched:
                continue
            if any(overlap(p.bbox, g.bbox) >= cfg.match_min_overlap for g in gt):
                continue
            if any(_inter(p.bbox, r) >= cfg.match_min_overlap * _area(p.bbox) for r in ignore):
                continue
            if any(overlap(p.bbox, d) >= 0.5 for d in decided):
                continue  # already decided while labelling text the SVG lacks
            cases.append({**base, "kind": "extra", "box": list(p.bbox), "gt": None, "pred": p.text,
                          "votes": words[pi].votes})
    # Only what is scored: no single characters or pure punctuation, and no differences that
    # vanish in scored form (typography, punctuation at the edges).
    def short(t):
        return t is None or len(scored_text(t, cfg)) < cfg.min_scored_chars
    cases = [c for c in cases
             if not (c["kind"] in ("mismatch", "missed") and short(c["gt"]))
             and not (c["kind"] == "extra" and short(c["pred"]))
             and not (c["kind"] == "mismatch" and scored_text(c["gt"], cfg) == scored_text(c["pred"], cfg))]
    for c in cases:
        c["box"] = [round(v, 1) for v in c["box"]]
    # Cases decided earlier (e.g. "the engine is wrong") are not asked again.
    cases = [c for c in cases if (c["kind"], tuple(c["box"])) not in adjudicated_all(gt_dir, c["json"])]
    for i, c in enumerate(cases):
        c["id"] = f"c{i:04d}"
    return cases


def adjudicated_all(gt_dir: Path, image_json: str) -> set:
    return {(r["kind"], tuple(r["box"])) for r in correction_records(gt_dir, image_json)}


def prepare(engine: ReplayEngine, gt_dir: Path, images: list[str], work: Path = WORK) -> list[dict]:
    cases = collect(engine, gt_dir, images)
    (work / "crops").mkdir(parents=True, exist_ok=True)
    for c in cases:
        x0, y0, x1, y1 = c["box"]
        h = max(10.0, y1 - y0)
        img = Image.open(gt_dir / Path(c["json"]).with_suffix(".png")).convert("RGB")
        cx0, cy0 = max(0, int(x0 - 3 * h)), max(0, int(y0 - 1.5 * h))
        crop = img.crop((cx0, cy0, int(x1 + 3 * h), int(y1 + 1.5 * h)))
        ImageDraw.Draw(crop).rectangle((x0 - cx0 - 2, y0 - cy0 - 2, x1 - cx0 + 2, y1 - cy0 + 2), outline=(220, 0, 0), width=2)
        crop.save(work / "crops" / f"{c['id']}.png")
        c["crop"] = f"crops/{c['id']}.png"
    (work / "candidates.json").write_text(json.dumps({"gt_dir": str(gt_dir), "run": engine.run_dir.name, "cases": cases},
                                                     ensure_ascii=False, indent=1), encoding="utf-8")
    return cases


def apply_decisions(work: Path, decisions: dict[str, dict], who: str = "user") -> Path:
    """Write the corrections file from decisions {case id: {"decision": ..., "text": ...}}."""
    cand = json.loads((work / "candidates.json").read_text(encoding="utf-8"))
    out = corrections_path(Path(cand["gt_dir"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    today = dt.date.today().isoformat()
    with out.open("w", encoding="utf-8") as f:
        for c in cand["cases"]:
            d = decisions.get(c["id"])
            if not d or d.get("decision") not in ("engine", "gt", "unreadable"):
                continue
            f.write(json.dumps({"json": c["json"], "kind": c["kind"], "box": c["box"], "gt": c["gt"], "pred": c["pred"],
                                "decision": d["decision"], "text": (d.get("text") or "").strip(),
                                "run": cand["run"], "by": who, "date": today}, ensure_ascii=False) + "\n")
    return out


_PAGE = """<!doctype html><html lang="pl"><head><meta charset="utf-8"><title>Rozstrzyganie</title>
<style>
body{font:14px system-ui,sans-serif;margin:16px;background:#fafafa;color:#222}
.card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:10px;margin:0 0 10px;display:flex;
flex-wrap:wrap;gap:12px;align-items:center}.card>div{flex:1 1 280px;min-width:0}
.card img{max-width:min(420px,100%);max-height:180px;border:1px solid #eee}
.card.engine{border-left:6px solid #1565c0}.card.gt{border-left:6px solid #2e7d32}.card.unreadable{border-left:6px solid #888}
.k{font-size:12px;text-transform:uppercase;color:#666}.t{font:15px monospace;background:#f3f3f3;padding:1px 4px}
input{font:15px monospace;width:100%;max-width:380px;padding:4px}
button{margin:4px 4px 0 0;padding:4px 10px;cursor:pointer}
#bar{position:sticky;top:0;background:#fafafa;padding:8px 0;z-index:1;border-bottom:1px solid #ddd;margin-bottom:10px}
</style></head><body>
<div id="bar"><b>Kto ma rację w czerwonej ramce: silnik czy tekst wzorcowy?</b>
„Silnik się myli” = wzorzec dobry. „Wzorzec się myli” = wpisz dokładnie, co widać (puste pole = nie ma tu tekstu).
<br><span id="count"></span> <button onclick="save()">Zapisz</button> <span id="msg"></span></div>
<div id="list"></div>
<script>
const KIND = {mismatch: 'silnik przeczytał inaczej niż wzorzec', missed: 'wzorzec ma tekst, silnik nic nie wykrył',
              extra: 'silnik coś przeczytał, wzorzec nie ma tu tekstu'};
let cand = null, dec = {};
const esc = s => (s ?? '—').replace(/&/g,'&amp;').replace(/</g,'&lt;');
function render(){
  const list = document.getElementById('list'); list.innerHTML = '';
  for (const c of cand.cases){
    const d = dec[c.id] || {}; const div = document.createElement('div');
    div.className = 'card ' + (d.decision || '');
    const votes = c.votes ? Object.entries(c.votes).map(([k,v]) => `${k}: <span class="t">${esc(v)}</span>`).join(' · ') : '';
    const proposal = d.text ?? (c.kind === 'missed' ? c.gt : (c.pred ?? ''));
    div.innerHTML = `<img src="${c.crop}"><div>
      <div class="k">${KIND[c.kind]}</div>
      <div>wzorzec: <span class="t">${esc(c.gt)}</span> &nbsp; silnik: <span class="t">${esc(c.pred)}</span></div>
      <div style="font-size:12px;color:#666">${votes}</div>
      <input value="${String(proposal).replace(/"/g,'&quot;')}">
      <div><button data-d="engine">silnik się myli</button><button data-d="gt">wzorzec się myli</button>
      <button data-d="unreadable">nieczytelne</button></div></div>`;
    div.querySelectorAll('button').forEach(b => b.onclick = () => {
      dec[c.id] = {decision: b.dataset.d, text: div.querySelector('input').value};
      div.className = 'card ' + b.dataset.d; upd();
    });
    div.querySelector('input').oninput = e => { if (dec[c.id]) dec[c.id].text = e.target.value; };
    list.appendChild(div);
  }
  upd();
}
function upd(){ document.getElementById('count').textContent = `Rozstrzygnięte: ${Object.keys(dec).length} z ${cand.cases.length}`; }
async function save(){
  const r = await fetch('save', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(dec)});
  document.getElementById('msg').textContent = r.ok ? 'Zapisano ' + new Date().toLocaleTimeString() : 'Błąd zapisu';
}
Promise.all([fetch('candidates.json').then(r=>r.json()), fetch('decisions.json').then(r=>r.ok?r.json():{})])
  .then(([c, d]) => { cand = c; dec = d; render(); });
</script></body></html>"""


def serve(work: Path = WORK, port: int = 8766):
    serve_page(work, port, _PAGE, lambda decisions: apply_decisions(work, decisions))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Adjudicate disagreements between a run and the ground truth.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--from-run", required=True)
    p.add_argument("--gt", type=Path, default=DEFAULT_GT)
    p.add_argument("--images", type=Path, help="limit to the images listed in this file")
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8766)
    args = ap.parse_args(argv)
    if args.cmd == "prepare":
        engine = ReplayEngine(RUNS / args.from_run, REPORTS, reapply_flags=False)
        images = ([r["json"] for r in images_from(args.images, load_index(args.gt))] if args.images
                  else list(engine.preds))
        cases = prepare(engine, args.gt, images)
        counts = {k: sum(c["kind"] == k for c in cases) for k in KINDS}
        print(f"{len(cases)} cases {counts} -> {WORK}")
    else:
        serve(port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
