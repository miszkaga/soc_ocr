"""Adjudicate disagreements between a run and the ground truth.

A wrong label only distorts the metrics where the engine's reading disagrees with it (if
both say the same, both are right or both are wrong in the very same way, which is rare).
So a human looks only at disagreements and decides who is right:

- mismatch  a group of overlapping ground-truth words and readings whose text, aligned
            character by character, differs where it was accepted,
- missed    ground truth nothing was read for,
- extra     an accepted reading where there is no ground truth at all.
How the engine splits words or places its boxes is not a disagreement (eval.metrics).

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
from eval.metrics import PredWord, evaluate_image, overlap
from eval.run_eval import (DEFAULT_GT, REPORTS, RUNS, ReplayEngine, correction_records, corrections_path,
                           images_from, load_gt, load_index, manual_records)
from ocr.config import load_config

REPO = Path(__file__).resolve().parent.parent
WORK = REPO / "artifacts" / "labeling" / "adjudicate"
KINDS = ("mismatch", "missed", "extra")


def _union(boxes):
    return [round(min(b[0] for b in boxes), 1), round(min(b[1] for b in boxes), 1),
            round(max(b[2] for b in boxes), 1), round(max(b[3] for b in boxes), 1)]


def collect(engine: ReplayEngine, gt_dir: Path, images: list[str]) -> list[dict]:
    """Disagreements as the metrics see them: groups of overlapping ground-truth words and
    readings whose aligned text differs (accepted readings only), ground truth nothing was read
    for, and accepted readings where there is no ground truth. How the engine splits words or
    places boxes is not a disagreement."""
    cfg = load_config().eval
    cases = []
    for image_json in images:
        if image_json not in engine.preds:
            continue
        _, gt, ignore = load_gt(gt_dir, {"json": image_json})
        words, _ = engine.read_row({"json": image_json})
        pred = [PredWord(w.text, w.bbox_px, w.conf, w.status) for w in words]
        res = evaluate_image(gt, pred, cfg, ignore)
        decided_regions = [tuple(m["box"]) for m in manual_records(gt_dir, image_json)]
        decided = correction_records(gt_dir, image_json)
        decided_boxes = [tuple(r["box"]) for r in decided]
        for d in res.disagreements:
            gws = [{"text": gt[i].text, "box": [round(v, 1) for v in gt[i].bbox]} for i in d["gt"]]
            pws = [words[i] for i in d["pred"]]
            boxes = [g["box"] for g in gws] or [list(w.bbox_px) for w in pws]
            case = {"json": image_json, "kind": d["kind"], "box": _union(boxes), "gt_words": gws,
                    "gt": " ".join(g["text"] for g in gws) or None,
                    "pred": " ".join(w.text for w in pws) or None,
                    "pred_boxes": [[round(v, 1) for v in w.bbox_px] for w in pws],
                    "votes": [w.votes for w in pws if w.votes] or None}
            # Already decided: this group, every wrong word of it (word-level decisions made
            # earlier), or for invented text the region labelled while labelling icons.
            if any(_same(tuple(case["box"]), b) for b in decided_boxes):
                continue
            if d["wrong"] and all(any(_same(gt[i].bbox, b) for b in decided_boxes) for i in d["wrong"]):
                continue
            if d["kind"] == "extra" and any(overlap(tuple(case["box"]), r) >= 0.5 for r in decided_regions):
                continue
            cases.append(case)
    for i, c in enumerate(cases):
        c["id"] = f"c{i:04d}"
    return cases


def _same(a, b, tol: float = 1.0) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def prepare(engine: ReplayEngine, gt_dir: Path, images: list[str], work: Path = WORK) -> list[dict]:
    cases = collect(engine, gt_dir, images)
    (work / "crops").mkdir(parents=True, exist_ok=True)
    for c in cases:
        boxes = [c["box"]] + c["pred_boxes"]
        x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
        x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
        h = max(10.0, c["box"][3] - c["box"][1])
        img = Image.open(gt_dir / Path(c["json"]).with_suffix(".png")).convert("RGB")
        cx0, cy0 = max(0, int(x0 - 3 * h)), max(0, int(y0 - 1.5 * h))
        crop = img.crop((cx0, cy0, int(x1 + 3 * h), int(y1 + 1.5 * h)))
        d = ImageDraw.Draw(crop)
        # Ground-truth words in green, the engine's readings in blue.
        for b, colour in [(g["box"], (0, 160, 0)) for g in c["gt_words"]] + [(b, (0, 70, 230)) for b in c["pred_boxes"]]:
            d.rectangle((b[0] - cx0 - 2, b[1] - cy0 - 2, b[2] - cx0 + 2, b[3] - cy0 + 2), outline=colour, width=2)
        crop.save(work / "crops" / f"{c['id']}.png")
        c["crop"] = f"crops/{c['id']}.png"
    (work / "candidates.json").write_text(json.dumps({"gt_dir": str(gt_dir), "set": work.name, "run": engine.run_dir.name,
                                                     "cases": cases},
                                                     ensure_ascii=False, indent=1), encoding="utf-8")
    return cases


def apply_decisions(work: Path, decisions: dict[str, dict], who: str = "user") -> Path:
    """Write the corrections file from decisions {case id: {"decision": ..., "text": ...}}."""
    cand = json.loads((work / "candidates.json").read_text(encoding="utf-8"))
    out = corrections_path(Path(cand["gt_dir"]))
    out.parent.mkdir(parents=True, exist_ok=True)
    today = dt.date.today().isoformat()
    adjudication_set = cand.get("set", work.name)
    # Dev and test adjudication share the file; replace only this set's records.
    kept = [l for l in (out.read_text(encoding="utf-8").splitlines() if out.exists() else [])
            if l.strip() and json.loads(l).get("set", "adjudicate") != adjudication_set]
    with out.open("w", encoding="utf-8") as f:
        for l in kept:
            f.write(l + "\n")
        for c in cand["cases"]:
            d = decisions.get(c["id"])
            if not d or d.get("decision") not in ("engine", "gt", "unreadable"):
                continue
            f.write(json.dumps({"json": c["json"], "kind": c["kind"], "box": c["box"], "gt": c["gt"], "pred": c["pred"],
                                "gt_words": c.get("gt_words"), "pred_boxes": c.get("pred_boxes"),
                                "decision": d["decision"], "text": (d.get("text") or "").strip(),
                                "run": cand["run"], "set": adjudication_set, "by": who, "date": today},
                               ensure_ascii=False) + "\n")
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
<div id="bar"><b>Kto ma rację: silnik czy tekst wzorcowy?</b> Ramka <span style="color:#090">zielona</span> = słowo
wzorca, <span style="color:#14e">niebieska</span> = słowo odczytane przez silnik.
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
    const votes = (c.votes || []).map(v => Object.entries(v).map(([k,x]) => `${k}: <span class="t">${esc(x)}</span>`).join(' · ')).join('<br>');
    const proposal = d.text ?? (c.kind === 'missed' ? c.gt : (c.pred ?? ''));
    div.innerHTML = `<img src="${c.crop}"><div>
      <div class="k">${c.id} · ${KIND[c.kind]}</div>
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
    p.add_argument("--work", type=Path, default=WORK, help="working directory (one per labelled set)")
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8766)
    s.add_argument("--work", type=Path, default=WORK)
    args = ap.parse_args(argv)
    if args.cmd == "prepare":
        engine = ReplayEngine(RUNS / args.from_run, REPORTS, reapply_flags=False)
        images = ([r["json"] for r in images_from(args.images, load_index(args.gt))] if args.images
                  else list(engine.preds))
        cases = prepare(engine, args.gt, images, args.work)
        counts = {k: sum(c["kind"] == k for c in cases) for k in KINDS}
        print(f"{len(cases)} cases {counts} -> {args.work}")
    else:
        serve(args.work, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
