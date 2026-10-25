#!/usr/bin/env bash
# Prepares a fresh clone to run the OCR engine on this machine (Linux laptop, see CLAUDE.md):
#   1. .venv with Python 3.11 and the pinned packages (ocr/requirements-cpu.txt);
#   2. Tesseract with Polish and English (checked; installing it needs sudo, so it is only shown);
#   3. the PaddleOCR models in artifacts/models/rapidocr: copied with --models-from DIR, otherwise
#      downloaded once by RapidOCR (checksums verified) - the only network access, packages and
#      models only; documents never leave the machine;
#   4. a check that the engine reads a test image.
#
# Usage:  ./setup.sh [--models-from DIR] [--eval]
#   --models-from DIR  copy the models from DIR (e.g. another clone's artifacts/models/rapidocr)
#   --eval             also install Chromium for the evaluation tools (eval/), not needed to run
# Then:   put files into input/ and run  .venv/bin/python -m ocr
set -euo pipefail
cd "$(dirname "$0")"

MODELS_FROM=""
EVAL=0
while [ $# -gt 0 ]; do
  case "$1" in
    --models-from) MODELS_FROM="${2:?--models-from needs a folder}"; shift 2 ;;
    --eval) EVAL=1; shift ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (see ./setup.sh --help)" >&2; exit 2 ;;
  esac
done

step() { printf '\n== %s\n' "$1"; }

step "Python 3.11 environment (.venv)"
if [ -x .venv/bin/python ] && .venv/bin/python -c 'import sys; sys.exit(sys.version_info[:2] != (3, 11))'; then
  echo ".venv exists (Python 3.11)"
elif command -v uv >/dev/null 2>&1 || [ -x "$HOME/.local/share/uv-bootstrap/bin/uv" ]; then
  UV="$(command -v uv || echo "$HOME/.local/share/uv-bootstrap/bin/uv")"
  "$UV" venv --python 3.11 .venv
elif command -v python3.11 >/dev/null 2>&1; then
  python3.11 -m venv .venv
else
  echo "Python 3.11 not found. Install uv (https://docs.astral.sh/uv/) or python3.11, then run again." >&2
  exit 1
fi
if command -v uv >/dev/null 2>&1 || [ -x "$HOME/.local/share/uv-bootstrap/bin/uv" ]; then
  UV="$(command -v uv || echo "$HOME/.local/share/uv-bootstrap/bin/uv")"
  "$UV" pip install --python .venv/bin/python -r ocr/requirements-cpu.txt
else
  .venv/bin/python -m pip install --upgrade pip >/dev/null
  .venv/bin/python -m pip install -r ocr/requirements-cpu.txt
fi

step "Tesseract (pol + eng)"
if ! command -v tesseract >/dev/null 2>&1; then
  echo "Tesseract is missing. Install it (needs sudo):" >&2
  echo "  sudo apt install tesseract-ocr tesseract-ocr-pol tesseract-ocr-eng" >&2
  exit 1
fi
langs="$(tesseract --list-langs 2>/dev/null)"
for l in pol eng; do
  if ! grep -qx "$l" <<<"$langs"; then
    echo "Tesseract has no '$l' language. Install it (needs sudo):" >&2
    echo "  sudo apt install tesseract-ocr-$l" >&2
    exit 1
  fi
done
tesseract --version | head -1

step "PaddleOCR models (artifacts/models/rapidocr)"
mkdir -p artifacts/models/rapidocr
if [ -n "$MODELS_FROM" ]; then
  cp -n "$MODELS_FROM"/*.onnx artifacts/models/rapidocr/ 2>/dev/null || true
  echo "copied from $MODELS_FROM"
fi
# The first use downloads whatever is missing (RapidOCR verifies the SHA-256 of every file).
.venv/bin/python - <<'EOF'
from PIL import Image, ImageDraw
from ocr.engines.registry import create
from ocr.config import load_config
engine = create(load_config().consensus.primary)
img = Image.new("RGB", (120, 40), "white")
ImageDraw.Draw(img).text((10, 12), "OCR test", fill="black")
engine.read(img)
v = engine.versions()
print("models:", ", ".join(f"{k}={v[k]}" for k in ("det_model", "rec_model") if k in v))
EOF

if [ "$EVAL" = 1 ]; then
  step "Chromium for the evaluation tools"
  .venv/bin/python -m playwright install chromium
fi

step "Check: the engine reads a test image"
.venv/bin/python - <<'EOF'
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from ocr.cli import run
import json

font = None
for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/TTF/DejaVuSans.ttf"):
    if Path(p).exists():
        font = ImageFont.truetype(p, 32)
with tempfile.TemporaryDirectory() as d:
    src = Path(d) / "in"
    src.mkdir()
    img = Image.new("RGB", (560, 80), "white")
    ImageDraw.Draw(img).text((20, 20), "Azure Firewall 10.0.0.1", font=font or ImageFont.load_default(), fill="black")
    img.save(src / "check.png")
    run(src, Path(d) / "out", log=open("/dev/null", "w"))
    doc = json.loads((Path(d) / "out" / "json" / "check.png.json").read_text())
    text = doc["pages"][0]["regions"][0]["text"]
    print("read:", repr(text))
    if text.split() != ["Azure", "Firewall", "10.0.0.1"]:
        raise SystemExit("the engine did not read the test image as expected")
EOF

echo
echo "Ready. Put PDFs and images into input/ and run:  .venv/bin/python -m ocr"
echo "Results: out/text.txt, out/boxes/, out/json/"
