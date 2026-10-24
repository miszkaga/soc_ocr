"""python -m ocr extract FOLDER (see ocr/cli.py)."""
import sys

from ocr.cli import main

raise SystemExit(main(sys.argv[1:]))
