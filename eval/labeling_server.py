"""Tiny local web server for the labelling pages (127.0.0.1 only).

Serves one HTML page plus the files of a work directory (crops, candidates.json) and
receives the decisions with POST /save.
"""
from __future__ import annotations

import http.server
import json
from pathlib import Path
from typing import Callable


def serve_page(work: Path, port: int, page_html: str, on_save: Callable[[dict], None]):
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(work), **kw)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                body = page_html.encode()
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
            on_save(decisions)
            self.send_response(204)
            self.end_headers()

        def log_message(self, *a):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"http://127.0.0.1:{port}/", flush=True)
    httpd.serve_forever()
