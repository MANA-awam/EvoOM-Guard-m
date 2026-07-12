"""`python -m app --port N` — serve the calculator over HTTP (stdlib only)."""

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from app import add


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 - stdlib naming
        url = urlparse(self.path)
        if url.path != "/add":
            self.send_response(404)
            self.end_headers()
            return
        q = parse_qs(url.query)
        try:
            result = add(float(q["a"][0]), float(q["b"][0]))
        except (KeyError, ValueError):
            self.send_response(400)
            self.end_headers()
            return
        body = json.dumps({"result": result}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # keep judge output clean
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    args = ap.parse_args()
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
