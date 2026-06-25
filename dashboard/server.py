"""Serve the standalone FAW_RNN progress dashboard on localhost."""
from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
from pathlib import Path
import re
import signal
import subprocess
import threading
from typing import Any

from dashboard.collector import DashboardState


DASHBOARD_DIR = Path(__file__).resolve().parent
STATIC_DIR = DASHBOARD_DIR / "static"
LAN_HOST_TOKEN = "private-lan"


def resolve_bind_host(host: str) -> str:
    """Resolve the special private-LAN token without binding VPN or loopback interfaces."""
    if host != LAN_HOST_TOKEN:
        return host
    for index in range(10):
        completed = subprocess.run(
            ["/sbin/ifconfig", f"en{index}"],
            text=True,
            capture_output=True,
            check=False,
        )
        match = re.search(r"^\s*inet (\d+(?:\.\d+){3})\b", completed.stdout, re.MULTILINE)
        if match is None:
            continue
        candidate = match.group(1)
        address = ipaddress.ip_address(candidate)
        if address.version == 4 and address.is_private and not address.is_link_local:
            return candidate
    raise RuntimeError("No private LAN IPv4 address found on en0-en9")


class DashboardHandler(SimpleHTTPRequestHandler):
    """Serve static assets and the cached status API."""

    state: DashboardState

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def _send_json(self, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        if self.path == "/api/status":
            self._send_json(self.state.snapshot())
            return
        if self.path == "/api/refresh":
            self.state.request_refresh()
            self._send_json({"accepted": True})
            return
        if self.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def log_message(self, format: str, *args: Any) -> None:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--once", action="store_true", help="Refresh once, print JSON, and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = DashboardState()
    if args.once:
        print(json.dumps(state.refresh(), indent=2))
        return

    bind_host = resolve_bind_host(args.host)
    DashboardHandler.state = state
    server = ThreadingHTTPServer((bind_host, args.port), DashboardHandler)
    state.start()

    def stop_server(*_: Any) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop_server)
    signal.signal(signal.SIGINT, stop_server)
    print(f"FAW_RNN dashboard listening on http://{bind_host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    finally:
        state.stop()
        server.server_close()


if __name__ == "__main__":
    main()
