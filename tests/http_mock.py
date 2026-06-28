"""Shared local HTTP mock-server helper for the OGC reader tests.

The WCS, WFS, and OGC API – Features test suites each need a throwaway localhost
HTTP server that returns a fixed body for every GET. This single helper replaces
the per-file copies so the boilerplate is not duplicated across suites.
"""

from __future__ import annotations

import http.server
import socketserver
import threading
from collections import Counter


def make_fixed_body_server(body: str, content_type: str, path: str = ""):
    """Start a local HTTP server returning `body` for every GET.

    Args:
        body: The exact response body returned for every GET request.
        content_type: The ``Content-Type`` header to send.
        path: Optional path/query suffix appended to the returned base URL (e.g.
            ``"/ows"`` or ``"/mapserv?map=/map/x.map"``).

    Returns:
        A ``(url, counter, httpd)`` tuple. ``counter["GET"]`` tracks request count;
        the caller must ``httpd.shutdown()`` / ``httpd.server_close()`` when done.
    """
    counter: Counter[str] = Counter()
    payload = body.encode("utf-8")

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            counter["GET"] += 1
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args, **kwargs):  # noqa: N802
            return

    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{httpd.server_address[1]}{path}", counter, httpd
