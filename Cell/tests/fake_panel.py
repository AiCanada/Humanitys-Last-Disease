"""A stand-in for the ``human`` option panel, for testing the Cell relay on Windows.

It follows the real panel's protocol as read from ``BrowserSession`` in
``src/rfproteina/disease/ui.py``: HTTP/1.0 answers with Content-Length, ``GET /``
redirecting to the current step, ``GET /step/N`` for an earlier step counting as a
rewind, a POST that blocks until the next page is ready, ``/progress`` JSON, and a
Close that answers with the window-closing page and then stops the server.

Run as a script it also plays the program: it prints ``Option panel: <address>``
the way the real one does, so the relay's supervisor can be tested with it.

    python fake_panel.py [--delay SECONDS] [--port N] [--close-after STEPS]
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fake panel</title>
</head>
<body>
<main>
<form data-step="{step}" class="landing" method="post" action="/" enctype="multipart/form-data">
<h1>Fake panel step {step}</h1>
<input type="text" name="disease" value="">
<input type="file" name="target_structure" accept=".pdb,.cif,.PDB,.CIF">
<button type="submit">Continue</button>
</form>
</main>
<script>/* panel script */</script>
</body>
</html>
"""

CLOSING = "<!DOCTYPE html><html><head><title>Closing</title></head><body><p>Closing the panel.</p></body></html>"


class FakePanel:
    def __init__(self, delay: float = 0.0, close_after: int = 0, port: int = 0) -> None:
        self.delay = delay
        self.close_after = close_after
        self.step = 0
        self.closed = False
        self.rewinds = 0
        self.posts: list[dict[str, list[str]]] = []
        self.active = False
        #: Seconds each /progress answer is held back, to play a panel that is busy.
        self.progress_delay = 0.0
        #: Headers of the last page request, lower-cased.
        self.last_get_headers: dict[str, str] = {}
        self.lock = threading.Lock()
        panel = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                path = self.path.split("?", 1)[0]
                if path == "/progress":
                    if panel.progress_delay:
                        time.sleep(panel.progress_delay)
                    with panel.lock:
                        body = {
                            "active": panel.active,
                            "done": 1 if panel.active else 0,
                            "total": 4 if panel.active else 0,
                            "label": "fake work" if panel.active else "",
                            "action": "Downloading" if panel.active else "",
                            "step": panel.step,
                            "closed": panel.closed,
                            # Not something the real panel sends. A Cell relay that is
                            # looking for a panel on this computer skips an answer that
                            # carries it, so a live relay never adopts a test stand-in.
                            "stand_in": True,
                        }
                    self._send(json.dumps(body).encode("utf-8"), "application/json")
                    return
                if path == "/viewer/3dmol.js":
                    self._send(b"/* fake viewer */" + b" " * 4000, "application/javascript")
                    return
                if path.startswith("/structure/"):
                    self._send(b"data_fake\n" * 300, "chemical/x-mmcif")
                    return
                if path == "/closed":
                    self._send(PAGE.format(step=panel.step).replace("<form ", '<form data-closed="true" ').encode())
                    return
                asked = None
                if path.startswith("/step/") and path[6:].isdigit():
                    asked = int(path[6:])
                mirror = "X-Cell-No-Rewind" in self.headers
                with panel.lock:
                    panel.last_get_headers = {name.lower(): value for name, value in self.headers.items()}
                    current = panel.step
                    if asked is not None and asked < current and not mirror:
                        # The real panel rewinds the run here, unless a mirror asked.
                        panel.rewinds += 1
                        panel.step = asked
                        current = asked
                if asked != current:
                    self._redirect(f"/step/{current}")
                    return
                self._send(PAGE.format(step=current).encode("utf-8"))

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                kind = self.headers.get("Content-Type", "")
                fields = parse_qs(raw.decode("utf-8", "replace")) if "urlencoded" in kind else {"raw": [str(len(raw))]}
                with panel.lock:
                    panel.posts.append(fields)
                    panel.active = True
                time.sleep(panel.delay)
                with panel.lock:
                    panel.active = False
                    closing = "close" in fields or (panel.close_after and len(panel.posts) >= panel.close_after)
                    if closing:
                        panel.closed = True
                    else:
                        panel.step += 1
                    current = panel.step
                if closing:
                    self._send(CLOSING.encode("utf-8"))
                    threading.Thread(target=panel.stop, daemon=True).start()
                    return
                self._redirect(f"/step/{current}")

            def log_message(self, *arguments: object) -> None:
                return

            def _redirect(self, location: str) -> None:
                self.send_response(303)
                self.send_header("Location", location)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", "0")
                self.end_headers()

            def _send(self, body: bytes, content_type: str = "text/html; charset=utf-8") -> None:
                # A caller that gave up waiting is not a failure, as in the real panel.
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except OSError:
                    return

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        time.sleep(0.3)
        self.server.shutdown()
        self.server.server_close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--close-after", type=int, default=0)
    arguments = parser.parse_args()
    print("Run log: fake.log")
    panel = FakePanel(arguments.delay, arguments.close_after, arguments.port)
    print(f"Option panel: http://127.0.0.1:{panel.port}/", flush=True)
    def listen() -> None:
        for line in sys.stdin:
            print("heard: " + line.strip(), flush=True)

    threading.Thread(target=listen, daemon=True).start()
    code = 0
    try:
        while panel.thread.is_alive():
            time.sleep(0.2)
        print("fake program finished", flush=True)
    except KeyboardInterrupt:
        print("Stopped.", flush=True)
        code = 130
    return code


if __name__ == "__main__":
    import os

    # The listener thread is still blocked on stdin, so leave without waiting for it.
    status = main()
    sys.stdout.flush()
    os._exit(status)
