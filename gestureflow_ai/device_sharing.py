"""Opt-in, same-network file handoff for GestureFlow AI.

The receiving device must start its receiver and approve every incoming file.
This first prototype uses HTTP on a trusted local network, so it is intended
for small, non-sensitive files and should not be exposed to the public internet.
"""

from __future__ import annotations

import hmac
import http.server
import json
import os
import queue
import secrets
import socket
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote


MAX_FILE_BYTES = 100 * 1024 * 1024
SHARE_PORT = 8765
APP_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RECEIVED_DIR = Path.home() / "Documents" / "GestureFlow AI" / "Received"
# Kept as an import-compatible default; each receiver uses its own destination.
RECEIVED_DIR = DEFAULT_RECEIVED_DIR


@dataclass
class IncomingShare:
    request_id: str
    filename: str
    size: int
    sender: str
    staged_path: Path
    decision: threading.Event = field(default_factory=threading.Event)
    accepted: bool = False
    saved_path: Path | None = None
    save_error: str = ""


class DeviceShareReceiver:
    """Small HTTP receiver with one-time pairing code and user approval."""

    def __init__(self, receive_directory: str | Path | None = None):
        self.code = ""
        self.address = ""
        self.port = 0
        self.receive_directory = Path(receive_directory or DEFAULT_RECEIVED_DIR).expanduser()
        self._server = None
        self._thread = None
        self._pending: queue.Queue[IncomingShare] = queue.Queue()
        self._requests: dict[str, IncomingShare] = {}
        self._lock = threading.Lock()
        self._remote_gesture: dict | None = None

    @property
    def running(self) -> bool:
        return self._server is not None

    def start(self) -> tuple[str, int, str]:
        if self.running:
            return self.address, self.port, self.code

        # Create the destination when receiving is enabled so the UI can point
        # to a folder that already exists on this laptop.
        self.receive_directory.mkdir(parents=True, exist_ok=True)

        receiver = self

        class Handler(http.server.BaseHTTPRequestHandler):
            server_version = "GestureFlowAIShare/0.1"

            def log_message(self, _format, *_args):
                return

            def _reply(self, status: int, message: str):
                payload = message.encode("utf-8", errors="replace")
                self.send_response(status)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                if self.path != "/health":
                    self._reply(404, "Not found")
                    return
                    self._reply(200, "GestureFlow AI receiver ready")

            def do_POST(self):
                if self.path == "/gesture":
                    supplied_code = self.headers.get("X-Share-Code", "")
                    if not hmac.compare_digest(supplied_code, receiver.code):
                        self._reply(401, "Pairing code was not accepted")
                        return
                    try:
                        length = int(self.headers.get("Content-Length", "0"))
                        if not 1 <= length <= 4096:
                            raise ValueError
                        payload = json.loads(self.rfile.read(length).decode("utf-8"))
                        state = payload.get("state")
                        filename = Path(str(payload.get("filename", ""))).name[:180]
                        if state not in {"holding", "released", "cancelled"}:
                            raise ValueError
                    except (ValueError, UnicodeError, json.JSONDecodeError):
                        self._reply(400, "Invalid gesture status")
                        return
                    with receiver._lock:
                        receiver._remote_gesture = {
                            "state": state,
                            "filename": filename,
                            "sender": self.client_address[0],
                            "updated_at": time.monotonic(),
                        }
                    self._reply(200, "Gesture status received")
                    return
                if self.path != "/share":
                    self._reply(404, "Not found")
                    return
                supplied_code = self.headers.get("X-Share-Code", "")
                if not hmac.compare_digest(supplied_code, receiver.code):
                    self._reply(401, "Pairing code was not accepted")
                    return
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    self._reply(400, "Invalid file size")
                    return
                if size <= 0 or size > MAX_FILE_BYTES:
                    self._reply(413, "File must be between 1 byte and 100 MB")
                    return

                raw_name = unquote(self.headers.get("X-File-Name", "shared-file"))
                filename = Path(raw_name.replace("\\", "/")).name
                filename = "".join(ch for ch in filename if ch.isprintable()).strip()
                if not filename or filename in {".", ".."}:
                    self._reply(400, "Invalid filename")
                    return

                staged = receiver.receive_directory / f".incoming-{secrets.token_hex(12)}.part"
                remaining = size
                try:
                    with staged.open("xb") as output:
                        while remaining:
                            chunk = self.rfile.read(min(1024 * 1024, remaining))
                            if not chunk:
                                raise ConnectionError("Sender disconnected during transfer")
                            output.write(chunk)
                            remaining -= len(chunk)
                except Exception:
                    staged.unlink(missing_ok=True)
                    self._reply(400, "Could not receive the complete file")
                    return

                request = IncomingShare(
                    request_id=secrets.token_hex(12),
                    filename=filename,
                    size=size,
                    sender=self.client_address[0],
                    staged_path=staged,
                )
                with receiver._lock:
                    receiver._requests[request.request_id] = request
                receiver._pending.put(request)
                if not request.decision.wait(timeout=120) or not request.accepted:
                    staged.unlink(missing_ok=True)
                    with receiver._lock:
                        receiver._requests.pop(request.request_id, None)
                    if request.save_error:
                        self._reply(500, f"Could not save the received file: {request.save_error}")
                    else:
                        self._reply(403, "Transfer declined or timed out")
                    return
                with receiver._lock:
                    receiver._requests.pop(request.request_id, None)
                saved_name = request.saved_path.name if request.saved_path else filename
                self._reply(200, f"Saved on receiver: {saved_name}")

        self.code = f"{secrets.randbelow(100_000_000):08d}"
        self._server = http.server.ThreadingHTTPServer(("0.0.0.0", SHARE_PORT), Handler)
        self._server.daemon_threads = True
        self.port = int(self._server.server_address[1])
        self.address = _local_address()
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self.address, self.port, self.code

    def next_request(self) -> IncomingShare | None:
        try:
            return self._pending.get_nowait()
        except queue.Empty:
            return None

    def remote_gesture_status(self) -> dict | None:
        with self._lock:
            status = self._remote_gesture
            if status is None:
                return None
            if time.monotonic() - status["updated_at"] > 30:
                self._remote_gesture = None
                return None
            return dict(status)

    def decide(self, request_id: str, accepted: bool) -> bool:
        with self._lock:
            request = self._requests.get(request_id)
            if request is None:
                return False
            if accepted:
                try:
                    self.receive_directory.mkdir(parents=True, exist_ok=True)
                    destination = _available_destination(
                        self.receive_directory / request.filename
                    )
                    # The staging file and final file share a directory, so the
                    # rename is atomic and the item appears only after approval.
                    os.replace(request.staged_path, destination)
                    request.saved_path = destination
                    request.accepted = True
                except OSError as exc:
                    request.save_error = str(exc)
                    request.accepted = False
            else:
                request.accepted = False
            request.decision.set()
            return True

    def stop(self) -> None:
        server, self._server = self._server, None
        self.code = ""
        self.address = ""
        self.port = 0
        with self._lock:
            self._remote_gesture = None
        if server is not None:
            server.shutdown()
            server.server_close()


def send_file(host: str, port: int, code: str, path: str | Path, timeout: int = 150) -> str:
    """Send one user-selected file and wait for the receiving user's decision."""
    import http.client
    from urllib.parse import quote

    file_path = Path(path).expanduser().resolve(strict=True)
    if not file_path.is_file():
        raise ValueError("Select a file to share")
    size = file_path.stat().st_size
    if size <= 0 or size > MAX_FILE_BYTES:
        raise ValueError("Choose a file between 1 byte and 100 MB")
    normalized_host = host.strip()
    if not normalized_host or any(char in normalized_host for char in "/\\@"):
        raise ValueError("Enter the receiving device's local IP address")
    if not 1 <= int(port) <= 65535:
        raise ValueError("Invalid receiver port")
    connection = http.client.HTTPConnection(normalized_host, int(port), timeout=timeout)
    try:
        connection.putrequest("POST", "/share")
        connection.putheader("Host", f"{normalized_host}:{int(port)}")
        connection.putheader("Content-Length", str(size))
        connection.putheader("Content-Type", "application/octet-stream")
        connection.putheader("X-Share-Code", code.strip())
        connection.putheader("X-File-Name", quote(file_path.name, safe=""))
        connection.endheaders()
        with file_path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                connection.send(chunk)
        response = connection.getresponse()
        message = response.read(1024).decode("utf-8", errors="replace")
        if response.status != 200:
            raise RuntimeError(message or f"Receiver returned HTTP {response.status}")
        return message
    finally:
        connection.close()


def send_gesture_state(host: str, port: int, code: str, state: str,
                       filename: str, timeout: int = 5) -> None:
    """Notify a paired receiver that a selected file is being held or released."""
    import http.client

    if state not in {"holding", "released", "cancelled"}:
        raise ValueError("Invalid handoff state")
    normalized_host = host.strip()
    if not normalized_host or any(char in normalized_host for char in "/\\@"):
        raise ValueError("Enter the receiving device's local IP address")
    body = json.dumps({"state": state, "filename": Path(filename).name}).encode("utf-8")
    connection = http.client.HTTPConnection(normalized_host, int(port), timeout=timeout)
    try:
        connection.request(
            "POST", "/gesture", body=body,
            headers={"Content-Type": "application/json", "Content-Length": str(len(body)),
                     "X-Share-Code": code.strip()},
        )
        response = connection.getresponse()
        message = response.read(1024).decode("utf-8", errors="replace")
        if response.status != 200:
            raise RuntimeError(message or f"Receiver returned HTTP {response.status}")
    finally:
        connection.close()


def _available_destination(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 10_000):
        candidate = path.with_name(f"{path.stem} ({index}){path.suffix}")
        if not candidate.exists():
            return candidate
    raise OSError("Too many files with this name already exist")


def _local_address() -> str:
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("192.0.2.1", 80))
        return str(probe.getsockname()[0])
    except OSError:
        return "127.0.0.1"
    finally:
        probe.close()
