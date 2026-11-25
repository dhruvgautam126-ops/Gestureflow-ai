"""One webcam owned by a background thread so the window stays responsive."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np


CAMERA_INDEX = 0
REQUESTED_WIDTH = 1280
REQUESTED_HEIGHT = 720
REQUESTED_FPS = 30


@dataclass(frozen=True)
class CameraStatus:
    """Snapshot of the camera for the status panel."""

    state: str
    message: str
    width: int
    height: int
    fps: float


@dataclass(frozen=True)
class CameraFramePacket:
    """A single camera frame with monotonic sequence number and capture timestamp.

    seq increments by 1 each time a new frame is published by the capture thread.
    The FrameCoordinator uses seq to detect duplicates and to match worker results
    to the frame they were computed from.

    timestamp is time.monotonic() at the moment the frame was stored.

    frame is the raw BGR uint8 ndarray.  Callers that may mutate the frame
    (e.g. the UI compositor) must copy it first with frame.copy().
    """

    seq: int
    timestamp: float
    frame: np.ndarray


class CameraCapture:
    """Open webcam 0, keep the latest frame, and release the device on stop."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame = None
        self._state = "off"
        self._message = ""
        self._width = 0
        self._height = 0
        self._fps = 0.0
        self._last_tick = 0.0
        self._seq: int = 0
        self._packet: CameraFramePacket | None = None

    def start(self) -> bool:
        """Begin opening the camera. Returns True on success."""
        with self._lock:
            thread = self._thread
            # Wait for any previous thread to fully terminate first
            if thread is not None and thread.is_alive():
                return False
            # Clear the thread reference so we can start fresh
            self._thread = None
            # Ensure stop event is cleared and state reset
            self._stop.clear()
            self._frame = None
            self._width = 0
            self._height = 0
            self._fps = 0.0
            self._last_tick = 0.0
            self._seq = 0
            self._packet = None
            self._state = "starting"
            self._message = "Opening webcam..."
            self._thread = threading.Thread(
                target=self._run,
                name="HomeGuardCamera",
                daemon=True,
            )
            self._thread.start()
            return True

    def stop(self) -> None:
        """Request the capture thread to stop. UI-layer compatibility wrapper."""
        self.request_stop()

    def request_stop(self) -> None:
        """Ask the capture thread to release the webcam. Safe to call from the UI thread."""
        self._stop.set()
        with self._lock:
            self._frame = None
            self._packet = None
            if self._state in ("starting", "on"):
                self._state = "stopping"
                self._message = "Releasing webcam..."

    def get_frame(self):
        """Return a copy of the latest BGR frame, or None when the camera is not on.

        Preserved for backward compatibility with legacy workers and all existing
        standalone tests.  The FrameCoordinator uses get_current_packet() instead.
        """
        with self._lock:
            if self._state != "on" or self._frame is None:
                return None
            return self._frame.copy()

    def get_current_packet(self) -> CameraFramePacket | None:
        """Return the latest CameraFramePacket, or None when no frame is available.

        Used by FrameCoordinator to obtain packets for the distributor thread.
        Returns None when the camera is off, shutting down, or no frames have been
        captured yet.  The packet itself is not copied; callers must copy the frame
        if they intend to mutate it.
        """
        with self._lock:
            if self._packet is None:
                return None
            return self._packet

    def snapshot(self) -> CameraStatus:
        with self._lock:
            return CameraStatus(
                state=self._state,
                message=self._message,
                width=self._width,
                height=self._height,
                fps=self._fps,
            )

    def shutdown(self, timeout: float = 3.0) -> None:
        """Stop capture and wait so the webcam is released before the process exits."""
        self.request_stop()
        with self._lock:
            thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        with self._lock:
            if self._thread is not None and not self._thread.is_alive():
                self._thread = None

    def _run(self) -> None:
        camera = None
        try:
            camera = self._open_camera()
            if camera is None:
                self._mark_open_failed()
                return

            width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            with self._lock:
                if self._stop.is_set():
                    self._state = "stopping"
                else:
                    self._state = "on"
                    self._message = "Live preview is running."
                    self._width = width
                    self._height = height

            while not self._stop.is_set():
                ok, frame = camera.read()
                if not ok or frame is None:
                    with self._lock:
                        self._frame = None
                        self._packet = None
                        self._state = "error"
                        self._message = "Camera frame could not be read."
                    break
                self._publish(frame)
        finally:
            if camera is not None:
                camera.release()
            with self._lock:
                self._frame = None
                self._packet = None
                self._width = 0
                self._height = 0
                self._fps = 0.0
                self._last_tick = 0.0
                if self._state != "error":
                    self._state = "off"
                    self._message = ""

    def _mark_open_failed(self) -> None:
        with self._lock:
            self._frame = None
            self._packet = None
            if self._stop.is_set():
                self._state = "off"
                self._message = ""
            else:
                self._state = "error"
                self._message = "Webcam could not be opened."

    def _publish(self, frame) -> None:
        now = time.perf_counter()
        with self._lock:
            if self._stop.is_set():
                return
            if self._last_tick:
                interval = now - self._last_tick
                if interval > 0:
                    measured = 1.0 / interval
                    self._fps = measured if self._fps == 0 else (self._fps * 0.9) + (measured * 0.1)
            self._last_tick = now
            self._frame = frame
            self._height, self._width = frame.shape[:2]
            self._seq += 1
            self._packet = CameraFramePacket(
                seq=self._seq,
                timestamp=time.monotonic(),
                frame=frame,
            )

    def _open_camera(self):
        camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not camera.isOpened():
            camera.release()
            camera = cv2.VideoCapture(CAMERA_INDEX)
        if not camera.isOpened():
            camera.release()
            return None

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, REQUESTED_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUESTED_HEIGHT)
        camera.set(cv2.CAP_PROP_FPS, REQUESTED_FPS)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return camera
