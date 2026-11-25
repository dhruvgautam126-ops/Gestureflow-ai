"""Pose-only human detection for the HOME GUARD AI desktop application."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

POSE_MODEL = Path(__file__).resolve().parent.parent / "pose_landmarker_lite.task"
POSE_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15),
    (15, 17), (15, 19), (15, 21),
    (17, 19),
    (12, 14), (14, 16),
    (16, 18), (16, 20), (16, 22),
    (18, 20),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27),
    (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28),
    (28, 30), (28, 32), (30, 32),
)


@dataclass(frozen=True)
class HumanDetectionStatus:
    state: str
    message: str
    people: int
    frame: np.ndarray | None


class HumanDetectionWorker:
    """Run MediaPipe pose detection on camera frames without blocking Tkinter."""

    def __init__(self, frame_provider: Callable[[], np.ndarray | None]) -> None:
        self._frame_provider = frame_provider
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = "off"
        self._message = ""
        self._people = 0
        self._frame: np.ndarray | None = None

    def start(self) -> bool:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stop.clear()
            self._state = "starting"
            self._message = "Starting human detection..."
            self._people = 0
            self._frame = None
            self._thread = threading.Thread(
                target=self._run,
                name="HomeGuardHumanDetection",
                daemon=True,
            )
            self._thread.start()
            return True

    def request_stop(self) -> None:
        self._stop.set()
        with self._lock:
            self._frame = None
            self._people = 0
            if self._state in ("starting", "on"):
                self._state = "stopping"
                self._message = "Stopping human detection..."

    def shutdown(self, timeout: float = 3.0) -> None:
        self.request_stop()
        with self._lock:
            thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=timeout)

    def snapshot(self) -> HumanDetectionStatus:
        with self._lock:
            return HumanDetectionStatus(
                state=self._state,
                message=self._message,
                people=self._people,
                frame=self._frame,
            )

    def _run(self) -> None:
        try:
            if not POSE_MODEL.is_file():
                raise FileNotFoundError(f"Pose model not found: {POSE_MODEL}")

            import mediapipe as mp

            base_options = mp.tasks.BaseOptions
            vision = mp.tasks.vision
            options = vision.PoseLandmarkerOptions(
                base_options=base_options(model_asset_path=str(POSE_MODEL)),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=2,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )

            with vision.PoseLandmarker.create_from_options(options) as detector:
                self._set_running()
                last_timestamp = 0
                while not self._stop.is_set():
                    frame = self._frame_provider()
                    if frame is None:
                        self._stop.wait(0.02)
                        continue

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb_frame,
                    )
                    timestamp = max(
                        time.monotonic_ns() // 1_000_000,
                        last_timestamp + 1,
                    )
                    last_timestamp = timestamp
                    result = detector.detect_for_video(mp_image, timestamp)

                    annotated = frame.copy()
                    people = _draw_poses(annotated, result)
                    _draw_detection_status(annotated, people)
                    self._publish(annotated, people)
        except Exception as exc:
            self._set_error(str(exc))
        finally:
            self._finish()

    def _set_running(self) -> None:
        with self._lock:
            if not self._stop.is_set():
                self._state = "on"
                self._message = "Human detection is running."

    def _publish(self, frame: np.ndarray, people: int) -> None:
        with self._lock:
            if not self._stop.is_set():
                self._frame = frame
                self._people = people

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._state = "error"
            self._message = f"Human detection failed: {message}"
            self._people = 0
            self._frame = None

    def _finish(self) -> None:
        with self._lock:
            self._frame = None
            self._people = 0
            if self._state != "error":
                self._state = "off"
                self._message = ""


def _draw_poses(frame: np.ndarray, result) -> int:
    height, width = frame.shape[:2]
    poses = result.pose_landmarks or ()

    for person_index, landmarks in enumerate(poses, start=1):
        points = [
            (
                min(width - 1, max(0, int(landmark.x * width))),
                min(height - 1, max(0, int(landmark.y * height))),
            )
            for landmark in landmarks
        ]
        visible = [
            getattr(landmark, "visibility", 0.0) > 0.5
            for landmark in landmarks
        ]

        for start, end in POSE_CONNECTIONS:
            if start < len(points) and end < len(points) and visible[start] and visible[end]:
                cv2.line(frame, points[start], points[end], (0, 255, 0), 2)

        visible_points = []
        for point, is_visible in zip(points, visible):
            if is_visible:
                cv2.circle(frame, point, 4, (0, 0, 255), -1)
                visible_points.append(point)

        if visible_points:
            x_values = [point[0] for point in visible_points]
            y_values = [point[1] for point in visible_points]
            x1, x2 = min(x_values), max(x_values)
            y1, y2 = min(y_values), max(y_values)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                frame,
                f"PERSON {person_index}",
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )

    return len(poses)


def _draw_detection_status(frame: np.ndarray, people: int) -> None:
    status = "HUMAN DETECTED" if people else "NO HUMAN DETECTED"
    color = (0, 255, 0) if people else (0, 0, 255)
    cv2.putText(
        frame,
        f"PEOPLE: {people}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        status,
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
    )
