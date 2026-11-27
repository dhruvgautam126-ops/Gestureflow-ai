"""Pose and facial-landmark detection module for Gesture.AI."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from app.face_landmarks import draw_face_landmarks, extract_face_landmark_draw_data

POSE_MODEL = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "human"
    / "pose_landmarker_full.task"
)
FACE_LANDMARK_MODEL = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "biometric"
    / "face_landmarker.task"
)
FACE_LANDMARK_INTERVAL_FRAMES = 3

# MediaPipe Pose 33 keypoint connections (COCO 17 skeleton mapped to 33 points)
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
    (28, 30), (30, 32), (32, 32),
)


@dataclass(frozen=True)
class HumanDetectionStatus:
    state: str
    message: str
    people: int
    frame: np.ndarray | None


class HumanDetectionWorker:
    """Run MediaPipe pose detection on camera frames without blocking Tkinter."""

    def __init__(
        self,
        frame_provider: Callable[[], np.ndarray | None] | None = None,
        *,
        coordinator=None,
        kind=None,
    ) -> None:
        self._frame_provider = frame_provider
        self._coordinator = coordinator
        self._kind = kind
        self._packet_queue: queue.Queue | None = (
            coordinator.get_packet_queue(kind) if coordinator is not None else None
        )
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

    def get_status(self) -> HumanDetectionStatus:
        with self._lock:
            return HumanDetectionStatus(
                state=self._state,
                message=self._message,
                people=self._people,
                frame=self._frame.copy() if self._frame is not None else None,
            )

    def _run(self) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
        except Exception as exc:
            with self._lock:
                self._state = "error"
                self._message = f"MediaPipe import failed: {exc}"
            return

        if not POSE_MODEL.exists():
            with self._lock:
                self._state = "error"
                self._message = f"Missing pose model: {POSE_MODEL.name}"
            return

        face_landmarker = None
        face_landmark_note = ""
        try:
            base_options = python.BaseOptions(model_asset_path=str(POSE_MODEL))
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                # VIDEO mode lets MediaPipe track detections across consecutive
                # camera frames instead of treating every frame as unrelated.
                running_mode=vision.RunningMode.VIDEO,
                num_poses=2,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            landmarker = vision.PoseLandmarker.create_from_options(options)
            if FACE_LANDMARK_MODEL.is_file():
                try:
                    face_options = vision.FaceLandmarkerOptions(
                        base_options=python.BaseOptions(
                            model_asset_path=str(FACE_LANDMARK_MODEL)
                        ),
                        running_mode=vision.RunningMode.VIDEO,
                        num_faces=2,
                        min_face_detection_confidence=0.5,
                        min_face_presence_confidence=0.5,
                        min_tracking_confidence=0.5,
                        output_face_blendshapes=False,
                        output_facial_transformation_matrixes=False,
                    )
                    face_landmarker = vision.FaceLandmarker.create_from_options(
                        face_options
                    )
                except Exception as exc:
                    face_landmark_note = f"Face landmarks unavailable: {exc}"
            else:
                face_landmark_note = (
                    f"Face landmark model missing: {FACE_LANDMARK_MODEL.name}"
                )
        except Exception as exc:
            with self._lock:
                self._state = "error"
                self._message = f"Failed to initialize PoseLandmarker: {exc}"
            return

        with self._lock:
            self._state = "on"
            self._message = (
                "Human and face landmark detection active."
                if face_landmarker is not None
                else f"Human detection active. {face_landmark_note}"
            )

        frame_index = 0
        last_timestamp_ms = 0
        latest_face_landmarks = ()
        try:
            while not self._stop.is_set():
                if self._coordinator is not None and self._packet_queue is not None:
                    try:
                        packet = self._packet_queue.get(timeout=0.05)
                    except queue.Empty:
                        continue

                    gen = self._coordinator.get_generation(self._kind)

                    frame = packet.frame
                    if frame is None or frame.size == 0:
                        continue

                    timestamp_ms = max(
                        int(packet.timestamp * 1000), last_timestamp_ms + 1
                    )
                    last_timestamp_ms = timestamp_ms

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)
                    frame_index += 1
                    if (
                        face_landmarker is not None
                        and frame_index % FACE_LANDMARK_INTERVAL_FRAMES == 0
                    ):
                        face_result = face_landmarker.detect_for_video(
                            mp_image, timestamp_ms
                        )
                        latest_face_landmarks = extract_face_landmark_draw_data(
                            face_result
                        )

                    people_count = 0
                    pose_draw_data_list = []

                    if result and result.pose_landmarks:
                        h_px, w_px = frame.shape[:2]
                        for landmarks in result.pose_landmarks:
                            points_norm = []
                            visible = []
                            for lm in landmarks:
                                # CameraCapture publishes the raw frame without a flip,
                                # and PoseLandmarker receives that same orientation.
                                points_norm.append((float(lm.x), float(lm.y)))
                                visible.append(bool(getattr(lm, 'visibility', 1.0) > 0.5))
                            pose_draw_data_list.append((tuple(points_norm), tuple(visible)))
                        people_count = len(result.pose_landmarks)

                    from app.coordinator import WorkerResult
                    worker_res = WorkerResult(
                        kind=self._kind,
                        source_seq=packet.seq,
                        source_ts=packet.timestamp,
                        generation=gen,
                        human_people=people_count,
                        human_pose_draw_data=tuple(pose_draw_data_list),
                        human_face_landmark_draw_data=latest_face_landmarks,
                    )
                    self._coordinator.publish_result(worker_res)

                    with self._lock:
                        self._people = people_count

                elif self._frame_provider is not None:
                    frame = self._frame_provider()
                    if frame is None:
                        time.sleep(0.01)
                        continue
                    timestamp_ms = max(
                        time.monotonic_ns() // 1_000_000,
                        last_timestamp_ms + 1,
                    )
                    last_timestamp_ms = timestamp_ms
                    annotated = frame.copy()
                    # Mirror the frame so model input matches displayed orientation
                    mirrored = cv2.flip(annotated, 1)
                    rgb = cv2.cvtColor(mirrored, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)
                    frame_index += 1
                    if (
                        face_landmarker is not None
                        and frame_index % FACE_LANDMARK_INTERVAL_FRAMES == 0
                    ):
                        face_result = face_landmarker.detect_for_video(
                            mp_image, timestamp_ms
                        )
                        latest_face_landmarks = extract_face_landmark_draw_data(
                            face_result
                        )
                        # This compatibility path draws on the unmirrored camera frame.
                        latest_face_landmarks = tuple(
                            tuple((1.0 - x, y) for x, y in face)
                            for face in latest_face_landmarks
                        )
                    people_count = _draw_poses(annotated, result)
                    draw_face_landmarks(annotated, latest_face_landmarks)
                    with self._lock:
                        self._people = people_count
                        self._frame = annotated
                    time.sleep(0.005)
                else:
                    time.sleep(0.05)
        finally:
            try:
                landmarker.close()
            except Exception:
                pass
            if face_landmarker is not None:
                try:
                    face_landmarker.close()
                except Exception:
                    pass
            with self._lock:
                self._state = "off"
                self._message = "Human detection stopped."


def _draw_poses(frame: np.ndarray, result) -> int:
    if not result or not result.pose_landmarks:
        return 0

    h_px, w_px = frame.shape[:2]
    poses = result.pose_landmarks
    people_count = len(poses)

    for person_index, landmarks in enumerate(poses, start=1):
        px_pts = []
        for lm in landmarks:
            # MediaPipe processes the mirrored frame (cv2.flip at line 201),
            # so normalize x must be mirrored back to match the non-mirrored display frame
            x = int((1.0 - lm.x) * w_px)
            y = int(lm.y * h_px)
            px_pts.append((x, y))

        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx < len(px_pts) and end_idx < len(px_pts):
                pt1 = px_pts[start_idx]
                pt2 = px_pts[end_idx]
                vis1 = getattr(landmarks[start_idx], "visibility", 1.0)
                vis2 = getattr(landmarks[end_idx], "visibility", 1.0)
                if vis1 > 0.5 and vis2 > 0.5:
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)

        for idx, (x, y) in enumerate(px_pts):
            vis = getattr(landmarks[idx], "visibility", 1.0)
            if vis > 0.5:
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1, cv2.LINE_AA)

        valid_x = [pt[0] for i, pt in enumerate(px_pts) if getattr(landmarks[i], "visibility", 1.0) > 0.5]
        valid_y = [pt[1] for i, pt in enumerate(px_pts) if getattr(landmarks[i], "visibility", 1.0) > 0.5]
        if valid_x and valid_y:
            x1 = max(0, min(valid_x) - 20)
            y1 = max(0, min(valid_y) - 20)
            x2 = min(w_px, max(valid_x) + 20)
            y2 = min(h_px, max(valid_y) + 20)
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

    return people_count
