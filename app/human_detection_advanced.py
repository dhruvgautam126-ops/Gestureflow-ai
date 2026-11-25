"""Standalone pose-based human detector with per-frame deduplication and tracking."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

POSE_MODEL = Path(__file__).resolve().parent.parent / "pose_landmarker_lite.task"
MAX_POSES = 2
POSE_CONFIDENCE = 0.5
LANDMARK_VISIBILITY_THRESHOLD = 0.50
MIN_VISIBLE_LANDMARKS = 8
MIN_MEAN_VISIBLE = 0.55
TORSO_LANDMARKS = (11, 12, 23, 24)
DUPLICATE_IOU_THRESHOLD = 0.60
DUPLICATE_CENTER_DIAGONAL_RATIO = 0.25
DUPLICATE_LANDMARK_IOU_THRESHOLD = 0.35
DUPLICATE_LANDMARK_DISTANCE_RATIO = 0.08
TRACK_MATCH_IOU_THRESHOLD = 0.10
TRACK_MATCH_CENTER_DIAGONAL_RATIO = 0.50
TRACK_CONFIRM_HITS = 3
TRACK_CONFIRM_WINDOW = 5
TRACK_MAX_MISSES = 4
UNSEQUENCED_FRAME_INTERVAL_SECONDS = 0.10
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

FrameSource = Callable[
    [],
    np.ndarray | tuple[int, np.ndarray] | None,
]


@dataclass(frozen=True)
class AdvancedDetectionStatus:
    """Immutable snapshot of the worker state and latest annotated frame."""

    state: str
    message: str
    people: int
    frame: np.ndarray | None


@dataclass
class _PoseCandidate:
    points: np.ndarray
    visibility: np.ndarray
    bbox: tuple[int, int, int, int]
    quality: float
    visible_count: int


@dataclass
class _Track:
    track_id: int
    bbox: tuple[int, int, int, int]
    hits: deque[int] = field(
        default_factory=lambda: deque(maxlen=TRACK_CONFIRM_WINDOW)
    )
    misses: int = 0
    confirmed: bool = False


class AdvancedHumanDetectionWorker:
    """Run pose inference, suppress duplicate poses, and stabilize person counts.

    The frame source may return a BGR ndarray, ``(sequence_number, BGR ndarray)``,
    or None. Supplying sequence numbers is preferred. For plain arrays, a
    unsequenced arrays are sampled at a maximum of 10 Hz. Sequence numbers
    provide exact new-frame detection and are preferred.
    """

    def __init__(self, frame_source: FrameSource) -> None:
        self._frame_source = frame_source
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = "off"
        self._message = ""
        self._people = 0
        self._frame: np.ndarray | None = None

    def start(self) -> bool:
        """Start the worker; return False while an earlier worker is still alive."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stop.clear()
            self._state = "starting"
            self._message = "Starting advanced human detection..."
            self._people = 0
            self._frame = None
            self._thread = threading.Thread(
                target=self._run,
                name="HomeGuardAdvancedHumanDetection",
                daemon=True,
            )
            self._thread.start()
            return True

    def request_stop(self) -> None:
        """Request a stop without blocking the caller."""
        self._stop.set()
        with self._lock:
            self._frame = None
            self._people = 0
            if self._state in ("starting", "on"):
                self._state = "stopping"
                self._message = "Stopping advanced human detection..."

    def shutdown(self, timeout: float = 3.0) -> None:
        """Request a stop and wait up to timeout seconds for worker cleanup."""
        self.request_stop()
        with self._lock:
            thread = self._thread
        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=timeout)

    def snapshot(self) -> AdvancedDetectionStatus:
        """Return the latest status snapshot."""
        with self._lock:
            return AdvancedDetectionStatus(
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

            vision = mp.tasks.vision
            options = vision.PoseLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=str(POSE_MODEL)
                ),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=MAX_POSES,
                min_pose_detection_confidence=POSE_CONFIDENCE,
                min_pose_presence_confidence=POSE_CONFIDENCE,
                min_tracking_confidence=POSE_CONFIDENCE,
            )

            with vision.PoseLandmarker.create_from_options(options) as detector:
                self._set_running()
                tracks: list[_Track] = []
                next_track_id = 1
                last_timestamp = 0
                last_sequence: int | None = None
                last_unsequenced_sample = 0.0

                while not self._stop.is_set():
                    source_value = self._frame_source()
                    if source_value is None:
                        self._stop.wait(0.01)
                        continue

                    sequence, frame = _unpack_frame(source_value)
                    if frame is None or frame.size == 0:
                        self._stop.wait(0.01)
                        continue

                    if sequence is not None:
                        if sequence == last_sequence:
                            self._stop.wait(0.005)
                            continue
                        last_sequence = sequence
                    else:
                        now = time.monotonic()
                        wait_time = (
                            UNSEQUENCED_FRAME_INTERVAL_SECONDS
                            - (now - last_unsequenced_sample)
                        )
                        if wait_time > 0:
                            self._stop.wait(wait_time)
                            continue
                        last_unsequenced_sample = now

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

                    candidates = _make_candidates(result, frame.shape[1], frame.shape[0])
                    candidates = _suppress_duplicates(candidates)
                    tracks, next_track_id = _update_tracks(
                        tracks,
                        candidates,
                        next_track_id,
                    )
                    people = sum(track.confirmed for track in tracks)
                    annotated = frame.copy()
                    _draw_candidates(annotated, candidates)
                    _draw_status(annotated, people)
                    self._publish(annotated, people)
        except Exception as exc:
            self._set_error(str(exc))
        finally:
            self._finish()

    def _set_running(self) -> None:
        with self._lock:
            if not self._stop.is_set():
                self._state = "on"
                self._message = "Advanced human detection is running."

    def _publish(self, frame: np.ndarray, people: int) -> None:
        with self._lock:
            if not self._stop.is_set():
                self._frame = frame
                self._people = people

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._state = "error"
            self._message = f"Advanced human detection failed: {message}"
            self._people = 0
            self._frame = None

    def _finish(self) -> None:
        with self._lock:
            self._frame = None
            self._people = 0
            if self._state != "error":
                self._state = "off"
                self._message = ""


def _unpack_frame(
    source_value: np.ndarray | tuple[int, np.ndarray],
) -> tuple[int | None, np.ndarray | None]:
    if isinstance(source_value, tuple):
        if len(source_value) != 2:
            raise ValueError("A sequenced frame must be a (sequence, frame) pair.")
        sequence, frame = source_value
        if not isinstance(sequence, int):
            raise TypeError("Frame sequence number must be an int.")
        return sequence, frame
    return None, source_value


def _make_candidates(result, width: int, height: int) -> list[_PoseCandidate]:
    candidates = []
    for landmarks in result.pose_landmarks or ():
        points = np.asarray(
            [(landmark.x * width, landmark.y * height) for landmark in landmarks],
            dtype=np.float32,
        )
        visibility = np.asarray(
            [
                float(getattr(landmark, "visibility", 0.0) or 0.0)
                for landmark in landmarks
            ],
            dtype=np.float32,
        )
        if len(points) != 33 or len(visibility) != 33:
            continue

        visible = visibility >= LANDMARK_VISIBILITY_THRESHOLD
        visible_count = int(np.count_nonzero(visible))
        if visible_count < MIN_VISIBLE_LANDMARKS:
            continue
        if not any(visible[index] for index in TORSO_LANDMARKS):
            continue
        mean_visible = float(np.mean(visibility[visible]))
        if mean_visible < MIN_MEAN_VISIBLE:
            continue

        visible_points = points[visible]
        x_min, y_min = np.min(visible_points, axis=0)
        x_max, y_max = np.max(visible_points, axis=0)
        box_width = max(1.0, float(x_max - x_min))
        box_height = max(1.0, float(y_max - y_min))
        pad_x = box_width * 0.05
        pad_y = box_height * 0.05
        x1 = max(0, min(width - 1, int(np.floor(x_min - pad_x))))
        y1 = max(0, min(height - 1, int(np.floor(y_min - pad_y))))
        x2 = max(0, min(width - 1, int(np.ceil(x_max + pad_x))))
        y2 = max(0, min(height - 1, int(np.ceil(y_max + pad_y))))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue

        quality = (
            0.7 * (visible_count / 33.0)
            + 0.3 * mean_visible
        )
        candidates.append(
            _PoseCandidate(
                points=points,
                visibility=visibility,
                bbox=(x1, y1, x2, y2),
                quality=quality,
                visible_count=visible_count,
            )
        )
    return candidates


def _suppress_duplicates(
    candidates: list[_PoseCandidate],
) -> list[_PoseCandidate]:
    ordered = sorted(candidates, key=lambda candidate: candidate.quality, reverse=True)
    retained: list[_PoseCandidate] = []
    for candidate in ordered:
        if not any(_is_duplicate(candidate, accepted) for accepted in retained):
            retained.append(candidate)
    return retained


def _is_duplicate(left: _PoseCandidate, right: _PoseCandidate) -> bool:
    iou = _box_iou(left.bbox, right.bbox)
    center_distance = _center_distance(left.bbox, right.bbox)
    diagonal = max(_box_diagonal(left.bbox), _box_diagonal(right.bbox), 1.0)

    if (
        iou >= DUPLICATE_IOU_THRESHOLD
        and center_distance <= DUPLICATE_CENTER_DIAGONAL_RATIO * diagonal
    ):
        return True

    matching = (
        (left.visibility >= LANDMARK_VISIBILITY_THRESHOLD)
        & (right.visibility >= LANDMARK_VISIBILITY_THRESHOLD)
    )
    if int(np.count_nonzero(matching)) < MIN_VISIBLE_LANDMARKS:
        return False
    distances = np.linalg.norm(left.points[matching] - right.points[matching], axis=1)
    return (
        iou >= DUPLICATE_LANDMARK_IOU_THRESHOLD
        and float(np.median(distances))
        <= DUPLICATE_LANDMARK_DISTANCE_RATIO * diagonal
    )


def _update_tracks(
    tracks: list[_Track],
    candidates: list[_PoseCandidate],
    next_track_id: int,
) -> tuple[list[_Track], int]:
    eligible_pairs = []
    for track_index, track in enumerate(tracks):
        if track.misses > TRACK_MAX_MISSES:
            continue
        for candidate_index, candidate in enumerate(candidates):
            iou = _box_iou(track.bbox, candidate.bbox)
            center_distance = _center_distance(track.bbox, candidate.bbox)
            diagonal = max(
                _box_diagonal(track.bbox),
                _box_diagonal(candidate.bbox),
                1.0,
            )
            if (
                iou >= TRACK_MATCH_IOU_THRESHOLD
                or center_distance <= TRACK_MATCH_CENTER_DIAGONAL_RATIO * diagonal
            ):
                eligible_pairs.append(
                    (-iou, center_distance / diagonal, track_index, candidate_index)
                )

    matched_tracks: set[int] = set()
    matched_candidates: set[int] = set()
    for _, _, track_index, candidate_index in sorted(eligible_pairs):
        if track_index in matched_tracks or candidate_index in matched_candidates:
            continue
        track = tracks[track_index]
        candidate = candidates[candidate_index]
        track.bbox = candidate.bbox
        track.hits.append(1)
        track.misses = 0
        if sum(track.hits) >= TRACK_CONFIRM_HITS:
            track.confirmed = True
        matched_tracks.add(track_index)
        matched_candidates.add(candidate_index)

    for track_index, track in enumerate(tracks):
        if track_index not in matched_tracks:
            track.hits.append(0)
            track.misses += 1

    for candidate_index, candidate in enumerate(candidates):
        if candidate_index in matched_candidates:
            continue
        track = _Track(track_id=next_track_id, bbox=candidate.bbox)
        track.hits.append(1)
        tracks.append(track)
        next_track_id += 1

    tracks = [track for track in tracks if track.misses <= TRACK_MAX_MISSES]
    return tracks, next_track_id


def _box_iou(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0
    left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0


def _center_distance(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
    left_center = ((left[0] + left[2]) / 2, (left[1] + left[3]) / 2)
    right_center = ((right[0] + right[2]) / 2, (right[1] + right[3]) / 2)
    return float(np.hypot(
        left_center[0] - right_center[0],
        left_center[1] - right_center[1],
    ))


def _box_diagonal(box: tuple[int, int, int, int]) -> float:
    return float(np.hypot(box[2] - box[0], box[3] - box[1]))


def _draw_candidates(frame: np.ndarray, candidates: list[_PoseCandidate]) -> None:
    for person_index, candidate in enumerate(candidates, start=1):
        visible = candidate.visibility >= LANDMARK_VISIBILITY_THRESHOLD
        points = np.rint(candidate.points).astype(np.int32)
        for start, end in POSE_CONNECTIONS:
            if visible[start] and visible[end]:
                cv2.line(
                    frame,
                    tuple(points[start]),
                    tuple(points[end]),
                    (0, 255, 0),
                    2,
                )
        for point, is_visible in zip(points, visible):
            if is_visible:
                cv2.circle(frame, tuple(point), 4, (0, 0, 255), -1)

        x1, y1, x2, _ = candidate.bbox
        cv2.rectangle(frame, (x1, y1), (x2, candidate.bbox[3]), (255, 0, 0), 2)
        cv2.putText(
            frame,
            f"PERSON {person_index}",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )


def _draw_status(frame: np.ndarray, people: int) -> None:
    text = "HUMAN DETECTED" if people else "NO HUMAN DETECTED"
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
        text,
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
    )
