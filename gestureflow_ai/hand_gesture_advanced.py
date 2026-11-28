"""Hand tracking and gesture recognition for GestureFlow AI."""

from __future__ import annotations

import queue
import threading
import time
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from gestureflow_ai.coordinator import DetectorKind, WorkerResult

APP_ROOT = Path(__file__).resolve().parent.parent
HAND_MODEL = APP_ROOT / "models" / "gesture" / "gesture_recognizer.task"

MAX_HANDS = 4
MAX_PEOPLE = 2
DETECTION_CONFIDENCE = 0.40
HAND_PRESENCE_CONFIDENCE = 0.35
HAND_TRACKING_CONFIDENCE = 0.45
HANDEDNESS_LABEL_THRESHOLD = 0.60
HANDEDNESS_TRACK_THRESHOLD = 0.75
HANDEDNESS_HISTORY_SIZE = 5
HANDEDNESS_CONFIRM_HITS = 4
HAND_DUPLICATE_IOU = 0.65
HAND_DUPLICATE_WRIST_RATIO = 0.20
HAND_DUPLICATE_LANDMARK_IOU = 0.35
HAND_DUPLICATE_LANDMARK_RATIO = 0.08
HAND_TRACK_GRACE_SECONDS = 2.0
HAND_TRACK_REID_MAX_SPEED = 1200.0
HAND_TRACK_REID_AMBIGUITY_MARGIN = 0.12
PERSON_TRACK_GRACE_SECONDS = 2.0
PERSON_TRACK_REID_MAX_SPEED = 900.0
PERSON_TRACK_AMBIGUITY_MARGIN = 0.15
GESTURE_HISTORY_SIZE = 5
GESTURE_CONFIRM_HITS = 3
FRAME_SAMPLE_INTERVAL = 1.0 / 20.0
POSE_INFERENCE_EVERY_N_FRAMES = 8
MAX_POSE_ASSOCIATION_AGE = 0.85
HAND_INFERENCE_MAX_WIDTH = 640
LANDMARK_IN_FRAME_TOLERANCE = 0.05

# --- Bounded lost-hand identity cache (temporary disappearance re-identification) ---
HAND_LOST_TRACK_GRACE_SECONDS = 8.0
HAND_LOST_TRACK_MAX = 8
HAND_REID_SIGNATURE_MAX = 0.35
HAND_REID_STRONG_SIGNATURE = 0.18
HAND_REID_POSITION_GATE_RATIO = 0.35
HAND_REID_STRICT_POSITION_RATIO = 0.20
HAND_REID_MIN_CONFIDENCE = 0.55
HAND_REID_GEOMETRY_ONLY_MIN_CONFIDENCE = 0.70
HAND_REID_AMBIGUITY_MARGIN = 0.10

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

POSE_WRIST_ELBOW = {
    "LEFT": (15, 13),
    "RIGHT": (16, 14),
}
POSE_SHOULDERS = (11, 12)
POSE_VISIBILITY_THRESHOLD = 0.35
UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class HandDetectionSnapshot:
    state: str
    message: str
    hand_count: int
    processing_fps: float
    frame: np.ndarray | None


@dataclass
class _HandCandidate:
    image_points: np.ndarray
    world_points: np.ndarray
    landmark_signature: np.ndarray
    bbox: tuple[int, int, int, int]
    wrist: np.ndarray
    handedness: str | None
    handedness_score: float
    quality: float
    person_id: int | None = None
    associated_side: str | None = None
    gesture: str = UNKNOWN


@dataclass
class _HandTrack:
    track_id: int
    bbox: tuple[int, int, int, int]
    wrist: np.ndarray
    velocity: np.ndarray
    landmark_signature: np.ndarray
    handedness: str | None
    handedness_score: float
    gesture_history: list[str] = field(default_factory=list)
    stable_gesture: str = UNKNOWN
    person_id: int | None = None
    associated_side: str | None = None
    misses: int = 0
    last_seen_at: float = 0.0
    handedness_history: list[tuple[str, float]] = field(default_factory=list)
    track_id_confident: bool = True
    last_confident_person_id: int | None = None


@dataclass(frozen=True)
class _LostHandIdentity:
    """Immutable snapshot of a hand track retained for bounded re-identification."""

    track_id: int
    bbox: tuple[int, int, int, int]
    wrist: np.ndarray
    velocity: np.ndarray
    landmark_signature: np.ndarray
    handedness: str | None
    handedness_score: float
    person_id: int | None
    last_confident_person_id: int | None
    associated_side: str | None
    stable_gesture: str
    gesture_history: tuple[str, ...]
    handedness_history: tuple[tuple[str, float], ...]
    last_seen_at: float


@dataclass
class _PoseCandidate:
    points: np.ndarray
    visibility: np.ndarray
    bbox: tuple[int, int, int, int]


@dataclass
class _PersonTrack:
    track_id: int
    points: np.ndarray
    visibility: np.ndarray
    bbox: tuple[int, int, int, int]
    misses: int = 0
    association_confident: bool = True
    anchor: np.ndarray | None = None
    velocity: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float32)
    )
    last_seen_at: float = 0.0


class AdvancedHandDetectionWorker:
    """Run hand and optional person-association pose inference in one worker."""

    def __init__(
        self,
        frame_provider,
        coordinator=None,
        kind=None,
    ) -> None:
        self._frame_provider = frame_provider
        self._coordinator = coordinator
        self._kind = kind if kind is not None else DetectorKind.HAND
        self._packet_queue = (
            coordinator.get_packet_queue(self._kind) if coordinator is not None else None
        )
        self._lock = threading.Lock()
        self._shutdown = threading.Event()
        self._thread: threading.Thread | None = None
        self._enabled = False
        self._state = "off"
        self._message = ""
        self._hand_count = 0
        self._processing_fps = 0.0
        self._frame: np.ndarray | None = None
        self._coordinator_generation = 0

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable inference without stopping the camera or UI."""
        with self._lock:
            if self._shutdown.is_set():
                return
            self._enabled = enabled
            if enabled:
                if self._thread is None or not self._thread.is_alive():
                    self._shutdown.clear()
                    self._state = "starting"
                    self._message = "Loading hand detection model..."
                    self._hand_count = 0
                    self._processing_fps = 0.0
                    self._frame = None
                    self._thread = threading.Thread(
                        target=self._run,
                        name="GestureFlowHandDetection",
                        daemon=True,
                    )
                    self._thread.start()
                    # Sync generation with coordinator so frames aren't discarded
                    if self._coordinator is not None:
                        self._coordinator_generation = self._coordinator.get_generation(self._kind)
                elif self._state != "error":
                    self._state = "starting"
                    self._message = "Resuming hand detection..."
                    if self._coordinator is not None:
                        self._coordinator_generation = self._coordinator.get_generation(
                            self._kind
                        )
            else:
                self._hand_count = 0
                self._processing_fps = 0.0
                self._frame = None
                if self._state != "error":
                    self._state = "off"
                    self._message = "Hand detection is off."

    def snapshot(self) -> HandDetectionSnapshot:
        with self._lock:
            return HandDetectionSnapshot(
                state=self._state,
                message=self._message,
                hand_count=self._hand_count,
                processing_fps=self._processing_fps,
                frame=self._frame,
            )

    def shutdown(self, timeout: float = 5.0) -> None:
        """Stop processing and wait for both MediaPipe tasks to be released."""
        self._shutdown.set()
        with self._lock:
            self._enabled = False
            self._frame = None
            self._hand_count = 0
        thread = self._thread
        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=timeout)

    def _run(self) -> None:
        try:
            if not HAND_MODEL.is_file():
                raise FileNotFoundError(f"Hand model not found: {HAND_MODEL}")

            import mediapipe as mp

            vision = mp.tasks.vision
            hand_options = vision.GestureRecognizerOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path=str(HAND_MODEL)
                ),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=MAX_HANDS,
                min_hand_detection_confidence=DETECTION_CONFIDENCE,
                min_hand_presence_confidence=HAND_PRESENCE_CONFIDENCE,
                min_tracking_confidence=HAND_TRACKING_CONFIDENCE,
            )

            with ExitStack() as stack:
                hand_detector = stack.enter_context(
                    vision.GestureRecognizer.create_from_options(hand_options)
                )
                self._set_running("Hand detection is running.")
                hand_tracks: list[_HandTrack] = []
                lost_hand_tracks: list[_LostHandIdentity] = []
                last_packet_seq: int | None = None
                next_hand_id = 1
                last_timestamp = 0
                last_sample_time = 0.0
                last_processed_time: float | None = None
                processing_fps = 0.0

                while not self._shutdown.is_set():
                    frame = None
                    packet_seq = 0
                    packet_ts = time.monotonic()
                    gen = 0

                    if self._coordinator is not None and self._packet_queue is not None:
                        try:
                            packet = self._packet_queue.get(timeout=0.05)
                        except queue.Empty:
                            continue

                        # Prefer the newest camera image to processing an old backlog.
                        while True:
                            try:
                                packet = self._packet_queue.get_nowait()
                            except queue.Empty:
                                break

                        last_packet_seq = _update_camera_session_state(
                            lost_hand_tracks,
                            last_packet_seq,
                            packet.seq,
                        )
                        gen = self._coordinator.get_generation(self._kind)
                        frame = packet.frame
                        packet_seq = packet.seq
                        packet_ts = packet.timestamp
                    else:
                        # Legacy fallback mode
                        frame = self._frame_provider()
                        if frame is None or frame.size == 0:
                            self._shutdown.wait(0.02)
                            continue
                        time.sleep(FRAME_SAMPLE_INTERVAL)

                    if frame is None or frame.size == 0:
                        continue

                    # Check if worker is still enabled (for coordinator mode, this is checked via generation)
                    if self._coordinator is not None and hasattr(self, '_coordinator_generation'):
                        if self._coordinator_generation != gen:
                            # Generation changed - worker was disabled/re-enabled, discard this frame
                            continue
                    elif not self._is_enabled():
                        continue

                    # Keep camera preview and desktop interaction responsive on
                    # CPU bound machines. The coordinator queue is latest-frame
                    # only, so skipped inference frames never build a backlog.
                    if self._coordinator is not None:
                        now = time.monotonic()
                        if (
                            last_sample_time > 0.0
                            and now - last_sample_time < FRAME_SAMPLE_INTERVAL
                        ):
                            continue
                        last_sample_time = now

                    height, width = frame.shape[:2]
                    inference_frame = frame
                    if width > HAND_INFERENCE_MAX_WIDTH:
                        inference_height = max(
                            1,
                            round(height * HAND_INFERENCE_MAX_WIDTH / width),
                        )
                        inference_frame = cv2.resize(
                            frame,
                            (HAND_INFERENCE_MAX_WIDTH, inference_height),
                            interpolation=cv2.INTER_AREA,
                        )
                    # Mirror after downscaling to avoid flipping every full-size
                    # camera frame on the inference path.
                    inference_frame = cv2.flip(inference_frame, 1)
                    rgb_frame = cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb_frame,
                    )
                    timestamp = max(
                        time.monotonic_ns() // 1_000_000,
                        last_timestamp + 1,
                    )
                    last_timestamp = timestamp

                    hand_result = hand_detector.recognize_for_video(
                        mp_image,
                        timestamp,
                    )
                    candidates = _make_hand_candidates(
                        hand_result,
                        width,
                        height,
                    )
                    candidates = _suppress_duplicate_hands(candidates)
                    hand_tracks, next_hand_id, visible_tracks = _update_hand_tracks(
                        hand_tracks,
                        candidates,
                        (),
                        next_hand_id,
                        now=time.monotonic(),
                        frame_width=width,
                        frame_height=height,
                        lost_tracks=lost_hand_tracks,
                    )

                    if self._coordinator is not None:
                        # Convert candidates and visible_tracks to tuples for WorkerResult
                        tuple_candidates = tuple(candidates)
                        tuple_visible_tracks = tuple(visible_tracks)

                        worker_res = WorkerResult(
                            kind=self._kind,
                            source_seq=packet_seq,
                            source_ts=packet_ts,
                            generation=gen,
                            hand_candidates=tuple_candidates,
                            hand_visible_tracks=tuple_visible_tracks,
                        )
                        self._coordinator.publish_result(worker_res)

                    if not self._is_enabled():
                        continue
                    completed_at = time.monotonic()
                    if last_processed_time is not None:
                        interval = completed_at - last_processed_time
                        if interval > 0:
                            measured = 1.0 / interval
                            processing_fps = (
                                measured
                                if processing_fps == 0
                                else processing_fps * 0.8 + measured * 0.2
                            )
                    last_processed_time = completed_at
                    self._publish(
                        inference_frame,
                        len(candidates),
                        processing_fps,
                    )
        except Exception as exc:
            self._set_error(str(exc))
        finally:
            with self._lock:
                if self._state != "error":
                    self._state = "off"
                    self._message = "Hand detection is off."
                    self._hand_count = 0
                    self._frame = None

    def _is_enabled(self) -> bool:
        with self._lock:
            return self._enabled and not self._shutdown.is_set()

    def _set_running(self, association_message: str) -> None:
        with self._lock:
            if self._enabled and not self._shutdown.is_set():
                self._state = "on"
                self._message = (
                    association_message
                    if association_message
                    else "Hand detection is running."
                )

    def _publish(
        self,
        frame: np.ndarray,
        hand_count: int,
        processing_fps: float,
    ) -> None:
        with self._lock:
            if self._enabled and not self._shutdown.is_set():
                self._frame = frame
                self._hand_count = hand_count
                self._processing_fps = processing_fps
                if not self._message.startswith("Person association unavailable"):
                    self._message = "Hand detection is running."

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._state = "error"
            self._message = f"Hand detection error: {message}"
            self._enabled = False
            self._hand_count = 0
            self._frame = None


def _make_hand_candidates(result, width: int, height: int) -> list[_HandCandidate]:
    candidates = []
    hand_landmarks = result.hand_landmarks or ()
    world_groups = result.hand_world_landmarks or ()
    handedness_groups = result.handedness or ()

    for index, landmarks in enumerate(hand_landmarks[:MAX_HANDS]):
        if len(landmarks) != 21:
            continue
        image_points = np.asarray(
            [(point.x, point.y, point.z) for point in landmarks],
            dtype=np.float32,
        )
        if not np.all(np.isfinite(image_points)):
            continue

        if index < len(world_groups) and len(world_groups[index]) == 21:
            world_points = np.asarray(
                [(point.x, point.y, point.z) for point in world_groups[index]],
                dtype=np.float32,
            )
        else:
            world_points = np.column_stack(
                (
                    image_points[:, 0] * width,
                    image_points[:, 1] * height,
                    image_points[:, 2] * width,
                )
            ).astype(np.float32)
        if not np.all(np.isfinite(world_points)):
            continue

        category_name = ""
        handedness_score = 0.0
        if index < len(handedness_groups) and handedness_groups[index]:
            category = handedness_groups[index][0]
            category_name = str(getattr(category, "category_name", ""))
            handedness_score = float(getattr(category, "score", 0.0) or 0.0)
        handedness = None
        if handedness_score >= HANDEDNESS_LABEL_THRESHOLD:
            if category_name.lower() == "left":
                handedness = "LEFT"
            elif category_name.lower() == "right":
                handedness = "RIGHT"

        visible_points = image_points[in_frame := (
            (image_points[:, 0] >= -LANDMARK_IN_FRAME_TOLERANCE)
            & (image_points[:, 0] <= 1.0 + LANDMARK_IN_FRAME_TOLERANCE)
            & (image_points[:, 1] >= -LANDMARK_IN_FRAME_TOLERANCE)
            & (image_points[:, 1] <= 1.0 + LANDMARK_IN_FRAME_TOLERANCE)
        )]
        # Keep partially visible hands when the wrist and enough of the
        # landmarks are still in view; fast movement often clips fingertips.
        if np.count_nonzero(in_frame) < 8:
            continue

        x_min, y_min = np.min(visible_points, axis=0)[:2]
        x_max, y_max = np.max(visible_points, axis=0)[:2]
        # image_points are normalized, while tracking and duplicate suppression
        # use pixel-space boxes. Convert before clipping to the frame bounds.
        x1 = max(0, min(width - 1, int(np.floor(x_min * width))))
        y1 = max(0, min(height - 1, int(np.floor(y_min * height))))
        x2 = max(0, min(width - 1, int(np.ceil(x_max * width))))
        y2 = max(0, min(height - 1, int(np.ceil(y_max * height))))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue

        quality = float(np.count_nonzero(in_frame) / 21.0)
        candidates.append(
            _HandCandidate(
                image_points=image_points,
                world_points=world_points,
                landmark_signature=_landmark_signature(world_points),
                bbox=(x1, y1, x2, y2),
                wrist=image_points[0, :2]
                * np.asarray([width, height], dtype=np.float32),
                handedness=handedness,
                handedness_score=handedness_score,
                quality=quality,
                gesture=(
                    _index_thumb_gesture(image_points, world_points)
                    or _gesture_label_for_hand(result, index)
                ),
            )
        )
    return candidates


def _gesture_label_for_hand(result, index: int) -> str:
    """Map the bundled model's per-hand class to the app's three labels."""
    gesture_groups = getattr(result, "gestures", ()) or ()
    if index >= len(gesture_groups) or not gesture_groups[index]:
        return UNKNOWN
    category = gesture_groups[index][0]
    if float(getattr(category, "score", 0.0) or 0.0) < 0.55:
        return UNKNOWN
    return {
        "Open_Palm": "OPEN HAND",
        "Closed_Fist": "CLOSED FIST",
        "Thumb_Up": "THUMBS UP",
    }.get(str(getattr(category, "category_name", "")), UNKNOWN)


def _index_thumb_gesture(
    image_points: np.ndarray,
    world_points: np.ndarray,
) -> str | None:
    """Recognize an index-and-thumb pointer pose and its pinch action.

    The index finger and thumb must be extended while the other three fingers
    are curled. Keeping the pinch label separate from a fist prevents a pinch
    from being treated as a right-click by the desktop controller.
    """
    image = np.asarray(image_points, dtype=np.float64)
    world = np.asarray(world_points, dtype=np.float64)
    if (
        image.ndim != 2
        or world.ndim != 2
        or image.shape[0] < 21
        or world.shape[0] < 21
        or image.shape[1] < 2
        or world.shape[1] < 3
        or not np.all(np.isfinite(image[:21, :2]))
        or not np.all(np.isfinite(world[:21, :3]))
    ):
        return None

    palm_width = float(np.linalg.norm(world[5] - world[17]))
    if palm_width < 1e-5:
        return None
    palm_center = np.mean(world[[0, 5, 9, 13, 17]], axis=0)

    index_pip = _joint_angle(world[5], world[6], world[7])
    index_dip = _joint_angle(world[6], world[7], world[8])
    index_tip_ratio = float(np.linalg.norm(world[8] - palm_center) / palm_width)
    index_extended = (
        np.isfinite(index_pip)
        and np.isfinite(index_dip)
        and index_pip >= 145.0
        and index_dip >= 135.0
        and index_tip_ratio >= 0.78
    )

    thumb_angle = _joint_angle(world[2], world[3], world[4])
    thumb_length_ratio = float(np.linalg.norm(world[4] - world[2]) / palm_width)
    thumb_extended = (
        np.isfinite(thumb_angle)
        and thumb_angle >= 140.0
        and thumb_length_ratio >= 0.48
    )

    curled_fingers = []
    for mcp, pip, dip, tip in (
        (9, 10, 11, 12),
        (13, 14, 15, 16),
        (17, 18, 19, 20),
    ):
        pip_angle = _joint_angle(world[mcp], world[pip], world[dip])
        tip_ratio = float(np.linalg.norm(world[tip] - palm_center) / palm_width)
        curled_fingers.append(
            np.isfinite(pip_angle) and pip_angle <= 145.0 and tip_ratio <= 1.25
        )

    if not (index_extended and thumb_extended and all(curled_fingers)):
        return None

    tip_gap_ratio = float(np.linalg.norm(world[4] - world[8]) / palm_width)
    if not np.isfinite(tip_gap_ratio):
        return None
    return "PINCH" if tip_gap_ratio <= 0.42 else "POINTING"


def _landmark_signature(world_points: np.ndarray) -> np.ndarray:
    bone_pairs = (
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
    )
    palm_pairs = (
        (0, 5), (0, 9), (0, 13), (0, 17),
        (5, 9), (5, 13), (5, 17), (9, 13), (9, 17), (13, 17),
    )
    palm_width = float(np.linalg.norm(world_points[5] - world_points[17]))
    if not np.isfinite(palm_width) or palm_width < 1e-5:
        return np.zeros(len(bone_pairs) + len(palm_pairs), dtype=np.float32)
    return np.concatenate(
        (
            np.asarray(
                [
                    np.linalg.norm(world_points[first] - world_points[second])
                    / palm_width
                    for first, second in bone_pairs
                ],
                dtype=np.float32,
            ),
            np.asarray(
                [
                    np.linalg.norm(world_points[first] - world_points[second])
                    / palm_width
                    for first, second in palm_pairs
                ],
                dtype=np.float32,
            ),
        )
    )


def _suppress_duplicate_hands(
    candidates: list[_HandCandidate],
) -> list[_HandCandidate]:
    ordered = sorted(
        candidates,
        key=lambda candidate: (
            candidate.quality,
            candidate.handedness_score,
        ),
        reverse=True,
    )
    kept: list[_HandCandidate] = []
    for candidate in ordered:
        if not any(_hands_are_duplicates(candidate, prior) for prior in kept):
            kept.append(candidate)
    return kept[:MAX_HANDS]


def _hands_are_duplicates(
    first: _HandCandidate,
    second: _HandCandidate,
) -> bool:
    iou = _box_iou(first.bbox, second.bbox)
    diagonal = max(_box_diagonal(first.bbox), _box_diagonal(second.bbox), 1.0)
    wrist_distance = float(np.linalg.norm(first.wrist - second.wrist))
    if (
        iou >= HAND_DUPLICATE_IOU
        and wrist_distance <= HAND_DUPLICATE_WRIST_RATIO * diagonal
    ):
        return True
    if iou < HAND_DUPLICATE_LANDMARK_IOU:
        return False

    first_points = first.image_points[:, :2]
    second_points = second.image_points[:, :2]
    valid = (
        np.all(np.isfinite(first_points), axis=1)
        & np.all(np.isfinite(second_points), axis=1)
    )
    if np.count_nonzero(valid) < 12:
        return False
    distances = np.linalg.norm(
        (first_points[valid] - second_points[valid])
        * np.asarray([diagonal, diagonal], dtype=np.float32),
        axis=1,
    )
    return float(np.median(distances)) <= HAND_DUPLICATE_LANDMARK_RATIO * diagonal


def _classify_gesture(
    candidate: _HandCandidate,
    width: int,
    height: int,
) -> str:
    image = candidate.image_points
    if image.shape[0] != 21:
        return UNKNOWN
    in_frame = (
        (image[:, 0] >= 0.0)
        & (image[:, 0] <= 1.0)
        & (image[:, 1] >= 0.0)
        & (image[:, 1] <= 1.0)
    )
    if np.count_nonzero(in_frame) < 16:
        return UNKNOWN
    critical = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
    if any(
        image[index, 0] < 0.0
        or image[index, 0] > 1.0
        or image[index, 1] < 0.0
        or image[index, 1] > 1.0
        for index in critical
    ):
        return UNKNOWN
    world = candidate.world_points.astype(np.float64, copy=False)
    palm_width = float(np.linalg.norm(world[5] - world[17]))
    if not np.isfinite(palm_width) or palm_width < 1e-5:
        return UNKNOWN

    finger_chains = ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
    finger_angles = [
        _joint_angle(world[mcp], world[pip], world[dip])
        for mcp, pip, dip, _tip in finger_chains
    ]
    palm_center = np.mean(world[[0, 5, 9, 13, 17]], axis=0)
    finger_tip_ratios = [
        float(np.linalg.norm(world[tip] - palm_center) / palm_width)
        for _mcp, _pip, _dip, tip in finger_chains
    ]
    if not all(np.isfinite(angle) for angle in finger_angles + finger_tip_ratios):
        return UNKNOWN

    fingers_extended = all(
        angle >= 155.0 and tip_ratio >= 0.75
        for angle, tip_ratio in zip(finger_angles, finger_tip_ratios)
    )
    fingers_folded = all(
        angle <= 125.0 and tip_ratio <= 1.0
        for angle, tip_ratio in zip(finger_angles, finger_tip_ratios)
    )
    thumb_angle = _joint_angle(world[2], world[3], world[4])
    thumb_length_ratio = float(np.linalg.norm(world[4] - world[1]) / palm_width)
    thumb_straight = (
        np.isfinite(thumb_angle)
        and thumb_angle >= 145.0
        and thumb_length_ratio >= 0.55
    )
    thumb_tip_ratio = float(np.linalg.norm(world[4] - palm_center) / palm_width)
    thumb_up = thumb_straight and _thumb_points_up(image, width, height)
    thumb_extended_for_open = thumb_straight and thumb_tip_ratio >= 0.75
    thumb_folded = np.isfinite(thumb_tip_ratio) and thumb_tip_ratio <= 0.85

    if fingers_extended and thumb_extended_for_open:
        return "OPEN HAND"
    if fingers_folded and thumb_folded and not thumb_up:
        return "CLOSED FIST"
    if fingers_folded and thumb_up:
        return "THUMBS UP"
    return UNKNOWN


def _joint_angle(first: np.ndarray, joint: np.ndarray, last: np.ndarray) -> float:
    first_vector = first - joint
    last_vector = last - joint
    denominator = float(np.linalg.norm(first_vector) * np.linalg.norm(last_vector))
    if denominator < 1e-8:
        return float("nan")
    cosine = float(np.dot(first_vector, last_vector) / denominator)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _thumb_points_up(image_points: np.ndarray, width: int, height: int) -> bool:
    thumb_base = image_points[2, :2] * np.asarray([width, height])
    thumb_tip = image_points[4, :2] * np.asarray([width, height])
    wrist = image_points[0, :2] * np.asarray([width, height])
    middle_mcp = image_points[9, :2] * np.asarray([width, height])
    palm_scale = max(float(np.linalg.norm(middle_mcp - wrist)), 1.0)
    dx, dy = thumb_tip - thumb_base
    return dy <= -0.20 * palm_scale and abs(dx) <= -dy


def _make_pose_candidates(result, width: int, height: int) -> list[_PoseCandidate]:
    candidates = []
    if result is None:
        return candidates
    for landmarks in (result.pose_landmarks or ())[:MAX_PEOPLE]:
        if len(landmarks) != 33:
            continue
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
        if not np.all(np.isfinite(points)):
            continue
        required = (13, 14, 15, 16)
        if sum(visibility[index] >= POSE_VISIBILITY_THRESHOLD for index in required) < 2:
            continue
        visible = visibility >= POSE_VISIBILITY_THRESHOLD
        visible_points = points[visible]
        if len(visible_points) < 4:
            continue
        x1 = max(0, min(width - 1, int(np.floor(np.min(visible_points[:, 0])))))
        y1 = max(0, min(height - 1, int(np.floor(np.min(visible_points[:, 1])))))
        x2 = max(0, min(width - 1, int(np.ceil(np.max(visible_points[:, 0])))))
        y2 = max(0, min(height - 1, int(np.ceil(np.max(visible_points[:, 1])))))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        candidates.append(
            _PoseCandidate(
                points=points,
                visibility=visibility,
                bbox=(x1, y1, x2, y2),
            )
        )
    return candidates


def _update_person_tracks(
    tracks: list[_PersonTrack],
    candidates: list[_PoseCandidate],
    next_id: int,
    now: float | None = None,
    frame_width: int | None = None,
    frame_height: int | None = None,
) -> tuple[list[_PersonTrack], int]:
    now = time.monotonic() if now is None else now
    tracks = [
        track
        for track in tracks
        if track.last_seen_at <= 0.0
        or now - track.last_seen_at <= PERSON_TRACK_GRACE_SECONDS
    ]
    for track in tracks:
        if track.last_seen_at <= 0.0:
            track.last_seen_at = now
        if track.anchor is None:
            track.anchor = _pose_anchor(track.points, track.visibility, track.bbox)

    frame_diagonal = (
        float(np.hypot(frame_width, frame_height))
        if frame_width is not None and frame_height is not None
        else None
    )
    costs: dict[tuple[int, int], float] = {}
    predictions: dict[int, tuple[np.ndarray, float]] = {}
    for track_index, track in enumerate(tracks):
        elapsed = max(0.0, now - track.last_seen_at)
        assert track.anchor is not None
        displacement = track.velocity * elapsed
        max_displacement = PERSON_TRACK_REID_MAX_SPEED * elapsed
        displacement_length = float(np.linalg.norm(displacement))
        if displacement_length > max_displacement > 0.0:
            displacement *= max_displacement / displacement_length
        predicted_anchor = track.anchor + displacement
        predicted_box = _translate_box(track.bbox, displacement)
        predictions[track_index] = (displacement, elapsed)
        for candidate_index, candidate in enumerate(candidates):
            candidate_anchor = _pose_anchor(
                candidate.points,
                candidate.visibility,
                candidate.bbox,
            )
            iou = _box_iou(predicted_box, candidate.bbox)
            diagonal = max(
                _box_diagonal(predicted_box),
                _box_diagonal(candidate.bbox),
                1.0,
            )
            gate = 0.60 * diagonal + 0.35 * PERSON_TRACK_REID_MAX_SPEED * elapsed
            if frame_diagonal is not None:
                gate = min(gate, 0.35 * frame_diagonal)
            gate = max(gate, 1.0)
            anchor_distance = float(np.linalg.norm(predicted_anchor - candidate_anchor))
            center_distance = _center_distance(predicted_box, candidate.bbox)
            if iou < 0.05 and anchor_distance > gate and center_distance > gate:
                continue

            torso_distance = _pose_torso_distance(
                track,
                candidate,
                displacement,
            )
            cost = (
                0.35 * min(anchor_distance / gate, 2.0)
                + 0.30 * min(center_distance / gate, 2.0)
                + 0.25 * (1.0 - iou)
                + 0.10 * min(torso_distance, 2.0)
                if torso_distance is not None
                else (
                    0.40 * min(anchor_distance / gate, 2.0)
                    + 0.35 * min(center_distance / gate, 2.0)
                    + 0.25 * (1.0 - iou)
                )
            )
            costs[(track_index, candidate_index)] = cost

    ambiguous_candidates: set[int] = set()
    for candidate_index in range(len(candidates)):
        scores = sorted(
            score
            for (track_index, index), score in costs.items()
            if index == candidate_index
        )
        if (
            len(scores) > 1
            and scores[1] - scores[0] < PERSON_TRACK_AMBIGUITY_MARGIN
        ):
            ambiguous_candidates.add(candidate_index)

    matched_tracks: set[int] = set()
    matched_candidates: set[int] = set()
    for track_index, candidate_index in _best_one_to_one_matches(
        costs,
        len(tracks),
        len(candidates),
    ):
        track = tracks[track_index]
        candidate = candidates[candidate_index]
        displacement, elapsed = predictions[track_index]
        candidate_anchor = _pose_anchor(
            candidate.points,
            candidate.visibility,
            candidate.bbox,
        )
        if elapsed > 1e-3:
            measured_velocity = (candidate_anchor - track.anchor) / elapsed
            speed = float(np.linalg.norm(measured_velocity))
            if speed > PERSON_TRACK_REID_MAX_SPEED:
                measured_velocity *= PERSON_TRACK_REID_MAX_SPEED / speed
            track.velocity = 0.65 * track.velocity + 0.35 * measured_velocity
        track.anchor = candidate_anchor
        track.points = candidate.points
        track.visibility = candidate.visibility
        track.bbox = candidate.bbox
        track.last_seen_at = now
        track.misses = 0
        track.association_confident = candidate_index not in ambiguous_candidates
        matched_tracks.add(track_index)
        matched_candidates.add(candidate_index)

    for index, track in enumerate(tracks):
        if index not in matched_tracks:
            track.misses += 1

    for index, candidate in enumerate(candidates):
        if index in matched_candidates:
            continue
        tracks.append(
            _PersonTrack(
                track_id=next_id,
                points=candidate.points,
                visibility=candidate.visibility,
                bbox=candidate.bbox,
                anchor=_pose_anchor(
                    candidate.points,
                    candidate.visibility,
                    candidate.bbox,
                ),
                last_seen_at=now,
                association_confident=index not in ambiguous_candidates,
            )
        )
        next_id += 1
    retained = [
        track
        for track in tracks
        if now - track.last_seen_at <= PERSON_TRACK_GRACE_SECONDS
    ]
    if len(retained) > MAX_PEOPLE:
        retained.sort(
            key=lambda track: (track.misses == 0, track.last_seen_at),
            reverse=True,
        )
        retained = retained[:MAX_PEOPLE]
    return retained, next_id


def _best_one_to_one_matches(
    costs: dict[tuple[int, int], float],
    track_count: int,
    candidate_count: int,
) -> list[tuple[int, int]]:
    best: list[tuple[int, int]] = []
    best_cost = float("inf")

    def search(
        candidate_index: int,
        used_tracks: set[int],
        matches: list[tuple[int, int]],
        total_cost: float,
    ) -> None:
        nonlocal best, best_cost
        if candidate_index == candidate_count:
            if len(matches) > len(best) or (
                len(matches) == len(best) and total_cost < best_cost
            ):
                best = matches.copy()
                best_cost = total_cost
            return

        search(
            candidate_index + 1,
            used_tracks,
            matches,
            total_cost,
        )
        for track_index in range(track_count):
            cost = costs.get((track_index, candidate_index))
            if cost is None or track_index in used_tracks:
                continue
            used_tracks.add(track_index)
            matches.append((track_index, candidate_index))
            search(
                candidate_index + 1,
                used_tracks,
                matches,
                total_cost + cost,
            )
            matches.pop()
            used_tracks.remove(track_index)

    search(0, set(), [], 0.0)
    return best


def _pose_anchor(
    points: np.ndarray,
    visibility: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    shoulders_visible = (
        visibility[11] >= POSE_VISIBILITY_THRESHOLD
        and visibility[12] >= POSE_VISIBILITY_THRESHOLD
    )
    hips_visible = (
        visibility[23] >= POSE_VISIBILITY_THRESHOLD
        and visibility[24] >= POSE_VISIBILITY_THRESHOLD
    )
    centers = []
    if shoulders_visible:
        centers.append((points[11] + points[12]) * 0.5)
    if hips_visible:
        centers.append((points[23] + points[24]) * 0.5)
    if centers:
        return np.mean(centers, axis=0).astype(np.float32)

    torso_indices = np.asarray((11, 12, 23, 24))
    visible = visibility[torso_indices] >= POSE_VISIBILITY_THRESHOLD
    if np.any(visible):
        return np.mean(points[torso_indices[visible]], axis=0).astype(np.float32)
    return np.asarray(
        ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5),
        dtype=np.float32,
    )


def _pose_torso_distance(
    track: _PersonTrack,
    candidate: _PoseCandidate,
    translation: np.ndarray,
) -> float | None:
    indices = np.asarray((11, 12, 23, 24))
    visible = (
        (track.visibility[indices] >= POSE_VISIBILITY_THRESHOLD)
        & (candidate.visibility[indices] >= POSE_VISIBILITY_THRESHOLD)
    )
    if np.count_nonzero(visible) < 2:
        return None
    predicted = track.points[indices] + translation
    distances = np.linalg.norm(
        predicted[visible] - candidate.points[indices][visible],
        axis=1,
    )
    track_shoulder_width = float(
        np.linalg.norm(track.points[11] - track.points[12])
    )
    candidate_shoulder_width = float(
        np.linalg.norm(candidate.points[11] - candidate.points[12])
    )
    scale = max(track_shoulder_width, candidate_shoulder_width, 20.0)
    return float(np.mean(distances) / scale)


def _translate_box(
    bbox: tuple[int, int, int, int],
    displacement: np.ndarray,
) -> tuple[int, int, int, int]:
    dx, dy = np.rint(displacement).astype(np.int32)
    return (
        bbox[0] + int(dx),
        bbox[1] + int(dy),
        bbox[2] + int(dx),
        bbox[3] + int(dy),
    )


def _update_hand_tracks(
    tracks: list[_HandTrack],
    candidates: list[_HandCandidate],
    people: list[_PersonTrack],
    next_id: int,
    now: float,
    frame_width: int,
    frame_height: int,
    lost_tracks: list[_LostHandIdentity] | None = None,
) -> tuple[list[_HandTrack], int, list[_HandTrack]]:
    costs: dict[tuple[int, int], float] = {}
    frame_diagonal = float(np.hypot(frame_width, frame_height))
    for track_index, track in enumerate(tracks):
        elapsed = max(0.0, now - track.last_seen_at)
        if elapsed > HAND_TRACK_GRACE_SECONDS:
            continue
        for candidate_index, candidate in enumerate(candidates):
            iou = _box_iou(track.bbox, candidate.bbox)
            diagonal = max(_box_diagonal(track.bbox), _box_diagonal(candidate.bbox), 1.0)
            predicted_wrist = track.wrist + track.velocity * elapsed
            prediction_distance = float(
                np.linalg.norm(predicted_wrist - candidate.wrist)
            )
            gate = max(
                1.5 * diagonal,
                min(
                    0.35 * frame_diagonal,
                    1.5 * diagonal + HAND_TRACK_REID_MAX_SPEED * elapsed * 0.35,
                ),
            )
            if iou < 0.08 and prediction_distance > gate:
                continue
            if track.landmark_signature.shape != candidate.landmark_signature.shape:
                continue
            signature_distance = float(
                np.mean(
                    np.abs(
                        track.landmark_signature
                        - candidate.landmark_signature
                    )
                )
            )
            if elapsed > 0.25:
                same_handedness = (
                    track.handedness is not None
                    and track.handedness == candidate.handedness
                    and track.handedness_score >= HANDEDNESS_TRACK_THRESHOLD
                    and candidate.handedness_score >= 0.80
                )
                same_person = (
                    track.last_confident_person_id is not None
                    and track.last_confident_person_id == candidate.person_id
                )
                proximity_gate = min(
                    0.25 * frame_diagonal,
                    max(
                        0.10 * frame_diagonal,
                        1.25 * diagonal
                        + 0.25 * HAND_TRACK_REID_MAX_SPEED * elapsed,
                    ),
                )
                close_to_prediction = prediction_distance <= proximity_gate
                has_identity_cue = (
                    same_handedness
                    or same_person
                    or close_to_prediction
                    or (
                        signature_distance <= 0.15
                        and prediction_distance <= 0.12 * frame_diagonal
                    )
                )
                if signature_distance > 0.50 or not has_identity_cue:
                    continue
            handedness_penalty = (
                0.15
                if track.handedness
                and candidate.handedness
                and track.handedness != candidate.handedness
                else 0.0
            )
            person_penalty = (
                0.12
                if track.last_confident_person_id is not None
                and candidate.person_id is not None
                and track.last_confident_person_id != candidate.person_id
                else 0.0
            )
            cost = (
                0.45 * min(prediction_distance / gate, 1.5)
                + 0.25 * (1.0 - iou)
                + 0.20 * min(signature_distance / 0.50, 1.5)
                + handedness_penalty
                + person_penalty
            )
            costs[(track_index, candidate_index)] = cost

    ambiguous_candidates: set[int] = set()
    for candidate_index in range(len(candidates)):
        candidate_scores = sorted(
            cost
            for (_track_index, index), cost in costs.items()
            if index == candidate_index
        )
        if (
            len(candidate_scores) > 1
            and candidate_scores[1] - candidate_scores[0]
            < HAND_TRACK_REID_AMBIGUITY_MARGIN
        ):
            ambiguous_candidates.add(candidate_index)

    matched_tracks: set[int] = set()
    matched_candidates: set[int] = set()
    visible_tracks: list[_HandTrack | None] = [None] * len(candidates)
    for track_index, candidate_index in _best_one_to_one_matches(
        costs,
        len(tracks),
        len(candidates),
    ):
        track = tracks[track_index]
        _apply_hand_candidate(
            track,
            candidates[candidate_index],
            people,
            now,
            track_id_confident=candidate_index not in ambiguous_candidates,
        )
        matched_tracks.add(track_index)
        matched_candidates.add(candidate_index)
        visible_tracks[candidate_index] = track

    for index, track in enumerate(tracks):
        if index not in matched_tracks:
            track.misses += 1

    if lost_tracks is not None:
        for track in tracks:
            if now - track.last_seen_at > HAND_TRACK_GRACE_SECONDS:
                _remember_lost_track(lost_tracks, track)
        _trim_lost_tracks(lost_tracks, now)

    tracks = [
        track
        for track in tracks
        if now - track.last_seen_at <= HAND_TRACK_GRACE_SECONDS
    ]

    reidentified_tracks: dict[int, _HandTrack] = {}
    if lost_tracks:
        reidentified_tracks = _reassociate_lost_tracks(
            tracks,
            candidates,
            matched_candidates,
            lost_tracks,
            people,
            now,
            frame_width,
            frame_height,
        )

    for candidate_index, candidate in enumerate(candidates):
        if candidate_index in matched_candidates:
            continue
        if candidate_index in reidentified_tracks:
            visible_tracks[candidate_index] = reidentified_tracks[candidate_index]
            continue
        track = _HandTrack(
            track_id=next_id,
            bbox=candidate.bbox,
            wrist=candidate.wrist.copy(),
            velocity=np.zeros(2, dtype=np.float32),
            landmark_signature=candidate.landmark_signature.copy(),
            handedness=None,
            handedness_score=0.0,
            track_id_confident=candidate_index not in ambiguous_candidates,
            last_seen_at=now,
        )
        _apply_hand_candidate(
            track,
            candidate,
            people,
            now,
            track_id_confident=candidate_index not in ambiguous_candidates,
        )
        tracks.append(track)
        visible_tracks[candidate_index] = track
        next_id += 1

    if len(tracks) > MAX_HANDS:
        tracks.sort(key=lambda track: (track.misses, -track.track_id))
        tracks = tracks[:MAX_HANDS]
    if lost_tracks is not None:
        _trim_lost_tracks(lost_tracks, now)
    return tracks, next_id, [track for track in visible_tracks if track is not None]


# ---------------------------------------------------------------------------
# Lost-hand identity cache helpers
# ---------------------------------------------------------------------------


def _snapshot_lost_track(track: _HandTrack) -> _LostHandIdentity:
    """Create an immutable, fully copied snapshot of a track for the lost cache."""
    return _LostHandIdentity(
        track_id=track.track_id,
        bbox=tuple(track.bbox),
        wrist=np.array(track.wrist, dtype=np.float32, copy=True),
        velocity=np.array(track.velocity, dtype=np.float32, copy=True),
        landmark_signature=np.array(
            track.landmark_signature, dtype=np.float32, copy=True
        ),
        handedness=track.handedness,
        handedness_score=float(track.handedness_score),
        person_id=track.person_id,
        last_confident_person_id=track.last_confident_person_id,
        associated_side=track.associated_side,
        stable_gesture=track.stable_gesture,
        gesture_history=tuple(track.gesture_history),
        handedness_history=tuple(track.handedness_history),
        last_seen_at=float(track.last_seen_at),
    )


def _remember_lost_track(
    lost_tracks: list[_LostHandIdentity],
    track: _HandTrack,
) -> None:
    """Add or refresh one lost identity, enforcing the cache bounds."""
    snapshot = _snapshot_lost_track(track)
    lost_tracks[:] = [
        entry for entry in lost_tracks if entry.track_id != snapshot.track_id
    ]
    lost_tracks.append(snapshot)


def _trim_lost_tracks(
    lost_tracks: list[_LostHandIdentity],
    now: float,
) -> None:
    """Expire old lost identities and keep at most HAND_LOST_TRACK_MAX entries."""
    fresh = [
        entry
        for entry in lost_tracks
        if now - entry.last_seen_at <= HAND_LOST_TRACK_GRACE_SECONDS
    ]
    fresh.sort(key=lambda entry: entry.last_seen_at, reverse=True)
    lost_tracks[:] = fresh[:HAND_LOST_TRACK_MAX]


def _lost_match_confidence(
    candidate: _HandCandidate,
    lost: _LostHandIdentity,
    now: float,
    frame_width: int,
    frame_height: int,
) -> float | None:
    """Return a conservative confidence for restoring ``lost`` to ``candidate``."""
    elapsed = max(0.0, now - lost.last_seen_at)
    if elapsed > HAND_LOST_TRACK_GRACE_SECONDS:
        return None
    if candidate.landmark_signature.shape != lost.landmark_signature.shape:
        return None
    signature_distance = float(
        np.mean(np.abs(candidate.landmark_signature - lost.landmark_signature))
    )
    if not np.isfinite(signature_distance) or signature_distance > HAND_REID_SIGNATURE_MAX:
        return None

    candidate_person = candidate.person_id
    lost_person = (
        lost.last_confident_person_id
        if lost.last_confident_person_id is not None
        else lost.person_id
    )
    if (
        candidate_person is not None
        and lost_person is not None
        and candidate_person != lost_person
    ):
        return None
    person_match = (
        candidate_person is not None
        and lost_person is not None
        and candidate_person == lost_person
    )

    diagonal = float(np.hypot(frame_width, frame_height))
    if diagonal <= 0.0:
        return None
    predicted_wrist = lost.wrist + lost.velocity * elapsed
    position_distance = float(np.linalg.norm(predicted_wrist - candidate.wrist))
    position_gate = max(
        HAND_REID_POSITION_GATE_RATIO * diagonal,
        HAND_TRACK_REID_MAX_SPEED * elapsed * 0.35,
    )
    position_gate = min(position_gate, 0.60 * diagonal)
    if position_distance > position_gate:
        return None

    handedness_match = (
        lost.handedness is not None
        and candidate.handedness is not None
        and lost.handedness == candidate.handedness
        and lost.handedness_score >= HANDEDNESS_TRACK_THRESHOLD
        and candidate.handedness_score >= 0.80
    )
    signature_match = signature_distance <= HAND_REID_STRONG_SIGNATURE
    position_match = position_distance <= HAND_REID_STRICT_POSITION_RATIO * diagonal

    cue_count = (
        int(person_match)
        + int(handedness_match)
        + int(signature_match)
        + int(position_match)
    )
    if cue_count < 2:
        return None

    if not person_match and not handedness_match:
        # No model-derived identity cue: require both strict geometric cues.
        if not (signature_match and position_match):
            return None
        minimum_confidence = HAND_REID_GEOMETRY_ONLY_MIN_CONFIDENCE
    else:
        minimum_confidence = HAND_REID_MIN_CONFIDENCE

    signature_similarity = 1.0 - min(
        signature_distance / HAND_REID_SIGNATURE_MAX, 1.0
    )
    position_similarity = 1.0 - min(position_distance / position_gate, 1.0)
    confidence = (
        0.45 * signature_similarity
        + 0.20 * position_similarity
        + 0.20 * (1.0 if person_match else 0.0)
        + 0.15 * (1.0 if handedness_match else 0.0)
    )
    if confidence < minimum_confidence:
        return None
    return confidence


def _restore_lost_track(
    lost: _LostHandIdentity,
    candidate: _HandCandidate,
    people: list[_PersonTrack],
    now: float,
) -> _HandTrack:
    """Rebuild an active track with the lost identity's original track_id."""
    track = _HandTrack(
        track_id=lost.track_id,
        bbox=lost.bbox,
        wrist=np.array(lost.wrist, dtype=np.float32, copy=True),
        velocity=np.zeros(2, dtype=np.float32),
        landmark_signature=np.array(
            lost.landmark_signature, dtype=np.float32, copy=True
        ),
        handedness=lost.handedness,
        handedness_score=lost.handedness_score,
        gesture_history=list(lost.gesture_history),
        stable_gesture=lost.stable_gesture,
        person_id=lost.person_id,
        associated_side=lost.associated_side,
        misses=0,
        last_seen_at=now,
        handedness_history=list(lost.handedness_history),
        track_id_confident=True,
        last_confident_person_id=lost.last_confident_person_id,
    )
    _apply_hand_candidate(track, candidate, people, now, track_id_confident=True)
    return track


def _reassociate_lost_tracks(
    tracks: list[_HandTrack],
    candidates: list[_HandCandidate],
    matched_candidates: set[int],
    lost_tracks: list[_LostHandIdentity],
    people: list[_PersonTrack],
    now: float,
    frame_width: int,
    frame_height: int,
) -> dict[int, _HandTrack]:
    """One-to-one reassociation of unmatched candidates to lost identities."""
    remaining = [
        candidate_index
        for candidate_index in range(len(candidates))
        if candidate_index not in matched_candidates
    ]
    if not remaining:
        return {}

    costs: dict[tuple[int, int], float] = {}
    confidences: dict[tuple[int, int], float] = {}
    for lost_index, lost in enumerate(lost_tracks):
        for local_index, candidate_index in enumerate(remaining):
            confidence = _lost_match_confidence(
                candidates[candidate_index],
                lost,
                now,
                frame_width,
                frame_height,
            )
            if confidence is None:
                continue
            costs[(lost_index, local_index)] = 1.0 - confidence
            confidences[(lost_index, local_index)] = confidence

    if not costs:
        return {}

    restored: dict[int, _HandTrack] = {}
    restored_ids: set[int] = set()
    for lost_index, local_index in _best_one_to_one_matches(
        costs,
        len(lost_tracks),
        len(remaining),
    ):
        lost = lost_tracks[lost_index]
        candidate_index = remaining[local_index]
        if any(track.track_id == lost.track_id for track in tracks):
            continue
        candidate_scores = sorted(
            (
                confidence
                for (confidence_lost, confidence_candidate), confidence in confidences.items()
                if confidence_candidate == local_index
            ),
            reverse=True,
        )
        if (
            len(candidate_scores) > 1
            and candidate_scores[0] - candidate_scores[1]
            < HAND_REID_AMBIGUITY_MARGIN
        ):
            continue
        track = _restore_lost_track(lost, candidates[candidate_index], people, now)
        tracks.append(track)
        restored[candidate_index] = track
        restored_ids.add(lost.track_id)

    if restored_ids:
        lost_tracks[:] = [
            entry for entry in lost_tracks if entry.track_id not in restored_ids
        ]
    return restored


def _update_camera_session_state(
    lost_tracks: list[_LostHandIdentity],
    last_packet_seq: int | None,
    packet_seq: int,
) -> int:
    """Clear lost identities when the camera packet sequence regresses.

    A CameraCapture session restart resets ``seq`` to zero and publishes
    ``seq == 1`` again, so a non-increasing sequence is a genuine camera
    session reset and must not reuse identities from the previous session.
    ``time.monotonic()``-based TTL expiry handles temporary interruptions.
    """
    if packet_seq <= 0:
        return last_packet_seq if last_packet_seq is not None else packet_seq
    if last_packet_seq is not None and packet_seq <= last_packet_seq:
        lost_tracks.clear()
    return packet_seq


def _apply_hand_candidate(
    track: _HandTrack,
    candidate: _HandCandidate,
    people: list[_PersonTrack],
    now: float,
    track_id_confident: bool = True,
) -> None:
    elapsed = now - track.last_seen_at
    if elapsed > 1e-3:
        measured_velocity = (candidate.wrist - track.wrist) / elapsed
        speed = float(np.linalg.norm(measured_velocity))
        if speed > HAND_TRACK_REID_MAX_SPEED:
            measured_velocity *= HAND_TRACK_REID_MAX_SPEED / speed
        track.velocity = 0.65 * track.velocity + 0.35 * measured_velocity
    track.bbox = candidate.bbox
    track.wrist = candidate.wrist.copy()
    track.landmark_signature = candidate.landmark_signature.copy()
    track.last_seen_at = now
    track.track_id_confident = track_id_confident
    if (
        candidate.handedness is not None
        and candidate.handedness_score >= HANDEDNESS_TRACK_THRESHOLD
    ):
        track.handedness_history.append(
            (candidate.handedness, candidate.handedness_score)
        )
        if len(track.handedness_history) > HANDEDNESS_HISTORY_SIZE:
            del track.handedness_history[0]
        votes: dict[str, list[float]] = {"LEFT": [], "RIGHT": []}
        for side, score in track.handedness_history:
            votes[side].append(score)
        stable_side, stable_scores = max(
            votes.items(),
            key=lambda item: len(item[1]),
        )
        if len(stable_scores) >= HANDEDNESS_CONFIRM_HITS:
            track.handedness = stable_side
            track.handedness_score = float(np.mean(stable_scores))
    track.misses = 0
    track.gesture_history.append(candidate.gesture)
    if len(track.gesture_history) > GESTURE_HISTORY_SIZE:
        del track.gesture_history[0]
    counts = {
        gesture: track.gesture_history.count(gesture)
        for gesture in (
            "OPEN HAND",
            "CLOSED FIST",
            "THUMBS UP",
            "POINTING",
            "PINCH",
            UNKNOWN,
        )
    }
    stable, votes = max(counts.items(), key=lambda item: item[1])
    if votes >= GESTURE_CONFIRM_HITS:
        track.stable_gesture = stable

    track.person_id, track.associated_side = _associate_person(
        candidate,
        people,
        track.last_confident_person_id,
        handedness=track.handedness,
    )
    if track.person_id is not None:
        track.last_confident_person_id = track.person_id


def _associate_person(
    candidate: _HandCandidate,
    people: list[_PersonTrack],
    previous_person_id: int | None,
    handedness: str | None = None,
) -> tuple[int | None, str | None]:
    scores: list[tuple[float, _PersonTrack, str]] = []
    allowed_sides = (
        (handedness,)
        if handedness is not None
        else ("LEFT", "RIGHT")
    )

    for person in people:
        if not person.association_confident:
            continue
        scale = max(
            float(np.linalg.norm(person.points[POSE_SHOULDERS[0]] - person.points[POSE_SHOULDERS[1]])),
            20.0,
        )
        for side in allowed_sides:
            wrist_index, elbow_index = POSE_WRIST_ELBOW[side]
            if (
                person.visibility[wrist_index] < POSE_VISIBILITY_THRESHOLD
                or person.visibility[elbow_index] < POSE_VISIBILITY_THRESHOLD
            ):
                continue
            wrist_point = person.points[wrist_index]
            elbow_point = person.points[elbow_index]
            arm_length = float(np.linalg.norm(wrist_point - elbow_point))
            local_scale = max(scale, arm_length, 20.0)
            wrist_distance = float(np.linalg.norm(candidate.wrist - wrist_point))
            segment_distance = _point_segment_distance(
                candidate.wrist,
                elbow_point,
                wrist_point,
            )
            score = (
                0.65 * wrist_distance / local_scale
                + 0.35 * segment_distance / local_scale
            )
            if wrist_distance <= 0.60 * local_scale and segment_distance <= 0.30 * local_scale:
                scores.append((score, person, side))

    if not scores:
        return None, None
    scores.sort(key=lambda item: item[0])
    best_score, best_person, best_side = scores[0]
    person_best: dict[int, tuple[float, str]] = {}
    for score, person, side in scores:
        current = person_best.get(person.track_id)
        if current is None or score < current[0]:
            person_best[person.track_id] = (score, side)
    ranked_people = sorted(
        (score, person_id, side)
        for person_id, (score, side) in person_best.items()
    )
    if (
        len(ranked_people) > 1
        and ranked_people[1][0] - ranked_people[0][0] < 0.12
    ):
        return None, None

    if previous_person_id is not None and previous_person_id in person_best:
        previous_score, previous_side = person_best[previous_person_id]
        if previous_score <= best_score + 0.10:
            previous_person = next(
                person for person in people if person.track_id == previous_person_id
            )
            return previous_person.track_id, (
                previous_side if handedness is not None else None
            )

    return best_person.track_id, (
        best_side if handedness is not None else None
    )


def _point_segment_distance(
    point: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> float:
    segment = end - start
    denominator = float(np.dot(segment, segment))
    if denominator < 1e-8:
        return float(np.linalg.norm(point - start))
    ratio = float(np.clip(np.dot(point - start, segment) / denominator, 0.0, 1.0))
    projection = start + ratio * segment
    return float(np.linalg.norm(point - projection))


def _box_iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_first = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    area_second = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union else 0.0


def _box_diagonal(box: tuple[int, int, int, int]) -> float:
    return float(np.hypot(box[2] - box[0], box[3] - box[1]))


def _center_distance(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    first_center = ((first[0] + first[2]) / 2, (first[1] + first[3]) / 2)
    second_center = ((second[0] + second[2]) / 2, (second[1] + second[3]) / 2)
    return float(
        np.hypot(
            first_center[0] - second_center[0],
            first_center[1] - second_center[1],
        )
    )


def _draw_hands(
    frame: np.ndarray,
    candidates: list[_HandCandidate],
    tracks: list[_HandTrack],
) -> None:
    height, width = frame.shape[:2]
    palette = (
        (0, 255, 0),
        (255, 180, 0),
        (255, 0, 255),
        (0, 220, 255),
    )
    for candidate, track in zip(candidates, tracks):
        color = palette[(track.track_id - 1) % len(palette)]
        points = np.rint(
            candidate.image_points[:, :2]
            * np.asarray([width, height], dtype=np.float32)
        ).astype(np.int32)
        inside = (
            (candidate.image_points[:, 0] >= 0.0)
            & (candidate.image_points[:, 0] <= 1.0)
            & (candidate.image_points[:, 1] >= 0.0)
            & (candidate.image_points[:, 1] <= 1.0)
        )
        for start, end in HAND_CONNECTIONS:
            if inside[start] and inside[end]:
                cv2.line(
                    frame,
                    tuple(points[start]),
                    tuple(points[end]),
                    color,
                    2,
                )
        for index, point in enumerate(points):
            if inside[index]:
                radius = 6 if index == 0 else 4
                cv2.circle(frame, tuple(point), radius, (0, 0, 255), -1)
        gesture = track.stable_gesture
        hand_text = (
            f"HAND {track.track_id}"
            if track.track_id_confident
            else "HAND ?"
        )
        if track.person_id is not None:
            side_text = (
                f"PERSON {track.person_id} - {track.associated_side} HAND"
                if track.associated_side
                else f"PERSON {track.person_id} - HAND"
            )
        elif track.handedness:
            side_text = f"PERSON? - {track.handedness} HAND"
        else:
            side_text = "UNKNOWN ASSOCIATION"
        label = f"{hand_text} | {side_text} | {gesture}"
        x1, y1, x2, _ = candidate.bbox
        text_y = max(24, y1 - 9)
        cv2.putText(
            frame,
            label,
            (max(4, x1), text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
        cv2.rectangle(frame, (x1, y1), (x2, candidate.bbox[3]), color, 1)
