"""Standalone face recognition worker for Gesture.AI (Stage 1).

Runs YuNet face detection and SFace feature extraction on a background thread.
Consumes frames via a provider callback (e.g. CameraCapture.get_frame), produces
thread-safe snapshot data, and enforces strict memory-only processing with zero
disk persistence and zero UI-thread blocking.

All detected faces are assigned the label 'UNKNOWN' in this initial stage.
"""

from __future__ import annotations

import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger("GestureAI.FaceWorker")

# Default model locations relative to project root
_DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_YUNET_PATH = _DEFAULT_MODELS_DIR / "face_detection_yunet_2023mar.onnx"
DEFAULT_SFACE_PATH = _DEFAULT_MODELS_DIR / "face_recognition_sface_2021dec.onnx"

# Default detector settings
DEFAULT_SCORE_THRESHOLD: float = 0.8
DEFAULT_NMS_THRESHOLD: float = 0.3
DEFAULT_TOP_K: int = 5000
DEFAULT_INFERENCE_INTERVAL: float = 0.05  # 20 Hz default, keeping 8 GB RAM usage minimal

# Threshold for detecting camera sequence counter resets or rollovers
SEQUENCE_RESET_THRESHOLD: int = 100

# Type for frame provider callback
FrameProvider = Callable[
    [],
    Union[
        np.ndarray,
        Tuple[bool, Optional[np.ndarray]],
        Tuple[int, Optional[np.ndarray]],
        None,
    ],
]


@dataclass(frozen=True)
class FaceDetection:
    """Immutable record of a detected face with normalized SFace embedding."""

    bbox: Tuple[int, int, int, int]  # Clamped to frame boundaries: (x, y, width, height)
    score: float
    landmarks: Tuple[Tuple[float, float], ...]  # 5 landmarks: right eye, left eye, nose, right mouth, left mouth
    embedding: np.ndarray  # (128,) float32 L2-normalized C-level immutable embedding
    label: str = "UNKNOWN"

    def __post_init__(self) -> None:
        # Enforce true immutable C-level buffer protection (cannot re-enable writeable flag)
        if isinstance(self.embedding, np.ndarray):
            if self.embedding.flags.writeable or not isinstance(self.embedding.base, bytes):
                immutable = np.frombuffer(self.embedding.tobytes(), dtype=np.float32)
                object.__setattr__(self, "embedding", immutable)

    def __repr__(self) -> str:
        emb_shape = self.embedding.shape if self.embedding is not None else None
        return (
            f"FaceDetection(bbox={self.bbox}, score={self.score:.3f}, "
            f"label='{self.label}', embedding_shape={emb_shape})"
        )


@dataclass(frozen=True)
class FaceWorkerSnapshot:
    """Thread-safe snapshot of the face recognition worker state and detections."""

    timestamp: float
    state: str  # "off", "starting", "running", "stopping", "error"
    message: str
    faces: Tuple[FaceDetection, ...] = ()
    frame_count: int = 0
    fps: float = 0.0
    error_message: Optional[str] = None


class FaceRecognitionWorker:
    """Background worker for offline face detection and feature extraction.

    Guarantees:
    - Never opens or controls a camera device directly.
    - Pure in-memory processing: zero face images, embeddings, or identities written to disk.
    - All detections labeled 'UNKNOWN' (no enrollment or identity matching at this stage).
    - Thread-safe snapshot access for UI or coordinator consumption.
    - Non-blocking execution on a dedicated daemon thread.
    - Safe handling and rigorous validation of camera frames.
    - Safe handling of camera sequence resets and rollovers.
    - Strict L2 normalization: zero-norm or invalid embeddings are discarded.
    - C-level immutable embedding buffers preventing all external mutation.
    - Published bounding boxes clamped strictly to frame boundaries.
    """

    def __init__(
        self,
        frame_provider: FrameProvider,
        yunet_path: Optional[Union[str, Path]] = None,
        sface_path: Optional[Union[str, Path]] = None,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        nms_threshold: float = DEFAULT_NMS_THRESHOLD,
        top_k: int = DEFAULT_TOP_K,
        inference_interval: float = DEFAULT_INFERENCE_INTERVAL,
    ) -> None:
        """Initialize the worker with a frame provider and model configurations.

        Args:
            frame_provider: Callable returning BGR ndarray, (success, frame), (seq, frame), or None.
            yunet_path: Path to YuNet ONNX model (defaults to models/face_detection_yunet_2023mar.onnx).
            sface_path: Path to SFace ONNX model (defaults to models/face_recognition_sface_2021dec.onnx).
            score_threshold: YuNet confidence score threshold (default: 0.8).
            nms_threshold: YuNet NMS suppression threshold (default: 0.3).
            top_k: YuNet maximum candidate faces before NMS (default: 5000).
            inference_interval: Minimum seconds between inference iterations (default: 0.05s).
        """
        if not callable(frame_provider):
            raise TypeError("frame_provider must be a callable returning camera frames")

        self._frame_provider = frame_provider
        self._yunet_path = Path(yunet_path) if yunet_path else DEFAULT_YUNET_PATH
        self._sface_path = Path(sface_path) if sface_path else DEFAULT_SFACE_PATH
        self._score_threshold = float(score_threshold)
        self._nms_threshold = float(nms_threshold)
        self._top_k = int(top_k)
        self._inference_interval = max(0.001, float(inference_interval))

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._last_frame_seq: Optional[int] = None
        self._snapshot = FaceWorkerSnapshot(
            timestamp=time.time(),
            state="off",
            message="Worker initialized",
            faces=(),
            frame_count=0,
            fps=0.0,
            error_message=None,
        )

    @property
    def is_alive(self) -> bool:
        """Return True if background worker thread is active."""
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        """Start the background face recognition worker thread.

        Returns:
            True if worker was started, False if already running.
        """
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False

            self._stop.clear()
            self._last_frame_seq = None
            self._snapshot = FaceWorkerSnapshot(
                timestamp=time.time(),
                state="starting",
                message="Starting face recognition worker...",
                faces=(),
                frame_count=0,
                fps=0.0,
                error_message=None,
            )
            self._thread = threading.Thread(
                target=self._run,
                name="HomeGuardFaceRecognitionWorker",
                daemon=True,
            )
            self._thread.start()
            return True

    def request_stop(self) -> None:
        """Signal the worker thread to terminate without blocking."""
        self._stop.set()
        with self._lock:
            if self._snapshot.state in ("starting", "running"):
                self._snapshot = FaceWorkerSnapshot(
                    timestamp=time.time(),
                    state="stopping",
                    message="Stopping face recognition worker...",
                    faces=(),
                    frame_count=self._snapshot.frame_count,
                    fps=self._snapshot.fps,
                    error_message=self._snapshot.error_message,
                )

    def shutdown(self, timeout: float = 3.0) -> None:
        """Signal the worker to stop and wait for thread termination."""
        self.request_stop()
        with self._lock:
            thread = self._thread
        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=timeout)

        with self._lock:
            # Check if the specific thread that was joined is still alive
            target_thread_alive = thread is not None and thread.is_alive()
            if target_thread_alive:
                # Do not report 'off' while worker thread is still alive after timeout
                self._snapshot = FaceWorkerSnapshot(
                    timestamp=time.time(),
                    state="stopping",
                    message="Shutdown timed out: worker thread still active",
                    faces=(),
                    frame_count=self._snapshot.frame_count,
                    fps=0.0,
                    error_message=self._snapshot.error_message,
                )
            else:
                # Thread terminated cleanly; clear reference if it was not replaced
                if self._thread is thread:
                    self._thread = None
                if self._snapshot.state != "error":
                    self._snapshot = FaceWorkerSnapshot(
                        timestamp=time.time(),
                        state="off",
                        message="Stopped",
                        faces=(),
                        frame_count=self._snapshot.frame_count,
                        fps=0.0,
                        error_message=self._snapshot.error_message,
                    )

    def snapshot(self) -> FaceWorkerSnapshot:
        """Return an immutable, thread-safe snapshot of the current state and detections."""
        with self._lock:
            return self._snapshot

    @staticmethod
    def _validate_frame(frame: Any) -> Optional[np.ndarray]:
        """Validate that frame is a valid 3-channel uint8 BGR image array.

        Enforces:
        - Must be a numpy.ndarray
        - Must have uint8 data type
        - Must have exactly 3 dimensions (height, width, channels)
        - Must have exactly 3 channels (BGR)
        - Height and width must be strictly greater than 0
        """
        if not isinstance(frame, np.ndarray):
            return None
        if frame.dtype != np.uint8:
            return None
        if frame.ndim != 3:
            return None
        if frame.shape[2] != 3:
            return None
        if frame.shape[0] <= 0 or frame.shape[1] <= 0:
            return None
        return frame

    def _extract_frame(self) -> Optional[np.ndarray]:
        """Fetch frame from frame_provider callback, validating format and sequencing."""
        try:
            raw = self._frame_provider()
        except Exception as exc:
            logger.warning("Frame provider callback raised exception: %s", exc)
            return None

        if raw is None:
            return None

        if isinstance(raw, np.ndarray):
            return self._validate_frame(raw)

        if isinstance(raw, (tuple, list)):
            if len(raw) == 0:
                return None
            if len(raw) == 2:
                first, second = raw
                # Format: (ret_bool, frame_array)
                if isinstance(first, bool):
                    if not first:
                        return None
                    return self._validate_frame(second)
                # Format: (sequence_number, frame_array)
                if isinstance(first, int):
                    # Handle camera sequence counter reset (e.g. camera restart, session reset, or large counter rollover)
                    is_reset = (
                        self._last_frame_seq is not None
                        and (first == 0 or (self._last_frame_seq - first >= SEQUENCE_RESET_THRESHOLD))
                    )
                    if is_reset:
                        self._last_frame_seq = None

                    # Reject duplicate (first == last) and out-of-order/stale (first < last) sequence numbers
                    if self._last_frame_seq is not None and first <= self._last_frame_seq:
                        return None
                    validated = self._validate_frame(second)
                    if validated is not None:
                        self._last_frame_seq = first
                    return validated
                # Format: generic tuple with ndarray
                val_second = self._validate_frame(second)
                if val_second is not None:
                    return val_second
                val_first = self._validate_frame(first)
                if val_first is not None:
                    return val_first

            # Search elements for valid ndarray
            for elem in raw:
                val = self._validate_frame(elem)
                if val is not None:
                    return val

        return None

    def _run(self) -> None:
        """Main worker thread loop: loads models, processes frames, updates snapshot."""
        # 1. Validate model files exist
        if not self._yunet_path.is_file():
            err_msg = f"YuNet model file not found: {self._yunet_path}"
            logger.error(err_msg)
            with self._lock:
                self._snapshot = FaceWorkerSnapshot(
                    timestamp=time.time(),
                    state="error",
                    message=err_msg,
                    faces=(),
                    error_message=err_msg,
                )
            return

        if not self._sface_path.is_file():
            err_msg = f"SFace model file not found: {self._sface_path}"
            logger.error(err_msg)
            with self._lock:
                self._snapshot = FaceWorkerSnapshot(
                    timestamp=time.time(),
                    state="error",
                    message=err_msg,
                    faces=(),
                    error_message=err_msg,
                )
            return

        # 2. Load OpenCV models
        try:
            detector = cv2.FaceDetectorYN.create(
                str(self._yunet_path),
                "",
                (320, 320),
                self._score_threshold,
                self._nms_threshold,
                self._top_k,
            )
            recognizer = cv2.FaceRecognizerSF.create(
                str(self._sface_path),
                "",
            )
        except Exception as exc:
            err_msg = f"Failed to instantiate OpenCV face models: {exc}"
            logger.error(err_msg, exc_info=True)
            with self._lock:
                self._snapshot = FaceWorkerSnapshot(
                    timestamp=time.time(),
                    state="error",
                    message=err_msg,
                    faces=(),
                    error_message=err_msg,
                )
            return

        with self._lock:
            self._snapshot = FaceWorkerSnapshot(
                timestamp=time.time(),
                state="running",
                message="Face recognition active",
                faces=(),
                frame_count=0,
                fps=0.0,
                error_message=None,
            )

        current_input_size = (320, 320)
        frame_counter = 0
        last_fps_time = time.time()
        fps_frame_count = 0
        current_fps = 0.0

        try:
            # 3. Worker loop
            while not self._stop.is_set():
                tick_start = time.time()

                frame = self._extract_frame()
                if frame is None:
                    # Idle sleep when no new frame is available or frame is invalid
                    self._stop.wait(min(0.01, self._inference_interval))
                    continue

                frame_h, frame_w = frame.shape[:2]

                # Update YuNet input size if frame dimensions changed
                if (frame_w, frame_h) != current_input_size:
                    try:
                        detector.setInputSize((frame_w, frame_h))
                        current_input_size = (frame_w, frame_h)
                    except Exception as exc:
                        logger.warning("Failed to update detector input size to (%d, %d): %s", frame_w, frame_h, exc)
                        self._stop.wait(min(0.01, self._inference_interval))
                        continue

                # Run face detection
                detected_faces: List[FaceDetection] = []
                try:
                    retval, raw_faces = detector.detect(frame)
                except Exception as exc:
                    logger.warning("Face detection failed on frame: %s", exc)
                    raw_faces = None

                if raw_faces is not None and len(raw_faces) > 0:
                    for face_row in raw_faces:
                        try:
                            # face_row layout: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rc, y_rc, x_lc, y_lc, score]
                            raw_x = float(face_row[0])
                            raw_y = float(face_row[1])
                            raw_w = float(face_row[2])
                            raw_h = float(face_row[3])
                            conf_score = float(face_row[14])

                            # Clamp bounding box strictly to frame boundaries
                            x1 = max(0, min(frame_w, int(round(raw_x))))
                            y1 = max(0, min(frame_h, int(round(raw_y))))
                            x2 = max(0, min(frame_w, int(round(raw_x + raw_w))))
                            y2 = max(0, min(frame_h, int(round(raw_y + raw_h))))
                            clamped_w = max(0, x2 - x1)
                            clamped_h = max(0, y2 - y1)

                            if clamped_w <= 0 or clamped_h <= 0:
                                # Bounding box is completely outside the frame boundary
                                continue

                            landmarks = (
                                (float(face_row[4]), float(face_row[5])),    # Right eye
                                (float(face_row[6]), float(face_row[7])),    # Left eye
                                (float(face_row[8]), float(face_row[9])),    # Nose tip
                                (float(face_row[10]), float(face_row[11])),  # Right mouth corner
                                (float(face_row[12]), float(face_row[13])),  # Left mouth corner
                            )

                            # Align and crop face for SFace (uses raw face_row for transformation accuracy)
                            aligned_face = recognizer.alignCrop(frame, face_row)
                            if aligned_face is None or not isinstance(aligned_face, np.ndarray) or aligned_face.size == 0:
                                continue

                            # Extract 128-dimensional embedding
                            raw_feature = recognizer.feature(aligned_face)
                            if raw_feature is None or not isinstance(raw_feature, np.ndarray) or raw_feature.size != 128:
                                continue

                            feat_flat = raw_feature.flatten().astype(np.float32)
                            if len(feat_flat) != 128 or not np.all(np.isfinite(feat_flat)):
                                # Non-finite or wrong-dimension embeddings must be discarded
                                continue

                            # L2 normalization
                            l2_norm = float(np.linalg.norm(feat_flat))
                            if l2_norm <= 1e-12 or not math.isfinite(l2_norm):
                                # Discard invalid or zero-norm embeddings, do not publish
                                continue

                            norm_embedding = (feat_flat / l2_norm).astype(np.float32)
                            if not math.isfinite(float(np.linalg.norm(norm_embedding))):
                                continue

                            # Enforce C-level immutable buffer protection
                            immutable_embedding = np.frombuffer(norm_embedding.tobytes(), dtype=np.float32)

                            # In Stage 1: All faces are strictly labelled UNKNOWN (no identity matching)
                            detected_faces.append(
                                FaceDetection(
                                    bbox=(x1, y1, clamped_w, clamped_h),
                                    score=conf_score,
                                    landmarks=landmarks,
                                    embedding=immutable_embedding,
                                    label="UNKNOWN",
                                )
                            )
                        except Exception as exc:
                            logger.warning("Error processing detected face candidate: %s", exc)
                            continue

                frame_counter += 1
                fps_frame_count += 1
                now = time.time()
                fps_elapsed = now - last_fps_time
                if fps_elapsed >= 1.0:
                    current_fps = round(fps_frame_count / fps_elapsed, 1)
                    fps_frame_count = 0
                    last_fps_time = now

                # Publish immutable snapshot under lock
                with self._lock:
                    self._snapshot = FaceWorkerSnapshot(
                        timestamp=now,
                        state="running",
                        message="Face recognition active",
                        faces=tuple(detected_faces),
                        frame_count=frame_counter,
                        fps=current_fps,
                        error_message=None,
                    )

                # Throttle to configured inference interval to prevent high CPU utilization
                elapsed_infer = time.time() - tick_start
                sleep_duration = max(0.001, self._inference_interval - elapsed_infer)
                self._stop.wait(sleep_duration)

        except Exception as exc:
            err_msg = f"Worker thread encountered unhandled exception: {exc}"
            logger.error(err_msg, exc_info=True)
            with self._lock:
                self._snapshot = FaceWorkerSnapshot(
                    timestamp=time.time(),
                    state="error",
                    message=err_msg,
                    faces=(),
                    frame_count=frame_counter,
                    fps=0.0,
                    error_message=str(exc),
                )
        finally:
            # Cleanup on loop termination: reliably set off if not in error state
            with self._lock:
                if self._snapshot.state != "error":
                    self._snapshot = FaceWorkerSnapshot(
                        timestamp=time.time(),
                        state="off",
                        message="Stopped",
                        faces=(),
                        frame_count=frame_counter,
                        fps=0.0,
                        error_message=None,
                    )
