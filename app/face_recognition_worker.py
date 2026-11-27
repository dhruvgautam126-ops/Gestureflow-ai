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
import queue
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
    """Background worker for offline face detection and feature extraction."""

    def __init__(
        self,
        frame_provider: Optional[FrameProvider] = None,
        *,
        yunet_path: Optional[Union[str, Path]] = None,
        sface_path: Optional[Union[str, Path]] = None,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        nms_threshold: float = DEFAULT_NMS_THRESHOLD,
        top_k: int = DEFAULT_TOP_K,
        inference_interval: float = DEFAULT_INFERENCE_INTERVAL,
        coordinator=None,
        kind=None,
    ) -> None:
        self._frame_provider = frame_provider
        self._yunet_path = Path(yunet_path) if yunet_path is not None else DEFAULT_YUNET_PATH
        self._sface_path = Path(sface_path) if sface_path is not None else DEFAULT_SFACE_PATH
        self._score_threshold = score_threshold
        self._nms_threshold = nms_threshold
        self._top_k = top_k
        self._inference_interval = inference_interval
        self._coordinator = coordinator
        self._kind = kind
        self._packet_queue = (
            coordinator.get_packet_queue(kind)
            if coordinator is not None and kind is not None
            else None
        )

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._snapshot = FaceWorkerSnapshot(
            timestamp=time.time(),
            state="off",
            message="Initialized",
            faces=(),
            frame_count=0,
            fps=0.0,
            error_message=None,
        )

    def start(self) -> bool:
        """Start the background recognition worker thread if not already running."""
        with self._lock:
            previous_thread = self._thread
            stopping_previous = self._stop_event.is_set()

        if previous_thread is not None and previous_thread.is_alive():
            if not stopping_previous:
                logger.warning("FaceRecognitionWorker already running.")
                return False
            previous_thread.join(timeout=3.0)
            if previous_thread.is_alive():
                logger.warning("Previous FaceRecognitionWorker is still stopping.")
                return False

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                logger.warning("FaceRecognitionWorker already running.")
                return False

            self._stop_event.clear()
            self._snapshot = FaceWorkerSnapshot(
                timestamp=time.time(),
                state="starting",
                message="Starting worker thread...",
                faces=(),
                frame_count=0,
                fps=0.0,
                error_message=None,
            )

            self._thread = threading.Thread(
                target=self._run_loop,
                name="HomeGuardFaceRecognition",
                daemon=True,
            )
            self._thread.start()
            logger.info("FaceRecognitionWorker thread started.")
            return True

    def request_stop(self) -> None:
        """Signal the background thread to stop without blocking."""
        self._stop_event.set()
        with self._lock:
            if self._snapshot.state in ("starting", "running"):
                self._snapshot = FaceWorkerSnapshot(
                    timestamp=self._snapshot.timestamp,
                    state="stopping",
                    message="Stopping worker thread...",
                    faces=self._snapshot.faces,
                    frame_count=self._snapshot.frame_count,
                    fps=self._snapshot.fps,
                    error_message=self._snapshot.error_message,
                )
        logger.info("FaceRecognitionWorker stop requested.")

    def shutdown(self, timeout: float = 3.0) -> None:
        """Request stop and join the background thread with a timeout."""
        self.request_stop()
        with self._lock:
            t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=timeout)
            if t.is_alive():
                logger.warning("FaceRecognitionWorker thread did not terminate within timeout.")
            else:
                logger.info("FaceRecognitionWorker thread joined successfully.")
        with self._lock:
            self._thread = None

    def get_snapshot(self) -> FaceWorkerSnapshot:
        """Return a thread-safe copy of the latest worker snapshot."""
        with self._lock:
            return self._snapshot

    def _run_loop(self) -> None:
        """Main execution loop running on the background daemon thread."""
        detector = None
        recognizer = None

        try:
            # 1. Validate models exist
            if not self._yunet_path.exists():
                raise FileNotFoundError(f"YuNet model file not found: {self._yunet_path}")
            if not self._sface_path.exists():
                raise FileNotFoundError(f"SFace model file not found: {self._sface_path}")

            # 2. Initialize OpenCV DNN Face Detector (YuNet) and Recognizer (SFace)
            # Note: We use default image size (320, 320) or dynamic; YuNet supports setInputSize.
            detector = cv2.FaceDetectorYN.create(
                model=str(self._yunet_path),
                config="",
                input_size=(320, 320),
                score_threshold=self._score_threshold,
                nms_threshold=self._nms_threshold,
                top_k=self._top_k,
            )
            recognizer = cv2.FaceRecognizerSF.create(
                model=str(self._sface_path),
                config="",
            )

            with self._lock:
                self._snapshot = FaceWorkerSnapshot(
                    timestamp=time.time(),
                    state="running",
                    message="Models loaded successfully. Processing frames...",
                    faces=(),
                    frame_count=0,
                    fps=0.0,
                    error_message=None,
                )
            logger.info("YuNet and SFace models loaded successfully.")

        except Exception as exc:
            logger.error("Failed to initialize FaceRecognitionWorker models: %s", exc, exc_info=True)
            with self._lock:
                self._snapshot = FaceWorkerSnapshot(
                    timestamp=time.time(),
                    state="error",
                    message=f"Initialization error: {exc}",
                    faces=(),
                    frame_count=0,
                    fps=0.0,
                    error_message=str(exc),
                )
            return

        frame_counter = 0
        fps_smoothing = 0.0
        last_time = time.time()

        try:
            while not self._stop_event.is_set():
                loop_start = time.time()
                frame = None
                packet_seq = 0
                packet_ts = loop_start

                if self._coordinator is not None and self._packet_queue is not None:
                    try:
                        packet = self._packet_queue.get(timeout=0.05)
                    except queue.Empty:
                        continue

                    gen = self._coordinator.get_generation(self._kind)
                    frame = packet.frame
                    packet_seq = packet.seq
                    packet_ts = packet.timestamp
                elif self._frame_provider is not None:
                    # Legacy fallback mode
                    raw_frame = self._frame_provider()
                    if raw_frame is None:
                        time.sleep(0.01)
                        continue

                    # Unpack provider return types
                    if isinstance(raw_frame, tuple):
                        if len(raw_frame) == 2:
                            _, frame = raw_frame
                        else:
                            frame = None
                    elif isinstance(raw_frame, np.ndarray):
                        frame = raw_frame
                    else:
                        frame = None

                    if frame is None or frame.size == 0:
                        time.sleep(0.01)
                        continue
                    time.sleep(self._inference_interval)
                else:
                    time.sleep(0.05)
                    continue

                if frame is None or frame.size == 0:
                    continue

                h, w = frame.shape[:2]
                detector.setInputSize((w, h))

                # Run YuNet face detection
                retval, faces_raw = detector.detect(frame)
                detections: List[FaceDetection] = []

                if retval > 0 and faces_raw is not None:
                    for face_row in faces_raw:
                        # YuNet output row format:
                        # [x1, y1, w, h, x_re, y_re, x_le, y_le, x_nose, y_nose, x_ra, y_ra, x_la, y_la, score]
                        x, y, fw, fh = map(int, face_row[0:4])
                        score = float(face_row[14])

                        # Clamp bounding box to frame boundaries
                        x_c = max(0, min(x, w - 1))
                        y_c = max(0, min(y, h - 1))
                        w_c = max(1, min(fw, w - x_c))
                        h_c = max(1, min(fh, h - y_c))
                        clamped_bbox = (x_c, y_c, w_c, h_c)

                        # Extract 5 facial landmarks
                        # right eye, left eye, nose, right mouth corner, left mouth corner
                        landmarks = (
                            (float(face_row[4]), float(face_row[5])),
                            (float(face_row[6]), float(face_row[7])),
                            (float(face_row[8]), float(face_row[9])),
                            (float(face_row[10]), float(face_row[11])),
                            (float(face_row[12]), float(face_row[13])),
                        )

                        # Align face and extract SFace embedding
                        aligned_face = recognizer.alignCrop(frame, face_row)
                        feature = recognizer.feature(aligned_face)  # shape (1, 128) float32

                        # SFace feature is already L2-normalized, but ensure clean 1D float32 array
                        emb_1d = feature.flatten().astype(np.float32)
                        
                        # Create immutable C-level buffer
                        immutable_emb = np.frombuffer(emb_1d.tobytes(), dtype=np.float32)

                        det = FaceDetection(
                            bbox=clamped_bbox,
                            score=score,
                            landmarks=landmarks,
                            embedding=immutable_emb,
                            label="UNKNOWN",
                        )
                        detections.append(det)

                frame_counter += 1
                now = time.time()
                dt = now - last_time
                last_time = now
                current_fps = 1.0 / dt if dt > 0 else 0.0
                if fps_smoothing == 0.0:
                    fps_smoothing = current_fps
                else:
                    fps_smoothing = 0.9 * fps_smoothing + 0.1 * current_fps

                tuple_detections = tuple(detections)

                if self._coordinator is not None:
                    from app.coordinator import WorkerResult
                    worker_res = WorkerResult(
                        kind=self._kind,
                        source_seq=packet_seq,
                        source_ts=packet_ts,
                        generation=gen,
                        face_detections=tuple_detections,
                    )
                    self._coordinator.publish_result(worker_res)

                with self._lock:
                    self._snapshot = FaceWorkerSnapshot(
                        timestamp=now,
                        state="running",
                        message=f"Processed frame {frame_counter}, detected {len(tuple_detections)} face(s).",
                        faces=tuple_detections,
                        frame_count=frame_counter,
                        fps=fps_smoothing,
                        error_message=None,
                    )

        except Exception as exc:
            logger.error("Error in FaceRecognitionWorker run loop: %s", exc, exc_info=True)
            with self._lock:
                self._snapshot = FaceWorkerSnapshot(
                    timestamp=time.time(),
                    state="error",
                    message=f"Runtime error: {exc}",
                    faces=(),
                    frame_count=frame_counter,
                    fps=0.0,
                    error_message=str(exc),
                )
        finally:
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
