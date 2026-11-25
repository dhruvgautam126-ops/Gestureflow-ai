"""Standalone test suite for HomeGuardAI Stage 1 Face Recognition Worker.

Verifies:
1. Model loading (YuNet face detector + SFace feature recognizer).
2. Worker lifecycle (start -> running -> shutdown -> off).
3. Frame robustness (None, empty, noise).
4. Synthetic face detection, 128-d finite embedding extraction, UNKNOWN label, and clamped bounding boxes.
5. Zero disk persistence (no DB, DPAPI, or biometric files created).
6. SEC-01: shutdown() thread reference race condition fix & timeout handling.
7. Issue 2 & 3: Invalid and zero-norm embeddings are discarded; all published embeddings are strictly 128-d, float32, finite, and L2-normalized.
8. SEC-03: C-level embedding immutability (write-flag cannot be re-enabled).
9. Issue 5: Frame validation safely rejects invalid dimensions, unsupported channels, and invalid data types.
10. SEC-02: Camera sequence resets (to 0) and rollovers (>100 frames) handled safely without dropping stream.
11. Issue 7: Worker cleanup executes reliably, including after exceptions.

Run with:
    python -m py_compile app/face_recognition_worker.py app/run_face_recognition_test.py
    python -m app.run_face_recognition_test
"""

from __future__ import annotations

import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from app.face_recognition_worker import (
    DEFAULT_SFACE_PATH,
    DEFAULT_YUNET_PATH,
    SEQUENCE_RESET_THRESHOLD,
    FaceDetection,
    FaceRecognitionWorker,
    FaceWorkerSnapshot,
)


def _scan_persistence_files(base_dir: Path) -> Set[Path]:
    """Scan directory for any database or biometric persistence files."""
    forbidden_extensions = {".db", ".sqlite", ".sqlite3", ".dpapi", ".key", ".bin"}
    found: Set[Path] = set()
    for root, _, files in os.walk(base_dir):
        for f in files:
            p = Path(root) / f
            if p.suffix.lower() in forbidden_extensions:
                found.add(p)
    return found


def _create_synthetic_face_frame(width: int = 320, height: int = 320) -> np.ndarray:
    """Generate a synthetic geometric face pattern (no real biometric data)."""
    img = np.full((height, width, 3), 210, dtype=np.uint8)
    # Head outline (skin-tone ellipse)
    center_x, center_y = width // 2, height // 2
    cv2.ellipse(img, (center_x, center_y), (70, 90), 0, 0, 360, (180, 150, 120), -1)
    # Eyes (white sclera + dark pupil)
    left_eye_x, right_eye_x = center_x - 25, center_x + 25
    eye_y = center_y - 20
    cv2.circle(img, (left_eye_x, eye_y), 10, (255, 255, 255), -1)
    cv2.circle(img, (right_eye_x, eye_y), 10, (255, 255, 255), -1)
    cv2.circle(img, (left_eye_x, eye_y), 4, (20, 20, 20), -1)
    cv2.circle(img, (right_eye_x, eye_y), 4, (20, 20, 20), -1)
    # Eyebrows
    cv2.line(img, (left_eye_x - 12, eye_y - 12), (left_eye_x + 10, eye_y - 10), (40, 30, 20), 2)
    cv2.line(img, (right_eye_x - 10, eye_y - 10), (right_eye_x + 12, eye_y - 12), (40, 30, 20), 2)
    # Nose
    nose_y = center_y + 15
    cv2.ellipse(img, (center_x, nose_y), (8, 14), 0, 0, 360, (140, 110, 80), -1)
    # Mouth
    mouth_y = center_y + 45
    cv2.ellipse(img, (center_x, mouth_y), (25, 10), 0, 0, 180, (40, 40, 180), -1)
    return img


def test_models_exist_and_load() -> Tuple[bool, str]:
    """Test 1: Verify model files exist and OpenCV initializes both models."""
    if not DEFAULT_YUNET_PATH.is_file():
        return False, f"YuNet model not found at: {DEFAULT_YUNET_PATH}"
    if not DEFAULT_SFACE_PATH.is_file():
        return False, f"SFace model not found at: {DEFAULT_SFACE_PATH}"

    try:
        detector = cv2.FaceDetectorYN.create(
            str(DEFAULT_YUNET_PATH), "", (320, 320), 0.8, 0.3, 5000
        )
        if detector is None:
            return False, "cv2.FaceDetectorYN.create returned None"
    except Exception as exc:
        return False, f"Error creating FaceDetectorYN: {exc}"

    try:
        recognizer = cv2.FaceRecognizerSF.create(str(DEFAULT_SFACE_PATH), "")
        if recognizer is None:
            return False, "cv2.FaceRecognizerSF.create returned None"
    except Exception as exc:
        return False, f"Error creating FaceRecognizerSF: {exc}"

    return True, "YuNet (detector) and SFace (recognizer) loaded successfully"


def test_worker_lifecycle() -> Tuple[bool, str]:
    """Test 2: Verify worker starts, transitions states, and shuts down cleanly."""
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def dummy_provider() -> np.ndarray:
        return dummy_frame

    worker = FaceRecognitionWorker(
        frame_provider=dummy_provider,
        inference_interval=0.02,
    )

    initial_snap = worker.snapshot()
    if initial_snap.state != "off":
        return False, f"Initial state should be 'off', got '{initial_snap.state}'"

    started = worker.start()
    if not started:
        return False, "worker.start() returned False on initial start"

    # Wait for worker to load models and transition to running state (up to 2.0s)
    wait_start = time.time()
    running_snap = worker.snapshot()
    while time.time() - wait_start < 2.0:
        running_snap = worker.snapshot()
        if running_snap.state == "running":
            break
        time.sleep(0.05)

    if running_snap.state != "running":
        worker.shutdown()
        return False, f"Expected running state, got '{running_snap.state}'"

    if not worker.is_alive:
        worker.shutdown()
        return False, "worker.is_alive returned False while running"

    worker.shutdown(timeout=2.0)
    if worker.is_alive:
        return False, "worker thread is still alive after shutdown"

    final_snap = worker.snapshot()
    if final_snap.state != "off":
        return False, f"Expected final state 'off', got '{final_snap.state}'"

    return True, "Worker lifecycle (start -> running -> shutdown -> off) passed"


def test_empty_and_synthetic_frame_robustness() -> Tuple[bool, str]:
    """Test 3: Verify worker handles None, empty, noise, and invalid frames without crashing."""
    frame_cycle = [
        None,
        (False, None),
        np.zeros((0, 0, 3), dtype=np.uint8),  # 0-dimension frame
        np.zeros((480, 640, 3), dtype=np.uint8),  # Blank black
        np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8),  # Noise
    ]
    idx = 0

    def cycling_provider():
        nonlocal idx
        item = frame_cycle[idx % len(frame_cycle)]
        idx += 1
        return item

    worker = FaceRecognitionWorker(
        frame_provider=cycling_provider,
        inference_interval=0.01,
    )

    try:
        worker.start()
        # Wait for worker to load models and transition to running state
        wait_start = time.time()
        while time.time() - wait_start < 2.0:
            if worker.snapshot().state == "running":
                break
            time.sleep(0.05)

        # Let it consume several cycles of diverse frames
        time.sleep(0.4)
        snap = worker.snapshot()
        if snap.state != "running":
            return False, f"Worker failed or crashed on malformed frames: state='{snap.state}' error='{snap.error_message}'"
        if snap.error_message is not None:
            return False, f"Worker reported unexpected error: {snap.error_message}"
    finally:
        worker.shutdown(timeout=2.0)

    return True, "Worker safely ingested None, empty, and noise frames without exceptions"


def test_face_detection_and_embedding_extraction() -> Tuple[bool, str]:
    """Test 4: Verify detection, 128-d finite embedding extraction, UNKNOWN label, and clamped bbox (SEC-04)."""
    frame_w, frame_h = 320, 320
    synthetic_face = _create_synthetic_face_frame(frame_w, frame_h)

    def synthetic_provider():
        return synthetic_face.copy()

    # Use low score_threshold (0.3) to detect synthetic geometric face reliably
    worker = FaceRecognitionWorker(
        frame_provider=synthetic_provider,
        score_threshold=0.3,
        inference_interval=0.02,
    )

    detected_faces: Tuple[FaceDetection, ...] = ()
    try:
        worker.start()
        # Poll snapshot for detected face (up to 2.5s)
        start_wait = time.time()
        while time.time() - start_wait < 2.5:
            snap = worker.snapshot()
            if len(snap.faces) > 0:
                detected_faces = snap.faces
                break
            time.sleep(0.05)
    finally:
        worker.shutdown(timeout=2.0)

    if len(detected_faces) == 0:
        return False, "No face detected in synthetic face pattern within timeout"

    face = detected_faces[0]

    # Verify label is strictly 'UNKNOWN'
    if face.label != "UNKNOWN":
        return False, f"Face label must be 'UNKNOWN', got '{face.label}'"

    # Verify embedding dimensions
    if face.embedding is None:
        return False, "Face embedding is None"

    if face.embedding.shape != (128,):
        return False, f"Expected embedding shape (128,), got {face.embedding.shape}"

    if face.embedding.dtype != np.float32:
        return False, f"Expected embedding dtype float32, got {face.embedding.dtype}"

    # Verify all embedding values are finite (no NaN or Inf)
    if not np.all(np.isfinite(face.embedding)):
        return False, "Embedding contains non-finite values (NaN or Inf)"

    # Verify L2 normalization
    norm = float(np.linalg.norm(face.embedding))
    if not math.isclose(norm, 1.0, rel_tol=1e-3, abs_tol=1e-3):
        return False, f"Embedding is not L2 normalized: norm={norm}"

    # Verify landmarks
    if len(face.landmarks) != 5:
        return False, f"Expected 5 facial landmarks, got {len(face.landmarks)}"

    # SEC-04 Verification: Verify bounding box is strictly clamped to frame boundaries
    bx, by, bw, bh = face.bbox
    if bx < 0 or by < 0:
        return False, f"Bounding box has negative coordinates: ({bx}, {by})"
    if bw <= 0 or bh <= 0:
        return False, f"Bounding box has invalid dimensions: ({bw}, {bh})"
    if bx + bw > frame_w:
        return False, f"Bounding box exceeds frame width: {bx} + {bw} > {frame_w}"
    if by + bh > frame_h:
        return False, f"Bounding box exceeds frame height: {by} + {bh} > {frame_h}"

    return True, (
        f"Detected {len(detected_faces)} synthetic face(s). "
        f"Embedding verified: 128-d, float32, L2-normalized (norm={norm:.4f}), "
        f"all values finite, label='{face.label}', bbox strictly clamped within ({frame_w}x{frame_h})"
    )


def test_zero_disk_persistence(initial_files: Set[Path]) -> Tuple[bool, str]:
    """Test 5: Verify zero database, biometric, or credential files were written to disk."""
    current_files = _scan_persistence_files(PROJECT_ROOT)
    new_files = current_files - initial_files

    if new_files:
        files_str = ", ".join(str(f) for f in new_files)
        return False, f"Forbidden persistence files were created on disk: {files_str}"

    return True, "Verified zero database, biometric, or credential files created on disk"


def test_sec01_shutdown_thread_reference_race_condition() -> Tuple[bool, str]:
    """SEC-01 Test: Verify shutdown() thread reference tracking and timeout behavior."""
    def dummy_provider():
        return None

    worker = FaceRecognitionWorker(frame_provider=dummy_provider)

    # 1. Test timeout preserves 'stopping' state and does not claim 'off'
    simulated_thread_stop = threading.Event()

    def stubborn_worker_target():
        simulated_thread_stop.wait(2.0)

    stubborn_thread = threading.Thread(target=stubborn_worker_target, daemon=True)
    stubborn_thread.start()

    with worker._lock:
        worker._thread = stubborn_thread
        worker._snapshot = FaceWorkerSnapshot(
            timestamp=time.time(),
            state="running",
            message="Simulated running worker",
        )

    try:
        worker.shutdown(timeout=0.05)
        snap = worker.snapshot()
        if snap.state == "off":
            return False, "shutdown() reported state 'off' even though worker thread was still alive!"
        if snap.state != "stopping":
            return False, f"Expected state 'stopping' during timeout, got '{snap.state}'"
    finally:
        simulated_thread_stop.set()
        stubborn_thread.join(timeout=1.0)

    # 2. Test clean shutdown clears _thread to None
    worker.shutdown(timeout=0.5)
    with worker._lock:
        if worker._thread is not None:
            return False, "worker._thread was not cleared to None after clean shutdown"

    # 3. Test concurrent start during shutdown does not get overwritten
    mock_t1 = threading.Thread(target=lambda: time.sleep(0.01), daemon=True)
    mock_t2 = threading.Thread(target=lambda: time.sleep(1.0), daemon=True)
    mock_t1.start()
    mock_t2.start()

    with worker._lock:
        worker._thread = mock_t1
        worker._snapshot = FaceWorkerSnapshot(timestamp=time.time(), state="running", message="T1 running")

    # While T1 shuts down, simulate another thread starting T2
    def simulate_concurrent_start():
        time.sleep(0.02)
        with worker._lock:
            worker._thread = mock_t2
            worker._snapshot = FaceWorkerSnapshot(timestamp=time.time(), state="running", message="T2 running")

    start_sim = threading.Thread(target=simulate_concurrent_start, daemon=True)
    start_sim.start()

    worker.shutdown(timeout=0.2)
    start_sim.join(timeout=1.0)

    with worker._lock:
        # T2 must not be overwritten or corrupted by T1's shutdown
        if worker._snapshot.message == "Shutdown timed out: worker thread still active":
            return False, "T1's shutdown falsely marked T2 as timed out/stopping!"
        if worker._thread is None:
            return False, "T1's shutdown erroneously cleared T2 from worker._thread!"

    return True, "SEC-01 verified: shutdown accurately tracks joined thread reference and avoids race conditions"


def test_issue2_and_3_zero_and_invalid_norm_embeddings_discarded() -> Tuple[bool, str]:
    """Issue 2 & 3 Test: Verify invalid or zero-norm embeddings are discarded and never published."""
    synthetic_face = _create_synthetic_face_frame(320, 320)

    worker = FaceRecognitionWorker(
        frame_provider=lambda: synthetic_face.copy(),
        score_threshold=0.3,
        inference_interval=0.02,
    )

    try:
        worker.start()
        # Wait for at least one detection
        wait_start = time.time()
        faces: Tuple[FaceDetection, ...] = ()
        while time.time() - wait_start < 2.5:
            snap = worker.snapshot()
            if len(snap.faces) > 0:
                faces = snap.faces
                break
            time.sleep(0.05)

        if len(faces) == 0:
            return False, "Failed to capture detected face snapshot"

        for idx, f in enumerate(faces):
            norm = float(np.linalg.norm(f.embedding))
            if norm <= 1e-6:
                return False, f"Face #{idx} has zero or near-zero norm: {norm} (must be discarded!)"
            if not math.isclose(norm, 1.0, abs_tol=1e-3):
                return False, f"Face #{idx} is not L2-normalized: norm={norm}"
            if f.embedding.shape != (128,):
                return False, f"Face #{idx} has wrong shape {f.embedding.shape}, expected (128,)"
            if f.embedding.dtype != np.float32:
                return False, f"Face #{idx} has dtype {f.embedding.dtype}, expected float32"
            if not np.all(np.isfinite(f.embedding)):
                return False, f"Face #{idx} contains non-finite embedding values"
    finally:
        worker.shutdown(timeout=2.0)

    return True, "All published embeddings are strictly 128-d, float32, finite, and L2-normalized; zero-norm discarded"


def test_sec03_c_level_embedding_immutability() -> Tuple[bool, str]:
    """SEC-03 Test: Verify C-level embedding immutability (write flag cannot be re-enabled)."""
    test_embedding = np.ones(128, dtype=np.float32) / math.sqrt(128.0)
    detection = FaceDetection(
        bbox=(10, 20, 50, 60),
        score=0.95,
        landmarks=((1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0), (5.0, 5.0)),
        embedding=test_embedding,
        label="UNKNOWN",
    )

    # 1. Direct in-place mutation must raise ValueError
    try:
        detection.embedding[0] = 999.0
        return False, "FaceDetection embedding allowed direct item assignment!"
    except ValueError as exc:
        if "read-only" not in str(exc).lower():
            return False, f"Expected 'read-only' error on direct assignment, got: {exc}"

    # 2. Attempting to re-enable writeable flag must be blocked at C/buffer level
    try:
        detection.embedding.setflags(write=True)
        return False, "detection.embedding.setflags(write=True) succeeded! (C-level immutability violated)"
    except ValueError as exc:
        if "cannot set writeable flag" not in str(exc).lower() and "cannot set writeable" not in str(exc).lower():
            return False, f"Expected buffer protocol rejection error, got: {exc}"

    return True, "SEC-03 verified: embeddings backed by immutable buffer; direct writes and setflags(write=True) blocked"


def test_issue5_frame_validation_rejects_invalid_types_channels_dimensions() -> Tuple[bool, str]:
    """Issue 5 Test: Verify frame validation rejects non-uint8, non-3-channel, and 0-dimension frames."""
    validator = FaceRecognitionWorker._validate_frame

    # 1. Reject non-ndarray
    if validator("not an array") is not None:
        return False, "Failed to reject string input"
    if validator(None) is not None:
        return False, "Failed to reject None input"
    if validator([1, 2, 3]) is not None:
        return False, "Failed to reject list input"

    # 2. Reject non-uint8 dtype
    float_frame = np.zeros((100, 100, 3), dtype=np.float32)
    if validator(float_frame) is not None:
        return False, "Failed to reject float32 frame"
    int32_frame = np.zeros((100, 100, 3), dtype=np.int32)
    if validator(int32_frame) is not None:
        return False, "Failed to reject int32 frame"

    # 3. Reject non-3-dimensional array
    gray_frame = np.zeros((100, 100), dtype=np.uint8)  # 2D grayscale
    if validator(gray_frame) is not None:
        return False, "Failed to reject 2D grayscale frame"
    four_d_frame = np.zeros((1, 100, 100, 3), dtype=np.uint8)  # 4D
    if validator(four_d_frame) is not None:
        return False, "Failed to reject 4D array"

    # 4. Reject unsupported channel counts
    rgba_frame = np.zeros((100, 100, 4), dtype=np.uint8)  # 4-channel
    if validator(rgba_frame) is not None:
        return False, "Failed to reject 4-channel RGBA frame"
    two_channel = np.zeros((100, 100, 2), dtype=np.uint8)  # 2-channel
    if validator(two_channel) is not None:
        return False, "Failed to reject 2-channel frame"

    # 5. Reject 0-dimension frames
    zero_h = np.zeros((0, 100, 3), dtype=np.uint8)
    if validator(zero_h) is not None:
        return False, "Failed to reject zero-height frame"
    zero_w = np.zeros((100, 0, 3), dtype=np.uint8)
    if validator(zero_w) is not None:
        return False, "Failed to reject zero-width frame"

    # 6. Accept valid 3-channel uint8 BGR frame
    valid_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    res = validator(valid_bgr)
    if res is None or res.shape != (480, 640, 3):
        return False, "Valid 3-channel uint8 frame was unexpectedly rejected"

    return True, "Frame validator strictly enforces 3-channel uint8 BGR arrays with positive dimensions"


def test_sec02_camera_sequence_reset_and_rollover() -> Tuple[bool, str]:
    """SEC-02 Test: Verify camera sequence counter resets and rollovers are handled safely."""
    valid_frame = np.zeros((240, 320, 3), dtype=np.uint8)

    stream = [
        (1, valid_frame.copy()),       # 1: valid sequence -> accept
        (2, valid_frame.copy()),       # 2: valid sequence -> accept
        (2, valid_frame.copy()),       # 2: duplicate sequence -> REJECT
        (1, valid_frame.copy()),       # 1: small out-of-order jitter -> REJECT
        (3, valid_frame.copy()),       # 3: valid sequence -> accept
        (0, valid_frame.copy()),       # 0: camera reset (first == 0) -> ACCEPT (SEC-02)
        (1, valid_frame.copy()),       # 1: sequence after reset -> accept
        (5000, valid_frame.copy()),    # 5000: sequence jump -> accept
        (5000, valid_frame.copy()),    # 5000: duplicate -> REJECT
        (4999, valid_frame.copy()),    # 4999: minor jitter (<100) -> REJECT
        (10, valid_frame.copy()),      # 10: large rollover/reset (5000 - 10 >= 100) -> ACCEPT (SEC-02)
        (11, valid_frame.copy()),      # 11: sequence after rollover -> accept
    ]
    idx = 0

    def sequenced_provider():
        nonlocal idx
        if idx < len(stream):
            item = stream[idx]
            idx += 1
            return item
        return None

    worker = FaceRecognitionWorker(frame_provider=sequenced_provider)

    # 1. Normal progress
    if worker._extract_frame() is None:
        return False, "Initial sequence 1 rejected"
    if worker._extract_frame() is None:
        return False, "Sequence 2 rejected"
    if worker._extract_frame() is not None:
        return False, "Duplicate sequence 2 was accepted (must reject)"
    if worker._extract_frame() is not None:
        return False, "Small out-of-order sequence 1 was accepted (must reject)"
    if worker._extract_frame() is None:
        return False, "Sequence 3 rejected"

    # 2. Camera reset to 0
    if worker._extract_frame() is None:
        return False, "Camera reset sequence 0 was rejected! (SEC-02 failure)"
    if worker._extract_frame() is None:
        return False, "Sequence 1 after reset was rejected"

    # 3. High sequence jump and rollover
    if worker._extract_frame() is None:
        return False, "Sequence 5000 rejected"
    if worker._extract_frame() is not None:
        return False, "Duplicate 5000 accepted (must reject)"
    if worker._extract_frame() is not None:
        return False, "Minor backward jitter 4999 accepted (must reject)"
    if worker._extract_frame() is None:
        return False, "Rollover/reset from 5000 to 10 rejected! (SEC-02 failure)"
    if worker._extract_frame() is None:
        return False, "Sequence 11 after rollover rejected"

    return True, "SEC-02 verified: camera resets to 0 and large rollovers accepted; minor jitter and duplicates rejected"


def test_issue7_worker_cleanup_on_exception() -> Tuple[bool, str]:
    """Issue 7 Test: Verify worker cleanup executes reliably and enters error state on unhandled exceptions."""
    def dummy_provider():
        return np.zeros((320, 320, 3), dtype=np.uint8)

    worker = FaceRecognitionWorker(
        frame_provider=dummy_provider,
        inference_interval=0.01,
    )

    # Inject an unhandled exception into worker._extract_frame to simulate unhandled internal loop failure
    call_count = 0
    def crashing_extract():
        nonlocal call_count
        call_count += 1
        if call_count >= 2:
            raise RuntimeError("Injected unhandled failure inside worker loop")
        return np.zeros((320, 320, 3), dtype=np.uint8)

    worker._extract_frame = crashing_extract

    try:
        worker.start()
        # Wait for unhandled exception to trigger inside worker thread
        wait_start = time.time()
        err_snap = worker.snapshot()
        while time.time() - wait_start < 2.0:
            err_snap = worker.snapshot()
            if err_snap.state == "error":
                break
            time.sleep(0.05)

        if err_snap.state != "error":
            return False, f"Expected state 'error' after unhandled exception, got '{err_snap.state}'"
        if not err_snap.error_message or "Injected unhandled failure" not in err_snap.error_message:
            return False, f"Expected error_message containing injected failure, got '{err_snap.error_message}'"
        if worker.is_alive:
            return False, "Worker thread is still alive after unhandled exception crashed it"
    finally:
        worker.shutdown(timeout=2.0)

    # After shutdown, error state must be preserved
    post_snap = worker.snapshot()
    if post_snap.state != "error":
        return False, f"Error state was overwritten on shutdown: got '{post_snap.state}'"

    return True, "Worker reliably catches unhandled exceptions, records error state, and cleans up without hanging"


def run_all_tests() -> bool:
    """Execute all tests and display structured diagnostic report."""
    print("=" * 72)
    print("HOMEGUARDAI — STAGE 1: COMPREHENSIVE FACE RECOGNITION WORKER TESTS")
    print("=" * 72)
    print(f"Python version : {sys.version.split()[0]}")
    print(f"OpenCV version : {cv2.__version__}")
    print(f"Project root   : {PROJECT_ROOT}")
    print(f"YuNet model    : {DEFAULT_YUNET_PATH}")
    print(f"SFace model    : {DEFAULT_SFACE_PATH}")
    print("-" * 72)

    initial_persistence_files = _scan_persistence_files(PROJECT_ROOT)

    tests = [
        ("Model loading (YuNet + SFace)", test_models_exist_and_load),
        ("Worker lifecycle (start / shutdown)", test_worker_lifecycle),
        ("Frame robustness (None, empty, noise)", test_empty_and_synthetic_frame_robustness),
        ("Synthetic face detection & clamped bbox (SEC-04)", test_face_detection_and_embedding_extraction),
        ("Zero disk persistence check", lambda: test_zero_disk_persistence(initial_persistence_files)),
        ("SEC-01: Shutdown thread race fix & timeout", test_sec01_shutdown_thread_reference_race_condition),
        ("Issue 2 & 3: Invalid/zero-norm discarded & normalized", test_issue2_and_3_zero_and_invalid_norm_embeddings_discarded),
        ("SEC-03: C-level embedding buffer immutability", test_sec03_c_level_embedding_immutability),
        ("Issue 5: Frame validation (types, channels, dimensions)", test_issue5_frame_validation_rejects_invalid_types_channels_dimensions),
        ("SEC-02: Camera sequence reset & rollover handling", test_sec02_camera_sequence_reset_and_rollover),
        ("Issue 7: Worker cleanup & exception handling", test_issue7_worker_cleanup_on_exception),
    ]

    all_passed = True
    results: List[Tuple[str, bool, str]] = []

    for name, test_func in tests:
        try:
            passed, msg = test_func()
        except Exception as exc:
            passed = False
            msg = f"Unhandled exception: {exc}"

        results.append((name, passed, msg))
        status_str = "PASS" if passed else "FAIL"
        print(f"[{status_str:4s}] {name}")
        print(f"       Details: {msg}")
        if not passed:
            all_passed = False

    print("-" * 72)
    print(f"Summary: {sum(1 for _, p, _ in results if p)}/{len(results)} tests passed.")
    if all_passed:
        print("ALL STAGE 1 FACE RECOGNITION TESTS PASSED SUCCESSFULLY.")
    else:
        print("SOME TESTS FAILED. PLEASE REVIEW DETAILS ABOVE.")
    print("=" * 72)

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
