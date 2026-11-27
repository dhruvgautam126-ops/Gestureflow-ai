"""Stage 3A integration audit tests for Gesture.AI.

These tests exercise the complete shared-frame pipeline:

    CameraFramePacket -> FrameCoordinator distributor -> worker queues
    -> WorkerResults -> compose() -> central rendering

They are written against the APPROVED Stage 3A requirements, so a failing
test is audit evidence that a requirement is not met.  Fast tests use fake
workers and synthetic frames; the ``TestRealWorkerPublishPath`` class loads
the real MediaPipe / OpenCV models and is therefore slower.

Run with:
    python -m unittest tests.test_stage3a_integration -v
"""

from __future__ import annotations

import inspect
import queue
import threading
import time
import unittest
from types import SimpleNamespace

import numpy as np

from app.camera.capture import CameraFramePacket
from app.coordinator import (
    HISTORY_SIZE,
    QUEUE_MAX_SIZE,
    DetectorKind,
    FrameCoordinator,
    WorkerResult,
)
from app.ui import main_window as mw


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_frame(value: int = 40) -> np.ndarray:
    return np.full((120, 160, 3), value, dtype=np.uint8)


def make_packet(seq: int, value: int = 40) -> CameraFramePacket:
    return CameraFramePacket(
        seq=seq,
        timestamp=1000.0 + seq,
        frame=make_frame(value),
    )


def make_result(kind: DetectorKind, seq: int, generation: int, **fields) -> WorkerResult:
    data = dict(
        kind=kind,
        source_seq=seq,
        source_ts=1000.0 + seq,
        generation=generation,
    )
    data.update(fields)
    return WorkerResult(**data)


def wait_until(predicate, timeout: float = 5.0, interval: float = 0.01) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


class FakeCamera:
    """Camera exposing both the Stage-3A and the legacy interfaces."""

    def __init__(self) -> None:
        self.packet: CameraFramePacket | None = None

    def get_current_packet(self) -> CameraFramePacket | None:
        return self.packet

    def get_frame(self) -> np.ndarray | None:
        return None if self.packet is None else self.packet.frame.copy()


class FakeWorker(threading.Thread):
    """Consumes coordinator packets and publishes WorkerResults like a real worker."""

    def __init__(self, coordinator: FrameCoordinator, kind: DetectorKind, delay: float = 0.0) -> None:
        super().__init__(daemon=True, name=f"FakeWorker-{kind.value}")
        self.coordinator = coordinator
        self.kind = kind
        self.delay = delay
        self._stop_event = threading.Event()
        self.published: list[WorkerResult] = []

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        packet_queue = self.coordinator.get_packet_queue(self.kind)
        while not self._stop_event.is_set():
            try:
                packet = packet_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            generation = self.coordinator.get_generation(self.kind)
            if self.delay:
                time.sleep(self.delay)
            if self.kind is DetectorKind.HUMAN:
                fields = dict(human_people=1, human_pose_draw_data=())
            elif self.kind is DetectorKind.FACE:
                fields = dict(face_detections=())
            else:
                fields = dict(hand_candidates=(), hand_visible_tracks=())
            result = make_result(self.kind, packet.seq, generation, **fields)
            self.coordinator.publish_result(result)
            self.published.append(result)


def inject(coordinator: FrameCoordinator, packet: CameraFramePacket, kinds, history: bool = True) -> None:
    """Inject a packet exactly as the distributor would (bounded queue drop-oldest)."""
    if history:
        with coordinator._lock:
            coordinator._frame_history.append(packet)
    for kind in kinds:
        target = coordinator.get_packet_queue(kind)
        try:
            target.put_nowait(packet)
        except queue.Full:
            try:
                target.get_nowait()
            except queue.Empty:
                pass
            try:
                target.put_nowait(packet)
            except queue.Full:
                pass


# ---------------------------------------------------------------------------
# Coordinator wiring (Stage 3A production wiring)
# ---------------------------------------------------------------------------


class TestCoordinatorWiring(unittest.TestCase):
    def test_distributor_survives_real_packet(self) -> None:
        """The distributor must copy a real CameraFramePacket into history."""
        camera = FakeCamera()
        coordinator = FrameCoordinator(camera_source=camera)
        coordinator.start()
        self.addCleanup(coordinator.stop)

        camera.packet = make_packet(1)

        published = wait_until(lambda: len(coordinator._frame_history) > 0, timeout=1.0)
        thread = coordinator._distributor_thread
        self.assertTrue(
            published,
            "distributor published no frames; it likely raised on the packet fields",
        )
        self.assertIsNotNone(thread)
        self.assertTrue(thread.is_alive(), "distributor thread died while processing a packet")

    def test_mainwindow_wiring_exposes_get_current_packet(self) -> None:
        """MainWindow must hand the coordinator an object with get_current_packet().

        The coordinator's intended input is a camera object exposing
        get_current_packet() returning a CameraFramePacket (as MainWindow does at
        app/ui/main_window.py line 72: FrameCoordinator(camera_source=self.camera)).
        A callable/camera.get_frame bound method is not the intended input.
        """
        camera = FakeCamera()
        # Exactly the wiring at app/ui/main_window.py line 72:
        coordinator = FrameCoordinator(camera_source=camera)
        coordinator.start()
        self.addCleanup(coordinator.stop)
        camera.packet = make_packet(1)
        time.sleep(0.2)  # give the distributor time to process the first packet

        thread = coordinator._distributor_thread
        self.assertIsNotNone(thread)
        self.assertTrue(
            thread.is_alive(),
            "coordinator._distributor_thread is not alive; camera_source must expose "
            "get_current_packet() returning CameraFramePacket",
        )

    def test_mainwindow_starts_coordinator(self) -> None:
        """MainWindow.__init__ must call coordinator.start()."""
        source = inspect.getsource(mw.MainWindow.__init__)
        self.assertIn(
            "self.coordinator.start()",
            source,
            "FrameCoordinator.start() is never called, so the distributor never runs",
        )

    def test_compose_returns_independent_copy(self) -> None:
        """compose() must return a writable copy, never the history frame itself."""
        camera = FakeCamera()
        coordinator = FrameCoordinator(camera_source=camera)
        packet = make_packet(1, value=77)
        with coordinator._lock:
            coordinator._frame_history.append(packet)

        frame, results, synced = coordinator.compose()
        self.assertIsNotNone(frame)
        self.assertFalse(np.shares_memory(frame, packet.frame))
        frame[:] = 0
        self.assertEqual(int(packet.frame[0, 0, 0]), 77)

    def test_history_and_workers_get_independent_frame_copies(self) -> None:
        """Each enabled worker must receive its own writable frame copy."""
        camera = FakeCamera()
        coordinator = FrameCoordinator(camera_source=camera)
        for kind in DetectorKind:
            coordinator.enable(kind)
        coordinator.start()
        self.addCleanup(coordinator.stop)
        camera.packet = make_packet(1)

        delivered = wait_until(
            lambda: all(
                coordinator.get_packet_queue(kind).qsize() > 0 for kind in DetectorKind
            ),
            timeout=1.0,
        )
        self.assertTrue(delivered, "distributor delivered no packets to worker queues")

        human_pkt = coordinator.get_packet_queue(DetectorKind.HUMAN).get_nowait()
        face_pkt = coordinator.get_packet_queue(DetectorKind.FACE).get_nowait()
        hand_pkt = coordinator.get_packet_queue(DetectorKind.HAND).get_nowait()
        self.assertIsNot(human_pkt.frame, face_pkt.frame)
        self.assertIsNot(human_pkt.frame, hand_pkt.frame)
        self.assertIsNot(face_pkt.frame, hand_pkt.frame)
        history_pkt = coordinator._frame_history[-1]
        self.assertIsNot(human_pkt.frame, history_pkt.frame)


# ---------------------------------------------------------------------------
# Full pipeline with three fake workers and synthetic frames
# ---------------------------------------------------------------------------


class TestThreeWorkerPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.camera = FakeCamera()
        self.coordinator = FrameCoordinator(camera_source=self.camera)
        for kind in DetectorKind:
            self.coordinator.enable(kind)
        self.workers = {
            kind: FakeWorker(self.coordinator, kind) for kind in DetectorKind
        }
        for worker in self.workers.values():
            worker.start()

    def tearDown(self) -> None:
        for worker in self.workers.values():
            worker.stop()
        for worker in self.workers.values():
            worker.join(timeout=2.0)
        self.coordinator.stop()

    def _all_results_at(self, seq: int) -> bool:
        frame, results, synced = self.coordinator.compose()
        return (
            frame is not None
            and synced
            and all(result is not None and result.source_seq == seq for result in results.values())
        )

    def test_three_workers_publish_synchronized_results(self) -> None:
        for seq in range(1, 6):
            inject(self.coordinator, make_packet(seq, value=10 + seq), DetectorKind)
            time.sleep(0.02)

        synced = wait_until(lambda: self._all_results_at(5), timeout=5.0)
        frame, results, synced_flag = self.coordinator.compose()
        self.assertTrue(synced, f"no synchronized result set; last results={results}")
        self.assertTrue(synced_flag)
        seqs = {kind: result.source_seq for kind, result in results.items()}
        self.assertEqual(set(seqs.values()), {5}, f"results from different frames: {seqs}")
        # The composed frame must be the clean camera frame (no worker annotation).
        self.assertEqual(int(frame[0, 0, 0]), 15)

    def test_slow_worker_cannot_break_sequence_synchronization(self) -> None:
        self.workers[DetectorKind.HAND].delay = 0.25
        for seq in range(1, 9):
            inject(self.coordinator, make_packet(seq), DetectorKind)
            time.sleep(0.005)

        synchronized = wait_until(
            lambda: self.coordinator.compose()[2]
            and all(result is not None for result in self.coordinator.compose()[1].values()),
            timeout=8.0,
        )
        self.assertTrue(synchronized, "slow worker prevented any synchronized composition")
        frame, results, synced_flag = self.coordinator.compose()
        self.assertTrue(synced_flag)
        self.assertEqual(
            len({result.source_seq for result in results.values()}),
            1,
            f"mixed source sequences in synchronized result: {results}",
        )

    def test_disable_re_enable_race_invalidates_old_generation(self) -> None:
        for seq in range(1, 4):
            inject(self.coordinator, make_packet(seq), DetectorKind)
        self.assertTrue(wait_until(lambda: self._all_results_at(3), timeout=3.0))

        old_generation = self.coordinator.get_generation(DetectorKind.HUMAN)
        self.coordinator.disable(DetectorKind.HUMAN)
        self.assertIsNone(self.coordinator.get_latest_result(DetectorKind.HUMAN))
        self.assertEqual(
            self.coordinator.get_packet_queue(DetectorKind.HUMAN).qsize(), 0
        )

        # A late result from the old generation must be rejected.
        self.coordinator.publish_result(
            make_result(DetectorKind.HUMAN, 3, old_generation, human_people=1)
        )
        self.assertIsNone(self.coordinator.get_latest_result(DetectorKind.HUMAN))

        self.coordinator.enable(DetectorKind.HUMAN)
        new_generation = self.coordinator.get_generation(DetectorKind.HUMAN)
        self.assertNotEqual(new_generation, old_generation)

        inject(self.coordinator, make_packet(9), DetectorKind)
        self.assertTrue(
            wait_until(lambda: self._all_results_at(9), timeout=3.0),
            "pipeline did not recover after disable/re-enable",
        )
        result = self.coordinator.get_latest_result(DetectorKind.HUMAN)
        self.assertIsNotNone(result)
        self.assertEqual(result.generation, new_generation)
        self.assertEqual(result.source_seq, 9)

    def test_queue_bounds_are_enforced(self) -> None:
        for kind in DetectorKind:
            self.assertEqual(
                self.coordinator.get_packet_queue(kind).maxsize, QUEUE_MAX_SIZE
            )
        self.assertLessEqual(
            len(list(self.coordinator._frame_history)), HISTORY_SIZE
        )

    def test_shutdown_stops_every_thread(self) -> None:
        # Mimic MainWindow.close(): coordinator first, then workers.
        self.coordinator.stop()
        for worker in self.workers.values():
            worker.stop()
        for worker in self.workers.values():
            worker.join(timeout=2.0)
            self.assertFalse(worker.is_alive(), f"{worker.name} leaked after shutdown")


# ---------------------------------------------------------------------------
# Central rendering
# ---------------------------------------------------------------------------


def make_hand_candidate(bbox=(10, 10, 50, 50), wrist=(30.0, 30.0)):
    image_points = np.zeros((21, 3), dtype=np.float32)
    image_points[:, 0] = 0.5
    image_points[:, 1] = 0.5
    image_points[:, 2] = 0.0
    return SimpleNamespace(
        image_points=image_points,
        world_points=np.zeros((21, 3), dtype=np.float32),
        landmark_signature=np.ones(30, dtype=np.float32),
        bbox=bbox,
        wrist=np.asarray(wrist, dtype=np.float32),
        handedness="RIGHT",
        handedness_score=0.9,
        quality=0.9,
    )


def make_hand_track(track_id=1):
    return SimpleNamespace(
        track_id=track_id,
        bbox=(10, 10, 50, 50),
        gesture_history=["OPEN HAND"],
        stable_gesture="OPEN HAND",
        person_id=None,
        associated_side=None,
        handedness="RIGHT",
        track_id_confident=True,
    )


class TestCentralRendering(unittest.TestCase):
    def test_face_overlay_returns_new_frame(self) -> None:
        frame = make_frame(5)
        faces = [SimpleNamespace(bbox=(10, 10, 20, 20), label="UNKNOWN")]
        annotated = mw._draw_face_overlays(frame, faces)
        self.assertIsNot(annotated, frame)
        self.assertTrue(np.array_equal(frame, np.full_like(frame, 5)))
        self.assertFalse(np.array_equal(annotated, frame))

    def test_hand_overlay_supports_refresh_call(self) -> None:
        """_refresh calls _draw_hand_overlays(frame, candidates, tracks)."""
        frame = make_frame(0)
        annotated = mw._draw_hand_overlays(
            frame, [make_hand_candidate()], [make_hand_track()]
        )
        self.assertIsNot(annotated, frame)
        self.assertFalse(np.array_equal(annotated, frame))

    def test_human_overlay_draws_pose_data(self) -> None:
        self.assertTrue(
            hasattr(mw, "_draw_human_overlays"),
            "no human pose renderer exists; human results are never drawn",
        )
        frame = make_frame(0)
        points = tuple((0.5, 0.5) for _ in range(33))
        visible = tuple(True for _ in range(33))
        annotated = mw._draw_human_overlays(frame, ((points, visible),))
        self.assertIsNot(annotated, frame)
        self.assertFalse(np.array_equal(annotated, frame))

    def test_refresh_isolates_detector_draw_errors(self) -> None:
        source = inspect.getsource(mw.MainWindow._refresh)
        self.assertIn(
            "except Exception",
            source,
            "no per-detector draw isolation in _refresh; one overlay error aborts the preview",
        )


# ---------------------------------------------------------------------------
# MainWindow layout smoke test (Tk)
# ---------------------------------------------------------------------------


class TestMainWindowSmoke(unittest.TestCase):
    def test_layout_and_controls_intact(self) -> None:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        window = None
        try:
            window = mw.MainWindow(root)
            window.pack(fill="both", expand=True)
            for name in (
                "toggle_button",
                "start_detection_button",
                "stop_detection_button",
                "start_face_button",
                "stop_face_button",
                "start_hand_button",
                "stop_hand_button",
                "preview_label",
                "detection_info",
                "face_info",
                "hand_info",
            ):
                self.assertTrue(hasattr(window, name), f"missing widget: {name}")
            self.assertIsInstance(window.coordinator, FrameCoordinator)
        finally:
            if window is not None:
                try:
                    window.close()
                except Exception:
                    pass
            try:
                root.destroy()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Hand re-identification across a short absence (live bug report)
# ---------------------------------------------------------------------------


class TestHandTrackReidentification(unittest.TestCase):
    """A hand that leaves the view briefly must keep the same track ID."""

    @staticmethod
    def _candidate():
        from app.hand_gesture_advanced import _HandCandidate

        image_points = np.zeros((21, 3), dtype=np.float32)
        image_points[:, 0] = 0.5
        image_points[:, 1] = 0.5
        return _HandCandidate(
            image_points=image_points,
            world_points=np.ones((21, 3), dtype=np.float32),
            landmark_signature=np.ones(30, dtype=np.float32),
            bbox=(100, 100, 200, 200),
            wrist=np.asarray([150.0, 150.0], dtype=np.float32),
            handedness="RIGHT",
            handedness_score=0.9,
            quality=0.9,
        )

    def _run_sequence(self, absent_frames: int):
        from app.hand_gesture_advanced import _update_hand_tracks

        tracks = []
        lost_tracks = []
        next_id = 1
        now = 100.0
        for _ in range(10):
            now += 0.05
            tracks, next_id, _ = _update_hand_tracks(
                tracks, [self._candidate()], [], next_id,
                now=now, frame_width=640, frame_height=480,
                lost_tracks=lost_tracks,
            )
        for _ in range(absent_frames):
            now += 0.05
            tracks, next_id, _ = _update_hand_tracks(
                tracks, [], [], next_id,
                now=now, frame_width=640, frame_height=480,
                lost_tracks=lost_tracks,
            )
        now += 0.05
        tracks, next_id, visible = _update_hand_tracks(
            tracks, [self._candidate()], [], next_id,
            now=now, frame_width=640, frame_height=480,
            lost_tracks=lost_tracks,
        )
        return [track.track_id for track in visible]

    def test_short_absence_keeps_track_id(self) -> None:
        self.assertEqual(self._run_sequence(absent_frames=10), [1])

    def test_three_second_absence_keeps_track_id(self) -> None:
        # 60 frames at 20 FPS == 3.0 s; the reported bug assigns HAND 2 here.
        self.assertEqual(self._run_sequence(absent_frames=60), [1])


# ---------------------------------------------------------------------------
# Real workers through the coordinator (slow: loads real models)
# ---------------------------------------------------------------------------


class TestRealWorkerPublishPath(unittest.TestCase):
    def setUp(self) -> None:
        self.camera = FakeCamera()
        self.coordinator = FrameCoordinator(camera_source=self.camera)
        self.packet = make_packet(1)

    def tearDown(self) -> None:
        self.coordinator.stop()

    def test_hand_worker_publishes_packet_timestamp(self) -> None:
        from app.hand_gesture_advanced import AdvancedHandDetectionWorker

        self.coordinator.enable(DetectorKind.HAND)
        worker = AdvancedHandDetectionWorker(
            None, coordinator=self.coordinator, kind=DetectorKind.HAND
        )
        self.addCleanup(worker.shutdown)
        worker.set_enabled(True)
        self.assertTrue(
            wait_until(lambda: worker.snapshot().state in ("on", "error"), timeout=30.0),
            "hand worker did not initialize",
        )
        self.assertEqual(
            worker.snapshot().state, "on", f"hand worker init failed: {worker.snapshot().message}"
        )
        self.coordinator.get_packet_queue(DetectorKind.HAND).put_nowait(self.packet)
        self.assertTrue(
            wait_until(
                lambda: self.coordinator.get_latest_result(DetectorKind.HAND) is not None,
                timeout=15.0,
            ),
            f"hand worker published no result; state={worker.snapshot().state} "
            f"message={worker.snapshot().message}",
        )
        result = self.coordinator.get_latest_result(DetectorKind.HAND)
        self.assertEqual(result.source_seq, 1)
        self.assertAlmostEqual(result.source_ts, self.packet.timestamp)

    def test_face_worker_publishes_packet_timestamp(self) -> None:
        from app.face_recognition_worker import FaceRecognitionWorker

        self.coordinator.enable(DetectorKind.FACE)
        worker = FaceRecognitionWorker(
            None, coordinator=self.coordinator, kind=DetectorKind.FACE
        )
        self.addCleanup(worker.shutdown)
        worker.start()
        self.assertTrue(
            wait_until(lambda: worker.get_snapshot().state in ("running", "error"), timeout=30.0),
            "face worker did not initialize",
        )
        self.assertEqual(
            worker.get_snapshot().state,
            "running",
            f"face worker init failed: {worker.get_snapshot().message}",
        )
        self.coordinator.get_packet_queue(DetectorKind.FACE).put_nowait(self.packet)
        self.assertTrue(
            wait_until(
                lambda: self.coordinator.get_latest_result(DetectorKind.FACE) is not None,
                timeout=15.0,
            ),
            f"face worker published no result; state={worker.get_snapshot().state} "
            f"message={worker.get_snapshot().message}",
        )
        result = self.coordinator.get_latest_result(DetectorKind.FACE)
        self.assertEqual(result.source_seq, 1)
        self.assertAlmostEqual(result.source_ts, self.packet.timestamp)

    def test_human_worker_publishes_packet_timestamp(self) -> None:
        from app.human_detection import HumanDetectionWorker

        self.coordinator.enable(DetectorKind.HUMAN)
        worker = HumanDetectionWorker(
            None, coordinator=self.coordinator, kind=DetectorKind.HUMAN
        )
        self.addCleanup(worker.shutdown)
        worker.start()
        self.assertTrue(
            wait_until(lambda: worker.get_status().state in ("on", "error"), timeout=30.0),
            "human worker did not initialize",
        )
        self.assertEqual(
            worker.get_status().state, "on", f"human worker init failed: {worker.get_status().message}"
        )
        self.coordinator.get_packet_queue(DetectorKind.HUMAN).put_nowait(self.packet)
        self.assertTrue(
            wait_until(
                lambda: self.coordinator.get_latest_result(DetectorKind.HUMAN) is not None,
                timeout=15.0,
            ),
            f"human worker published no result; state={worker.get_status().state} "
            f"message={worker.get_status().message}",
        )
        result = self.coordinator.get_latest_result(DetectorKind.HUMAN)
        self.assertEqual(result.source_seq, 1)
        self.assertAlmostEqual(result.source_ts, self.packet.timestamp)

    def test_hand_worker_legacy_provider_mode_annotates_frame(self) -> None:
        """Standalone (no coordinator) hand worker must still operate."""
        from app.hand_gesture_advanced import AdvancedHandDetectionWorker

        frame = make_frame(0)
        worker = AdvancedHandDetectionWorker(lambda: frame.copy())
        self.addCleanup(worker.shutdown)
        worker.set_enabled(True)
        self.assertTrue(
            wait_until(lambda: worker.snapshot().state == "on", timeout=30.0),
            f"legacy hand worker did not initialize: {worker.snapshot().message}",
        )
        self.assertTrue(
            wait_until(lambda: worker.snapshot().frame is not None, timeout=10.0),
            "legacy hand worker produced no annotated frame",
        )
        self.assertEqual(worker.snapshot().frame.shape, frame.shape)


if __name__ == "__main__":
    unittest.main()
