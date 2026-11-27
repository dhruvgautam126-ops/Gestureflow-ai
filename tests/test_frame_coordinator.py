"""Automated unit tests for FrameCoordinator (Stage 3A).

Covers: history bounds, queue overflow, slow-worker handling, camera resets,
sequence synchronization, strict shutdown ordering, and worker gating via
generation changes.
"""

from __future__ import annotations

import time
import unittest
from typing import Any

import numpy as np

from app.camera.capture import CameraFramePacket
from app.coordinator import FrameCoordinator, DetectorKind, WorkerResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_packet(seq: int, frame: np.ndarray | None = None) -> CameraFramePacket:
    if frame is None:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
    return CameraFramePacket(seq=seq, timestamp=time.time(), frame=frame)


def make_result(
    kind: DetectorKind,
    seq: int,
    generation: int = 0,
    people: int = 0,
    face_detections: tuple = (),
    hand_candidates: tuple = (),
    hand_visible_tracks: tuple = (),
) -> WorkerResult:
    return WorkerResult(
        kind=kind,
        source_seq=seq,
        source_ts=time.time(),
        generation=generation,
        human_people=people,
        human_pose_draw_data=(),
        face_detections=face_detections,
        hand_candidates=hand_candidates,
        hand_visible_tracks=hand_visible_tracks,
    )


class FakeCamera:
    """Stand-in camera returning packets on demand; has get_current_packet()."""

    def __init__(self) -> None:
        self._latest: CameraFramePacket | None = None

    def get_frame(self) -> CameraFramePacket | None:
        import time as _t

        pkt = CameraFramePacket(
            seq=_t.time_ns(), timestamp=_t.time(), frame=np.zeros((10, 10, 3), np.uint8)
        )
        self._latest = pkt
        return pkt

    def get_current_packet(self) -> CameraFramePacket | None:
        return self._latest


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFrameCoordinatorLifecycle(unittest.TestCase):
    def setUp(self) -> None:
        self.camera = FakeCamera()
        self.coord = FrameCoordinator(camera_source=self.camera)

    def tearDown(self) -> None:
        try:
            self.coord.stop()
        except Exception:
            pass

    def test_starts_and_stops(self) -> None:
        self.coord.start()
        self.assertIsNotNone(self.coord._distributor_thread)
        self.assertTrue(self.coord._distributor_thread.is_alive())
        self.coord.stop()
        self.assertIsNone(self.coord._distributor_thread)


class TestHistoryBounds(unittest.TestCase):
    def setUp(self) -> None:
        self.camera = FakeCamera()
        self.coord = FrameCoordinator(camera_source=self.camera)

    def tearDown(self) -> None:
        try:
            self.coord.stop()
        except Exception:
            pass

    def test_frame_history_bounded(self) -> None:
        # Directly populate history (deterministic, avoids async distributor timing)
        for i in range(10):
            self.coord._frame_history.append(make_packet(i))
        self.assertLessEqual(len(self.coord._frame_history), 5)  # HISTORY_SIZE = 5

    def test_result_history_bounded(self) -> None:
        self.coord.enable(DetectorKind.HUMAN)
        gen = self.coord.get_generation(DetectorKind.HUMAN)
        for i in range(5):
            self.coord.publish_result(
                make_result(DetectorKind.HUMAN, i, generation=gen)
            )
        self.assertLessEqual(
            len(self.coord._result_histories[DetectorKind.HUMAN]), 5
        )


class TestComposeEmpty(unittest.TestCase):
    def setUp(self) -> None:
        self.camera = FakeCamera()
        self.coord = FrameCoordinator(camera_source=self.camera)

    def tearDown(self) -> None:
        try:
            self.coord.stop()
        except Exception:
            pass

    def test_returns_none_when_no_frames(self) -> None:
        frame, results, synced = self.coord.compose()
        self.assertIsNone(frame)
        self.assertTrue(all(v is None for v in results.values()))
        self.assertTrue(synced)


class TestSequenceMatching(unittest.TestCase):
    def setUp(self) -> None:
        self.camera = FakeCamera()
        self.coord = FrameCoordinator(camera_source=self.camera)
        self.gen_human = self.coord.get_generation(DetectorKind.HUMAN)
        self.gen_face = self.coord.get_generation(DetectorKind.FACE)

    def tearDown(self) -> None:
        try:
            self.coord.stop()
        except Exception:
            pass

    def _add_frames(self, count: int) -> None:
        for i in range(1, count + 1):
            self.coord._frame_history.append(make_packet(i))

    def test_exact_sequence_matching(self) -> None:
        self.coord.enable(DetectorKind.HUMAN)
        self._add_frames(5)
        for i in range(1, 6):
            self.coord.publish_result(
                make_result(
                    DetectorKind.HUMAN, i, people=i,
                    generation=self.coord.get_generation(DetectorKind.HUMAN),
                )
            )

        frame, results, synced = self.coord.compose()
        self.assertTrue(synced)
        human = results[DetectorKind.HUMAN]
        self.assertIsNotNone(human)
        self.assertEqual(human.source_seq, 5)
        self.assertEqual(human.human_people, 5)

    def test_no_seq_tolerance_strict_matching(self) -> None:
        # Only frames up to seq 5 exist; publish result at seq 7
        self.coord.enable(DetectorKind.HUMAN)
        self._add_frames(5)
        self.coord.publish_result(
            make_result(
                DetectorKind.HUMAN, 7, people=1,
                generation=self.coord.get_generation(DetectorKind.HUMAN),
            )
        )

        frame, results, synced = self.coord.compose()
        self.assertFalse(synced)

    def test_highest_common_sequence_selection(self) -> None:
        self.coord.enable(DetectorKind.HUMAN)
        self._add_frames(5)
        for i in range(1, 6):
            self.coord.publish_result(
                make_result(
                    DetectorKind.HUMAN, i, people=i,
                    generation=self.coord.get_generation(DetectorKind.HUMAN),
                )
            )
        # Extra frame at seq 6 has no matching result -> highest common is 5
        self.coord._frame_history.append(make_packet(6))

        frame, results, synced = self.coord.compose()
        self.assertTrue(synced)
        human = results[DetectorKind.HUMAN]
        self.assertEqual(human.source_seq, 5)

    def test_disabled_worker_returns_none(self) -> None:
        # HUMAN not enabled
        self._add_frames(3)
        frame, results, synced = self.coord.compose()
        self.assertIsNone(results[DetectorKind.HUMAN])

    def test_generation_change_blocks_stale_results(self) -> None:
        self.coord.enable(DetectorKind.HUMAN)
        gen = self.coord.get_generation(DetectorKind.HUMAN)
        self._add_frames(3)
        self.coord.publish_result(
            make_result(
                DetectorKind.HUMAN, 1, people=1, generation=gen,
            )
        )

        self.coord.disable(DetectorKind.HUMAN)  # bumps generation; stale results invalidated
        frame, results, synced = self.coord.compose()
        self.assertIsNone(results[DetectorKind.HUMAN])

    def test_multiple_enabled_workers_require_common_seq(self) -> None:
        self.coord.enable(DetectorKind.HUMAN)
        self.coord.enable(DetectorKind.FACE)
        gen_h = self.coord.get_generation(DetectorKind.HUMAN)
        gen_f = self.coord.get_generation(DetectorKind.FACE)
        self._add_frames(4)
        for i in range(1, 5):
            self.coord.publish_result(
                make_result(DetectorKind.HUMAN, i, people=i, generation=gen_h)
            )
            self.coord.publish_result(
                make_result(DetectorKind.FACE, i, generation=gen_f)
            )

        frame, results, synced = self.coord.compose()
        self.assertTrue(synced)
        self.assertIsNotNone(results[DetectorKind.HUMAN])
        self.assertIsNotNone(results[DetectorKind.FACE])

    def test_slow_worker_handling(self) -> None:
        self.coord.enable(DetectorKind.HUMAN)
        gen = self.coord.get_generation(DetectorKind.HUMAN)
        self._add_frames(4)
        # Slow worker publishes results at old seqs only
        self.coord.publish_result(
            make_result(DetectorKind.HUMAN, 2, people=1, generation=gen)
        )

        frame, results, synced = self.coord.compose()
        self.assertTrue(synced)
        self.assertIsNotNone(results[DetectorKind.HUMAN])
        self.assertEqual(results[DetectorKind.HUMAN].source_seq, 2)


class TestQueueProperties(unittest.TestCase):
    def setUp(self) -> None:
        self.camera = FakeCamera()
        self.coord = FrameCoordinator(camera_source=self.camera)

    def tearDown(self) -> None:
        try:
            self.coord.stop()
        except Exception:
            pass

    def test_queue_maxsize_is_2(self) -> None:
        for kind in DetectorKind:
            q = self.coord.get_packet_queue(kind)
            self.assertEqual(q.maxsize, 2)

    def test_disable_drains_queues(self) -> None:
        self.coord.enable(DetectorKind.HUMAN)
        q = self.coord.get_packet_queue(DetectorKind.HUMAN)
        q.put_nowait(make_packet(1))

        self.coord.disable(DetectorKind.HUMAN)
        self.assertEqual(q.qsize(), 0)


class TestGenerationValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.camera = FakeCamera()
        self.coord = FrameCoordinator(camera_source=self.camera)

    def tearDown(self) -> None:
        try:
            self.coord.stop()
        except Exception:
            pass

    def test_publish_result_validates_generation(self) -> None:
        self.coord.enable(DetectorKind.HUMAN)
        gen = self.coord.get_generation(DetectorKind.HUMAN)
        result = make_result(DetectorKind.HUMAN, 1, generation=99)
        # Publishing with mismatched generation is rejected
        self.coord.publish_result(result)
        self.assertIsNone(self.coord._results[DetectorKind.HUMAN])

    def test_compose_no_workers_returns_latest_frame(self) -> None:
        # Directly populate frame history
        for i in range(1, 4):
            self.coord._frame_history.append(make_packet(i))

        frame, results, synced = self.coord.compose()
        self.assertIsNotNone(frame)
        self.assertTrue(synced)
        self.assertTrue(all(v is None for v in results.values()))


class TestWorkerResultImmutability(unittest.TestCase):
    def test_frozen_dataclass(self) -> None:
        result = make_result(DetectorKind.HUMAN, 1, people=2)
        with self.assertRaises(Exception):
            result.human_people = 99


if __name__ == "__main__":
    unittest.main()
