"""Hand identity persistence tests for Gesture.AI.

These tests exercise the bounded lost-hand identity cache in
``app/hand_gesture_advanced.py`` using synthetic landmarks only.  No webcam
and no MediaPipe model execution is required.

Run with:
    python -m unittest tests.test_hand_identity_persistence -v
"""

from __future__ import annotations

import inspect
import unittest

import numpy as np

from app.hand_gesture_advanced import (
    HAND_LOST_TRACK_GRACE_SECONDS,
    HAND_LOST_TRACK_MAX,
    AdvancedHandDetectionWorker,
    _HandCandidate,
    _HandTrack,
    _LostHandIdentity,
    _snapshot_lost_track,
    _update_camera_session_state,
    _update_hand_tracks,
)

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_INTERVAL = 0.05  # 20 FPS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_candidate(
    center: tuple[int, int] = (150, 150),
    signature: float = 1.0,
    handedness: str = "RIGHT",
    handedness_score: float = 0.9,
    person_id: int | None = None,
    gesture: str = "OPEN HAND",
) -> _HandCandidate:
    half = 50
    bbox = (
        center[0] - half,
        center[1] - half,
        center[0] + half,
        center[1] + half,
    )
    image_points = np.zeros((21, 3), dtype=np.float32)
    image_points[:, 0] = center[0] / FRAME_WIDTH
    image_points[:, 1] = center[1] / FRAME_HEIGHT
    candidate = _HandCandidate(
        image_points=image_points,
        world_points=np.ones((21, 3), dtype=np.float32),
        landmark_signature=np.full(30, signature, dtype=np.float32),
        bbox=bbox,
        wrist=np.asarray(center, dtype=np.float32),
        handedness=handedness,
        handedness_score=handedness_score,
        quality=0.9,
    )
    candidate.person_id = person_id
    candidate.gesture = gesture
    return candidate


def make_track(
    track_id: int,
    center: tuple[int, int] = (150, 150),
    signature: float = 1.0,
    last_seen_at: float = 0.0,
    handedness: str = "RIGHT",
    person_id: int | None = None,
    stable_gesture: str = "OPEN HAND",
) -> _HandTrack:
    half = 50
    bbox = (
        center[0] - half,
        center[1] - half,
        center[0] + half,
        center[1] + half,
    )
    return _HandTrack(
        track_id=track_id,
        bbox=bbox,
        wrist=np.asarray(center, dtype=np.float32),
        velocity=np.zeros(2, dtype=np.float32),
        landmark_signature=np.full(30, signature, dtype=np.float32),
        handedness=handedness,
        handedness_score=0.9,
        gesture_history=["OPEN HAND"] * 3,
        stable_gesture=stable_gesture,
        person_id=person_id,
        misses=0,
        last_seen_at=last_seen_at,
        handedness_history=[("RIGHT", 0.9)] * 3,
        last_confident_person_id=person_id,
    )


class TrackerHarness:
    """Drives ``_update_hand_tracks`` with a persistent lost-identity cache."""

    def __init__(self) -> None:
        self.tracks: list[_HandTrack] = []
        self.lost: list[_LostHandIdentity] = []
        self.next_id = 1
        self.now = 100.0

    def step(self, candidates, people=()):
        self.tracks, self.next_id, visible = _update_hand_tracks(
            self.tracks,
            list(candidates),
            list(people),
            self.next_id,
            now=self.now,
            frame_width=FRAME_WIDTH,
            frame_height=FRAME_HEIGHT,
            lost_tracks=self.lost,
        )
        return visible

    def run_frames(self, count: int, candidates_factory=None):
        visible = []
        for _ in range(count):
            self.now += FRAME_INTERVAL
            candidates = [] if candidates_factory is None else candidates_factory()
            visible = self.step(candidates)
        return visible


# ---------------------------------------------------------------------------
# Repeated disappearance / re-entry
# ---------------------------------------------------------------------------


class TestRepeatedReentry(unittest.TestCase):
    def test_repeated_disappearance_and_reentry_keeps_id(self) -> None:
        tracker = TrackerHarness()
        visible = tracker.run_frames(10, lambda: [make_candidate()])
        self.assertEqual([track.track_id for track in visible], [1])

        for cycle in range(3):
            tracker.run_frames(60, lambda: [])  # 3.0 s absence
            visible = tracker.run_frames(1, lambda: [make_candidate()])
            self.assertEqual(
                [track.track_id for track in visible],
                [1],
                f"identity changed on re-entry cycle {cycle}",
            )
        self.assertEqual(tracker.next_id, 2, "restored identities must not consume new IDs")

    def test_short_interruption_uses_active_track(self) -> None:
        tracker = TrackerHarness()
        tracker.run_frames(10, lambda: [make_candidate()])
        tracker.run_frames(10, lambda: [])  # 0.5 s: inside 2 s active grace
        visible = tracker.run_frames(1, lambda: [make_candidate()])
        self.assertEqual([track.track_id for track in visible], [1])

    def test_restored_track_retains_gesture_and_handedness(self) -> None:
        tracker = TrackerHarness()
        tracker.run_frames(10, lambda: [make_candidate(gesture="OPEN HAND")])
        self.assertEqual(tracker.tracks[0].stable_gesture, "OPEN HAND")
        tracker.run_frames(60, lambda: [])
        visible = tracker.run_frames(1, lambda: [make_candidate(gesture="OPEN HAND")])
        restored = visible[0]
        self.assertEqual(restored.track_id, 1)
        self.assertEqual(restored.stable_gesture, "OPEN HAND")
        self.assertEqual(restored.handedness, "RIGHT")


# ---------------------------------------------------------------------------
# Multiple hands
# ---------------------------------------------------------------------------


class TestMultipleHands(unittest.TestCase):
    def _left(self):
        return make_candidate((150, 150), signature=1.0, handedness="RIGHT")

    def _right(self):
        return make_candidate((450, 150), signature=1.6, handedness="LEFT")

    def test_two_simultaneous_hands_keep_distinct_ids(self) -> None:
        tracker = TrackerHarness()
        visible = tracker.run_frames(10, lambda: [self._left(), self._right()])
        self.assertEqual({track.track_id for track in visible}, {1, 2})

        # Left hand disappears for 3 s while the right hand stays visible.
        for _ in range(60):
            tracker.now += FRAME_INTERVAL
            visible = tracker.step([self._right()])
            self.assertEqual([track.track_id for track in visible], [2])

        visible = tracker.run_frames(1, lambda: [self._left(), self._right()])
        ids = {track.track_id for track in visible}
        self.assertEqual(ids, {1, 2}, f"identity collision or swap: {ids}")

    def test_two_simultaneous_hands_never_share_an_id(self) -> None:
        tracker = TrackerHarness()
        for _ in range(90):
            tracker.now += FRAME_INTERVAL
            visible = tracker.step([self._left(), self._right()])
            ids = [track.track_id for track in visible]
            self.assertEqual(len(ids), 2)
            self.assertEqual(len(set(ids)), 2, f"duplicate Hand ID: {ids}")

    def test_ambiguous_match_assigns_new_id(self) -> None:
        tracker = TrackerHarness()
        # Two expired identities with identical appearance/position: any match
        # is inherently ambiguous, so a new ID must be assigned.
        tracker.tracks = [
            make_track(1, last_seen_at=tracker.now - 3.0),
            make_track(2, last_seen_at=tracker.now - 3.0),
        ]
        tracker.next_id = 3
        visible = tracker.step([make_candidate()])
        self.assertEqual([track.track_id for track in visible], [3])
        self.assertEqual(len({entry.track_id for entry in tracker.lost}), 2)


# ---------------------------------------------------------------------------
# New hands and uncertainty
# ---------------------------------------------------------------------------


class TestNewHandAssignment(unittest.TestCase):
    def test_genuinely_new_hand_gets_new_id(self) -> None:
        tracker = TrackerHarness()
        tracker.run_frames(10, lambda: [make_candidate((150, 150), signature=1.0)])
        tracker.run_frames(60, lambda: [])
        # Different bone signature (distance 0.6 > 0.35) and different position.
        visible = tracker.run_frames(
            1, lambda: [make_candidate((300, 150), signature=1.6)]
        )
        self.assertEqual([track.track_id for track in visible], [2])
        self.assertEqual(tracker.next_id, 3)

    def test_different_person_never_restores_previous_id(self) -> None:
        tracker = TrackerHarness()
        # Seed an expired identity that was confidently associated with person 7.
        tracker.tracks = [
            make_track(1, last_seen_at=tracker.now - 3.0, person_id=7)
        ]
        tracker.next_id = 2
        # Same geometry but a confidently different person: hard reject.
        visible = tracker.step([make_candidate(person_id=8)])
        self.assertEqual([track.track_id for track in visible], [2])

    def test_matching_person_is_an_independent_cue(self) -> None:
        tracker = TrackerHarness()
        tracker.tracks = [
            make_track(1, last_seen_at=tracker.now - 3.0, person_id=7)
        ]
        tracker.next_id = 2
        visible = tracker.step([make_candidate(person_id=7)])
        self.assertEqual([track.track_id for track in visible], [1])

    def test_signature_alone_is_not_enough(self) -> None:
        tracker = TrackerHarness()
        # Confirmed handedness and a close signature, but far away position.
        tracker.run_frames(
            10, lambda: [make_candidate((150, 150), signature=1.0, handedness="RIGHT")]
        )
        tracker.run_frames(60, lambda: [])
        visible = tracker.run_frames(
            1, lambda: [make_candidate((590, 150), signature=1.0, handedness="LEFT")]
        )
        self.assertEqual([track.track_id for track in visible], [2])


# ---------------------------------------------------------------------------
# Timeout and bounded memory
# ---------------------------------------------------------------------------


class TestExpiryAndBounds(unittest.TestCase):
    def test_tracking_timeout_expires_lost_identity(self) -> None:
        tracker = TrackerHarness()
        tracker.run_frames(10, lambda: [make_candidate()])
        tracker.run_frames(180, lambda: [])  # 9.0 s > 8 s TTL
        self.assertEqual(tracker.lost, [])
        visible = tracker.run_frames(1, lambda: [make_candidate()])
        self.assertEqual([track.track_id for track in visible], [2])

    def test_lost_cache_is_bounded(self) -> None:
        tracker = TrackerHarness()
        tracker.tracks = [
            make_track(track_id, last_seen_at=tracker.now - 3.0)
            for track_id in range(1, 11)
        ]
        tracker.next_id = 11
        tracker.step([])
        self.assertEqual(len(tracker.lost), HAND_LOST_TRACK_MAX)
        self.assertLessEqual(len(tracker.lost), HAND_LOST_TRACK_MAX)
        self.assertEqual(len({entry.track_id for entry in tracker.lost}), len(tracker.lost))

    def test_lost_snapshot_copies_mutable_state(self) -> None:
        track = make_track(1, last_seen_at=100.0)
        snapshot = _snapshot_lost_track(track)
        track.wrist[0] = 999.0
        track.gesture_history.append("CLOSED FIST")
        self.assertNotEqual(float(snapshot.wrist[0]), 999.0)
        self.assertNotIn("CLOSED FIST", snapshot.gesture_history)


# ---------------------------------------------------------------------------
# Camera interruption versus genuine camera-session reset
# ---------------------------------------------------------------------------


class TestCameraSessionReset(unittest.TestCase):
    def _lost_cache(self) -> TrackerHarness:
        tracker = TrackerHarness()
        tracker.run_frames(10, lambda: [make_candidate()])
        tracker.run_frames(60, lambda: [])  # 3 s absence -> cached identity
        self.assertEqual(len(tracker.lost), 1)
        return tracker

    def test_temporary_interruption_preserves_and_restores_identity(self) -> None:
        tracker = self._lost_cache()
        tracker.now += 2.0  # temporary interruption: no frames, well under 8 s TTL
        sequence: int | None = 100
        sequence = _update_camera_session_state(tracker.lost, sequence, 150)
        self.assertEqual(len(tracker.lost), 1)
        visible = tracker.run_frames(1, lambda: [make_candidate()])
        self.assertEqual([track.track_id for track in visible], [1])

    def test_genuine_camera_reset_clears_lost_identities(self) -> None:
        tracker = self._lost_cache()
        # A restarted CameraCapture session publishes seq == 1 again.
        _update_camera_session_state(tracker.lost, 100, 1)
        self.assertEqual(tracker.lost, [])
        visible = tracker.run_frames(1, lambda: [make_candidate()])
        self.assertEqual([track.track_id for track in visible], [2])

    def test_duplicate_sequence_is_treated_as_reset(self) -> None:
        tracker = self._lost_cache()
        _update_camera_session_state(tracker.lost, 50, 50)
        self.assertEqual(tracker.lost, [])

    def test_invalid_sequence_is_ignored(self) -> None:
        tracker = self._lost_cache()
        sequence = _update_camera_session_state(tracker.lost, 42, 0)
        self.assertEqual(sequence, 42)
        self.assertEqual(len(tracker.lost), 1)

    def test_worker_restart_creates_empty_cache(self) -> None:
        source = inspect.getsource(AdvancedHandDetectionWorker._run)
        self.assertIn(
            "lost_hand_tracks: list[_LostHandIdentity] = []",
            source,
            "worker _run must start with an empty lost-identity cache",
        )


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility(unittest.TestCase):
    def test_old_three_tuple_api_still_works_without_cache(self) -> None:
        tracks: list[_HandTrack] = []
        next_id = 1
        now = 100.0
        for _ in range(10):
            now += FRAME_INTERVAL
            tracks, next_id, visible = _update_hand_tracks(
                tracks,
                [make_candidate()],
                [],
                next_id,
                now=now,
                frame_width=FRAME_WIDTH,
                frame_height=FRAME_HEIGHT,
            )
        self.assertEqual([track.track_id for track in visible], [1])

        for _ in range(10):  # 0.5 s: still inside active grace
            now += FRAME_INTERVAL
            tracks, next_id, visible = _update_hand_tracks(
                tracks,
                [],
                [],
                next_id,
                now=now,
                frame_width=FRAME_WIDTH,
                frame_height=FRAME_HEIGHT,
            )
        now += FRAME_INTERVAL
        tracks, next_id, visible = _update_hand_tracks(
            tracks,
            [make_candidate()],
            [],
            next_id,
            now=now,
            frame_width=FRAME_WIDTH,
            frame_height=FRAME_HEIGHT,
        )
        self.assertEqual([track.track_id for track in visible], [1])

    def test_long_absence_without_cache_keeps_legacy_new_id_behavior(self) -> None:
        tracks: list[_HandTrack] = []
        next_id = 1
        now = 100.0
        for _ in range(10):
            now += FRAME_INTERVAL
            tracks, next_id, _ = _update_hand_tracks(
                tracks,
                [make_candidate()],
                [],
                next_id,
                now=now,
                frame_width=FRAME_WIDTH,
                frame_height=FRAME_HEIGHT,
            )
        for _ in range(60):
            now += FRAME_INTERVAL
            tracks, next_id, _ = _update_hand_tracks(
                tracks,
                [],
                [],
                next_id,
                now=now,
                frame_width=FRAME_WIDTH,
                frame_height=FRAME_HEIGHT,
            )
        now += FRAME_INTERVAL
        tracks, next_id, visible = _update_hand_tracks(
            tracks,
            [make_candidate()],
            [],
            next_id,
            now=now,
            frame_width=FRAME_WIDTH,
            frame_height=FRAME_HEIGHT,
        )
        self.assertEqual([track.track_id for track in visible], [2])
        self.assertEqual(next_id, 3)


if __name__ == "__main__":
    unittest.main()
