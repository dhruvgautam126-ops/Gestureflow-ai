from __future__ import annotations

import enum
import queue
import threading
import time
from dataclasses import dataclass, field
from collections import deque
from typing import Any

HISTORY_SIZE = 5
QUEUE_MAX_SIZE = 2


class DetectorKind(enum.Enum):
    HUMAN = "human"
    FACE = "face"
    HAND = "hand"


@dataclass(frozen=True)
class WorkerResult:
    kind: DetectorKind
    source_seq: int
    source_ts: float
    generation: int
    human_people: int | None = None
    human_pose_draw_data: tuple | None = None
    human_face_landmark_draw_data: tuple | None = None
    face_detections: tuple | None = None
    hand_candidates: tuple | None = None
    hand_visible_tracks: tuple | None = None


class FrameCoordinator:
    def __init__(self, camera_source):
        if hasattr(camera_source, "get_current_packet"):
            self._camera_getter = camera_source.get_current_packet
        else:
            self._camera_getter = camera_source
        self._running = False
        self._lock = threading.Lock()
        self._shutdown = threading.Event()
        self._queues: dict[DetectorKind, queue.Queue] = {
            kind: queue.Queue(maxsize=QUEUE_MAX_SIZE) for kind in DetectorKind
        }
        self._enabled: dict[DetectorKind, bool] = {kind: False for kind in DetectorKind}
        self._generations: dict[DetectorKind, int] = {kind: 0 for kind in DetectorKind}
        self._results: dict[DetectorKind, WorkerResult | None] = {kind: None for kind in DetectorKind}
        self._frame_history: deque = deque(maxlen=HISTORY_SIZE)
        self._result_histories: dict[DetectorKind, deque] = {
            kind: deque(maxlen=HISTORY_SIZE) for kind in DetectorKind
        }
        self._distributor_thread: threading.Thread | None = None

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._shutdown.clear()
            self._distributor_thread = threading.Thread(
                target=self._distribute,
                name="FrameCoordinator-Distributor",
                daemon=True,
            )
            self._distributor_thread.start()

    def stop(self) -> None:
        self._shutdown.set()
        with self._lock:
            self._running = False
        if self._distributor_thread and self._distributor_thread.is_alive():
            self._distributor_thread.join(timeout=2.0)
        self._distributor_thread = None

    def enable(self, kind: DetectorKind) -> int:
        with self._lock:
            self._enabled[kind] = True
            self._generations[kind] += 1
            self._results[kind] = None
            self._drain(self._queues[kind])
            return self._generations[kind]

    def disable(self, kind: DetectorKind) -> None:
        with self._lock:
            self._enabled[kind] = False
            self._generations[kind] += 1
            self._results[kind] = None
            self._drain(self._queues[kind])
            self._result_histories[kind].clear()

    def is_enabled(self, kind: DetectorKind) -> bool:
        with self._lock:
            return self._enabled[kind]

    def get_generation(self, kind: DetectorKind) -> int:
        with self._lock:
            return self._generations[kind]

    def get_packet_queue(self, kind: DetectorKind) -> queue.Queue:
        return self._queues[kind]

    def publish_result(self, result: WorkerResult) -> None:
        with self._lock:
            if not self._enabled[result.kind]:
                return
            if result.generation != self._generations[result.kind]:
                return
            self._results[result.kind] = result
            self._result_histories[result.kind].append(result)

    def compose(self, *, prefer_latest_frame: bool = False) -> tuple:
        """Return a camera frame plus the newest detector results.

        By default, the frame is selected from the same camera packet as the
        detector results. Live previews may request ``prefer_latest_frame`` so
        the video stays current while slower detectors process fewer frames.
        In that mode, ``synced`` is false when the preview frame and results
        came from different camera packets.
        """
        with self._lock:
            if not self._frame_history:
                return None, {k: None for k in DetectorKind}, True

            enabled_kinds = [k for k, v in self._enabled.items() if v]
            frame_seqs = {pkt.seq: pkt for pkt in self._frame_history}
            if not frame_seqs:
                return None, {k: None for k in DetectorKind}, True

            if not enabled_kinds:
                latest_pkt = self._frame_history[-1]
                return latest_pkt.frame.copy(), {k: None for k in DetectorKind}, True

            common_seqs = set(frame_seqs.keys())
            for kind in enabled_kinds:
                worker_seqs = {
                    res.source_seq
                    for res in self._result_histories[kind]
                    if res.generation == self._generations[kind]
                }
                common_seqs &= worker_seqs

            if not common_seqs:
                latest_pkt = self._frame_history[-1]
                res_dict = {
                    k: (self._results[k] if k in enabled_kinds else None)
                    for k in DetectorKind
                }
                return latest_pkt.frame.copy(), res_dict, False

            target_seq = max(common_seqs)
            target_pkt = frame_seqs[target_seq]

            res_dict = {}
            for kind in DetectorKind:
                if kind in enabled_kinds:
                    matching_res = next(
                        (
                            res
                            for res in self._result_histories[kind]
                            if res.source_seq == target_seq
                            and res.generation == self._generations[kind]
                        ),
                        None,
                    )
                    res_dict[kind] = matching_res
                else:
                    res_dict[kind] = None

            preview_pkt = (
                self._frame_history[-1]
                if prefer_latest_frame
                else target_pkt
            )
            return (
                preview_pkt.frame.copy(),
                res_dict,
                preview_pkt.seq == target_pkt.seq,
            )

    def get_latest_result(self, kind: DetectorKind) -> WorkerResult | None:
        with self._lock:
            return self._results[kind]

    def get_current_packet(self):
        return self._camera_getter()

    def _drain(self, q: queue.Queue) -> None:
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break

    def _distribute(self) -> None:
        last_seq = -1
        while not self._shutdown.is_set():
            packet = self._camera_getter()
            if packet is None or packet.seq == last_seq:
                self._shutdown.wait(0.005)
                continue
            last_seq = packet.seq
            # Keep one owned copy per capture packet and share it between the
            # preview history and read-only detector inputs. The workers create
            # their own transformed buffers before resizing/flipping frames.
            shared_frame = packet.frame.copy()
            packet_for_distribution = type(packet)(
                packet.seq, packet.timestamp, shared_frame
            )

            with self._lock:
                self._frame_history.append(packet_for_distribution)
                enabled_kinds = [k for k, v in self._enabled.items() if v]

            for kind in enabled_kinds:
                q = self._queues[kind]
                try:
                    q.put_nowait(packet_for_distribution)
                except queue.Full:
                    try:
                        q.get_nowait()
                        q.put_nowait(packet_for_distribution)
                    except (queue.Empty, queue.Full):
                        pass


def _drain(q: queue.Queue) -> None:
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            break
