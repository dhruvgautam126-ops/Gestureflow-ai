"""Optional Phase 3 inference stack; the current app UI does not import it yet.

Run from ``.phase3_env``. ONNX models use the DirectML adapter selected by
``device_id``; tracking and MediaPipe hand gestures remain on the CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading


PROJECT_ROOT = Path(__file__).resolve().parent.parent
YOLO26_MODEL = PROJECT_ROOT / "models" / "human" / "yolo26n.onnx"
RTMPOSE_MODEL = PROJECT_ROOT / "models" / "human" / "rtmpose_s.onnx"
INSIGHTFACE_ROOT = PROJECT_ROOT / "models" / "face" / "insightface"
DML_PROVIDER = "DmlExecutionProvider"


@dataclass(frozen=True)
class PersonPose:
    track_id: int
    box_xyxy: tuple[float, float, float, float]
    keypoints_xy: tuple[tuple[float, float], ...]
    keypoint_scores: tuple[float, ...]
    detection_score: float


@dataclass(frozen=True)
class FaceMatchInput:
    box_xyxy: tuple[float, float, float, float]
    detection_score: float
    embedding: tuple[float, ...]


@dataclass(frozen=True)
class Phase3FrameResult:
    people: tuple[PersonPose, ...]
    faces: tuple[FaceMatchInput, ...]


class _DetectionBatch:
    """Small Boxes-compatible object for Ultralytics' ByteTrack implementation."""

    def __init__(self, xywh, confidence, classes):
        self.xywh = xywh
        self.conf = confidence
        self.cls = classes

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, selection):
        return _DetectionBatch(
            self.xywh[selection], self.conf[selection], self.cls[selection]
        )


def _letterbox(frame, input_size: int = 640):
    """Resize and pad BGR input to the fixed square YOLO model size."""
    import cv2
    import numpy as np

    height, width = frame.shape[:2]
    scale = min(input_size / width, input_size / height)
    resized_width = round(width * scale)
    resized_height = round(height * scale)
    resized = cv2.resize(frame, (resized_width, resized_height))
    pad_x = (input_size - resized_width) / 2
    pad_y = (input_size - resized_height) / 2
    left, top = round(pad_x - 0.1), round(pad_y - 0.1)
    right, bottom = input_size - resized_width - left, input_size - resized_height - top
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    tensor = padded[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return tensor, scale, left, top


def _make_rtmpose_directml_ready(device_id: int):
    """Give RTMLib's session factory DirectML's required session settings."""
    import onnxruntime as ort
    from rtmlib.tools.base import RTMLIB_SETTINGS

    if not getattr(ort.InferenceSession, "_homeguard_dml_wrapper", False):
        original_session = ort.InferenceSession

        def inference_session(*args, **kwargs):
            providers = kwargs.get("providers") or ()
            uses_directml = any(
                provider == DML_PROVIDER
                or (isinstance(provider, tuple) and provider[0] == DML_PROVIDER)
                for provider in providers
            )
            if uses_directml and kwargs.get("sess_options") is None:
                options = ort.SessionOptions()
                options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                options.enable_mem_pattern = False
                kwargs["sess_options"] = options
            return original_session(*args, **kwargs)

        inference_session._homeguard_dml_wrapper = True
        ort.InferenceSession = inference_session

    RTMLIB_SETTINGS["onnxruntime"]["homeguard_directml"] = (
        DML_PROVIDER,
        {"device_id": int(device_id)},
    )


class Phase3VisionStack:
    """DirectML YOLO26 + ByteTrack + RTMPose + SCRFD/InsightFace pipeline.

    The adapter index defaults to 1 because that is the AMD Radeon R7 M260 on
    this computer. Other computers may need a different adapter index. Calls
    must be serialized because the person tracker and DirectML sessions belong
    to one video stream.
    """

    def __init__(
        self,
        *,
        max_people: int = 2,
        confidence: float = 0.35,
        device_id: int = 1,
    ) -> None:
        import onnxruntime as ort

        if max_people < 1:
            raise ValueError("max_people must be at least 1")
        required = (YOLO26_MODEL, RTMPOSE_MODEL)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("Missing Phase 3 model(s): " + ", ".join(missing))
        face_model_dir = INSIGHTFACE_ROOT / "models" / "buffalo_sc"
        if not face_model_dir.is_dir():
            raise FileNotFoundError(f"Missing InsightFace model folder: {face_model_dir}")
        if DML_PROVIDER not in ort.get_available_providers():
            raise RuntimeError(
                "DirectML is unavailable. Run this stack with the Phase 3 environment."
            )

        from insightface.app import FaceAnalysis
        from rtmlib import RTMPose
        from ultralytics.cfg import get_cfg
        from ultralytics.trackers.byte_tracker import BYTETracker
        import ultralytics

        self.max_people = min(int(max_people), 2)
        self.confidence = float(confidence)
        self.device_id = int(device_id)
        self._lock = threading.Lock()

        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.enable_mem_pattern = False
        providers = [(DML_PROVIDER, {"device_id": self.device_id}), "CPUExecutionProvider"]

        self._person_session = ort.InferenceSession(
            str(YOLO26_MODEL), sess_options=options, providers=providers
        )
        if DML_PROVIDER not in self._person_session.get_providers():
            raise RuntimeError("YOLO26 did not activate DirectML on the selected GPU.")
        self._person_input = self._person_session.get_inputs()[0].name

        _make_rtmpose_directml_ready(self.device_id)

        self._pose_model = RTMPose(
            str(RTMPOSE_MODEL),
            model_input_size=(192, 256),
            backend="onnxruntime",
            device="homeguard_directml",
        )
        if DML_PROVIDER not in self._pose_model.session.get_providers():
            raise RuntimeError("RTMPose did not activate DirectML on the selected GPU.")

        self._face_app = FaceAnalysis(
            name="buffalo_sc",
            root=str(INSIGHTFACE_ROOT),
            allowed_modules=["detection", "recognition"],
            providers=providers,
        )
        # InsightFace uses a negative context ID to force its sessions to CPU.
        self._face_app.prepare(ctx_id=self.device_id, det_size=(320, 320))
        if any(
            DML_PROVIDER not in model.session.get_providers()
            for model in self._face_app.models.values()
        ):
            raise RuntimeError("InsightFace did not activate DirectML on the selected GPU.")

        tracker_config_path = (
            Path(ultralytics.__file__).resolve().parent
            / "cfg"
            / "trackers"
            / "bytetrack.yaml"
        )
        tracker_config = get_cfg(tracker_config_path)
        tracker_config.track_buffer = 30
        self._tracker = BYTETracker(tracker_config)

    def _person_detections(self, frame):
        import cv2
        import numpy as np

        tensor, scale, pad_x, pad_y = _letterbox(frame)
        output = self._person_session.run(None, {self._person_input: tensor})[0]
        predictions = output[0].T
        # The exported YOLO26 detect head contains xywh followed by 80 COCO
        # class probabilities. Class 0 is person.
        scores = predictions[:, 4]
        selected = np.flatnonzero(scores >= self.confidence)
        boxes_xywh = []
        kept_scores = []
        for index in selected:
            cx, cy, box_width, box_height = predictions[index, :4]
            x1 = max(0.0, (cx - box_width / 2 - pad_x) / scale)
            y1 = max(0.0, (cy - box_height / 2 - pad_y) / scale)
            x2 = min(float(frame.shape[1]), (cx + box_width / 2 - pad_x) / scale)
            y2 = min(float(frame.shape[0]), (cy + box_height / 2 - pad_y) / scale)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])
            kept_scores.append(float(scores[index]))

        if not boxes_xywh:
            return _DetectionBatch(
                np.empty((0, 4), np.float32),
                np.empty((0,), np.float32),
                np.empty((0,), np.float32),
            )

        nms_input = [
            [int(x), int(y), int(width), int(height)]
            for x, y, width, height in boxes_xywh
        ]
        keep = cv2.dnn.NMSBoxes(
            nms_input, kept_scores, self.confidence, 0.45
        )
        keep = np.asarray(keep).reshape(-1).astype(int)
        boxes_xywh = np.asarray(boxes_xywh, dtype=np.float32)[keep]
        kept_scores = np.asarray(kept_scores, dtype=np.float32)[keep]
        centers = boxes_xywh.copy()
        centers[:, 0] += centers[:, 2] / 2
        centers[:, 1] += centers[:, 3] / 2
        return _DetectionBatch(
            centers,
            kept_scores,
            np.zeros(len(kept_scores), dtype=np.float32),
        )

    def analyze(self, frame) -> Phase3FrameResult:
        """Analyze one BGR frame and return person poses plus face embeddings."""
        import numpy as np

        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3:
            raise ValueError("frame must be a BGR image array")

        with self._lock:
            detections = self._person_detections(frame)
            tracked = self._tracker.update(detections, img=frame)
            if tracked is None or len(tracked) == 0:
                people: tuple[PersonPose, ...] = ()
            else:
                order = np.argsort(tracked[:, 5])[::-1][: self.max_people]
                selected_tracks = tracked[order]
                selected_boxes = selected_tracks[:, :4].astype(np.float32, copy=False)
                keypoints, keypoint_scores = self._pose_model(frame, bboxes=selected_boxes)
                people = tuple(
                    PersonPose(
                        track_id=int(track[4]),
                        box_xyxy=tuple(float(v) for v in track[:4]),
                        keypoints_xy=tuple(
                            (float(x), float(y)) for x, y in keypoints[index]
                        ),
                        keypoint_scores=tuple(
                            float(v) for v in keypoint_scores[index]
                        ),
                        detection_score=float(track[5]),
                    )
                    for index, track in enumerate(selected_tracks)
                )

            face_detections = self._face_app.get(frame, max_num=self.max_people)
            faces = tuple(
                FaceMatchInput(
                    box_xyxy=tuple(float(v) for v in face.bbox),
                    detection_score=float(face.det_score),
                    embedding=tuple(float(v) for v in face.normed_embedding),
                )
                for face in face_detections
                if face.embedding is not None
            )

        return Phase3FrameResult(people=people, faces=faces)
