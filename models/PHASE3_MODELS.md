# Phase 3 model files

These assets are for the local, non-distributed prototype with the project team.
Do not package or publish them until their current model and software terms have
been reviewed for the intended release.

| Feature | File(s) | Use |
| --- | --- | --- |
| Person detection | `human/yolo26n.onnx` | YOLO26 nano ONNX export; DirectML on this computer's AMD adapter. |
| Person tracking | Ultralytics `bytetrack.yaml` | ByteTrack keeps person IDs; this light association step runs on CPU. |
| Pose | `human/rtmpose_s.onnx` | 17 body landmarks per person; DirectML with CPU fallback for unsupported nodes. |
| Face detection and embeddings | `face/insightface/models/buffalo_sc/` | SCRFD 500MF detector and MBF face embeddings; DirectML with CPU fallback. |
| Hands and gestures | `hand/gesture_recognizer.task` | Existing MediaPipe task supports four hands and stable labels; its Python GPU delegate is unavailable on Windows, so it runs on CPU. |

## Runtime

The extra runtime is isolated in `.phase3_env`; it does not replace packages in
the app's existing Python installation. Use that interpreter to import
`app.phase3_vision.Phase3VisionStack`. The existing UI is not connected to this
stack yet, so the normal app still uses its current CPU workers. The adapter
selects DirectML device 1, which was verified as the AMD Radeon R7 M260 on this
computer; another computer may need a different device index.

Installed in the isolated environment:

- `onnxruntime-directml==1.24.4` (active DirectML provider, plus CPU fallback)
- `onnxruntime==1.32.0` (RTMLib dependency; DirectML is installed last)
- `rtmlib==0.0.16`
- `ultralytics==8.4.219`
- `lap==0.5.14` (ByteTrack assignment dependency)
- `insightface==2.1`

The YOLO26, RTMPose, and InsightFace ONNX sessions initialized and ran on
DirectML device 1. The Intel adapter (device 0) failed this model check, so the
AMD adapter is selected. This computer has no NVIDIA GPU, so TensorRT is not an
option here. GPU acceleration does not make every part parallel: DirectML
inference runs sequentially, ByteTrack stays on CPU, and MediaPipe hands stay
on CPU on Windows. On one supplied screenshot's camera-preview crop, the full
Phase 3 pass took about 193–205 ms after warm-up (roughly 5 frames per second).
That is a prototype measurement, not a guarantee for a live camera.

## Terms to revisit before sharing a release

- Ultralytics publishes YOLO26 under AGPL-3.0, with a separate Enterprise
  licensing option. Review those terms before distributing the app.
- InsightFace states that its pretrained model-zoo assets are for
  non-commercial research use. The `buffalo_sc` files must not be included in a
  shared release unless the project obtains terms that cover that release.
- MMPose code is Apache-2.0, but the license for this specific RTMPose
  checkpoint has not been confirmed. Verify it before distributing the file.
- ByteTrack is MIT-licensed. OpenCV, MediaPipe, and the existing face models
  keep their own notices and terms.

These notes are a project checklist, not legal advice.
