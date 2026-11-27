# GestureFlow AI

GestureFlow AI is a Windows desktop prototype for controlling the mouse with
hand gestures and sharing files with another GestureFlow AI device on the same
Wi-Fi network.

## Run the app

Open PowerShell in this folder and run:

```powershell
.\.gestureflow_env\Scripts\python.exe -m gestureflow_ai
```

This uses the existing local Python environment. No package installation is
needed to start it on the computer where that environment is already set up.

## Included features

- Live camera preview with optional image noise reduction.
- Hand landmark tracking and per-hand gesture labels, with a limit of four
  tracked hands.
- Gesture based cursor movement and mouse actions.
- Water-inspired cursor and interface effects.
- Optional file sharing with a nearby device, with received files saved under
  `Documents\GestureFlow AI\Received`.

## Gestures

The hand recognizer uses MediaPipe's bundled gesture model. The app uses the
open palm, closed fist, and thumbs-up categories for its hand controls.

## Model and license

The active hand model is `models/gesture/gesture_recognizer.task`. Its source,
checksum, and license are recorded in
[`models/gesture/README.md`](models/gesture/README.md). The model is licensed
under Apache 2.0. This version focuses on hand gestures and contains no face or person recognition modules.

## Local development

Runtime packages are listed in [`requirements.txt`](requirements.txt). The
`.gestureflow_env` folder is a local environment and is not published to GitHub.
The project does not install packages automatically.
