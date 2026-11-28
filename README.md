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

## Use a phone camera over USB

GestureFlow reads USB webcam sources exposed by Windows through OpenCV. Connect
the phone with a data-capable USB cable, unlock it, open the USB options
notification, and choose **Webcam** (some phones call this **USB webcam**).
Keep the phone's webcam preview active, click **Find** in GestureFlow, select
the phone's camera input, and turn the camera on. If the phone offers a webcam
preview, choose its rear camera for a wider, clearer view. [Google's Pixel
instructions](https://support.google.com/pixelcamera/answer/14274129) show the
USB menu flow; the exact menu depends on the phone.

USB debugging alone does not send camera video to the laptop. The phone must
support USB webcam/UVC mode in its Android version and manufacturer software.
Android's platform includes UVC webcam support on Android 14 and newer devices,
but each phone maker must enable it, so the option is not available on every
phone ([Android platform documentation](https://source.android.com/docs/core/camera/webcam)).
If the phone has no **Webcam** USB option, it needs a phone-side camera streaming
app or a vendor-provided webcam mode; laptop-only Python cannot bypass Android's
camera permission. No extra PC package is installed by GestureFlow.

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
