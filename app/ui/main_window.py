import os
import ctypes
import tkinter as tk
from tkinter import ttk
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

from app.coordinator import DetectorKind, FrameCoordinator
from app.camera.capture import CameraCapture as CameraService, CameraStatus
from app.human_detection import HumanDetectionWorker
from app.face_recognition_worker import FaceRecognitionWorker
from app.hand_gesture_advanced import AdvancedHandDetectionWorker


# --- Visual constants ---
PREVIEW_BG = "#0f1620"
VIDEO_BG = "#0a0f14"
TEXT = "#e8eaed"
MUTED = "#6a7b8d"
ACCENT = "#2a7ae2"
ACCENT_HOVER = "#1d5fc4"
DANGER = "#b42318"
DANGER_HOVER = "#8a1a13"
FACE_BOX_COLOR = (0, 200, 255)
FACE_LABEL_COLOR = (0, 200, 255)
HAND_CANDIDATE_COLOR = (0, 255, 0)
HAND_TRACK_COLOR = (0, 200, 255)


def _enable_windows_dpi() -> None:
    """Set Windows DPI awareness for proper scaling on high-DPI displays."""
    import sys
    if sys.platform == "win32":
        try:
            ctypes.shcore.SetProcessDpiAwareness(1)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
HUMAN_POSE_COLOR = (0, 255, 255)
HUMAN_POSE_POINT_COLOR = (255, 0, 255)
OFF_MESSAGE = "Camera is off"
REQUESTED_WIDTH = 1280
REQUESTED_HEIGHT = 720
REQUESTED_FPS = 30


def _placeholder_image(width: int, height: int) -> Image.Image:
    """Create a placeholder image for the camera preview area."""
    import numpy as np
    from PIL import Image
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:] = (40, 44, 48)  # PREVIEW_BG color
    return Image.fromarray(arr)


def _draw_camera_icon(draw: ImageDraw.ImageDraw, center_x: int, center_y: int) -> None:
    """Draw a camera icon on the given draw context."""
    color = "#6a7b8d"
    body = (center_x - 54, center_y - 34, center_x + 54, center_y + 34)
    draw.rounded_rectangle(body, radius=12, outline=color, width=5)
    draw.rounded_rectangle(
        (center_x + 18, center_y - 52, center_x + 48, center_y - 32),
        radius=4,
        outline=color,
        width=5,
    )
    draw.ellipse(
        (center_x - 20, center_y - 20, center_x + 20, center_y + 20),
        outline=color,
        width=5,
    )
    draw.line(
        (center_x - 70, center_y + 48, center_x + 70, center_y - 52),
        fill="#9aabbd",
        width=5,
    )


def _fit_image(image: Image.Image, width: int, height: int, background: str) -> Image.Image:
    """Fit an image into a canvas with the given background."""
    canvas = Image.new("RGB", (width, height), background)
    scale = min(width / image.width, height / image.height)
    resized = image.resize(
        (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
        Image.Resampling.BILINEAR,
    )
    x = (width - resized.width) // 2
    y = (height - resized.height) // 2
    canvas.paste(resized, (x, y))
    return canvas


def _fit_font(draw: ImageDraw.ImageDraw, text: str, max_width: int, size: int) -> ImageFont.ImageFont:
    """Find a font size that fits within max_width."""
    font = _font(size)
    while size > 12:
        bbox = draw.textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= max_width:
            return font
        size -= 1
        font = _font(size)
    return font


def _font(size: int) -> ImageFont.ImageFont:
    """Load a Segoe UI font at the given size, falling back to default."""
    windows = os.environ.get("WINDIR", r"C:\Windows")
    path = os.path.join(windows, "Fonts", "segoeui.ttf")
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def main() -> None:
    _enable_windows_dpi()
    root = tk.Tk()
    root.title("HOME GUARD AI")
    root.geometry("1180x760")
    root.minsize(1020, 680)
    root.configure(bg=PREVIEW_BG)
    window = MainWindow(root)
    window.pack(fill="both", expand=True)
    root.protocol("WM_DELETE_WINDOW", window._on_close)
    root.mainloop()


class MainWindow(ttk.Frame):
    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)
        self.parent = parent

        self.camera = CameraService()
        self.coordinator = FrameCoordinator(self.camera)
        self._closing = False
        self._placeholder: Optional[ImageTk.PhotoImage] = None
        self._placeholder_size: Optional[tuple[int, int]] = None
        self._photo: Optional[ImageTk.PhotoImage] = None

        # Worker instances (create once, start/stop via coordinator enable/disable)
        self.human_detection = HumanDetectionWorker(coordinator=self.coordinator, kind=DetectorKind.HUMAN)
        self.face_recognition = FaceRecognitionWorker(
            frame_provider=self.camera.get_frame,
            coordinator=self.coordinator,
            kind=DetectorKind.FACE,
        )
        self.hand_detection = AdvancedHandDetectionWorker(
            frame_provider=self.camera.get_frame,
            coordinator=self.coordinator,
            kind=DetectorKind.HAND,
        )

        self.coordinator.start()

        self._build()
        self._enable_dpi_awareness()
        self.after(50, self._tick)

    def _enable_dpi_awareness(self) -> None:
        _enable_windows_dpi()

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # --- Header bar ---
        header = ttk.Frame(self)
        header.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))
        header.columnconfigure(1, weight=1)

        ttk.Label(header, text="HomeGuard AI", font=("Segoe UI", 16, "bold")).grid(row=0, column=0, sticky="w")

        status_frame = ttk.Frame(header)
        status_frame.grid(row=0, column=1, sticky="e")
        status_frame.columnconfigure((0, 2, 4), weight=1)

        ttk.Label(status_frame, text="State:", font=("Segoe UI", 9)).grid(row=0, column=0, sticky="e", padx=(0, 4))
        self.state_value = ttk.Label(status_frame, text="OFF", font=("Segoe UI", 9, "bold"), foreground=MUTED)
        self.state_value.grid(row=0, column=1, sticky="w", padx=(0, 16))

        ttk.Label(status_frame, text="Resolution:", font=("Segoe UI", 9)).grid(row=0, column=2, sticky="e", padx=(0, 4))
        self.resolution_value = ttk.Label(status_frame, text="—", font=("Segoe UI", 9))
        self.resolution_value.grid(row=0, column=3, sticky="w", padx=(0, 16))

        ttk.Label(status_frame, text="FPS:", font=("Segoe UI", 9)).grid(row=0, column=4, sticky="e", padx=(0, 4))
        self.fps_value = ttk.Label(status_frame, text="—", font=("Segoe UI", 9))
        self.fps_value.grid(row=0, column=5, sticky="w")

        # --- Video preview ---
        preview_container = ttk.Frame(self)
        preview_container.grid(row=1, column=0, sticky="nsew", padx=12, pady=6)
        preview_container.columnconfigure(0, weight=1)
        preview_container.rowconfigure(0, weight=1)

        self.preview_label = ttk.Label(preview_container, background=VIDEO_BG, anchor="center")
        self.preview_label.grid(row=0, column=0, sticky="nsew")

        # --- Notice bar ---
        notice_frame = ttk.Frame(self)
        notice_frame.grid(row=2, column=0, sticky="ew", padx=12, pady=(0, 6))
        notice_frame.columnconfigure(0, weight=1)
        self.notice_value = ttk.Label(notice_frame, text="", font=("Segoe UI", 9), foreground=MUTED, anchor="center")
        self.notice_value.grid(row=0, column=0, sticky="ew")

        # --- Control panel ---
        controls = ttk.Frame(self)
        controls.grid(row=3, column=0, sticky="ew", padx=12, pady=(0, 12))
        controls.columnconfigure((0, 1, 2), weight=1)

        # Camera toggle
        cam_frame = ttk.LabelFrame(controls, text="Camera", padding=8)
        cam_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        cam_frame.columnconfigure(0, weight=1)

        self.toggle_button = ttk.Button(
            cam_frame,
            text="Turn Camera ON",
            command=self._toggle_camera,
            width=18,
        )
        self.toggle_button.grid(row=0, column=0, pady=4, sticky="ew")

        # Human detection
        det_frame = ttk.LabelFrame(controls, text="Human Detection", padding=8)
        det_frame.grid(row=0, column=1, sticky="nsew", padx=6)
        det_frame.columnconfigure(0, weight=1)

        self.detection_info = ttk.Label(det_frame, text="Turn the camera on before starting detection.", font=("Segoe UI", 9), foreground=MUTED, anchor="center", wraplength=180)
        self.detection_info.grid(row=0, column=0, pady=(0, 6), sticky="ew")

        btn_row = ttk.Frame(det_frame)
        btn_row.grid(row=1, column=0, sticky="ew")
        btn_row.columnconfigure((0, 1), weight=1)

        self.start_detection_button = ttk.Button(btn_row, text="Start", command=lambda: self._set_detector(DetectorKind.HUMAN, True), width=10, state="disabled")
        self.start_detection_button.grid(row=0, column=0, padx=(0, 4), sticky="ew")
        self.stop_detection_button = ttk.Button(btn_row, text="Stop", command=lambda: self._set_detector(DetectorKind.HUMAN, False), width=10, state="disabled")
        self.stop_detection_button.grid(row=0, column=1, padx=(4, 0), sticky="ew")

        # Face recognition
        face_frame = ttk.LabelFrame(controls, text="Face Recognition", padding=8)
        face_frame.grid(row=0, column=2, sticky="nsew", padx=(6, 0))
        face_frame.columnconfigure(0, weight=1)

        self.face_info = ttk.Label(face_frame, text="Turn the camera on before starting face recognition.", font=("Segoe UI", 9), foreground=MUTED, anchor="center", wraplength=180)
        self.face_info.grid(row=0, column=0, pady=(0, 6), sticky="ew")

        face_btn_row = ttk.Frame(face_frame)
        face_btn_row.grid(row=1, column=0, sticky="ew")
        face_btn_row.columnconfigure((0, 1), weight=1)

        self.start_face_button = ttk.Button(face_btn_row, text="Start", command=lambda: self._set_detector(DetectorKind.FACE, True), width=10, state="disabled")
        self.start_face_button.grid(row=0, column=0, padx=(0, 4), sticky="ew")
        self.stop_face_button = ttk.Button(face_btn_row, text="Stop", command=lambda: self._set_detector(DetectorKind.FACE, False), width=10, state="disabled")
        self.stop_face_button.grid(row=0, column=1, padx=(4, 0), sticky="ew")

        # Hand detection (row 1)
        hand_frame = ttk.LabelFrame(controls, text="Hand Detection", padding=8)
        hand_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 6), pady=(6, 0))
        hand_frame.columnconfigure(0, weight=1)

        self.hand_info = ttk.Label(hand_frame, text="Turn the camera on before starting hand detection.", font=("Segoe UI", 9), foreground=MUTED, anchor="center", wraplength=180)
        self.hand_info.grid(row=0, column=0, pady=(0, 6), sticky="ew")

        hand_btn_row = ttk.Frame(hand_frame)
        hand_btn_row.grid(row=1, column=0, sticky="ew")
        hand_btn_row.columnconfigure((0, 1), weight=1)

        self.start_hand_button = ttk.Button(hand_btn_row, text="Start", command=lambda: self._set_detector(DetectorKind.HAND, True), width=10, state="disabled")
        self.start_hand_button.grid(row=0, column=0, padx=(0, 4), sticky="ew")
        self.stop_hand_button = ttk.Button(hand_btn_row, text="Stop", command=lambda: self._set_detector(DetectorKind.HAND, False), width=10, state="disabled")
        self.stop_hand_button.grid(row=0, column=1, padx=(4, 0), sticky="ew")

        # Quit button
        quit_frame = ttk.Frame(controls)
        quit_frame.grid(row=1, column=2, sticky="nsew", padx=(6, 0), pady=(6, 0))
        quit_frame.columnconfigure(0, weight=1)
        ttk.Button(quit_frame, text="Quit", command=self._on_close, width=10).grid(row=0, column=0, sticky="e")

    def _toggle_camera(self) -> None:
        status = self.camera.snapshot()
        if status.state == "on":
            self.camera.stop()
            # Stop all detector workers when camera turns off
            self.human_detection.request_stop()
            self.face_recognition.request_stop()
            self.hand_detection.set_enabled(False)
            self.coordinator.disable(DetectorKind.HUMAN)
            self.coordinator.disable(DetectorKind.FACE)
            self.coordinator.disable(DetectorKind.HAND)
        else:
            # Try to start the camera; if it fails, keep it off but refresh UI
            started = self.camera.start()
            if not started:
                # Camera could not start — keep it off but re-enable detectors
                # so they can be started later when camera is available
                self.coordinator.enable(DetectorKind.HUMAN)
                self.coordinator.enable(DetectorKind.FACE)
                self.coordinator.enable(DetectorKind.HAND)
                self._refresh()
                return
            # Re-enable all detectors and start workers after camera starts
            self.coordinator.enable(DetectorKind.HUMAN)
            self.coordinator.enable(DetectorKind.FACE)
            self.coordinator.enable(DetectorKind.HAND)
            # Start the worker threads
            self.human_detection.start()
            self.face_recognition.start()
            self.hand_detection.set_enabled(True)
        self._refresh()

    def _set_detector(self, kind: DetectorKind, enable: bool) -> None:
        import traceback
        try:
            if enable:
                self.coordinator.enable(kind)
                # Start the corresponding worker
                if kind == DetectorKind.HUMAN:
                    self.human_detection.start()
                elif kind == DetectorKind.FACE:
                    self.face_recognition.start()
                elif kind == DetectorKind.HAND:
                    self.hand_detection.set_enabled(True)
            else:
                self.coordinator.disable(kind)
                # Stop the corresponding worker
                if kind == DetectorKind.HUMAN:
                    self.human_detection.request_stop()
                elif kind == DetectorKind.FACE:
                    self.face_recognition.request_stop()
                elif kind == DetectorKind.HAND:
                    self.hand_detection.set_enabled(False)
        except Exception as e:
            print(f"[DIAG EXCEPTION] {kind.name} {enable=}: {e}")
            traceback.print_exc()
        self._refresh()

    def _on_close(self) -> None:
        self._closing = True
        self.human_detection.shutdown(timeout=1.0)
        self.face_recognition.shutdown(timeout=1.0)
        self.hand_detection.shutdown(timeout=1.0)
        self.coordinator.stop()
        self.camera.shutdown(timeout=2.0)
        self.parent.destroy()

    def _tick(self) -> None:
        try:
            self._refresh()
        except Exception as exc:
            self.detection_info.configure(text=f"UI error: {exc}", foreground="#b42318")
        finally:
            if not self._closing:
                self.after(33, self._tick)

    def _refresh(self) -> None:
        status = self.camera.snapshot()
        frame, results, synced = self.coordinator.compose()

        # Extract results for display updates
        human_result = results.get(DetectorKind.HUMAN)
        face_result = results.get(DetectorKind.FACE)
        hand_result = results.get(DetectorKind.HAND)

        # Stop workers if camera is off
        if status.state != "on":
            if human_result and human_result.human_people is not None:
                self.coordinator.disable(DetectorKind.HUMAN)
            if face_result and face_result.face_detections is not None:
                self.coordinator.disable(DetectorKind.FACE)
            if hand_result and hand_result.hand_candidates is not None:
                self.coordinator.disable(DetectorKind.HAND)

        # Update display status
        self._update_detection_display(status.state, human_result)
        self._update_face_display(status.state, face_result)
        self._update_hand_display(hand_result)
        self._set_status_display(
            status.state,
            status.message,
            status.width,
            status.height,
            status.fps,
        )
        if status.state == "on" and frame is not None:
            display = frame
            if face_result and face_result.face_detections:
                try:
                    display = self._draw_face_overlays(display, face_result.face_detections)
                except Exception:
                    pass
            if hand_result and hand_result.hand_candidates is not None:
                try:
                    candidates = hand_result.hand_candidates or ()
                    tracks = hand_result.hand_visible_tracks or ()
                    display = self._draw_hand_overlays(display, candidates, tracks)
                except Exception:
                    pass
            if human_result and human_result.human_pose_draw_data:
                try:
                    display = self._draw_human_overlays(display, human_result.human_pose_draw_data)
                except Exception:
                    pass
            if human_result and human_result.human_face_landmark_draw_data:
                try:
                    from app.face_landmarks import draw_face_landmarks
                    display = draw_face_landmarks(
                        display,
                        human_result.human_face_landmark_draw_data,
                    )
                except Exception:
                    pass
            self._show_frame(display)
            return
        self._show_placeholder()

    def _update_detection_display(self, camera_state: str, human_result) -> None:
        if human_result is None:
            worker_state = self.human_detection.get_status()
            if camera_state == "on" and self.coordinator.is_enabled(DetectorKind.HUMAN):
                text = worker_state.message or "Human detection is starting."
                color = DANGER if worker_state.state == "error" else ACCENT
                can_start = False
                can_stop = worker_state.state != "off"
            elif camera_state == "on":
                text = "Human detection is off."
                color = MUTED
                can_start = True
                can_stop = False
            else:
                text = "Turn the camera on before starting detection."
                color = MUTED
                can_start = False
                can_stop = False
        else:
            text = (
                f"{'HUMAN DETECTED' if human_result.human_people else 'NO HUMAN DETECTED'} "
                f"— People: {human_result.human_people or 0}"
            )
            color = "#1b7f3a" if human_result.human_people else MUTED
            can_start = False
            can_stop = True
        self.detection_info.configure(text=text, foreground=color)

        self.start_detection_button.configure(
            state="normal" if can_start else "disabled"
        )
        self.stop_detection_button.configure(
            state="normal" if can_stop else "disabled"
        )

    def _update_face_display(self, camera_state: str, face_result) -> None:
        if face_result is None:
            worker_state = self.face_recognition.get_snapshot()
            if camera_state == "on" and self.coordinator.is_enabled(DetectorKind.FACE):
                if worker_state.state in ("off", "error"):
                    text = (
                        worker_state.message
                        if worker_state.state == "error"
                        else "Face worker stopped. Press Start to retry."
                    )
                    color = DANGER if worker_state.state == "error" else MUTED
                    can_start = True
                    can_stop = False
                elif worker_state.state == "stopping":
                    text = worker_state.message or "Face recognition is stopping."
                    color = ACCENT
                    can_start = False
                    can_stop = False
                else:
                    text = worker_state.message or "Face recognition is starting."
                    color = ACCENT
                    can_start = False
                    can_stop = True
            elif camera_state == "on":
                text = "Face recognition is off."
                color = MUTED
                can_start = True
                can_stop = False
            else:
                text = "Turn the camera on before starting face recognition."
                color = MUTED
                can_start = False
                can_stop = False
        else:
            count = len(face_result.face_detections) if face_result.face_detections else 0
            text = f"Faces: {count} — all labeled UNKNOWN"
            color = "#1b7f3a" if count else MUTED
            can_start = False
            can_stop = True
        self.face_info.configure(text=text, foreground=color)

        self.start_face_button.configure(
            state="normal" if can_start else "disabled"
        )
        self.stop_face_button.configure(
            state="normal" if can_stop else "disabled"
        )

    def _update_hand_display(self, hand_result) -> None:
        camera_on = self.camera.snapshot().state == "on"
        if hand_result is None:
            worker_state = self.hand_detection.snapshot()
            if camera_on and self.coordinator.is_enabled(DetectorKind.HAND):
                text = worker_state.message or "Hand detection is starting."
                color = DANGER if worker_state.state == "error" else ACCENT
                can_start = False
                can_stop = worker_state.state != "off"
            elif camera_on:
                text = "Hand detection is off."
                color = MUTED
                can_start = True
                can_stop = False
            else:
                text = "Turn the camera on before starting hand detection."
                color = MUTED
                can_start = False
                can_stop = False
        else:
            candidates = hand_result.hand_candidates or ()
            text = f"Detected hands: {len(candidates)}"
            color = "#1b7f3a" if len(candidates) > 0 else MUTED
            can_start = False
            can_stop = True
        self.hand_info.configure(text=text, foreground=color)

        self.start_hand_button.configure(
            state="normal" if can_start else "disabled"
        )
        self.stop_hand_button.configure(
            state="normal" if can_stop else "disabled"
        )

    def _set_status_display(
        self,
        state: str,
        message: str,
        width: int,
        height: int,
        fps: float,
    ) -> None:
        self.state_value.configure(
            text={"off": "OFF", "starting": "STARTING", "on": "ON", "stopping": "STOPPING", "error": "ERROR"}.get(state, state.upper()),
            foreground={ "off": MUTED, "starting": ACCENT, "on": "#1b7f3a", "stopping": "#e8a838", "error": DANGER }.get(state, TEXT),
        )
        if state == "on" and width > 0 and height > 0:
            self.resolution_value.configure(text=f"{width} × {height}")
        else:
            self.resolution_value.configure(text="—")
        if state == "on" and fps > 0:
            self.fps_value.configure(text=f"{fps:.0f} FPS")
        else:
            self.fps_value.configure(text="—")

        requested = f"Requested {REQUESTED_WIDTH} × {REQUESTED_HEIGHT} at {REQUESTED_FPS} FPS."
        if state == "error" and message:
            notice = message
        elif message and state != "off":
            notice = message
        else:
            notice = requested
        self.notice_value.configure(text=notice)

        if state == "starting":
            self.toggle_button.configure(state="disabled", text="Opening Camera…")
        elif state == "stopping":
            self.toggle_button.configure(state="disabled", text="Turning Camera OFF")
        elif state == "on":
            self.toggle_button.configure(state="normal", text="Turn Camera OFF")
        else:
            self.toggle_button.configure(state="normal", text="Turn Camera ON")

    def _preview_size(self) -> tuple[int, int]:
        self.preview_label.update_idletasks()
        width = self.preview_label.winfo_width()
        height = self.preview_label.winfo_height()
        if width < 200 or height < 120:
            return 800, 450
        return width, height

    def _show_placeholder(self) -> None:
        size = self._preview_size()
        if self._placeholder is None or self._placeholder_size != size:
            image = _placeholder_image(size[0], size[1])
            self._placeholder = ImageTk.PhotoImage(image)
            self._placeholder_size = size
        self.preview_label.configure(image=self._placeholder)
        self._photo = self._placeholder

    def _show_frame(self, frame) -> None:
        width, height = self._preview_size()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        fitted = _fit_image(image, width, height, VIDEO_BG)
        photo = ImageTk.PhotoImage(fitted)
        self.preview_label.configure(image=photo)
        self._photo = photo

    def _draw_face_overlays(self, frame, faces):
        """Copy frame and draw UNKNOWN face boxes on the UI thread."""
        annotated = frame.copy()
        for face in faces:
            x, y, width, height = face.bbox
            x2 = x + width
            y2 = y + height
            cv2.rectangle(annotated, (x, y), (x2, y2), FACE_BOX_COLOR, 2)
            label = face.label if face.label else "UNKNOWN"
            cv2.putText(
                annotated,
                label,
                (x, max(22, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                FACE_LABEL_COLOR,
                2,
            )
        return annotated

    def _draw_hand_overlays(self, frame, candidates, tracks):
        """Copy frame and draw hand candidates/tracks on the UI thread."""
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        def _candidate_point(candidate):
            # Try .wrist first (pixel coordinates from MediaPipe)
            wrist = getattr(candidate, "wrist", None)
            if wrist is not None:
                try:
                    x, y = float(wrist[0]), float(wrist[1])
                    if 0 <= x < w and 0 <= y < h:
                        return int(x), int(y)
                except Exception:
                    pass
            # Try .x, .y (normalized)
            try:
                x_norm = float(candidate.x)
                y_norm = float(candidate.y)
                x = int(x_norm * w)
                y = int(y_norm * h)
                if 0 <= x < w and 0 <= y < h:
                    return x, y
            except Exception:
                pass
            # Try .bbox as pixel coordinates [x1, y1, x2, y2]
            bbox = getattr(candidate, "bbox", None)
            if bbox is not None:
                try:
                    x1, y1, x2, y2 = bbox
                    x = int((x1 + x2) / 2)
                    y = int((y1 + y2) / 2)
                    if 0 <= x < w and 0 <= y < h:
                        return x, y
                except Exception:
                    pass
            # Try .bbox as normalized coordinates
            if bbox is not None:
                try:
                    x1, y1, x2, y2 = bbox
                    x = int((x1 + x2) / 2 * w)
                    y = int((y1 + y2) / 2 * h)
                    if 0 <= x < w and 0 <= y < h:
                        return w - x, y
                except Exception:
                    pass
            # Dict or subscriptable formats (normalized).
            try:
                x_norm, y_norm = float(candidate[0]), float(candidate[1])
                x = int(x_norm * w)
                y = int(y_norm * h)
                if 0 <= x < w and 0 <= y < h:
                    return w - x, y
            except Exception:
                pass
            return None

        for candidate in candidates or ():
            try:
                point = _candidate_point(candidate)
                if point is None:
                    continue
                x, y = point
                cv2.circle(annotated, (x, y), 6, (0, 255, 0), 2)
            except Exception:
                continue
        for track in tracks or ():
            try:
                # Try tuple/list format first
                if isinstance(track, (tuple, list)) and len(track) >= 2:
                    x, y = int(track[0]), int(track[1])
                else:
                    # Try object with bbox attribute (pixel coordinates)
                    bbox = getattr(track, "bbox", None)
                    if bbox is not None:
                        x1, y1, x2, y2 = bbox
                        x = int((x1 + x2) / 2)
                        y = int((y1 + y2) / 2)
                    else:
                        continue
                    w = annotated.shape[1]
                    x = w - x
                    cv2.circle(annotated, (x, y), 4, (0, 200, 255), -1)
            except Exception:
                continue
        return annotated

    def _draw_human_overlays(self, frame, draw_data):
        """Copy frame and draw human pose skeleton on the UI thread.

        draw_data is a tuple of (points_norm, visible) per person:
        - points_norm: tuple of 33 (x, y) normalized coordinates (0-1)
        - visible: tuple of 33 booleans indicating keypoint visibility
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # COCO 17 keypoint indices and names (MediaPipe Pose has 33 keypoints)
        keypoint_names = [
            "nose", "left_eye_inner", "left_eye", "left_eye_outer",
            "right_eye_inner", "right_eye", "right_eye_outer",
            "left_ear", "right_ear",
            "mouth_left", "mouth_right",
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
            "left_pinky", "right_pinky",
            "left_index", "right_index",
            "left_thumb", "right_thumb",
            "left_hip", "right_hip",
            "left_knee", "right_knee",
            "left_ankle", "right_ankle",
            "left_heel", "right_heel",
            "left_foot_index", "right_foot_index"
        ]

        # COCO 17 skeleton edges using keypoint_names indices
        edges = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # nose - eyes
            (5, 7), (6, 8),                    # ears
            (9, 11), (10, 12),                 # mouth - shoulders
            (11, 13), (13, 15),                # left arm
            (12, 14), (14, 16),                # right arm
            (11, 12),                          # shoulders
            (11, 23), (12, 24),                # shoulders - hips
            (23, 24),                          # hips
            (23, 25), (25, 27),                # left leg
            (24, 26), (26, 28),                # right leg
        ]

        for person in draw_data:
            # Handle both tuple format (points_norm, visible) and dict format
            if isinstance(person, tuple) and len(person) == 2:
                points_norm, visible = person
            elif isinstance(person, dict):
                points_norm = person.get("points", [])
                visible = person.get("visible", [])
            else:
                continue
                
            if not points_norm or len(points_norm) < 33:
                continue
                
            # Draw keypoints that are visible
            kps = {}
            for i, (x_norm, y_norm) in enumerate(points_norm):
                if i < len(visible) and visible[i]:
                    if i < len(keypoint_names):
                        name = keypoint_names[i]
                        px = int(x_norm * w)
                        py = int(y_norm * h)
                        kps[name] = (px, py)
                        # Draw keypoint
                        cv2.circle(annotated, (px, py), 4, HUMAN_POSE_POINT_COLOR, -1)

            # Draw skeleton edges
            for a_idx, b_idx in edges:
                if a_idx < len(keypoint_names) and b_idx < len(keypoint_names):
                    a_name = keypoint_names[a_idx]
                    b_name = keypoint_names[b_idx]
                    if a_name in kps and b_name in kps:
                        cv2.line(annotated, kps[a_name], kps[b_name], HUMAN_POSE_COLOR, 2)
                    
        return annotated


def main() -> None:
    _enable_windows_dpi()
    root = tk.Tk()
    root.title("HOME GUARD AI")
    root.geometry("1180x760")
    root.minsize(1020, 680)
    root.configure(bg=PREVIEW_BG)
    window = MainWindow(root)
    window.pack(fill="both", expand=True)
    root.protocol("WM_DELETE_WINDOW", window._on_close)
    root.mainloop()
