import os
import ctypes
import tkinter as tk
from tkinter import ttk
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

from gestureflow_ai.coordinator import DetectorKind, FrameCoordinator
from gestureflow_ai.camera.capture import CameraCapture as CameraService, CameraStatus
from gestureflow_ai.hand_gesture_advanced import AdvancedHandDetectionWorker
from gestureflow_ai.camera.noise_cleaner import CameraNoiseCleaner
from gestureflow_ai.gesture_mouse_control import GestureMouseController
from gestureflow_ai.ui.water_effects import WaterCursorOverlay


# --- Visual constants ---
PREVIEW_BG = "#0f1620"
VIDEO_BG = "#0a0f14"
TEXT = "#e8eaed"
MUTED = "#6a7b8d"
ACCENT = "#2a7ae2"
ACCENT_HOVER = "#1d5fc4"
DANGER = "#b42318"
DANGER_HOVER = "#8a1a13"
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
    root.title("GestureFlow AI")
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
        self._photo_size: Optional[tuple[int, int]] = None
        self.preview_noise_cleaner_enabled = False
        self.camera_noise_cleaner = CameraNoiseCleaner()

        # GestureFlow AI runs the hand and gesture worker only.
        self.hand_detection = AdvancedHandDetectionWorker(
            frame_provider=self.camera.get_frame,
            coordinator=self.coordinator,
            kind=DetectorKind.HAND,
        )

        self.coordinator.start()

        self._build()
        self.water_cursor = WaterCursorOverlay(self.parent)
        self.gesture_mouse = GestureMouseController(
            self.parent,
            protected_widgets=(self.quit_button,),
            on_click=self._on_gesture_click,
            on_button=self.water_cursor.set_pressed,
            on_active=self.water_cursor.set_active,
        )
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

        ttk.Label(header, text="GestureFlow AI", font=("Segoe UI", 16, "bold")).grid(row=0, column=0, sticky="w")

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

        # Hand detection
        hand_frame = ttk.LabelFrame(controls, text="Hand Detection", padding=8)
        hand_frame.grid(row=0, column=1, sticky="nsew", padx=6)
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

        self.gesture_control_button = ttk.Button(
            hand_frame,
            text="Enable Gesture Control",
            command=self._toggle_gesture_control,
            state="disabled",
        )
        self.gesture_control_button.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        self.gesture_guide = ttk.Label(
            hand_frame,
            text="POINTING: index + thumb open; aim with index tip\nPINCH: hold/drag  ·  OPEN HAND: release + wrist-twist scroll\nTHUMBS UP: left-click  ·  CLOSED FIST: right-click",
            font=("Segoe UI", 8),
            foreground=MUTED,
            anchor="center",
            justify="center",
            wraplength=270,
        )
        self.gesture_guide.grid(row=3, column=0, sticky="ew", pady=(5, 0))
        self.gesture_control_info = ttk.Label(
            hand_frame,
            text="Enable gesture control to use the cursor across your desktop.",
            font=("Segoe UI", 8),
            foreground=MUTED,
            anchor="center",
            wraplength=240,
        )
        self.gesture_control_info.grid(row=4, column=0, sticky="ew", pady=(4, 0))

        # Quit button
        quit_frame = ttk.Frame(controls)
        quit_frame.grid(row=0, column=2, sticky="nsew", padx=(6, 0))
        quit_frame.columnconfigure(0, weight=1)
        self.quit_button = ttk.Button(quit_frame, text="Quit", command=self._on_close, width=10)
        self.quit_button.grid(row=0, column=0, sticky="e")

    def _toggle_gesture_control(self) -> None:
        if self.gesture_mouse.active:
            self.gesture_mouse.stop()
            self.gesture_control_button.configure(text="Enable Desktop Gesture Control")
            self.gesture_control_info.configure(
                text="Gesture control is off.",
                foreground=MUTED,
            )
            return

        if self.camera.snapshot().state != "on":
            self.gesture_control_info.configure(
                text="Turn on the camera and hand detection first.",
                foreground=DANGER,
            )
            return
        if not self.coordinator.is_enabled(DetectorKind.HAND):
            self._set_detector(DetectorKind.HAND, True)
        if self.gesture_mouse.start():
            self.gesture_control_button.configure(text="Disable Desktop Gesture Control")
            self.gesture_control_info.configure(
                text="Active across the desktop · show one clear hand to begin.",
                foreground="#1b7f3a",
            )
        else:
            self.gesture_control_info.configure(
                text="Gesture control currently requires Windows.",
                foreground=DANGER,
            )

    def _on_gesture_click(self, screen_point=None) -> None:
        if hasattr(self, "water_cursor"):
            self.water_cursor.splash(screen_point)
        water_surface = getattr(self, "water_backdrop", None)
        if water_surface is None:
            return
        try:
            if screen_point is None:
                water_surface.tap()
                return
            left, top = water_surface.winfo_rootx(), water_surface.winfo_rooty()
            x, y = screen_point[0] - left, screen_point[1] - top
            if 0 <= x < water_surface.winfo_width() and 0 <= y < water_surface.winfo_height():
                water_surface.tap(x, y)
        except Exception:
            water_surface.tap()

    def _toggle_camera(self) -> None:
        status = self.camera.snapshot()
        if status.state == "on":
            self.gesture_mouse.stop()
            self.camera.stop()
            self.hand_detection.set_enabled(False)
            self.coordinator.disable(DetectorKind.HAND)
        else:
            # Try to start the camera; if it fails, keep it off but refresh UI
            started = self.camera.start()
            if not started:
                self._refresh()
                return
            # Start only the gesture detector when the camera becomes ready.
            self.coordinator.enable(DetectorKind.HAND)
            self.hand_detection.set_enabled(True)
        self._refresh()

    def _set_detector(self, kind: DetectorKind, enable: bool) -> None:
        import traceback
        if kind != DetectorKind.HAND:
            return
        try:
            if enable:
                self.coordinator.enable(kind)
                self.hand_detection.set_enabled(True)
            else:
                self.coordinator.disable(kind)
                if kind == DetectorKind.HAND:
                    self.hand_detection.set_enabled(False)
                    self.gesture_mouse.stop()
        except Exception as e:
            print(f"[DIAG EXCEPTION] {kind.name} {enable=}: {e}")
            traceback.print_exc()
        self._refresh()

    def _on_close(self) -> None:
        self._closing = True
        self.gesture_mouse.stop()
        self.hand_detection.shutdown(timeout=1.0)
        self.coordinator.stop()
        self.camera.shutdown(timeout=2.0)
        self.parent.destroy()

    def _tick(self) -> None:
        try:
            self._refresh()
        except Exception as exc:
            self.hand_info.configure(text=f"UI error: {exc}", foreground="#b42318")
        finally:
            if not self._closing:
                self.after(33, self._tick)

    def _refresh(self) -> None:
        status = self.camera.snapshot()
        # Keep the camera preview on the newest capture frame while hand
        # inference runs at a lower rate to avoid overloading the machine.
        frame, results, synced = self.coordinator.compose(prefer_latest_frame=True)

        # GestureFlow AI composes only the hand detector with the camera frame.
        hand_result = results.get(DetectorKind.HAND)
        self._process_device_share_gestures(hand_result)

        if self.gesture_mouse.active:
            if status.state != "on" or not self.coordinator.is_enabled(DetectorKind.HAND):
                self.gesture_mouse.stop()
                self.gesture_control_button.configure(text="Enable Gesture Control")
                self.gesture_control_info.configure(
                    text="Gesture control stopped because the camera or hand detection is off.",
                    foreground=MUTED,
                )
            else:
                candidates = hand_result.hand_candidates if hand_result else ()
                tracks = hand_result.hand_visible_tracks if hand_result else ()
                message = self.gesture_mouse.update(
                    candidates or (),
                    tracks or (),
                    source_seq=hand_result.source_seq if hand_result else None,
                )
                color = "#1b7f3a" if message.startswith("Grab active") else ACCENT
                self.gesture_control_info.configure(text=message, foreground=color)

        # Stop workers if camera is off
        if status.state != "on":
            if self.coordinator.is_enabled(DetectorKind.HAND):
                self.coordinator.disable(DetectorKind.HAND)

        # Update display status
        self._update_hand_display(hand_result)
        self._set_status_display(
            status.state,
            status.message,
            status.width,
            status.height,
            status.fps,
        )
        if status.state == "on" and frame is not None:
            display = (
                self.camera_noise_cleaner.apply(frame)
                if self.preview_noise_cleaner_enabled else frame
            )
            if hand_result and hand_result.hand_candidates is not None:
                try:
                    candidates = hand_result.hand_candidates or ()
                    tracks = hand_result.hand_visible_tracks or ()
                    display = self._draw_hand_overlays(
                        display, candidates, tracks, copy_frame=False
                    )
                except Exception:
                    pass
            self._show_frame(display)
            return
        self._show_placeholder()

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
            tracks = hand_result.hand_visible_tracks or ()
            gestures = []
            for index, candidate in enumerate(candidates):
                track = tracks[index] if index < len(tracks) else None
                label = getattr(track, "stable_gesture", "UNKNOWN") if track else "UNKNOWN"
                if label == "UNKNOWN":
                    label = getattr(candidate, "gesture", "UNKNOWN")
                if label != "UNKNOWN" and label not in gestures:
                    gestures.append(str(label))
            gesture_text = f" · {', '.join(gestures)}" if gestures else " · recognizing…"
            worker_state = self.hand_detection.snapshot()
            fps_text = (
                f" · {worker_state.processing_fps:.0f} FPS"
                if worker_state.processing_fps > 0
                else ""
            )
            text = f"Detected hands: {len(candidates)}{gesture_text}{fps_text}"
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
        self.gesture_control_button.configure(
            state="normal" if camera_on else "disabled",
            text="Disable Desktop Gesture Control" if self.gesture_mouse.active else "Enable Desktop Gesture Control",
        )

    def _process_device_share_gestures(self, hand_result) -> None:
        """Hook for opt-in handoff windows provided by the liquid-glass UI."""

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
        if self._photo is not self._placeholder:
            self.preview_label.configure(image=self._placeholder)
        self._photo = self._placeholder
        self._photo_size = size

    def _show_frame(self, frame) -> None:
        width, height = self._preview_size()
        source_height, source_width = frame.shape[:2]
        scale = min(width / source_width, height / source_height)
        display_width = max(1, int(source_width * scale))
        display_height = max(1, int(source_height * scale))
        if (display_width, display_height) != (source_width, source_height):
            frame = cv2.resize(
                frame,
                (display_width, display_height),
                interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
            )
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        fitted = Image.new("RGB", (width, height), VIDEO_BG)
        fitted.paste(image, ((width - display_width) // 2, (height - display_height) // 2))
        size = (width, height)
        if self._photo is None or self._photo is self._placeholder or self._photo_size != size:
            self._photo = ImageTk.PhotoImage(fitted)
            self.preview_label.configure(image=self._photo)
            self._photo_size = size
        else:
            # Reuse the Tk image object between frames to reduce allocations
            # and avoid small preview flashes while the live feed is updating.
            self._photo.paste(fitted)

    def _draw_hand_overlays(self, frame, candidates, tracks, *, copy_frame: bool = True):
        """Draw hand candidates/tracks on the composed UI frame."""
        annotated = frame.copy() if copy_frame else frame
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

def main() -> None:
    _enable_windows_dpi()
    root = tk.Tk()
    root.title("GestureFlow AI")
    root.geometry("1180x760")
    root.minsize(1020, 680)
    root.configure(bg=PREVIEW_BG)
    window = MainWindow(root)
    window.pack(fill="both", expand=True)
    root.protocol("WM_DELETE_WINDOW", window._on_close)
    root.mainloop()
