"""HOME GUARD AI Phase 1 window: camera toggle, preview, and status."""

from __future__ import annotations

import ctypes
import os
import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageDraw, ImageFont, ImageTk

from app.camera.capture import REQUESTED_FPS, REQUESTED_HEIGHT, REQUESTED_WIDTH, CameraCapture
from app.face_recognition_worker import FaceRecognitionWorker
from app.hand_gesture_advanced import AdvancedHandDetectionWorker
from app.human_detection import HumanDetectionWorker

BG = "#f4f7fb"
CARD = "#ffffff"
HEADER = "#123a63"
BORDER = "#d5deea"
TEXT = "#1b2838"
MUTED = "#5d6b7c"
ACCENT = "#1565c0"
PREVIEW_BG = "#e7eef6"
VIDEO_BG = "#12181f"
OFF_MESSAGE = "CAMERA IS OFF — ENABLE CAMERA TO VIEW"
FACE_BOX_COLOR = (0, 200, 255)
FACE_LABEL_COLOR = (0, 200, 255)

FEATURES = (
    "Hand Tracking",
    "Ball Tracking",
    "Video Recording",
)

STATE_TEXT = {
    "off": "OFF",
    "starting": "STARTING",
    "on": "ON",
    "stopping": "TURNING OFF",
    "error": "ERROR",
}

STATE_COLOR = {
    "off": MUTED,
    "starting": ACCENT,
    "on": "#1b7f3a",
    "stopping": ACCENT,
    "error": "#b42318",
}


def main() -> None:
    _enable_windows_dpi()
    root = tk.Tk()
    root.title("HOME GUARD AI")
    root.geometry("1180x760")
    root.minsize(1020, 680)
    root.configure(bg=BG)
    window = MainWindow(root)
    window.pack(fill="both", expand=True)
    root.protocol("WM_DELETE_WINDOW", window.close)
    root.mainloop()


class MainWindow(ttk.Frame):
    def __init__(self, master: tk.Tk) -> None:
        super().__init__(master)
        self.camera = CameraCapture()
        self.human_detection = HumanDetectionWorker(self.camera.get_frame)
        self.face_recognition = FaceRecognitionWorker(self.camera.get_frame)
        self.hand_detection = AdvancedHandDetectionWorker(self.camera.get_frame)
        self._closing = False
        self._pending_start = False
        self._photo: ImageTk.PhotoImage | None = None
        self._placeholder_size: tuple[int, int] | None = None
        self._placeholder: ImageTk.PhotoImage | None = None
        self._tick_error: str | None = None

        self._apply_styles()
        self._build()
        self._show_placeholder()
        self.after(33, self._tick)

    def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._pending_start = False
        # Workers first, then camera — face, hand and human share the camera owner.
        self.hand_detection.shutdown()
        self.face_recognition.shutdown()
        self.human_detection.shutdown()
        self.camera.shutdown()
        self.master.destroy()

    def _apply_styles(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background=BG, font=("Segoe UI", 10))
        style.configure("App.TFrame", background=BG)
        style.configure("Card.TFrame", background=CARD)
        style.configure("Header.TFrame", background=HEADER)
        style.configure(
            "HeaderTitle.TLabel",
            background=HEADER,
            foreground="#ffffff",
            font=("Segoe UI", 22, "bold"),
        )
        style.configure(
            "HeaderSub.TLabel",
            background=HEADER,
            foreground="#d5e4f2",
            font=("Segoe UI", 10),
        )
        style.configure(
            "Section.TLabel",
            background=CARD,
            foreground=TEXT,
            font=("Segoe UI", 12, "bold"),
        )
        style.configure(
            "Key.TLabel",
            background=CARD,
            foreground=MUTED,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Value.TLabel",
            background=CARD,
            foreground=TEXT,
            font=("Segoe UI", 10, "bold"),
        )
        style.configure(
            "Notice.TLabel",
            background=CARD,
            foreground=MUTED,
            font=("Segoe UI", 9),
        )
        style.configure(
            "Feature.TLabel",
            background=CARD,
            foreground=TEXT,
            font=("Segoe UI", 10),
        )
        style.configure(
            "Coming.TLabel",
            background=CARD,
            foreground="#8b98a8",
            font=("Segoe UI", 9),
        )
        style.configure("Card.TSeparator", background="#e6edf5")
        style.configure(
            "Primary.TButton",
            background=ACCENT,
            foreground="#ffffff",
            font=("Segoe UI", 11, "bold"),
            padding=(12, 10),
            borderwidth=0,
            focuscolor=ACCENT,
        )
        style.map(
            "Primary.TButton",
            background=[
                ("pressed", "#0d47a1"),
                ("active", "#0d5cb3"),
                ("disabled", "#b7c3d1"),
            ],
            foreground=[("disabled", "#f7f9fb"), ("!disabled", "#ffffff")],
        )
        self.configure(style="App.TFrame")

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        header = ttk.Frame(self, style="Header.TFrame", padding=(24, 16))
        header.grid(row=0, column=0, sticky="ew")
        ttk.Label(header, text="HOME GUARD AI", style="HeaderTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            header,
            text="Live camera preview",
            style="HeaderSub.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        body = ttk.Frame(self, style="App.TFrame", padding=16)
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)

        preview_shell = tk.Frame(body, bg=BORDER, bd=0, highlightthickness=0)
        preview_shell.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        preview_card = ttk.Frame(preview_shell, style="Card.TFrame", padding=12)
        preview_card.pack(fill="both", expand=True, padx=1, pady=1)
        preview_card.rowconfigure(0, weight=1)
        preview_card.columnconfigure(0, weight=1)

        self.preview_label = tk.Label(
            preview_card,
            bg=PREVIEW_BG,
            bd=0,
            highlightthickness=0,
            anchor="center",
        )
        self.preview_label.grid(row=0, column=0, sticky="nsew")

        side_scroll = tk.Canvas(body, bg=BG, highlightthickness=0)
        side_scroll.grid(row=0, column=1, sticky="nsew")
        side_scroll.columnconfigure(0, weight=1)

        side_scrollbar = ttk.Scrollbar(body, orient="vertical", command=side_scroll.yview)
        side_scrollbar.grid(row=0, column=2, sticky="ns")

        side = ttk.Frame(side_scroll, style="App.TFrame")
        side.columnconfigure(0, weight=1)

        side_scroll.create_window((0, 0), window=side, anchor="nw", tags="side_frame")

        def _configure_scroll_region(event):
            side_scroll.configure(scrollregion=side_scroll.bbox("all"))

        side.bind("<Configure>", _configure_scroll_region)

        def _on_mousewheel(event):
            side_scroll.yview_scroll(int(-1 * (event.delta / 120)), "units")

        side_scroll.bind_all("<MouseWheel>", _on_mousewheel)

        side_scroll.bind("<Configure>", lambda e: side_scroll.itemconfig("side_frame", width=e.width))

        camera_shell, camera_card = self._card(side)
        camera_shell.grid(row=0, column=0, sticky="ew")
        ttk.Label(camera_card, text="Camera", style="Section.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            camera_card,
            text="Turn the webcam on to show the live feed in this window.",
            style="Notice.TLabel",
            wraplength=270,
        ).grid(row=1, column=0, sticky="ew", pady=(4, 12))
        self.toggle_button = ttk.Button(
            camera_card,
            text="Turn Camera ON",
            style="Primary.TButton",
            command=self._toggle_camera,
        )
        self.toggle_button.grid(row=2, column=0, sticky="ew")

        status_shell, status_card = self._card(side)
        status_shell.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        ttk.Label(status_card, text="Status", style="Section.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )
        self.state_value = self._status_row(status_card, 1, "Camera")
        self.resolution_value = self._status_row(status_card, 2, "Resolution")
        self.fps_value = self._status_row(status_card, 3, "Frame rate")
        self.notice_value = ttk.Label(
            status_card,
            text="",
            style="Notice.TLabel",
            wraplength=270,
        )
        self.notice_value.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        features_shell, features_card = self._card(side)
        features_shell.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        ttk.Label(features_card, text="Features", style="Section.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        ttk.Label(
            features_card,
            text="Turn the camera on before starting pose-based detection.",
            style="Notice.TLabel",
            wraplength=270,
        ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 10))

        self.start_detection_button = ttk.Button(
            features_card,
            text="Start Human Detection",
            style="Primary.TButton",
            command=self._start_human_detection,
        )
        self.start_detection_button.grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(0, 6)
        )
        self.stop_detection_button = ttk.Button(
            features_card,
            text="Stop Human Detection",
            command=self._stop_human_detection,
        )
        self.stop_detection_button.grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=(0, 8)
        )

        # Hand detection buttons
        self.start_hand_button = ttk.Button(
            features_card,
            text="Start Hand Detection",
            style="Primary.TButton",
            command=self._start_hand_detection,
        )
        self.start_hand_button.grid(
            row=8, column=0, columnspan=2, sticky="ew", pady=(0, 6)
        )
        self.stop_hand_button = ttk.Button(
            features_card,
            text="Stop Hand Detection",
            command=self._stop_hand_detection,
        )
        self.stop_hand_button.grid(
            row=9, column=0, columnspan=2, sticky="ew", pady=(0, 8)
        )

        self.detection_info = ttk.Label(
            features_card,
            text="Human detection is off.",
            style="Notice.TLabel",
            wraplength=270,
        )
        self.detection_info.grid(
            row=4, column=0, columnspan=2, sticky="ew", pady=(0, 8)
        )

        # Hand detection labels
        self.hand_count = ttk.Label(
            features_card,
            text="Detected hands: 0",
            font=("Segoe UI", 12, "bold"),
        )
        self.hand_count.grid(
            row=10, column=0, sticky="w", pady=(0, 4)
        )
        self.hand_info = ttk.Label(
            features_card,
            text="Hand detection is off.",
            style="Notice.TLabel",
            wraplength=270,
        )
        self.hand_info.grid(
            row=11, column=0, columnspan=2, sticky="ew", pady=(0, 8)
        )

        self.start_face_button = ttk.Button(
            features_card,
            text="Start Face Recognition",
            style="Primary.TButton",
            command=self._start_face_recognition,
        )
        self.start_face_button.grid(
            row=5, column=0, columnspan=2, sticky="ew", pady=(0, 6)
        )
        self.stop_face_button = ttk.Button(
            features_card,
            text="Stop Face Recognition",
            command=self._stop_face_recognition,
        )
        self.stop_face_button.grid(
            row=6, column=0, columnspan=2, sticky="ew", pady=(0, 8)
        )
        self.face_info = ttk.Label(
            features_card,
            text="Face recognition is off.",
            style="Notice.TLabel",
            wraplength=270,
        )
        self.face_info.grid(
            row=7, column=0, columnspan=2, sticky="ew", pady=(0, 8)
        )

        for index, name in enumerate(FEATURES):
            row = 12 + index
            ttk.Label(features_card, text=name, style="Feature.TLabel").grid(
                row=row, column=0, sticky="w", pady=4
            )
            ttk.Label(features_card, text="Not started", style="Coming.TLabel").grid(
                row=row, column=1, sticky="e", padx=(12, 0), pady=4
            )
        features_card.columnconfigure(0, weight=1)

        self._set_status_display("off", "", 0, 0, 0.0)

    def _card(self, parent: ttk.Frame) -> tuple[tk.Frame, ttk.Frame]:
        shell = tk.Frame(parent, bg=BORDER, bd=0, highlightthickness=0)
        card = ttk.Frame(shell, style="Card.TFrame", padding=14)
        card.pack(fill="both", expand=True, padx=1, pady=1)
        card.columnconfigure(0, weight=1)
        return shell, card

    def _status_row(self, parent: ttk.Frame, row: int, label: str) -> ttk.Label:
        ttk.Label(parent, text=label, style="Key.TLabel").grid(
            row=row, column=0, sticky="w", pady=3
        )
        value = ttk.Label(parent, text="—", style="Value.TLabel")
        value.grid(row=row, column=1, sticky="e", pady=3)
        parent.columnconfigure(1, weight=1)
        return value

    def _toggle_camera(self) -> None:
        status = self.camera.snapshot()
        if status.state in ("off", "error"):
            self._pending_start = not self.camera.start()
        elif status.state in ("starting", "on"):
            self._pending_start = False
            self.face_recognition.request_stop()
            self.human_detection.request_stop()
            self.hand_detection.set_enabled(False)
            self.camera.request_stop()
            self._show_placeholder()

    def _start_human_detection(self) -> None:
        if self.camera.snapshot().state == "on":
            self.human_detection.start()

    def _stop_human_detection(self) -> None:
        self.human_detection.request_stop()

    def _start_face_recognition(self) -> None:
        if self.camera.snapshot().state == "on":
            self.face_recognition.start()

    def _stop_face_recognition(self) -> None:
        self.face_recognition.request_stop()

    def _start_hand_detection(self) -> None:
        if self.camera.snapshot().state == "on":
            self.hand_detection.set_enabled(True)

    def _stop_hand_detection(self) -> None:
        self.hand_detection.set_enabled(False)

    def _tick(self) -> None:
        if self._closing:
            return
        try:
            if self._pending_start and self.camera.start():
                self._pending_start = False
            self._refresh()
        except Exception as exc:
            self._tick_error = str(exc)
            self.hand_info.configure(text=f"UI error: {exc}", foreground="#b42318")
        finally:
            if not self._closing:
                self.after(33, self._tick)

    def _refresh(self) -> None:
        status = self.camera.snapshot()
        detection = self.human_detection.snapshot()
        faces = self.face_recognition.snapshot()
        hand = self.hand_detection.snapshot()
        if status.state != "on" and detection.state in ("starting", "on"):
            self.human_detection.request_stop()
            detection = self.human_detection.snapshot()
        if status.state != "on" and faces.state in ("starting", "running"):
            self.face_recognition.request_stop()
            faces = self.face_recognition.snapshot()
        if status.state != "on" and hand.state in ("starting", "on"):
            self.hand_detection.set_enabled(False)
            hand = self.hand_detection.snapshot()
        self._update_detection_display(status.state, detection)
        self._update_face_display(status.state, faces)
        self._update_hand_display(hand)
        self._set_status_display(
            status.state,
            status.message,
            status.width,
            status.height,
            status.fps,
        )
        if status.state == "on":
            frame = self.camera.get_frame()
            if frame is not None:
                preview_frame = detection.frame if detection.state == "on" else None
                display = preview_frame if preview_frame is not None else frame
                if faces.state == "running" and faces.faces:
                    display = _draw_face_overlays(display, faces.faces)
                if hand.state == "on" and hand.frame is not None:
                    display = hand.frame.copy()
                self._show_frame(display)
            return
        self._show_placeholder()

    def _update_detection_display(self, camera_state: str, detection) -> None:
        if detection.state == "error":
            text = detection.message
            color = "#b42318"
        elif detection.state == "starting":
            text = "Starting human detection..."
            color = ACCENT
        elif detection.state == "stopping":
            text = "Stopping human detection..."
            color = ACCENT
        elif detection.state == "on":
            text = (
                f"{'HUMAN DETECTED' if detection.people else 'NO HUMAN DETECTED'} "
                f"— People: {detection.people}"
            )
            color = "#1b7f3a" if detection.people else MUTED
        elif camera_state == "on":
            text = "Human detection is off."
            color = MUTED
        else:
            text = "Turn the camera on before starting detection."
            color = MUTED
        self.detection_info.configure(text=text, foreground=color)

        can_start = camera_state == "on" and detection.state in ("off", "error")
        can_stop = detection.state in ("starting", "on")
        self.start_detection_button.configure(
            state="normal" if can_start else "disabled"
        )
        self.stop_detection_button.configure(
            state="normal" if can_stop else "disabled"
        )

    def _update_face_display(self, camera_state: str, faces) -> None:
        if faces.state == "error":
            text = faces.message
            color = "#b42318"
        elif faces.state == "starting":
            text = "Starting face recognition..."
            color = ACCENT
        elif faces.state == "stopping":
            text = "Stopping face recognition..."
            color = ACCENT
        elif faces.state == "running":
            count = len(faces.faces)
            text = f"Faces: {count} — all labeled UNKNOWN"
            color = "#1b7f3a" if count else MUTED
        elif camera_state == "on":
            text = "Face recognition is off."
            color = MUTED
        else:
            text = "Turn the camera on before starting face recognition."
            color = MUTED
        self.face_info.configure(text=text, foreground=color)

        can_start = camera_state == "on" and faces.state in ("off", "error")
        can_stop = faces.state in ("starting", "running")
        self.start_face_button.configure(
            state="normal" if can_start else "disabled"
        )
        self.stop_face_button.configure(
            state="normal" if can_stop else "disabled"
        )

    def _update_hand_display(self, hand) -> None:
        if hand.state == "error":
            text = hand.message
            color = "#b42318"
        elif hand.state == "starting":
            text = "Starting hand detection..."
            color = ACCENT
        elif hand.state == "stopping":
            text = "Stopping hand detection..."
            color = ACCENT
        elif hand.state == "on":
            text = f"Detected hands: {hand.hand_count}"
            color = "#1b7f3a" if hand.hand_count > 0 else MUTED
        elif self.camera.snapshot().state == "on":
            text = "Hand detection is off."
            color = MUTED
        else:
            text = "Turn the camera on before starting hand detection."
            color = MUTED
        self.hand_info.configure(text=text, foreground=color)

        can_start = self.camera.snapshot().state == "on" and hand.state in ("off", "error")
        can_stop = hand.state in ("starting", "on")
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
            text=STATE_TEXT.get(state, state.upper()),
            foreground=STATE_COLOR.get(state, TEXT),
        )
        if state == "on" and width > 0 and height > 0:
            self.resolution_value.configure(text=f"{width} × {height}")
        else:
            self.resolution_value.configure(text="—")
        if state == "on" and fps > 0:
            self.fps_value.configure(text=f"{fps:.0f} FPS")
        else:
            self.fps_value.configure(text="—")

        requested = (
            f"Requested {REQUESTED_WIDTH} × {REQUESTED_HEIGHT} at {REQUESTED_FPS} FPS."
        )
        if state == "error" and message:
            notice = message
        elif message and state != "off":
            notice = message
        else:
            notice = requested
        self.notice_value.configure(text=notice)

        if state == "stopping":
            self.toggle_button.configure(state="disabled", text="Turning Camera OFF")
        elif state in ("starting", "on"):
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


def _draw_face_overlays(frame, faces):
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


def _placeholder_image(width: int, height: int) -> Image.Image:
    image = Image.new("RGB", (width, height), PREVIEW_BG)
    draw = ImageDraw.Draw(image)
    center_x = width // 2
    center_y = max(90, height // 2 - 36)
    _draw_camera_icon(draw, center_x, center_y)

    font = _fit_font(draw, OFF_MESSAGE, width - 48, 22)
    bbox = draw.textbbox((0, 0), OFF_MESSAGE, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    text_x = (width - text_width) // 2
    text_y = min(height - text_height - 24, center_y + 78)
    draw.text((text_x, text_y), OFF_MESSAGE, font=font, fill="#3d4d60")
    return image


def _draw_camera_icon(draw: ImageDraw.ImageDraw, center_x: int, center_y: int) -> None:
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
    font = _font(size)
    while size > 12:
        bbox = draw.textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= max_width:
            return font
        size -= 1
        font = _font(size)
    return font


def _font(size: int) -> ImageFont.ImageFont:
    windows = os.environ.get("WINDIR", r"C:\Windows")
    path = os.path.join(windows, "Fonts", "segoeui.ttf")
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def _enable_windows_dpi() -> None:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass

def _draw_hand_overlays(frame, hand_info_text: str) -> np.ndarray:
    """Draw hand count overlay on frame."""
    import cv2
    import numpy as np
    annotated = frame.copy()
    cv2.putText(
        annotated,
        hand_info_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    return annotated

def _enable_windows_dpi() -> None:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass
