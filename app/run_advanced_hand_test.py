"""Standalone Phase 3 hand detection test window.

Run from the project root with: python -m app.run_advanced_hand_test
"""

from __future__ import annotations

import time
import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

from app.camera.capture import CameraCapture
from app.hand_gesture_advanced import AdvancedHandDetectionWorker

PREVIEW_WIDTH = 800
PREVIEW_HEIGHT = 450
BACKGROUND = "#12181f"


class AdvancedHandTestWindow(ttk.Frame):
    def __init__(self, master: tk.Tk) -> None:
        super().__init__(master, padding=12)
        self.master = master
        self.camera = CameraCapture()
        self.detector = AdvancedHandDetectionWorker(self.camera.get_frame)
        self._closing = False
        self._detection_enabled = False
        self._photo: ImageTk.PhotoImage | None = None
        self._last_refresh_time: float | None = None
        self._display_fps = 0.0

        self._build()
        self.camera.start()
        self.after(33, self._refresh)

    def _build(self) -> None:
        self.master.title("Gesture.AI - Advanced Hand Test")
        self.master.geometry("1160x650")
        self.master.minsize(1000, 560)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.preview_container = tk.Frame(
            self,
            width=PREVIEW_WIDTH,
            height=PREVIEW_HEIGHT,
            background=BACKGROUND,
        )
        self.preview_container.grid(
            row=0,
            column=0,
            sticky="nw",
            padx=(0, 14),
        )
        self.preview_container.grid_propagate(False)
        self.preview = tk.Label(
            self.preview_container,
            text="Waiting for camera...",
            background=BACKGROUND,
            foreground="#ffffff",
            anchor="center",
        )
        self.preview.pack(fill="both", expand=True)

        controls = ttk.Frame(self, padding=8)
        controls.grid(row=0, column=1, sticky="n")
        ttk.Label(
            controls,
            text="Phase 3 hand test",
            font=("Segoe UI", 13, "bold"),
        ).grid(row=0, column=0, sticky="w", pady=(0, 12))

        self.camera_status = ttk.Label(
            controls,
            text="Camera: starting",
            wraplength=265,
        )
        self.camera_status.grid(row=1, column=0, sticky="w", pady=(0, 10))
        self.camera_button = ttk.Button(
            controls,
            text="Retry Camera",
            command=self._retry_camera,
        )
        self.camera_button.grid(row=2, column=0, sticky="ew", pady=(0, 16))

        self.detection_button = ttk.Button(
            controls,
            text="Turn Hand Detection ON",
            command=self._toggle_detection,
        )
        self.detection_button.grid(row=3, column=0, sticky="ew")
        self.hand_count = ttk.Label(
            controls,
            text="Detected hands: 0",
            font=("Segoe UI", 12, "bold"),
        )
        self.hand_count.grid(row=4, column=0, sticky="w", pady=(14, 4))
        self.detection_status = ttk.Label(
            controls,
            text="Hand detection is off.",
            wraplength=265,
        )
        self.detection_status.grid(row=5, column=0, sticky="w", pady=(0, 8))
        self.camera_rate = ttk.Label(controls, text="Camera capture rate: —")
        self.camera_rate.grid(row=6, column=0, sticky="w")
        self.processing_rate = ttk.Label(controls, text="Detection processing rate: —")
        self.processing_rate.grid(row=7, column=0, sticky="w")
        self.display_rate = ttk.Label(controls, text="UI display rate: —")
        self.display_rate.grid(row=8, column=0, sticky="w", pady=(0, 16))

        ttk.Label(
            controls,
            text=(
                "Each gesture is labeled beside its hand. Person labels are "
                "shown only when pose association is confident. Close the main "
                "app before testing to avoid webcam conflicts."
            ),
            wraplength=265,
        ).grid(row=9, column=0, sticky="w")
        controls.columnconfigure(0, weight=1)

    def _retry_camera(self) -> None:
        status = self.camera.snapshot()
        if status.state in ("off", "error"):
            if not self.camera.start():
                self.camera_status.configure(
                    text="Camera is still stopping; try again shortly."
                )

    def _toggle_detection(self) -> None:
        if self._detection_enabled:
            self._detection_enabled = False
            self.detector.set_enabled(False)
        else:
            if self.camera.snapshot().state != "on":
                self.detection_status.configure(
                    text="Turn on the camera before enabling hand detection."
                )
                return
            self._detection_enabled = True
            self.detector.set_enabled(True)

    def _refresh(self) -> None:
        if self._closing:
            return

        now = time.perf_counter()
        if self._last_refresh_time is not None:
            interval = now - self._last_refresh_time
            if interval > 0:
                measured = 1.0 / interval
                self._display_fps = (
                    measured
                    if self._display_fps == 0
                    else self._display_fps * 0.8 + measured * 0.2
                )
        self._last_refresh_time = now

        camera_status = self.camera.snapshot()
        detection = self.detector.snapshot()
        if camera_status.state != "on" and self._detection_enabled:
            self._detection_enabled = False
            self.detector.set_enabled(False)
            detection = self.detector.snapshot()

        if camera_status.state == "on":
            self.camera_status.configure(text="Camera: ON")
            self.camera_button.configure(state="disabled")
        elif camera_status.state == "starting":
            self.camera_status.configure(text="Camera: starting...")
            self.camera_button.configure(state="disabled")
        elif camera_status.state == "stopping":
            self.camera_status.configure(text="Camera: stopping...")
            self.camera_button.configure(state="disabled")
        elif camera_status.state == "error":
            self.camera_status.configure(
                text=f"Camera error: {camera_status.message}"
            )
            self.camera_button.configure(state="normal")
        else:
            self.camera_status.configure(text="Camera: OFF")
            self.camera_button.configure(state="normal")

        if detection.state == "error":
            detection_text = detection.message
        elif not self._detection_enabled:
            detection_text = "Hand detection is off."
        elif detection.state == "starting":
            detection_text = detection.message or "Loading hand model..."
        elif detection.message.startswith("Person association unavailable"):
            detection_text = detection.message
        else:
            detection_text = "Hand detection is running."
        self.detection_status.configure(text=detection_text)
        self.detection_button.configure(
            text=(
                "Turn Hand Detection OFF"
                if self._detection_enabled
                else "Turn Hand Detection ON"
            ),
            state=(
                "normal"
                if camera_status.state == "on"
                else "disabled"
            ),
        )
        self.hand_count.configure(
            text=f"Detected hands: {detection.hand_count if self._detection_enabled else 0}"
        )
        if camera_status.state == "on" and camera_status.fps > 0:
            self.camera_rate.configure(
                text=f"Camera capture rate: {camera_status.fps:.1f} FPS"
            )
        else:
            self.camera_rate.configure(text="Camera capture rate: —")
        if self._detection_enabled and detection.processing_fps > 0:
            self.processing_rate.configure(
                text=(
                    "Detection processing rate: "
                    f"{detection.processing_fps:.1f} FPS"
                )
            )
        else:
            self.processing_rate.configure(text="Detection processing rate: —")
        if self._display_fps > 0:
            self.display_rate.configure(
                text=f"UI display rate: {self._display_fps:.1f} FPS"
            )
        else:
            self.display_rate.configure(text="UI display rate: —")

        if camera_status.state == "on":
            if self._detection_enabled and detection.frame is not None:
                preview_frame = detection.frame
            else:
                frame = self.camera.get_frame()
                preview_frame = cv2.flip(frame, 1) if frame is not None else None
            if preview_frame is not None:
                self._show_frame(preview_frame)
        else:
            self._photo = None
            self.preview.configure(
                image="",
                text=(
                    camera_status.message
                    if camera_status.state == "error"
                    else "Camera is off"
                ),
            )

        self.after(33, self._refresh)

    def _show_frame(self, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        scale = min(
            PREVIEW_WIDTH / image.width,
            PREVIEW_HEIGHT / image.height,
        )
        resized = image.resize(
            (
                max(1, int(image.width * scale)),
                max(1, int(image.height * scale)),
            ),
            Image.Resampling.BILINEAR,
        )
        canvas = Image.new("RGB", (PREVIEW_WIDTH, PREVIEW_HEIGHT), BACKGROUND)
        canvas.paste(
            resized,
            (
                (PREVIEW_WIDTH - resized.width) // 2,
                (PREVIEW_HEIGHT - resized.height) // 2,
            ),
        )
        photo = ImageTk.PhotoImage(canvas)
        self.preview.configure(image=photo, text="")
        self._photo = photo

    def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self.detector.shutdown()
        self.camera.shutdown()
        self.master.destroy()


def main() -> None:
    root = tk.Tk()
    window = AdvancedHandTestWindow(root)
    window.grid(row=0, column=0, sticky="nsew")
    root.protocol("WM_DELETE_WINDOW", window.close)
    root.mainloop()


if __name__ == "__main__":
    main()
