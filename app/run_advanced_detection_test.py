"""Run a standalone preview for testing the advanced human detector.

Launch from the project root with: python -m app.run_advanced_detection_test
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

from app.camera.capture import CameraCapture
from app.human_detection_advanced import AdvancedHumanDetectionWorker

PREVIEW_WIDTH = 720
PREVIEW_HEIGHT = 405


class AdvancedDetectionTestWindow(ttk.Frame):
    def __init__(self, master: tk.Tk) -> None:
        super().__init__(master, padding=12)
        self.master = master
        self.camera = CameraCapture()
        self.detector = AdvancedHumanDetectionWorker(self.camera.get_frame)
        self._closing = False
        self._photo: ImageTk.PhotoImage | None = None

        self._build()
        self.camera.start()
        self.after(33, self._refresh)

    def _build(self) -> None:
        self.master.title("Advanced Human Detection Test")
        self.master.geometry("1100x700")
        self.master.minsize(800, 520)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        preview_container = tk.Frame(
            self,
            width=PREVIEW_WIDTH,
            height=PREVIEW_HEIGHT,
            background="#12181f",
        )
        preview_container.grid(
            row=0,
            column=0,
            sticky="nw",
            padx=(0, 12),
        )
        preview_container.grid_propagate(False)
        self.preview = tk.Label(
            preview_container,
            text="Waiting for camera...",
            background="#12181f",
            foreground="#ffffff",
            anchor="center",
        )
        self.preview.pack(fill="both", expand=True)

        controls = ttk.Frame(self, padding=8)
        controls.grid(row=0, column=1, sticky="n")
        ttk.Label(
            controls,
            text="Standalone detector test",
            font=("Segoe UI", 13, "bold"),
        ).grid(row=0, column=0, sticky="w", pady=(0, 12))

        self.camera_state = ttk.Label(controls, text="Camera: starting")
        self.camera_state.grid(row=1, column=0, sticky="w", pady=(0, 8))
        camera_buttons = ttk.Frame(controls)
        camera_buttons.grid(row=2, column=0, sticky="ew", pady=(0, 16))
        self.camera_on_button = ttk.Button(
            camera_buttons,
            text="Camera ON",
            command=self._start_camera,
        )
        self.camera_on_button.grid(row=0, column=0, sticky="ew")
        self.camera_off_button = ttk.Button(
            camera_buttons,
            text="Camera OFF",
            command=self._stop_camera,
        )
        self.camera_off_button.grid(row=1, column=0, sticky="ew", pady=(6, 0))

        self.detection_state = ttk.Label(
            controls,
            text="Confirmed people: 0\nDetection is off.",
            justify="left",
            wraplength=250,
        )
        self.detection_state.grid(row=3, column=0, sticky="w", pady=(0, 8))
        self.detection_on_button = ttk.Button(
            controls,
            text="Start Human Detection",
            command=self._start_detection,
        )
        self.detection_on_button.grid(row=4, column=0, sticky="ew")
        self.detection_off_button = ttk.Button(
            controls,
            text="Stop Human Detection",
            command=self._stop_detection,
        )
        self.detection_off_button.grid(row=5, column=0, sticky="ew", pady=(6, 0))

        ttk.Label(
            controls,
            text="Close the main app before opening this test so it can use the webcam.",
            wraplength=250,
        ).grid(row=6, column=0, sticky="w", pady=(18, 0))
        controls.columnconfigure(0, weight=1)
        camera_buttons.columnconfigure(0, weight=1)

    def _start_camera(self) -> None:
        status = self.camera.snapshot()
        if status.state in ("off", "error"):
            if not self.camera.start():
                self.camera_state.configure(
                    text="Camera is still stopping; try again shortly."
                )

    def _stop_camera(self) -> None:
        self.detector.request_stop()
        self.camera.request_stop()

    def _start_detection(self) -> None:
        if self.camera.snapshot().state == "on":
            self.detector.start()

    def _stop_detection(self) -> None:
        self.detector.request_stop()

    def _refresh(self) -> None:
        if self._closing:
            return

        camera_status = self.camera.snapshot()
        detection = self.detector.snapshot()
        if camera_status.state != "on" and detection.state in ("starting", "on"):
            self.detector.request_stop()
            detection = self.detector.snapshot()

        if camera_status.state == "on":
            self.camera_state.configure(text="Camera: ON")
        elif camera_status.state == "starting":
            self.camera_state.configure(text="Camera: starting")
        elif camera_status.state == "stopping":
            self.camera_state.configure(text="Camera: stopping")
        elif camera_status.state == "error":
            self.camera_state.configure(
                text=f"Camera error: {camera_status.message}"
            )
        else:
            self.camera_state.configure(text="Camera: OFF")

        if detection.state == "error":
            detection_text = f"Detection error: {detection.message}"
        elif detection.state == "on":
            detection_text = (
                f"Confirmed people: {detection.people}\n"
                f"{'HUMAN DETECTED' if detection.people else 'NO HUMAN DETECTED'}"
            )
        elif detection.state == "starting":
            detection_text = "Starting human detection..."
        elif detection.state == "stopping":
            detection_text = "Stopping human detection..."
        else:
            detection_text = "Confirmed people: 0\nDetection is off."
        self.detection_state.configure(text=detection_text)

        self.camera_on_button.configure(
            state=(
                "normal"
                if camera_status.state in ("off", "error")
                else "disabled"
            )
        )
        self.camera_off_button.configure(
            state=(
                "normal"
                if camera_status.state in ("starting", "on")
                else "disabled"
            )
        )
        self.detection_on_button.configure(
            state=(
                "normal"
                if camera_status.state == "on"
                and detection.state in ("off", "error")
                else "disabled"
            )
        )
        self.detection_off_button.configure(
            state=(
                "normal"
                if detection.state in ("starting", "on")
                else "disabled"
            )
        )

        if camera_status.state == "on":
            frame = detection.frame if detection.state == "on" else None
            if frame is None:
                frame = self.camera.get_frame()
            if frame is not None:
                self._show_frame(frame)
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
        photo = ImageTk.PhotoImage(resized)
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
    window = AdvancedDetectionTestWindow(root)
    window.grid(row=0, column=0, sticky="nsew")
    root.protocol("WM_DELETE_WINDOW", window.close)
    root.mainloop()


if __name__ == "__main__":
    main()
