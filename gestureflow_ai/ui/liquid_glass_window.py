"""Liquid-glass-inspired desktop UI for GestureFlow AI.

This screen reuses the existing camera and detector orchestration. It uses
Tkinter and Pillow already present in the project, so no new UI dependency is
needed. Tkinter does not provide true backdrop blur; cards use layered color,
soft borders, and highlights to give a similar frosted-glass feel.
"""

from __future__ import annotations

import ctypes
import sys
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

from gestureflow_ai.hand_gesture_advanced import HAND_CONNECTIONS
from gestureflow_ai.ui.main_window import MainWindow as DetectionWindow, _font
from gestureflow_ai.ui.device_share_window import DeviceShareWindow
from gestureflow_ai.ui.water_effects import WaterBackdrop


BG = "#091627"
SURFACE = "#213a55"
SURFACE_ALT = "#294866"
STROKE = "#82b9d1"
TEXT = "#edf5ff"
MUTED = "#9aadc0"
ACCENT = "#83e8ff"
ACCENT_DARK = "#173b59"
GREEN = "#65e6b3"
ORANGE = "#f5bd70"
VIDEO_BG = "#09121e"


def configure_glass_style(root: tk.Tk) -> None:
    root.configure(bg=BG)
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    style.configure("App.TFrame", background=BG)
    style.configure("Glass.TFrame", background=SURFACE)
    style.configure("GlassAlt.TFrame", background=SURFACE_ALT)
    style.configure("Brand.TLabel", background=SURFACE, foreground=TEXT,
                    font=("Segoe UI", 19, "bold"))
    style.configure("Subtitle.TLabel", background=SURFACE, foreground=MUTED,
                    font=("Segoe UI", 9))
    style.configure("CardTitle.TLabel", background=SURFACE, foreground=TEXT,
                    font=("Segoe UI", 11, "bold"))
    style.configure("Body.TLabel", background=SURFACE, foreground=TEXT,
                    font=("Segoe UI", 10))
    style.configure("Muted.TLabel", background=SURFACE, foreground=MUTED,
                    font=("Segoe UI", 9))
    style.configure("Metric.TLabel", background=SURFACE, foreground=ACCENT,
                    font=("Segoe UI", 10, "bold"))
    style.configure("Preview.TLabel", background=VIDEO_BG, foreground=TEXT)
    style.configure("StatusValue.TLabel", background=SURFACE, foreground=GREEN,
                    font=("Segoe UI", 10, "bold"))
    style.configure("Primary.TButton", background="#167b9a", foreground="#f5fcff",
                    bordercolor="#42b7d6", lightcolor="#42b7d6", darkcolor="#10536c",
                    padding=(12, 8), font=("Segoe UI", 9, "bold"))
    style.map("Primary.TButton", background=[("active", "#219aba"), ("disabled", "#26384a")],
              foreground=[("disabled", "#75899c")])
    style.configure("Secondary.TButton", background="#23364a", foreground=TEXT,
                    bordercolor=STROKE, lightcolor=STROKE, darkcolor="#152437",
                    padding=(10, 7), font=("Segoe UI", 9))
    style.map("Secondary.TButton", background=[("active", "#304960"), ("disabled", "#1a2938")],
              foreground=[("disabled", "#75899c")])


class GlassCard(tk.Canvas):
    """Rounded, softly outlined panel with a regular Tk child frame inside."""

    def __init__(self, parent, *, padding: int = 16, fill: str = SURFACE, **kwargs):
        super().__init__(
            parent,
            background=BG,
            borderwidth=0,
            highlightthickness=0,
            **kwargs,
        )
        self._padding = padding
        self._fill = fill
        self._shape_ids: list[int] = []
        self.body = ttk.Frame(self, style="Glass.TFrame")
        self._body_id = self.create_window(
            padding, padding, anchor="nw", window=self.body
        )
        self.bind("<Configure>", self._resize)

    def _resize(self, event) -> None:
        width, height = max(1, event.width), max(1, event.height)
        for item in self._shape_ids:
            self.delete(item)
        self._shape_ids.clear()

        inset = 1
        radius = 22
        x1, y1, x2, y2 = inset, inset, max(inset + 1, width - inset), max(inset + 1, height - inset)
        points = (
            x1 + radius, y1, x2 - radius, y1,
            x2, y1, x2, y1 + radius,
            x2, y2 - radius, x2, y2,
            x2 - radius, y2, x1 + radius, y2,
            x1, y2, x1, y2 - radius,
            x1, y1 + radius, x1, y1,
        )
        self._shape_ids.append(
            self.create_polygon(
                *points,
                smooth=True,
                splinesteps=12,
                fill=self._fill,
                outline=STROKE,
                width=1,
            )
        )
        self._shape_ids.append(
            self.create_line(
                x1 + 28, y1 + 1, x2 - 28, y1 + 1,
                fill="#b7e7f5", width=1,
            )
        )
        available_width = max(1, width - self._padding * 2)
        available_height = max(1, height - self._padding * 2)
        self.itemconfigure(self._body_id, width=available_width, height=available_height)
        self.tag_raise(self._body_id)


class LiquidGlassWindow(DetectionWindow):
    """New dashboard layout backed by the existing detection window logic."""

    def _build(self) -> None:
        self.configure(style="App.TFrame")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        header = GlassCard(self, height=88, padding=16)
        header.grid(row=0, column=0, sticky="ew", padx=18, pady=(16, 10))
        header.body.columnconfigure(1, weight=1)

        brand = ttk.Frame(header.body, style="Glass.TFrame")
        brand.grid(row=0, column=0, sticky="w")
        ttk.Label(brand, text="GestureFlow AI", style="Brand.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(brand, text="HAND GESTURE CONTROL  ·  LOCAL PROCESSING",
                  style="Subtitle.TLabel").grid(row=1, column=0, sticky="w", pady=(1, 0))

        telemetry = ttk.Frame(header.body, style="Glass.TFrame")
        telemetry.grid(row=0, column=1, sticky="e", padx=(20, 16))
        for column in (1, 3, 5):
            telemetry.columnconfigure(column, minsize=76)
        for col, title in ((0, "CAMERA"), (2, "RESOLUTION"), (4, "CAPTURE")):
            ttk.Label(telemetry, text=title, style="Muted.TLabel").grid(
                row=0, column=col, sticky="w", padx=(0, 7)
            )
        self.state_value = ttk.Label(telemetry, text="OFF", style="StatusValue.TLabel")
        self.state_value.grid(row=0, column=1, sticky="w", padx=(0, 14))
        self.resolution_value = ttk.Label(telemetry, text="—", style="Body.TLabel")
        self.resolution_value.grid(row=0, column=3, sticky="w", padx=(0, 14))
        self.fps_value = ttk.Label(telemetry, text="—", style="Body.TLabel")
        self.fps_value.grid(row=0, column=5, sticky="w")

        self.toggle_button = ttk.Button(
            header.body,
            text="Turn Camera ON",
            command=self._toggle_camera,
            style="Primary.TButton",
        )
        self.toggle_button.grid(row=0, column=2, sticky="e")

        main = WaterBackdrop(self)
        self.water_backdrop = main
        main.grid(row=1, column=0, sticky="nsew", padx=18, pady=4)

        preview = GlassCard(main, padding=16)
        preview.body.columnconfigure(0, weight=1)
        preview.body.rowconfigure(1, weight=1)
        preview_header = ttk.Frame(preview.body, style="Glass.TFrame")
        preview_header.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        preview_header.columnconfigure(0, weight=1)
        ttk.Label(preview_header, text="LIVE CAMERA", style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.noise_cleaner_button = ttk.Button(
            preview_header,
            text="Noise Cleaner: OFF",
            command=self._toggle_noise_cleaner,
            style="Secondary.TButton",
        )
        self.noise_cleaner_button.grid(row=0, column=1, sticky="e")
        self.preview_label = ttk.Label(
            preview.body,
            text="",
            anchor="center",
            style="Preview.TLabel",
        )
        self.preview_label.grid(row=1, column=0, sticky="nsew")

        hand_card = self._build_detector_card(
            main, 0, "GESTURE CONTROL", "Up to 4 tracked hands · local processing",
            "hand_info", "start_hand_button", "stop_hand_button",
            lambda: self._set_detector(self._hand_kind(), True),
            lambda: self._set_detector(self._hand_kind(), False),
        )
        self._preview_window_id = main.create_window(12, 12, anchor="nw", window=preview)
        self._hand_window_id = main.create_window(12, 12, anchor="nw", window=hand_card)
        main.bind("<Configure>", self._layout_main, add="+")

        footer = GlassCard(self, height=58, padding=12, fill=SURFACE_ALT)
        footer.grid(row=2, column=0, sticky="ew", padx=18, pady=(10, 16))
        footer.body.columnconfigure(0, weight=1)
        self.notice_value = ttk.Label(
            footer.body,
            text="Camera is off · Turn it on to begin.",
            style="Muted.TLabel",
            anchor="w",
        )
        self.notice_value.grid(row=0, column=0, sticky="ew")
        self.share_button = ttk.Button(
            footer.body,
            text="Device Share",
            command=self._open_device_share,
            style="Secondary.TButton",
        )
        self.share_button.grid(row=0, column=1, sticky="e", padx=(12, 0))
        self.quit_button = ttk.Button(
            footer.body,
            text="Exit",
            command=self._on_close,
            style="Secondary.TButton",
        )
        self.quit_button.grid(row=0, column=2, sticky="e", padx=(12, 0))

    def _open_device_share(self) -> None:
        window = getattr(self, "_device_share_window", None)
        try:
            if window is not None and window.winfo_exists():
                window.deiconify()
                window.lift()
                return
        except Exception:
            pass
        self._device_share_window = DeviceShareWindow(
            self.parent, on_arm=self._arm_device_share_gesture
        )

    def _arm_device_share_gesture(self) -> bool:
        if self.camera.snapshot().state != "on":
            return False
        if self.gesture_mouse.active:
            self.gesture_mouse.stop()
            self.gesture_control_button.configure(text="Enable Desktop Gesture Control")
            self.gesture_control_info.configure(
                text="Desktop cursor control paused for file handoff.", foreground=MUTED
            )
        if not self.coordinator.is_enabled(self._hand_kind()):
            self._set_detector(self._hand_kind(), True)
        return self.coordinator.is_enabled(self._hand_kind())

    def _process_device_share_gestures(self, hand_result) -> None:
        window = getattr(self, "_device_share_window", None)
        try:
            if window is not None and window.winfo_exists():
                candidates = hand_result.hand_candidates if hand_result else ()
                tracks = hand_result.hand_visible_tracks if hand_result else ()
                source_seq = hand_result.source_seq if hand_result else None
                window.process_hand_gestures(
                    candidates or (), tracks or (), source_seq=source_seq
                )
        except Exception:
            # A closing share window must not interrupt camera preview updates.
            return

    def _toggle_noise_cleaner(self) -> None:
        self.preview_noise_cleaner_enabled = not self.preview_noise_cleaner_enabled
        state = "ON" if self.preview_noise_cleaner_enabled else "OFF"
        self.noise_cleaner_button.configure(text=f"Noise Cleaner: {state}")

    @staticmethod
    def _hand_kind():
        from gestureflow_ai.coordinator import DetectorKind
        return DetectorKind.HAND

    def _build_detector_card(
        self,
        parent,
        row: int,
        title: str,
        detail: str,
        info_name: str,
        start_name: str,
        stop_name: str,
        start_command,
        stop_command,
    ) -> None:
        card = GlassCard(parent, padding=14)
        card.body.columnconfigure(0, weight=1)
        ttk.Label(card.body, text=title, style="CardTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(card.body, text=detail, style="Muted.TLabel").grid(
            row=1, column=0, sticky="w", pady=(3, 8)
        )
        info = ttk.Label(
            card.body,
            text="Turn the camera on to begin.",
            style="Body.TLabel",
            wraplength=250,
            justify="left",
        )
        info.grid(row=2, column=0, sticky="ew", pady=(0, 9))
        setattr(self, info_name, info)

        buttons = ttk.Frame(card.body, style="Glass.TFrame")
        buttons.grid(row=3, column=0, sticky="ew")
        buttons.columnconfigure((0, 1), weight=1)
        start_button = ttk.Button(
            buttons, text="Start", command=start_command,
            style="Primary.TButton", state="disabled",
        )
        start_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        stop_button = ttk.Button(
            buttons, text="Stop", command=stop_command,
            style="Secondary.TButton", state="disabled",
        )
        stop_button.grid(row=0, column=1, sticky="ew", padx=(5, 0))
        setattr(self, start_name, start_button)
        setattr(self, stop_name, stop_button)

        if info_name == "hand_info":
            self.gesture_control_button = ttk.Button(
                card.body,
                text="Enable Desktop Gesture Control",
                command=self._toggle_gesture_control,
                style="Secondary.TButton",
                state="disabled",
            )
            self.gesture_control_button.grid(row=4, column=0, sticky="ew", pady=(9, 0))
            self.gesture_guide = ttk.Label(
                card.body,
                text="PALM: move cursor  ·  THUMBS UP: left-click\nCLOSED FIST: right-click  ·  PINCH: hold/drag\nOPEN HAND: release  ·  OPEN HAND + WRIST TWIST: scroll",
                style="Muted.TLabel",
                wraplength=280,
                justify="left",
            )
            self.gesture_guide.grid(row=5, column=0, sticky="ew", pady=(7, 0))
            self.gesture_control_info = ttk.Label(
                card.body,
                text="Enable gesture control to use the cursor across your desktop.",
                style="Muted.TLabel",
                wraplength=260,
                justify="left",
            )
            self.gesture_control_info.grid(row=6, column=0, sticky="ew", pady=(5, 0))
        return card

    def _layout_main(self, event) -> None:
        width, height = max(1, event.width), max(1, event.height)
        margin, gap = 12, 14
        hand_width = max(300, min(370, round(width * 0.30)))
        preview_width = max(1, width - margin * 2 - gap - hand_width)
        content_height = max(1, height - margin * 2)
        self.water_backdrop.coords(self._preview_window_id, margin, margin)
        self.water_backdrop.itemconfigure(
            self._preview_window_id,
            width=preview_width,
            height=content_height,
        )
        self.water_backdrop.coords(
            self._hand_window_id,
            margin + preview_width + gap,
            margin,
        )
        self.water_backdrop.itemconfigure(
            self._hand_window_id,
            width=hand_width,
            height=content_height,
        )

    def _show_placeholder(self) -> None:
        width, height = self._preview_size()
        if self._placeholder is None or self._placeholder_size != (width, height):
            top = np.asarray((12, 24, 38), dtype=np.float32)
            bottom = np.asarray((18, 39, 56), dtype=np.float32)
            ramp = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
            pixels = np.repeat(top[None, None, :] * (1 - ramp) + bottom[None, None, :] * ramp,
                               width, axis=1).astype(np.uint8)
            image = Image.fromarray(pixels, mode="RGB")
            draw = ImageDraw.Draw(image)
            cx, cy = width // 2, height // 2
            draw.rounded_rectangle(
                (cx - 36, cy - 26, cx + 36, cy + 26),
                radius=13, outline="#5c8397", width=2,
            )
            draw.ellipse((cx - 11, cy - 11, cx + 11, cy + 11), outline=ACCENT, width=2)
            draw.text((cx, cy + 48), "CAMERA OFF", fill=TEXT, font=_font(15), anchor="mm")
            draw.text((cx, cy + 72), "Turn on the camera to view live detection",
                      fill="#9aadc0", font=_font(10), anchor="mm")
            self._placeholder = ImageTk.PhotoImage(image)
            self._placeholder_size = (width, height)
        if self._photo is not self._placeholder:
            self.preview_label.configure(image=self._placeholder)
        self._photo = self._placeholder
        self._photo_size = (width, height)

    def _draw_hand_overlays(self, frame, candidates, tracks, *, copy_frame: bool = True):
        """Draw each tracked hand independently on the shared preview frame."""
        annotated = frame.copy() if copy_frame else frame
        height, width = annotated.shape[:2]
        palette = (
            (94, 217, 242),
            (112, 226, 179),
            (239, 183, 111),
            (195, 151, 255),
        )
        for index, candidate in enumerate(candidates or ()):
            track = tracks[index] if tracks and index < len(tracks) else None
            points = getattr(candidate, "image_points", None)
            if points is None or len(points) != 21:
                continue
            xy: list[tuple[int, int]] = []
            visible: list[bool] = []
            for point in points:
                x_norm, y_norm = float(point[0]), float(point[1])
                in_frame = 0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0
                x = max(0, min(width - 1, int(round((1.0 - x_norm) * (width - 1)))))
                y = max(0, min(height - 1, int(round(y_norm * (height - 1)))))
                xy.append((x, y))
                visible.append(in_frame)

            color = palette[index % len(palette)]
            for first, second in HAND_CONNECTIONS:
                if visible[first] and visible[second]:
                    cv2.line(annotated, xy[first], xy[second], color, 2, cv2.LINE_AA)
            for point, is_visible in zip(xy, visible):
                if is_visible:
                    cv2.circle(annotated, point, 4, (245, 251, 255), -1, cv2.LINE_AA)
                    cv2.circle(annotated, point, 6, color, 1, cv2.LINE_AA)

            bbox = getattr(candidate, "bbox", None)
            if bbox is not None and len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)
                left = max(0, min(width - 1, width - x2))
                right = max(0, min(width - 1, width - x1))
                top = max(0, min(height - 1, y1))
                bottom = max(0, min(height - 1, y2))
                cv2.rectangle(annotated, (left, top), (right, bottom), color, 1)
            else:
                left, top = xy[0]

            gesture = getattr(track, "stable_gesture", "UNKNOWN") if track else "UNKNOWN"
            if gesture == "UNKNOWN":
                gesture = getattr(candidate, "gesture", "UNKNOWN")
            person_id = getattr(track, "person_id", None) if track else None
            side = getattr(track, "associated_side", None) if track else None
            if person_id is not None and side:
                label = f"PERSON {person_id}  {side} HAND  ·  {gesture}"
            elif side:
                label = f"{side} HAND  ·  {gesture}"
            else:
                track_id = getattr(track, "track_id", index + 1) if track else index + 1
                label = f"HAND {track_id}  ·  {gesture}"

            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
            )
            label_x = max(4, min(width - text_width - 12, left))
            label_y = max(text_height + 10, top - 8)
            cv2.rectangle(
                annotated,
                (label_x, label_y - text_height - 8),
                (label_x + text_width + 12, label_y + baseline + 2),
                (16, 31, 47),
                -1,
            )
            cv2.rectangle(
                annotated,
                (label_x, label_y - text_height - 8),
                (label_x + text_width + 12, label_y + baseline + 2),
                color,
                1,
            )
            cv2.putText(
                annotated, label, (label_x + 6, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (241, 248, 255), 1, cv2.LINE_AA,
            )
        return annotated


def main() -> None:
    if sys.platform == "win32":
        try:
            ctypes.shcore.SetProcessDpiAwareness(1)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass
    root = tk.Tk()
    root.title("GestureFlow AI · Hand Control")
    root.geometry("1440x900")
    root.minsize(1120, 720)
    configure_glass_style(root)
    window = LiquidGlassWindow(root)
    window.pack(fill="both", expand=True)
    root.protocol("WM_DELETE_WINDOW", window._on_close)
    root.mainloop()
