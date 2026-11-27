"""Lightweight water backdrop and click-through bubble cursor effects."""

from __future__ import annotations

import ctypes
import math
import sys
import time
import tkinter as tk
from ctypes import wintypes

import numpy as np
from PIL import Image, ImageTk


class WaterBackdrop(tk.Canvas):
    """Procedural water-colored backdrop with tap ripples."""

    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            background="#10223a",
            borderwidth=0,
            highlightthickness=0,
            **kwargs,
        )
        self._image_id = self.create_image(0, 0, anchor="nw")
        self._photo = None
        self._texture_size = (0, 0)
        self._ripples: list[tuple[float, float, float, int, int]] = []
        self._ambient_lines = [
            self.create_line(
                *([0, 0] * 33), smooth=True, splinesteps=12,
                fill=color, width=2 if band == 1 else 1,
            )
            for band, color in enumerate(("#4daac5", "#a4e6f2", "#4e77b3"))
        ]
        self._ambient_glints = [
            self.create_oval(0, 0, 0, 0, fill="#b9f4ff", outline="")
            for _ in range(5)
        ]
        self.bind("<Configure>", self._on_resize, add="+")
        self.after(60, self._animate)

    def tap(self, x: float | None = None, y: float | None = None) -> None:
        width, height = max(1, self.winfo_width()), max(1, self.winfo_height())
        ripple_x = float(width * 0.5 if x is None else x)
        ripple_y = float(height * 0.48 if y is None else y)
        outer = self.create_oval(ripple_x, ripple_y, ripple_x, ripple_y, outline="#c9f3ff", width=2)
        inner = self.create_oval(ripple_x, ripple_y, ripple_x, ripple_y, outline="#a8e8f8", width=1)
        self._ripples.append(
            (ripple_x, ripple_y, time.monotonic(), outer, inner)
        )
        for old_ripple in self._ripples[:-5]:
            self.delete(old_ripple[3], old_ripple[4])
        self._ripples = self._ripples[-5:]

    def _on_resize(self, _event=None) -> None:
        width, height = self.winfo_width(), self.winfo_height()
        if width < 2 or height < 2:
            return
        # Build a smaller texture and let Tk scale its placement with the window.
        texture_width = max(320, min(960, width // 2))
        texture_height = max(200, round(texture_width * height / width))
        if (texture_width, texture_height) == self._texture_size:
            return
        self._texture_size = texture_width, texture_height
        yy, xx = np.mgrid[0:1:complex(texture_height), 0:1:complex(texture_width)]
        stops = np.asarray(
            [
                (25, 91, 145),
                (83, 104, 177),
                (150, 104, 174),
                (39, 139, 164),
            ],
            dtype=np.float32,
        )
        segment = yy * (len(stops) - 1)
        low = np.floor(segment).astype(np.int32).clip(0, len(stops) - 2)
        mix = (segment - low)[..., None]
        color = stops[low] * (1.0 - mix) + stops[low + 1] * mix
        waves = (
            np.sin(xx * 13.0 + yy * 5.0)
            + 0.65 * np.sin(yy * 18.0 - xx * 8.0)
            + 0.34 * np.cos((xx - yy) * 27.0)
        )
        highlight = np.exp(-(((xx - 0.20) / 0.38) ** 2 + ((yy - 0.12) / 0.42) ** 2))
        blue_glow = np.exp(-(((xx - 0.84) / 0.38) ** 2 + ((yy - 0.78) / 0.56) ** 2))
        lilac_glow = np.exp(-(((xx - 0.78) / 0.45) ** 2 + ((yy - 0.20) / 0.55) ** 2))
        color = color * (0.91 + 0.055 * waves[..., None]) + 28.0 * highlight[..., None]
        color += np.asarray((21, 64, 94), dtype=np.float32) * blue_glow[..., None]
        color += np.asarray((27, 8, 42), dtype=np.float32) * lilac_glow[..., None]
        pixels = np.clip(color, 0, 255).astype(np.uint8)
        water_image = Image.fromarray(pixels, mode="RGB").resize(
            (width, height), Image.Resampling.BILINEAR
        )
        self._photo = ImageTk.PhotoImage(water_image)
        self.itemconfigure(self._image_id, image=self._photo)

    def _animate(self) -> None:
        now = time.monotonic()
        width, height = max(1, self.winfo_width()), max(1, self.winfo_height())
        phase = now * 0.75
        # Ambient water bands drift slowly behind the glass cards.
        for band, (base_y, amplitude) in enumerate((
            (0.70, 13),
            (0.76, 9),
            (0.86, 16),
        )):
            points = []
            for step in range(33):
                x = width * step / 32
                xf = step / 32
                y = height * base_y + amplitude * math.sin(xf * 9.0 + phase + band * 1.7)
                y += amplitude * 0.38 * math.sin(xf * 17.0 - phase * 0.7 + band)
                points.extend((x, y))
            self.coords(self._ambient_lines[band], *points)
        for index, item in enumerate(self._ambient_glints):
            x = width * ((index * 0.213 + now * (0.012 + index * 0.001)) % 1.0)
            y = height * (0.20 + 0.62 * ((index * 0.37) % 1.0))
            radius = 2.0 + 1.2 * (0.5 + 0.5 * math.sin(now * 1.5 + index * 2.1))
            self.coords(item, x - radius, y - radius, x + radius, y + radius)
        active = []
        for x, y, started, outer, inner in self._ripples:
            age = now - started
            if age >= 1.0:
                self.delete(outer, inner)
                continue
            active.append((x, y, started, outer, inner))
            radius = 12 + age * 170
            fade = max(0.0, 1.0 - age)
            color = "#%02x%02x%02x" % (
                int(76 + 125 * fade),
                int(148 + 90 * fade),
                int(190 + 65 * fade),
            )
            wobble = 1.0 + 0.04 * math.sin(age * 28.0)
            outer_radius = radius * wobble
            inner_radius = radius * 0.68 * wobble
            self.coords(outer, x - outer_radius, y - outer_radius * 0.34,
                        x + outer_radius, y + outer_radius * 0.34)
            self.itemconfigure(outer, outline=color)
            self.coords(inner, x - inner_radius, y - inner_radius * 0.34,
                        x + inner_radius, y + inner_radius * 0.34)
            self.itemconfigure(inner, outline="#%02x%02x%02x" % (
                int(76 + 85 * fade), int(148 + 70 * fade), int(190 + 45 * fade)
            ))
        self._ripples = active
        self.after(45, self._animate)


class WaterCursorOverlay:
    """Show a transparent bubble around the live Windows cursor while enabled."""

    KEY = "#ff00fe"
    SIZE = 66

    def __init__(self, parent):
        self.parent = parent
        self.active = False
        self.pressed = False
        self._ripples: list[float] = []
        self._last_cursor: tuple[int, int, float] | None = None
        self._speed = 0.0
        self._direction = 1.0
        self._tick_after_id = None
        self._window = tk.Toplevel(parent)
        self._window.overrideredirect(True)
        self._window.configure(bg=self.KEY)
        self._window.attributes("-topmost", True)
        try:
            self._window.attributes("-transparentcolor", self.KEY)
        except tk.TclError:
            pass
        self._window.geometry(f"{self.SIZE}x{self.SIZE}+0+0")
        self._canvas = tk.Canvas(
            self._window,
            width=self.SIZE,
            height=self.SIZE,
            bg=self.KEY,
            borderwidth=0,
            highlightthickness=0,
        )
        self._canvas.pack()
        center = self.SIZE / 2
        self._wake_arcs = [
            self._canvas.create_arc(
                center - 13, center - 17, center + 13, center + 17,
                start=65, extent=110, style="arc", outline="#6599b8", width=1,
            )
            for _ in range(2)
        ]
        self._wake_dots = [
            self._canvas.create_oval(0, 0, 0, 0, fill="#c8f7ff", outline="", state="hidden")
            for _ in range(2)
        ]
        self._cursor_ring = self._canvas.create_oval(0, 0, 0, 0, outline="#427aa0", width=1)
        self._cursor_bubble = self._canvas.create_oval(
            0, 0, 0, 0, fill="#7ccbec", outline="#e7fbff", width=2
        )
        self._cursor_highlight = self._canvas.create_oval(0, 0, 0, 0, fill="#f2fdff", outline="")
        self._cursor_shine = self._canvas.create_arc(
            0, 0, 0, 0, start=205, extent=190, style="arc", outline="#338ab2", width=2
        )
        self._ripple_items = [
            self._canvas.create_oval(0, 0, 0, 0, outline="#9beaff", width=2, state="hidden")
            for _ in range(4)
        ]
        self._window.withdraw()
        self._apply_click_through()

    def set_active(self, active: bool) -> None:
        if sys.platform != "win32":
            return
        active = bool(active)
        if self.active == active:
            return
        self.active = active
        if self.active:
            self._apply_click_through()
            self._window.deiconify()
            self._apply_click_through()
            self._tick()
        else:
            if self._tick_after_id is not None:
                try:
                    self._window.after_cancel(self._tick_after_id)
                except tk.TclError:
                    pass
                self._tick_after_id = None
            self._last_cursor = None
            self._window.withdraw()

    def set_pressed(self, pressed: bool) -> None:
        self.pressed = bool(pressed)

    def splash(self, _screen_point=None) -> None:
        self._ripples.append(time.monotonic())
        self._ripples = self._ripples[-4:]

    def destroy(self) -> None:
        try:
            if self._tick_after_id is not None:
                self._window.after_cancel(self._tick_after_id)
            self._window.destroy()
        except tk.TclError:
            pass

    def _apply_click_through(self) -> None:
        if sys.platform != "win32":
            return
        try:
            user32 = ctypes.windll.user32
            hwnd = wintypes.HWND(self._window.winfo_id())
            get_long = user32.GetWindowLongW
            get_long.argtypes = [wintypes.HWND, ctypes.c_int]
            get_long.restype = ctypes.c_long
            set_long = user32.SetWindowLongW
            set_long.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_long]
            set_long.restype = ctypes.c_long
            style = get_long(hwnd, -20)
            style |= 0x00000020 | 0x00000080 | 0x00080000 | 0x08000000
            set_long(hwnd, -20, style)
        except Exception:
            pass

    def _tick(self) -> None:
        if not self.active:
            self._tick_after_id = None
            return
        self._tick_after_id = None
        try:
            user32 = ctypes.windll.user32
            point = wintypes.POINT()
            get_cursor_pos = user32.GetCursorPos
            get_cursor_pos.argtypes = [ctypes.POINTER(wintypes.POINT)]
            get_cursor_pos.restype = wintypes.BOOL
            get_cursor_pos(ctypes.byref(point))
            now = time.monotonic()
            if self._last_cursor is not None:
                old_x, old_y, old_time = self._last_cursor
                dt = max(0.008, now - old_time)
                dx, dy = point.x - old_x, point.y - old_y
                # Smooth the speed estimate so tiny OS pointer jitter does not
                # make the water wake flicker.
                measured = min(1.0, math.hypot(dx, dy) / (dt * 1700.0))
                self._speed = self._speed * 0.72 + measured * 0.28
                if abs(dx) > 1:
                    self._direction = 1.0 if dx > 0 else -1.0
            self._last_cursor = (point.x, point.y, now)
            hwnd = wintypes.HWND(self._window.winfo_id())
            set_window_pos = user32.SetWindowPos
            set_window_pos.argtypes = [
                wintypes.HWND,
                wintypes.HWND,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                wintypes.UINT,
            ]
            set_window_pos.restype = wintypes.BOOL
            set_window_pos(
                hwnd,
                wintypes.HWND(-1),
                point.x - self.SIZE // 2,
                point.y - self.SIZE // 2,
                self.SIZE,
                self.SIZE,
                0x0010 | 0x0040,
            )
            self._draw()
        except Exception:
            pass
        if self.active and self._tick_after_id is None:
            self._tick_after_id = self._window.after(33, self._tick)

    def _draw(self) -> None:
        now = time.monotonic()
        canvas = self._canvas
        center = self.SIZE / 2
        radius = 19 if self.pressed else 16
        strength = min(1.0, self._speed)
        spread = 18 + strength * 15
        # The bubble pushes a pair of soft wave crests outward as it moves.
        # Keep the center clear so the real desktop pointer remains precise.
        phase = now * 7.0
        for index, side in enumerate((-1.0, 1.0)):
            direction = side * self._direction
            x_offset = direction * spread
            sway = math.sin(phase + side) * (2.0 + 7.0 * strength)
            alpha_color = "#78d8f3" if strength > 0.12 else "#6599b8"
            arc = self._wake_arcs[index]
            canvas.coords(
                arc,
                center + x_offset - 13, center - 17 + sway,
                center + x_offset + 13, center + 17 + sway,
            )
            canvas.itemconfigure(
                arc,
                start=65 if direction > 0 else 245,
                extent=110,
                outline=alpha_color,
                width=1 + int(strength * 2),
            )
            if strength > 0.2:
                dot_x = center + direction * (spread + 8 + 5 * math.sin(phase * 0.7 + side))
                dot_y = center + 8 * math.sin(phase + side * 2)
                dot_r = 1.2 + strength * 1.5
                dot = self._wake_dots[index]
                canvas.coords(
                    dot,
                    dot_x - dot_r, dot_y - dot_r, dot_x + dot_r, dot_y + dot_r,
                )
                canvas.itemconfigure(dot, state="normal")
            else:
                canvas.itemconfigure(self._wake_dots[index], state="hidden")
        canvas.coords(
            self._cursor_ring,
            center - radius - 5, center - radius - 5,
            center + radius + 5, center + radius + 5,
        )
        canvas.coords(
            self._cursor_bubble,
            center - radius, center - radius,
            center + radius, center + radius,
        )
        canvas.coords(
            self._cursor_highlight,
            center - radius + 5, center - radius + 4,
            center + 1, center + 1,
        )
        canvas.coords(
            self._cursor_shine,
            center - radius + 4, center - radius + 4,
            center + radius - 4, center + radius - 4,
        )
        alive = []
        for index, started in enumerate(self._ripples):
            age = now - started
            item = self._ripple_items[index]
            if age >= 0.75:
                canvas.itemconfigure(item, state="hidden")
                continue
            alive.append(started)
            r = radius + 5 + age * 54
            canvas.coords(item, center - r, center - r * 0.72, center + r, center + r * 0.72)
            canvas.itemconfigure(item, state="normal")
        self._ripples = alive
