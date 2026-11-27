"""Opt-in whole-desktop mouse control from tracked hand landmarks."""

from __future__ import annotations

import ctypes
import math
import os
import time
from ctypes import wintypes

import numpy as np


class GestureMouseController:
    """Map one tracked hand to the Windows desktop cursor and mouse.

    A palm-base anchor moves the pointer relative to its initial hand position.
    Using wrist and knuckle landmarks keeps the pointer steady as fingers open,
    close, or form a thumbs-up. A pinch grabs and drags; an open palm releases.
    Thumbs-up left-clicks and a closed fist right-clicks.
    Wrist rotation while grabbing scrolls. Input is active only after the user
    enables it in the app, and the held button is released if the hand is lost.
    """

    PINCH_DOWN_RATIO = 0.38
    PINCH_UP_RATIO = 0.62
    CONFIRM_FRAMES = 2
    LOST_HAND_RELEASE_SECONDS = 0.30
    HAND_RESELECT_SECONDS = 0.8
    TWIST_STEP_RADIANS = math.radians(18)
    TWIST_NOISE_RADIANS = math.radians(1.5)
    CURSOR_SMOOTHING = 0.36
    CURSOR_GAIN = 1.65
    HAND_DEADZONE = 0.003

    def __init__(self, window, protected_widgets=(), on_click=None, on_button=None, on_active=None):
        self.window = window
        self.protected_widgets = list(protected_widgets)
        self.on_click = on_click
        self.on_button = on_button
        self.on_active = on_active
        self.active = False
        self._pressed = False
        self._grab_hits = 0
        self._release_hits = 0
        self._track_id: int | None = None
        self._last_hand_seen = 0.0
        self._cursor: tuple[float, float] | None = None
        self._pointer_hand_position: tuple[float, float] | None = None
        self._grab_origin_wrist: tuple[float, float] | None = None
        self._grab_origin_cursor: tuple[float, float] | None = None
        self._last_twist_angle: float | None = None
        self._twist_accumulator = 0.0
        self._previous_gesture = "UNKNOWN"
        self._last_source_seq: int | None = None

    def start(self) -> bool:
        if os.name != "nt":
            return False
        self.active = True
        self._reset_tracking(release=True)
        self._cursor = self._read_cursor_position()
        self._notify_active(True)
        return True

    def stop(self) -> None:
        self.active = False
        self._reset_tracking(release=True)
        self._notify_active(False)

    def update(self, candidates, tracks, source_seq: int | None = None) -> str:
        if not self.active:
            return "Gesture control is off."

        hand = self._select_hand(candidates or (), tracks or ())
        now = time.monotonic()
        if hand is None:
            # Preserve the OS cursor and reacquire a fresh hand anchor later.
            self._pointer_hand_position = None
            if self._last_hand_seen and now - self._last_hand_seen >= self.LOST_HAND_RELEASE_SECONDS:
                self._reset_tracking(release=True, keep_track=True)
            if now - self._last_hand_seen >= self.HAND_RESELECT_SECONDS:
                self._track_id = None
            return "Show one hand clearly to control the cursor."

        if source_seq is not None:
            if source_seq == self._last_source_seq:
                return "Hand is steady; waiting for the next camera frame."
            self._last_source_seq = source_seq

        candidate, track = hand
        self._last_hand_seen = now
        points = np.asarray(candidate.image_points, dtype=np.float32)
        world = np.asarray(candidate.world_points, dtype=np.float32)
        if points.shape[0] < 21 or world.shape[0] < 21:
            return "Hand landmarks are incomplete."

        anchor_x, anchor_y = self._palm_anchor(points)
        wrist_x, wrist_y = map(float, points[0, :2])
        if not all(0.0 <= value <= 1.0 for value in (anchor_x, anchor_y, wrist_x, wrist_y)):
            self._reset_tracking(release=True, keep_track=True)
            return "Keep the hand in view."

        stable_gesture = str(getattr(track, "stable_gesture", "UNKNOWN") or "UNKNOWN")
        stable_gesture = stable_gesture.upper().replace("_", " ")
        gesture = stable_gesture
        if gesture == "UNKNOWN":
            gesture = str(getattr(candidate, "gesture", "UNKNOWN") or "UNKNOWN")
            gesture = gesture.upper().replace("_", " ")

        palm_width = float(np.linalg.norm(world[5] - world[17]))
        pinch_distance = float(np.linalg.norm(world[4] - world[8]))
        pinch_ratio = (
            pinch_distance / palm_width
            if np.isfinite(palm_width) and palm_width > 1e-5
            else math.inf
        )
        open_palm = stable_gesture == "OPEN HAND"
        fist_pose = gesture == "CLOSED FIST"
        grab_signal = not fist_pose and pinch_ratio <= self.PINCH_DOWN_RATIO
        release_signal = open_palm or pinch_ratio >= self.PINCH_UP_RATIO
        gesture_holds_cursor = gesture in {"CLOSED FIST", "THUMBS UP"}

        if self._cursor is None:
            self._cursor = self._read_cursor_position()
        if self._cursor is None:
            return "Could not read the current cursor position."
        first_hand_frame = self._pointer_hand_position is None
        if first_hand_frame:
            # Anchor the hand at the current OS cursor when first seen. This
            # preserves the user's cursor position instead of snapping to center.
            self._pointer_hand_position = (anchor_x, anchor_y)
        else:
            hand_dx = anchor_x - self._pointer_hand_position[0]
            hand_dy = anchor_y - self._pointer_hand_position[1]
            self._pointer_hand_position = (anchor_x, anchor_y)
            if (
                not gesture_holds_cursor
                and (abs(hand_dx) > self.HAND_DEADZONE or abs(hand_dy) > self.HAND_DEADZONE)
            ):
                _, _, screen_width, screen_height = self._virtual_screen()
                self._move_cursor(
                    round(self._cursor[0] + hand_dx * screen_width * self.CURSOR_GAIN),
                    round(self._cursor[1] + hand_dy * screen_height * self.CURSOR_GAIN),
                )
        screen_point = (round(self._cursor[0]), round(self._cursor[1]))

        if not self._pressed:
            if grab_signal:
                self._grab_hits += 1
                self._release_hits = 0
            else:
                self._grab_hits = 0
            if self._grab_hits >= self.CONFIRM_FRAMES:
                if not self._is_protected_target(*screen_point):
                    self._mouse_button(down=True)
                    self._pressed = True
                    self._grab_origin_wrist = (wrist_x, wrist_y)
                    cursor_origin = self._cursor or screen_point
                    self._grab_origin_cursor = (
                        float(cursor_origin[0]),
                        float(cursor_origin[1]),
                    )
                    self._last_twist_angle = self._palm_angle(points)
                    self._twist_accumulator = 0.0
                    self._notify_button(True)
                    self._notify_click(screen_point)
        else:
            if release_signal:
                self._release_hits += 1
            else:
                self._release_hits = 0
            if self._release_hits >= (1 if open_palm else self.CONFIRM_FRAMES):
                self._release_button()
            elif self._grab_origin_wrist and self._grab_origin_cursor:
                dx = wrist_x - self._grab_origin_wrist[0]
                dy = wrist_y - self._grab_origin_wrist[1]
                left, top, width, height = self._virtual_screen()
                self._move_cursor(
                    round(self._grab_origin_cursor[0] + dx * width),
                    round(self._grab_origin_cursor[1] + dy * height),
                )
                self._apply_twist_scroll(points)

        if first_hand_frame:
            # Do not click merely because a click pose was already held on activation.
            self._previous_gesture = stable_gesture
        elif stable_gesture == "THUMBS UP" and self._previous_gesture != "THUMBS UP" and not self._pressed:
            if not self._is_protected_target(*screen_point):
                self._click(left=True)
                self._notify_click(screen_point)
        elif stable_gesture == "CLOSED FIST" and self._previous_gesture != "CLOSED FIST" and not self._pressed:
            if not self._is_protected_target(*screen_point):
                self._click(left=False)
                self._notify_click(screen_point)
        self._previous_gesture = stable_gesture

        if self._pressed:
            return "Left button held: move to drag; open hand to release."
        if gesture in {"OPEN HAND", "CLOSED FIST", "THUMBS UP"}:
            return f"Detected {gesture}; palm center controls the pointer."
        return "Palm center moves the pointer; pinch to hold and drag."

    @staticmethod
    def _palm_anchor(points: np.ndarray) -> tuple[float, float]:
        """Return a hand-shape-stable point from the palm and knuckle bases.

        Fingertips shift substantially between pointing, open-hand, fist, and
        thumbs-up poses. The wrist and MCP joints (landmarks 0, 5, 9, 13, 17)
        stay near the palm, so their centroid avoids that pose-induced cursor
        jump while still following actual hand movement.
        """
        palm_indices = (0, 5, 9, 13, 17)
        anchor = np.mean(points[list(palm_indices), :2], axis=0)
        return float(anchor[0]), float(anchor[1])

    def _select_hand(self, candidates, tracks):
        pairs = []
        for candidate, track in zip(candidates, tracks):
            if float(getattr(candidate, "quality", 0.0)) < 0.45:
                continue
            pairs.append((candidate, track))

        if self._track_id is not None:
            for candidate, track in pairs:
                if track.track_id == self._track_id:
                    return candidate, track
            return None
        if not pairs:
            return None

        right_hands = [
            pair for pair in pairs
            if getattr(pair[1], "handedness", None) == "RIGHT"
            and float(getattr(pair[1], "handedness_score", 0.0)) >= 0.65
        ]
        if right_hands:
            selected = max(right_hands, key=lambda pair: pair[0].quality)
        elif len(pairs) == 1:
            selected = pairs[0]
        else:
            selected = max(pairs, key=lambda pair: pair[0].quality)
        self._track_id = selected[1].track_id
        return selected

    @staticmethod
    def _virtual_screen() -> tuple[int, int, int, int]:
        user32 = ctypes_user32()
        return (
            user32.GetSystemMetrics(76),
            user32.GetSystemMetrics(77),
            max(1, user32.GetSystemMetrics(78)),
            max(1, user32.GetSystemMetrics(79)),
        )

    @staticmethod
    def _read_cursor_position() -> tuple[float, float] | None:
        try:
            point = wintypes.POINT()
            user32 = ctypes_user32()
            user32.GetCursorPos.argtypes = [ctypes.POINTER(wintypes.POINT)]
            user32.GetCursorPos.restype = wintypes.BOOL
            if user32.GetCursorPos(ctypes.byref(point)):
                return float(point.x), float(point.y)
        except Exception:
            pass
        return None

    def _move_cursor(self, x: int, y: int) -> None:
        left, top, width, height = self._virtual_screen()
        x = min(left + width - 1, max(left, x))
        y = min(top + height - 1, max(top, y))
        if self._cursor is None:
            smooth_x, smooth_y = float(x), float(y)
        else:
            alpha = self.CURSOR_SMOOTHING
            smooth_x = self._cursor[0] + alpha * (x - self._cursor[0])
            smooth_y = self._cursor[1] + alpha * (y - self._cursor[1])
        self._cursor = smooth_x, smooth_y
        ctypes_user32().SetCursorPos(round(smooth_x), round(smooth_y))

    def _is_protected_target(self, x: int, y: int) -> bool:
        for widget in self.protected_widgets:
            try:
                if widget.winfo_exists() and widget.winfo_viewable():
                    left, top = widget.winfo_rootx(), widget.winfo_rooty()
                    if left <= x < left + widget.winfo_width() and top <= y < top + widget.winfo_height():
                        return True
            except Exception:
                continue
        return False

    @staticmethod
    def _mouse_button(*, down: bool) -> None:
        flags = 0x0002 if down else 0x0004
        ctypes_user32().mouse_event(flags, 0, 0, 0, 0)

    def _release_button(self) -> None:
        if self._pressed:
            self._mouse_button(down=False)
            self._notify_button(False)
        self._pressed = False
        self._grab_origin_wrist = None
        self._grab_origin_cursor = None
        self._last_twist_angle = None
        self._twist_accumulator = 0.0
        self._release_hits = 0

    def _notify_button(self, pressed: bool) -> None:
        if self.on_button:
            try:
                self.on_button(pressed)
            except Exception:
                pass

    def _notify_click(self, screen_point) -> None:
        if self.on_click:
            try:
                self.on_click(screen_point)
            except Exception:
                pass

    def _notify_active(self, active: bool) -> None:
        if self.on_active:
            try:
                self.on_active(active)
            except Exception:
                pass

    def _click(self, *, left: bool) -> None:
        user32 = ctypes_user32()
        down, up = (0x0002, 0x0004) if left else (0x0008, 0x0010)
        user32.mouse_event(down, 0, 0, 0, 0)
        user32.mouse_event(up, 0, 0, 0, 0)

    def _apply_twist_scroll(self, points: np.ndarray) -> None:
        angle = self._palm_angle(points)
        if angle is None:
            return
        if self._last_twist_angle is None:
            self._last_twist_angle = angle
            return
        delta = (angle - self._last_twist_angle + math.pi) % (2 * math.pi) - math.pi
        self._last_twist_angle = angle
        magnitude = abs(delta) - self.TWIST_NOISE_RADIANS
        if magnitude <= 0:
            return
        filtered_delta = math.copysign(magnitude, delta)
        # Convert wrist rotation to smooth wheel units. Tiny values are held
        # until enough motion accumulates; one frame can never trigger a jump.
        self._twist_accumulator += (
            filtered_delta * 120.0 / self.TWIST_STEP_RADIANS
        )
        wheel_units = int(self._twist_accumulator)
        if abs(wheel_units) >= 6:
            wheel_units = max(-120, min(120, wheel_units))
            wheel_delta = ctypes.c_uint32(wheel_units).value
            ctypes_user32().mouse_event(0x0800, 0, 0, wheel_delta, 0)
            self._twist_accumulator -= wheel_units

    @staticmethod
    def _palm_angle(points: np.ndarray) -> float | None:
        dx = float(points[5, 0] - points[17, 0])
        dy = float(points[5, 1] - points[17, 1])
        if not np.isfinite(dx + dy) or math.hypot(dx, dy) < 1e-4:
            return None
        return math.atan2(dy, dx)

    def _reset_tracking(self, *, release: bool, keep_track: bool = False) -> None:
        if release:
            self._release_button()
        self._grab_hits = 0
        self._release_hits = 0
        self._pointer_hand_position = None
        self._previous_gesture = "UNKNOWN"
        self._last_source_seq = None
        if not keep_track:
            self._track_id = None


def ctypes_user32():
    user32 = ctypes.windll.user32
    user32.GetSystemMetrics.argtypes = [ctypes.c_int]
    user32.GetSystemMetrics.restype = ctypes.c_int
    user32.SetCursorPos.argtypes = [ctypes.c_int, ctypes.c_int]
    user32.SetCursorPos.restype = wintypes.BOOL
    user32.mouse_event.argtypes = [
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_size_t,
    ]
    user32.mouse_event.restype = None
    return user32
