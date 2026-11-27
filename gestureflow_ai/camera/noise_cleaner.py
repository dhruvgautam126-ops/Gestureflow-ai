"""Lightweight, preview-only camera noise reduction using OpenCV."""

from __future__ import annotations

import cv2
import numpy as np


class CameraNoiseCleaner:
    """Reduce visible camera noise without changing frames used by detectors.

    The bilateral filter runs at half resolution to keep the added preview cost
    modest. A small blend with the original frame helps retain fine detail.
    """

    def __init__(self, scale: float = 0.5, blend: float = 0.78) -> None:
        self.scale = min(1.0, max(0.25, float(scale)))
        self.blend = min(1.0, max(0.0, float(blend)))

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Return a cleaned copy; never mutate the camera frame."""
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.size == 0:
            return frame
        height, width = frame.shape[:2]
        if height < 2 or width < 2:
            return frame.copy()

        small_size = (max(1, round(width * self.scale)), max(1, round(height * self.scale)))
        small = cv2.resize(frame, small_size, interpolation=cv2.INTER_AREA)
        filtered = cv2.bilateralFilter(small, d=5, sigmaColor=28, sigmaSpace=5)
        cleaned = cv2.resize(filtered, (width, height), interpolation=cv2.INTER_LINEAR)
        if self.blend < 1.0:
            cleaned = cv2.addWeighted(cleaned, self.blend, frame, 1.0 - self.blend, 0.0)
        return cleaned
