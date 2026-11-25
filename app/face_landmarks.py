"""Small helpers for displaying MediaPipe FaceLandmarker output."""

from __future__ import annotations

import cv2
import numpy as np


_FACE_CONTOURS = (
    # Face oval
    ((10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
      379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
      234, 127, 162, 21, 54, 103, 67, 109), True),
    # Eyes
    ((33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160,
      161, 246), True),
    ((362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386,
      385, 384, 398), True),
    # Lips
    ((61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318,
      402, 317, 14, 87, 178, 88, 95, 78), True),
    # Nose bridge
    ((168, 6, 197, 195, 5, 4, 1, 19, 94), False),
)

FACE_LANDMARK_CONNECTIONS = tuple(
    (first, second)
    for contour, closed in _FACE_CONTOURS
    for first, second in zip(
        contour,
        contour[1:] + (contour[:1] if closed else ()),
    )
)


def extract_face_landmark_draw_data(result, max_faces: int = 2) -> tuple:
    """Return normalized face landmarks as immutable coordinate tuples."""
    groups = getattr(result, "face_landmarks", None) or ()
    extracted = []
    for landmarks in groups[:max_faces]:
        if len(landmarks) < 468:
            continue
        points = tuple(
            (float(point.x), float(point.y))
            for point in landmarks[:468]
        )
        if all(np.isfinite(x) and np.isfinite(y) for x, y in points):
            extracted.append(points)
    return tuple(extracted)


def draw_face_landmarks(
    frame: np.ndarray,
    faces: tuple,
    *,
    mirror_x: bool = False,
) -> np.ndarray:
    """Draw face contours over a BGR image, optionally mirroring X coordinates."""
    if not faces:
        return frame
    height, width = frame.shape[:2]
    color = (220, 155, 255)
    for face in faces:
        if len(face) < 468:
            continue
        points: list[tuple[int, int] | None] = []
        for x_norm, y_norm in face:
            if not (0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0):
                points.append(None)
                continue
            x = 1.0 - x_norm if mirror_x else x_norm
            points.append((
                max(0, min(width - 1, int(round(x * (width - 1))))),
                max(0, min(height - 1, int(round(y_norm * (height - 1))))),
            ))
        for first, second in FACE_LANDMARK_CONNECTIONS:
            point_a, point_b = points[first], points[second]
            if point_a is not None and point_b is not None:
                cv2.line(frame, point_a, point_b, color, 1, cv2.LINE_AA)
        # A few bright points make the imported landmark layer easy to see.
        for index in (1, 4, 33, 133, 263, 362, 61, 291):
            point = points[index]
            if point is not None:
                cv2.circle(frame, point, 2, (245, 245, 255), -1, cv2.LINE_AA)
    return frame
