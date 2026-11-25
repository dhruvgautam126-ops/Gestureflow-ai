def _draw_hand_overlays(frame, hand_info_text: str) -> object:
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
