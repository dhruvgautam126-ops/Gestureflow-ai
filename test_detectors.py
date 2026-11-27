"""Test script to verify detector execution paths work correctly."""
import tkinter as tk
import time
from app.coordinator import DetectorKind, FrameCoordinator
from app.camera.capture import CameraCapture, CameraStatus
from app.ui.main_window import MainWindow


def check_thread(window, worker_name):
    """Check worker thread status."""
    worker = getattr(window, worker_name)
    thread = worker._thread
    enabled = worker._enabled if hasattr(worker, '_enabled') else 'N/A'
    state = worker._state if hasattr(worker, '_state') else 'N/A'
    print(f"  {worker_name}: thread={thread is not None and thread.is_alive() if thread else False}, "
          f"_enabled={enabled}, _state={state}")


def main():
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    window = MainWindow(root)

    print("Test 1: Worker start/stop via coordinator enable/disable")
    print(f"  Human enabled: {window.coordinator._enabled[DetectorKind.HUMAN]}")
    print(f"  Face enabled: {window.coordinator._enabled[DetectorKind.FACE]}")
    print(f"  Hand enabled: {window.coordinator._enabled[DetectorKind.HAND]}")

    # Enable human detector - should start worker
    window._set_detector(DetectorKind.HUMAN, True)
    root.after(200, lambda: None)
    root.update()
    print(f"  After enable Human - enabled: {window.coordinator._enabled[DetectorKind.HUMAN]}")
    check_thread(window, 'human_detection')

    # Enable face detector
    window._set_detector(DetectorKind.FACE, True)
    root.after(200, lambda: None)
    root.update()
    print(f"  After enable Face - enabled: {window.coordinator._enabled[DetectorKind.FACE]}")
    check_thread(window, 'face_recognition')

    # Enable hand detector
    window._set_detector(DetectorKind.HAND, True)
    root.after(200, lambda: None)
    root.update()
    print(f"  After enable Hand - enabled: {window.coordinator._enabled[DetectorKind.HAND]}")
    check_thread(window, 'hand_detection')

    # Wait a moment for threads to be fully started
    root.after(500, lambda: None)
    root.update()

    # Now test disabling with proper wait
    print("\nTest 2: Stop workers")
    window._set_detector(DetectorKind.HUMAN, False)
    root.after(200, lambda: None)
    root.update()
    print(f"  After disable Human - enabled: {window.coordinator._enabled[DetectorKind.HUMAN]}")
    check_thread(window, 'human_detection')

    window._set_detector(DetectorKind.FACE, False)
    root.after(200, lambda: None)
    root.update()
    print(f"  After disable Face - enabled: {window.coordinator._enabled[DetectorKind.FACE]}")
    check_thread(window, 'face_recognition')

    window._set_detector(DetectorKind.HAND, False)
    root.after(200, lambda: None)
    root.update()
    print(f"  After disable Hand - enabled: {window.coordinator._enabled[DetectorKind.HAND]}")
    check_thread(window, 'hand_detection')

    root.after(2000, lambda: None)
    root.update()

    root.destroy()
    print("\nAll execution path tests completed!")


if __name__ == "__main__":
    main()