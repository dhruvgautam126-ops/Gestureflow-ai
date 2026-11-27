"""Test repeated start/stop cycles for all three detectors."""
import tkinter as tk
import time
from app.coordinator import DetectorKind
from app.ui.main_window import MainWindow


def main():
    root = tk.Tk()
    root.withdraw()
    window = MainWindow(root)

    print("=== Repeated Start/Stop Cycles Test ===\n")

    # Cycle 1: Enable all, then disable all
    print("Cycle 1: Enable all detectors")
    window._set_detector(DetectorKind.HUMAN, True)
    window._set_detector(DetectorKind.FACE, True)
    window._set_detector(DetectorKind.HAND, True)
    root.after(500, lambda: None)
    root.update()

    print("\nCycle 2: Disable all detectors")
    window._set_detector(DetectorKind.HUMAN, False)
    window._set_detector(DetectorKind.FACE, False)
    window._set_detector(DetectorKind.HAND, False)
    root.after(1000, lambda: None)
    root.update()

    # Cycle 3: Enable all again
    print("\nCycle 3: Enable all detectors again")
    window._set_detector(DetectorKind.HUMAN, True)
    window._set_detector(DetectorKind.FACE, True)
    window._set_detector(DetectorKind.HAND, True)
    root.after(500, lambda: None)
    root.update()

    print("\nCycle 4: Disable all detectors again")
    window._set_detector(DetectorKind.HUMAN, False)
    window._set_detector(DetectorKind.FACE, False)
    window._set_detector(DetectorKind.HAND, False)
    root.after(1000, lambda: None)
    root.update()

    # Cycle 5: Enable human only, then re-enable all
    print("\nCycle 5: Enable human only, then all three")
    window._set_detector(DetectorKind.HUMAN, True)
    root.after(300, lambda: None)
    root.update()
    window._set_detector(DetectorKind.FACE, True)
    root.after(300, lambda: None)
    root.update()
    window._set_detector(DetectorKind.HAND, True)
    root.after(300, lambda: None)
    root.update()

    print("\nCycle 6: Disable them in reverse order")
    window._set_detector(DetectorKind.HAND, False)
    root.after(300, lambda: None)
    root.update()
    window._set_detector(DetectorKind.FACE, False)
    root.after(300, lambda: None)
    root.update()
    window._set_detector(DetectorKind.HUMAN, False)
    root.after(300, lambda: None)
    root.update()

    root.after(500, lambda: None)
    root.update()
    root.destroy()
    print("\n=== All repeated start/stop cycles completed successfully! ===")


if __name__ == "__main__":
    main()