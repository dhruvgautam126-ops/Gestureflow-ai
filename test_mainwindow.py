#!/usr/bin/env python
"""Test full app startup"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tkinter as tk
from app.ui.main_window import MainWindow

root = tk.Tk()
root.title("Gesture.AI")
root.geometry("1180x760")

try:
    window = MainWindow(root)
    print("MainWindow instantiated OK")
    window.pack(fill="both", expand=True)
    print("Window packed OK")

    # Try to start coordinator
    print(f"Coordinator running: {window.coordinator._running}")
    print(f"Coordinator enabled: HUMAN={window.coordinator._enabled[window.coordinator.DetectorKind.HUMAN]}")

except Exception as e:
    print(f"Error during init: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

root.destroy()
print("Test complete")
