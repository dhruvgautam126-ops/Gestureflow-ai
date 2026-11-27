#!/usr/bin/env python
"""Test app startup and camera toggle"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import tkinter as tk
from app.ui.main_window import MainWindow

root = tk.Tk()
root.title("Gesture.AI")
root.geometry("1180x760")

window = MainWindow(root)
print("MainWindow instantiated OK")
print("Camera type:", type(window.camera).__name__)
print("Has stop method:", hasattr(window.camera, 'stop'))
print("Has request_stop method:", hasattr(window.camera, 'request_stop'))

# Test toggle
print("\nTesting camera start...")
window.camera.start()
print("Camera started OK")

print("\nTesting camera stop...")
window.camera.stop()
print("Camera stopped OK")

root.destroy()
print("Test complete - app works!")
