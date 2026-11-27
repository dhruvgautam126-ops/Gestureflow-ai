#!/usr/bin/env python
"""Test app startup"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.ui.main_window import main
print("main imported successfully")
