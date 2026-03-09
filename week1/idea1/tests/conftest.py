"""
Pytest configuration for local package imports.

Ensures the project root is available on sys.path so tests can import
`envs` and `networks` reliably across different pytest import modes.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
root_str = str(ROOT_DIR)
if root_str not in sys.path:
    sys.path.insert(0, root_str)
