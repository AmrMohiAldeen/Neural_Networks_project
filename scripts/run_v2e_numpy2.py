"""Local NumPy 2 compatibility launcher for the pinned v2e-1.7.0.

Linux instructions use NumPy 1.26 and do not need this launcher.
"""
import math
from pathlib import Path
import runpy
import sys

import numpy as np

if not hasattr(np, 'math'):
    np.math = math
entry = Path(__file__).resolve().parents[1] / 'third_party/v2e/v2e.py'
sys.path.insert(0, str(entry.parent))
sys.argv[0] = str(entry)
runpy.run_path(str(entry), run_name='__main__')
