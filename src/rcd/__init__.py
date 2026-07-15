"""Rostrocaudal complexity and EEG transportability analysis."""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rcd-matplotlib"))

__version__ = "0.2.0"
