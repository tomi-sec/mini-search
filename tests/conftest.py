"""Puts src/ on sys.path so tests can `import ranking`, `import indexer`, etc.

The engine's modules import each other flatly and rely on running with
src/ as the script directory (see main.py). Tests aren't run that way,
so this is the one place that bridges the gap.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
