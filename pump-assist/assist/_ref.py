"""Make the read-only reference package (original-challenge/src) importable. Never edit that package."""
import sys
from .splits import REPO_ROOT

_SRC = str(REPO_ROOT / "original-challenge" / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
