import sys
from pathlib import Path

# robop's package __init__ imports estimator dependencies (einops, NeMO). The
# tested modules are dependency-light by design, so import them directly from
# src/robop without going through the package. The repo root makes the
# evaluation package importable.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src" / "robop"))
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))
