"""Import the unchanged parent QUBO implementation from this optional extension."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
if str(PARENT) not in sys.path: sys.path.append(str(PARENT))
