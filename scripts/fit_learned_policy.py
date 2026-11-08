"""Fit the learned review policy on val. Runs from a repo clone without pip install.

python scripts/fit_learned_policy.py [--config configs/default.yaml] [--weights FILE.json]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roadcond.routing.compare import fit_main  # noqa: E402

if __name__ == "__main__":
    fit_main()
