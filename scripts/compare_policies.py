"""Compare the learned and threshold review policies on test. Runs without pip install.

python scripts/compare_policies.py [--config configs/default.yaml] [--archive-existing]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roadcond.routing.compare import compare_main  # noqa: E402

if __name__ == "__main__":
    compare_main()
