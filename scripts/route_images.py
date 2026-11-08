"""Route test images to review modes. Runs from a repo clone without pip install.

python scripts/route_images.py [--probs artifacts/results/baseline_probs.npz] [--archive-existing]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roadcond.routing.run import main  # noqa: E402

if __name__ == "__main__":
    main()
