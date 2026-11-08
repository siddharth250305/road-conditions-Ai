"""Save the exported model's val/test probabilities. Runs from a repo clone without pip install.

python scripts/save_bundle_probs.py --root <RSCD> [--bundle serving] [--out FILE.npz]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roadcond.bundle_probs import main  # noqa: E402

if __name__ == "__main__":
    main()
