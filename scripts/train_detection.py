"""Cross-country damage detection (PRD D1-D5). Runs from a repo clone without pip install.

python scripts/train_detection.py prepare --root /path/to/RDD2022
python scripts/train_detection.py source | source --augment | fewshot | pseudo | report
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roadcond.detection.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
