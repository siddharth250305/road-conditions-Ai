"""Train and evaluate a surface classifier. Runs from a repo clone without pip install.

python scripts/train_classifier.py --model resnet50 --root /path/to/RSCD [--eval-test]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roadcond.train import main  # noqa: E402

if __name__ == "__main__":
    main()
