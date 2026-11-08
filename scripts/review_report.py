"""Report on live review routing. Runs from a repo clone without pip install.

python scripts/review_report.py [--log logs/decisions.jsonl] [--bundle serving] [--out FILE]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roadcond.routing.monitor import main  # noqa: E402

if __name__ == "__main__":
    main()
