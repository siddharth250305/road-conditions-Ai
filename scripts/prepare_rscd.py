"""Write RSCD split files and the leak report. Runs from a repo clone without pip install.

python scripts/prepare_rscd.py --root /path/to/RSCD
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roadcond.data.rscd import main  # noqa: E402

if __name__ == "__main__":
    main()
