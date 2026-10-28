"""Export a trained classifier to ONNX. Runs from a repo clone without pip install.

python scripts/export_model.py bundle --checkpoint best.pt --results resnet50.json --out serving
python scripts/export_model.py study --results-dir artifacts/results --out serving/study.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roadcond.export import main  # noqa: E402

if __name__ == "__main__":
    main()
