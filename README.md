# Road Condition AI

Road surface (friction) classification on RSCD and a cross-country damage detection study on RDD2022.

## Setup

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # add ,train / ,detect / ,export / ,serve as needed
pytest
```

Datasets go in `data/raw/` and are not committed. Saved split file lists go in `data/splits/` and are committed.
