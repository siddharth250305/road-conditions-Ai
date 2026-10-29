"""Write the results tables into the README (and optionally a file) from saved result files.

python scripts/render_results.py --results-dir artifacts/results            # print
python scripts/render_results.py --results-dir artifacts/results --readme README.md
Runs from a repo clone without pip install.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from roadcond.results_tables import render, update_readme  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-dir", default="artifacts/results")
    parser.add_argument(
        "--readme", help="update the block between the results markers in this file"
    )
    parser.add_argument("--section", choices=["all", "classification", "detection"], default="all")
    parser.add_argument("--out", help="also write the tables to this markdown file")
    args = parser.parse_args()

    tables = render(args.results_dir, args.section)
    print(tables)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(tables + "\n")
    if args.readme and not update_readme(args.readme, tables):
        print(f"\nNo results markers in {args.readme}: nothing updated.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
