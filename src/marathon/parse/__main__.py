"""Parse a summarizedActivities export to a tidy parquet file.

Usage: python -m marathon.parse <export.json> <out.parquet>
"""

from __future__ import annotations

import sys
from pathlib import Path

from marathon.parse.activities import load_export


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    src, dst = Path(argv[0]), Path(argv[1])
    df = load_export(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, index=False)
    by_discipline = df["discipline"].value_counts().to_dict()
    print(f"{len(df)} activities -> {dst}")
    print(f"by discipline: {by_discipline}")
    print(f"races: {int(df['is_race'].sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
