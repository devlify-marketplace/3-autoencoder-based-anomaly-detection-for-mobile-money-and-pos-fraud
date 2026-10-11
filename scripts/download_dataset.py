#!/usr/bin/env python3
"""Download the public OpenML credit-card fraud benchmark."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fraud_autoencoder.data import RAW_NAME, download_dataset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--overwrite", action="store_true", help="Replace the existing source file.")
    args = parser.parse_args()
    destination = args.root / "data" / "raw" / RAW_NAME
    manifest = download_dataset(destination, overwrite=args.overwrite)
    print(json.dumps(manifest, indent=2))
    print(f"\nDataset saved to: {destination}")


if __name__ == "__main__":
    main()
