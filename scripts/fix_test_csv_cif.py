"""
Fix test.csv by replacing the cif_structure column with correct CIF content
fetched from the Materials Project cache.

Usage:
    python fix_test_csv_cif.py \
        --test-csv /data/reproducibility/data/test.csv \
        --cif-cache /data/cgcnn_mp_test_cache/cif
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", required=True, help="Path to test.csv to update in-place")
    parser.add_argument("--cif-cache", required=True, help="Directory containing per-material .cif files named {material_id}.cif")
    return parser.parse_args()


def main():
    args = parse_args()
    test_csv = Path(args.test_csv)
    cif_cache = Path(args.cif_cache)

    csv.field_size_limit(10 * 1024 * 1024)

    # Read original
    with test_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    print(f"Loaded {len(rows)} rows from {test_csv}")

    # Replace cif_structure
    replaced = 0
    dropped = 0
    kept_rows = []
    for row in rows:
        mid = row["material_id"].strip()
        cif_path = cif_cache / f"{mid}.cif"
        if cif_path.exists():
            row["cif_structure"] = cif_path.read_text(encoding="utf-8")
            replaced += 1
            kept_rows.append(row)
        else:
            dropped += 1  # drop rows with no cached CIF

    print(f"Replaced: {replaced}, Dropped (no cache): {dropped}")
    print(f"Remaining rows: {len(kept_rows)}")

    # Backup original
    backup = test_csv.with_suffix(".csv.bak")
    shutil.copy2(test_csv, backup)
    print(f"Backup saved to {backup}")

    # Overwrite
    with test_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept_rows)

    print(f"Overwrote {test_csv}")


if __name__ == "__main__":
    main()
