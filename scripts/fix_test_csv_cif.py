"""
Repair the broken row alignment of the Materials Project test.csv from
LLM4Mat-Bench.

Each row of this file should hold one material, but the metadata columns
and the structure columns were written out independently and simply placed
side by side.

- Metadata columns (material_id, formula_pretty, description, properties):
  there are 9947 distinct materials, but 53 material_id values are repeated
  eight times each, which inflates the file by 53 * 7 = 371 rows to 10318.
  The repeated rows all fall within the 424 rows 5088-5511 (0-based).

- Structure columns (structure, cif_structure):
  the 9947 structures are packed from the first row with no gaps, and the
  last 371 rows (9947-10317) are empty strings that pad the row count.

Because of this mismatch, the structure columns run 371 rows ahead of the
metadata columns after the inflated block, so from row 5088 onward the
material_id and the structure on the same CSV row do not belong together.

The repair does not match rows by row number. It relies only on both sides
listing the materials in the same order:

    metadata side  : collapse duplicates, keeping first occurrences -> 9947 rows
    structure side : drop the empty strings                         -> 9947 entries

The two sides then have the same order and length, so pairing them row by
row restores the original correspondence. This is verified by checking that
the chemical formula on the data_ line of every CIF equals formula_pretty
for all 9947 materials.

No material is dropped: all 9947 materials are restored with their
structures.

The description of a repeated material_id is mostly identical across its
copies, but some copies differ in wording without differing in meaning, e.g.
    "corner, face, and edge-sharing FePr3Y3Fe6 cuboctahedra"
    "edge, face, and corner-sharing FePr3Y3Fe6 cuboctahedra"
Only the order of the listed items differs, so the first occurrence is kept.

na_filter=False keeps empty strings as empty strings and stops the formula
"NaN" (sodium nitride) from being read as a missing value.

Usage:
    python scripts/fix_test_csv_cif.py \
        --test-csv data/raw/llm4mat_bench/mp/test.csv \
        --out-csv data/raw/llm4mat_bench/mp/test_fixed.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


N_ROWS = 10_318      # rows in the broken file
N_MATERIALS = 9_947  # distinct materials
N_INFLATED = 371     # extra rows, equal to the offset between the two sides


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-csv", required=True, help="test.csv of the Materials Project split as distributed (10,318 rows); left unchanged")
    parser.add_argument("--out-csv", default=None, help="Where to write the repaired split (9,947 rows); defaults to test_fixed.csv next to --test-csv")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.test_csv)
    output_path = Path(args.out_csv) if args.out_csv else input_path.with_name("test_fixed.csv")

    df = pd.read_csv(input_path, na_filter=False)

    assert len(df) == N_ROWS

    # --- Confirm how the file is broken ---

    # Metadata side: 53 materials appear eight times each, adding 371 rows
    is_duplicate = df["material_id"].duplicated(keep=False)
    duplicate_counts = df.loc[is_duplicate, "material_id"].value_counts()

    assert duplicate_counts.eq(8).all()
    assert len(duplicate_counts) == 53
    assert df["material_id"].nunique() == N_MATERIALS
    assert len(df) - N_MATERIALS == N_INFLATED

    # Structure side: 9947 entries packed from the top, then 371 empty rows
    has_cif = df["cif_structure"].str.strip().ne("")
    has_structure = df["structure"].str.strip().ne("")

    assert has_cif.eq(has_structure).all()
    assert int(has_cif.sum()) == N_MATERIALS

    # The empty rows must form one contiguous block at the end, i.e. padding
    empty_positions = df.index[~has_cif]
    assert empty_positions.min() == N_MATERIALS
    assert empty_positions.max() == N_ROWS - 1
    assert empty_positions.to_series().diff().dropna().eq(1).all()

    # --- Repair ---

    # Metadata side: collapse duplicates, keeping first occurrences
    fixed = df.drop_duplicates("material_id", keep="first").reset_index(drop=True)

    # Structure side: drop the empty strings
    structures = df.loc[has_cif, ["structure", "cif_structure"]].reset_index(drop=True)

    assert len(fixed) == len(structures) == N_MATERIALS

    fixed[["structure", "cif_structure"]] = structures

    # --- Verify ---

    # Take the formula from the data_ line of each CIF and compare it with
    # formula_pretty for every material
    cif_formula = fixed["cif_structure"].str.extract(
        r"(?m)^data_([^\r\n]+)",
        expand=False,
    )

    assert fixed["formula_pretty"].eq(cif_formula).all(), (
        "Metadata and structures are not aligned after the repair."
    )

    assert fixed["material_id"].is_unique
    assert len(fixed) == N_MATERIALS
    assert fixed["formula_pretty"].str.strip().ne("").all()
    assert fixed["description"].str.strip().ne("").all()
    assert fixed["cif_structure"].str.strip().ne("").all()

    fixed.to_csv(output_path, index=False)

    print(f"raw rows:            {len(df)}")
    print(f"inflated rows:       {N_INFLATED}")
    print(f"repaired materials:  {len(fixed)}")
    print(f"materials dropped:   0")
    print(f"saved:               {output_path}")


if __name__ == "__main__":
    main()
