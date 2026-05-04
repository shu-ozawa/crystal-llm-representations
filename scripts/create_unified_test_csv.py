"""
Create a unified test.csv that works for all modalities.

- Base: test_llm4mat.csv (10,318 rows, correct ordering for non-cif modalities)
- Source of correct cif_structure: test.csv (10,223 rows, CIF-dataset order)
- Matching key: (material_id, structure) — confirmed zero duplicates in both files

Output:
  - data/test.csv       <- overwritten with unified version (10,318 rows)
  - data/test_cif_backup.csv <- renamed from old test.csv
"""
import argparse
import hashlib
import shutil
from pathlib import Path

import pandas as pd


def row_key(material_id: str, structure: str) -> str:
    """Stable hash of (material_id, structure) for fast lookup."""
    raw = f"{material_id}||{structure}"
    return hashlib.md5(raw.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge CIF structures into the LLM4Mat-Bench test split")
    parser.add_argument("--llm4mat-csv", required=True, help="test_llm4mat.csv (10,318 rows, correct ordering for non-CIF modalities)")
    parser.add_argument("--cif-csv", required=True, help="test.csv with correct cif_structure column (10,223 rows, CIF-dataset order)")
    parser.add_argument("--out-csv", required=True, help="Output path for unified test.csv (10,318 rows)")
    args = parser.parse_args()

    LLM4MAT_CSV = Path(args.llm4mat_csv)
    CIF_CSV = Path(args.cif_csv)
    OUT_CSV = Path(args.out_csv)
    BACKUP_CSV = OUT_CSV.with_name(OUT_CSV.stem + "_cif_backup.csv")

    print(f"Loading {LLM4MAT_CSV} ...")
    df_llm = pd.read_csv(LLM4MAT_CSV)
    print(f"  rows: {len(df_llm)}")

    print(f"Loading {CIF_CSV} ...")
    df_cif = pd.read_csv(CIF_CSV)
    print(f"  rows: {len(df_cif)}")

    # Build lookup: key → correct cif_structure (from df_cif)
    print("Building cif_structure lookup ...")
    cif_lookup: dict[str, str] = {}
    for _, row in df_cif.iterrows():
        k = row_key(str(row["material_id"]), str(row["structure"]))
        cif_lookup[k] = str(row["cif_structure"])
    print(f"  lookup size: {len(cif_lookup)}")

    # Map correct cif_structure into df_llm
    print("Transplanting cif_structure ...")
    n_matched = 0
    n_missing = 0
    new_cif = []
    for _, row in df_llm.iterrows():
        k = row_key(str(row["material_id"]), str(row["structure"]))
        if k in cif_lookup:
            new_cif.append(cif_lookup[k])
            n_matched += 1
        else:
            new_cif.append("")   # no valid CIF for this material
            n_missing += 1

    df_llm["cif_structure"] = new_cif
    print(f"  matched: {n_matched}  missing (no CIF): {n_missing}")

    # Sanity checks
    assert len(df_llm) == 10318, f"Expected 10318 rows, got {len(df_llm)}"
    assert df_llm["material_id"].iloc[0] == "mp-1519998", "index 0 material_id mismatch"
    assert df_llm["formula_pretty"].iloc[500] == "HgH2NCl", \
        f"index 500 formula mismatch: {df_llm['formula_pretty'].iloc[500]}"

    # Backup old test.csv then write unified version
    print(f"\nBacking up {CIF_CSV} → {BACKUP_CSV} ...")
    shutil.copy2(CIF_CSV, BACKUP_CSV)

    print(f"Writing unified test.csv → {OUT_CSV} ...")
    df_llm.to_csv(OUT_CSV, index=False)
    print("Done.")

    # Verify output
    df_out = pd.read_csv(OUT_CSV)
    print(f"\nVerification:")
    print(f"  rows: {len(df_out)}")
    for i in [0, 500, 1000]:
        cif_preview = str(df_out["cif_structure"].iloc[i])[:40] if df_out["cif_structure"].iloc[i] else "(empty)"
        print(f"  [{i}] formula={df_out['formula_pretty'].iloc[i]:20s}  cif_ok={bool(cif_preview.strip())}  cif[:40]={cif_preview}")


if __name__ == "__main__":
    main()
