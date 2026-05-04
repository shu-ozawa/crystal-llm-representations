"""
Remap `index` fields in CIF records.jsonl files.

After create_unified_test_csv.py runs, test.csv is in test_llm4mat.csv order (10,318 rows).
CIF records.jsonl (seed43/44/45) were generated with old test.csv order (10,223 rows, CIF order).
This script rewrites each CIF record's `index` to match the new unified test.csv row number.

Mapping:
  old_index  →  (material_id, structure)  [via test_cif_backup.csv]
             →  new_index                 [via new unified test.csv]

Target files: CIF records.jsonl for seed43, 44, 45 (band_gap + formation_energy, 1B + 8B)
Excludes seed42 (9,947 records = old wrong CIF dataset).
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


def row_key(material_id: str, structure: str) -> str:
    raw = f"{material_id}||{structure}"
    return hashlib.md5(raw.encode()).hexdigest()


def build_old_index_to_key(backup_csv: Path) -> dict[int, str]:
    """old_index (0-based row number in old test.csv) → (mat_id, structure) hash key"""
    df = pd.read_csv(backup_csv)
    return {i: row_key(str(row["material_id"]), str(row["structure"]))
            for i, row in df.iterrows()}


def build_key_to_new_index(new_csv: Path) -> dict[str, int]:
    """(mat_id, structure) hash key → new_index (0-based row in unified test.csv)"""
    df = pd.read_csv(new_csv)
    return {row_key(str(row["material_id"]), str(row["structure"])): i
            for i, row in df.iterrows()}


def remap_records_file(records_path: Path,
                       old_to_key: dict[int, str],
                       key_to_new: dict[str, int]) -> dict:
    """Rewrite records.jsonl with remapped indices. Returns stats."""
    lines_in = records_path.read_text(encoding="utf-8").splitlines()

    out_lines = []
    stats = {"total": 0, "remapped": 0, "unmapped": 0, "skipped": 0}

    for line in lines_in:
        line = line.strip()
        if not line:
            continue
        stats["total"] += 1
        try:
            rec = json.loads(line)
        except Exception:
            stats["skipped"] += 1
            out_lines.append(line)
            continue

        old_idx = rec.get("index")
        try:
            old_idx = int(old_idx)
        except Exception:
            stats["skipped"] += 1
            out_lines.append(line)
            continue

        key = old_to_key.get(old_idx)
        if key is None:
            stats["unmapped"] += 1
            out_lines.append(line)
            continue

        new_idx = key_to_new.get(key)
        if new_idx is None:
            stats["unmapped"] += 1
            out_lines.append(line)
            continue

        rec["index"] = new_idx
        out_lines.append(json.dumps(rec, ensure_ascii=False))
        stats["remapped"] += 1

    # Write back
    backup = records_path.with_suffix(".jsonl.pre_remap")
    shutil.copy2(records_path, backup)
    records_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return stats


def find_cif_records(runs_root: Path) -> list[Path]:
    """Find CIF records.jsonl for seed43/44/45 only."""
    paths = []
    # band_gap: .../band_gap/Bg{1B,8B}-cif-*-seed{43,44,45}/test/records.jsonl
    bg_root = runs_root / "band_gap"
    if bg_root.exists():
        for p in sorted(bg_root.glob("Bg*-cif-*-seed4[345]/test/records.jsonl")):
            paths.append(p)

    # formation_energy: .../G{1B,8B}-cif-*-seed{43,44,45}/test/records.jsonl
    for p in sorted(runs_root.glob("G*-cif-*-seed4[345]/test/records.jsonl")):
        paths.append(p)

    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Remap index fields in CIF records.jsonl to match unified test.csv ordering")
    parser.add_argument("--backup-csv", required=True, help="Old CIF-ordered test.csv (10,223 rows) — output of create_unified_test_csv.py backup")
    parser.add_argument("--new-csv", required=True, help="Unified test.csv (10,318 rows) — output of create_unified_test_csv.py")
    parser.add_argument("--runs-root", required=True, type=Path, help="Root directory containing inference run outputs")
    args = parser.parse_args()

    BACKUP_CSV = Path(args.backup_csv)
    NEW_CSV = Path(args.new_csv)
    RUNS_ROOT = args.runs_root

    print("Building index lookup tables ...")
    old_to_key = build_old_index_to_key(BACKUP_CSV)
    key_to_new = build_key_to_new_index(NEW_CSV)
    print(f"  old test.csv rows: {len(old_to_key)}")
    print(f"  new test.csv rows: {len(key_to_new)}")

    target_files = find_cif_records(RUNS_ROOT)
    print(f"\nFound {len(target_files)} CIF records.jsonl files to remap:")
    for p in target_files:
        print(f"  {p.relative_to(RUNS_ROOT)}")

    print()
    total_remapped = 0
    total_unmapped = 0
    for records_path in target_files:
        # Skip if not 10,223 records (= old wrong CIF seed42 runs that slipped through)
        n_lines = sum(1 for ln in records_path.read_text(encoding="utf-8").splitlines() if ln.strip())
        if n_lines < 10000:
            print(f"  [SKIP] {records_path.name} parent={records_path.parent.parent.name} "
                  f"(only {n_lines} records, likely old wrong CIF)")
            continue

        stats = remap_records_file(records_path, old_to_key, key_to_new)
        status = "OK" if stats["unmapped"] == 0 else "WARN"
        print(f"  [{status}] {records_path.parent.parent.name}: "
              f"remapped={stats['remapped']}  unmapped={stats['unmapped']}  "
              f"skipped={stats['skipped']}")
        total_remapped += stats["remapped"]
        total_unmapped += stats["unmapped"]

    print(f"\nTotal remapped: {total_remapped}  unmapped: {total_unmapped}")
    if total_unmapped > 0:
        print("WARNING: some records could not be remapped — check unmapped entries.")
    else:
        print("All records remapped successfully.")


if __name__ == "__main__":
    main()
