import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import evaluation.evaluation_utils as F
from evaluation.evaluate_parse_integrity import discover_runs

TARGET_SPLIT = "test"


def compute_metrics(run_meta: dict, y_true_df: pd.DataFrame) -> dict:
    property_name = run_meta["property_name"]
    y_true = y_true_df[property_name].to_numpy(dtype=float)

    records_path = Path(run_meta["records_path"])

    errors = []
    n_total = 0
    n_parsed = 0

    with records_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue

            n_total += 1
            idx = row.get("index")
            text = row.get("generated_text")

            if text is None:
                continue
            pred = F.extract_property_value(str(text))
            if pred is None:
                continue

            try:
                idx_int = int(idx)
            except Exception:
                continue
            if not (0 <= idx_int < len(y_true)):
                continue

            gt = y_true[idx_int]
            if not math.isfinite(gt):
                continue

            errors.append(abs(pred - gt))
            n_parsed += 1

    errors_arr = np.array(errors, dtype=float)
    mae  = float(np.mean(errors_arr))  if len(errors_arr) > 0 else None
    rmse = float(np.sqrt(np.mean(errors_arr ** 2))) if len(errors_arr) > 0 else None

    return {
        "run_id":        run_meta["run_id"],
        "seed":          run_meta["seed"],
        "property_name": run_meta["property_name"],
        "modality":      run_meta["modality"],
        "variant":       run_meta["variant"],
        "model_size":    run_meta["model_size"],
        "n_total":       n_total,
        "n_parsed":      n_parsed,
        "parse_rate":    n_parsed / n_total if n_total > 0 else None,
        "mae":           mae,
        "rmse":          rmse,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute MAE/RMSE for all discovered runs")
    parser.add_argument("--runs_root", type=Path, required=True, help="Root directory containing inference run outputs")
    parser.add_argument("--test_csv", type=Path, required=True, help="Test CSV with ground-truth property values")
    parser.add_argument("--out_dir", type=Path, required=True, help="Directory to write metrics_table.csv")
    args = parser.parse_args()

    y_true_df = pd.read_csv(args.test_csv)
    runs = discover_runs(runs_root=args.runs_root, split=TARGET_SPLIT, require_records=True)
    print(f"[INFO] {len(runs)} runs discovered")

    rows = []
    for run_meta in runs:
        row = compute_metrics(run_meta, y_true_df)
        rows.append(row)
        print(f"  {row['run_id']:45s}  mae={row['mae']:.4f}" if row["mae"] is not None else f"  {row['run_id']:45s}  mae=None")

    df = pd.DataFrame(rows)
    df = df.sort_values(["property_name", "modality", "model_size", "variant", "seed"]).reset_index(drop=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "metrics_table.csv"
    df.to_csv(out_path, index=False)
    print(f"[INFO] wrote: {out_path}")


if __name__ == "__main__":
    main()
