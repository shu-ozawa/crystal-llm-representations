import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import evaluation.evaluation_utils as F

TARGET_SPLIT = "test"

MODALITIES = ("formula", "descr1st", "description", "descr_last", "cif")
PROPERTIES = ("formation_energy_per_atom", "band_gap", "bulk_modulus_kv")

# Expected test-set sizes. MP has two, depending on the representation. The CIF
# runs use test_fixed.csv, the repaired split: one row per material, 9,947 of
# them, every one carrying a CIF. The other representations still run over
# test.csv as distributed, which repeats 53 materials eight times each and so
# has 10,318 rows for the same 9,947 materials; evaluate_metrics collapses that
# back down by keeping the first occurrence, which is why both end up reported
# over 9,947 materials. JARVIS (bulk modulus) uses a single test split
# restricted to the 2,340 materials that carry an elastic tensor, so every
# representation shares it (see data/processed/jarvis_test_labeled/test.csv).
_MP_TEST_SIZE = {"cif": 9947}
_MP_TEST_SIZE_DEFAULT = 10318
_JARVIS_TEST_SIZE = 2340


def _discover_repro_runs(runs_root: Path, split: str, require_records: bool) -> list[dict]:
    rows: list[dict] = []
    # property_name is inferred from the directory location plus the run_id
    # prefix. Getting this wrong scores a run against another property's
    # ground truth: no error is raised and the resulting MAE still looks
    # plausible. Always change this together with the writing side
    # (scripts/*/eval_runs.sh), which decides where runs are placed.
    roots = (
        ("formation_energy_per_atom", runs_root, "G*"),
        ("band_gap", runs_root / "band_gap", "Bg*"),
        ("bulk_modulus_kv", runs_root / "bulk_modulus_kv", "Bk*"),
    )
    for property_name, root_dir, pattern in roots:
        if not root_dir.exists():
            continue
        for run_dir in sorted(p for p in root_dir.glob(pattern) if p.is_dir()):
            run_id = run_dir.name
            parts = run_id.split("-")
            last = parts[-1]
            if not (last.startswith("seed") and last[4:].isdigit()):
                continue
            seed = last[4:]
            core_parts = parts[:-1]
            prefix = core_parts[0]
            modality = core_parts[1]
            variant = "-".join(core_parts[2:])
            if modality not in MODALITIES:
                continue
            model_size = prefix[2:] if prefix.startswith(("Bg", "Bk")) else prefix[1:]
            records_path = run_dir / split / "records.jsonl"
            records_exists = records_path.is_file()
            if require_records and not records_exists:
                continue
            rows.append({
                "run_id": run_id,
                "run_dir": str(run_dir),
                "source": "repro",
                "seed": seed,
                "property_name": property_name,
                "model_size": model_size,
                "modality": modality,
                "variant": variant,
                "split": split,
                "records_path": str(records_path),
                "records_exists": records_exists,
                "expected_test_size": (
                    _JARVIS_TEST_SIZE
                    if property_name == "bulk_modulus_kv"
                    else _MP_TEST_SIZE.get(modality, _MP_TEST_SIZE_DEFAULT)
                ),
            })
    return rows


def discover_runs(
    runs_root: Path,
    split: str = TARGET_SPLIT,
    require_records: bool = True,
) -> list[dict]:
    rows = _discover_repro_runs(runs_root, split, require_records)
    rows.sort(key=lambda x: x["run_id"])
    return rows


def check_basic_integrity(run_meta: dict) -> dict:
    out = dict(run_meta)

    out.update(
        {
            "n_records": 0,
            "n_json_error": 0,
            "n_index_valid": 0,
            "n_index_unique": 0,
            "n_index_dup": 0,
            "index_min": None,
            "index_max": None,
            "n_index_missing_between_min_max": None,
            "n_hidden_path_nonempty": 0,
            "n_hidden_exists": 0,
            "n_hidden_missing": 0,
            "n_mean_nll_notna": 0,
            "n_mean_nll_nan": 0,
            "is_complete_by_count": False,
        }
    )

    records_path = Path(str(out.get("records_path", "")))
    if not records_path.is_file():
        return out

    index_values: list[int] = []

    with records_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            out["n_records"] += 1
            line = line.strip()
            if not line:
                out["n_json_error"] += 1
                continue

            try:
                row = json.loads(line)
            except Exception:
                out["n_json_error"] += 1
                continue

            idx = row.get("index")
            try:
                idx_int = int(idx)
                index_values.append(idx_int)
            except Exception:
                pass

            hidden_path = row.get("hidden_last_token_path")
            if isinstance(hidden_path, str) and hidden_path.strip():
                out["n_hidden_path_nonempty"] += 1
                if Path(hidden_path).exists():
                    out["n_hidden_exists"] += 1

            mean_nll = row.get("mean_nll")
            is_notna = False
            if mean_nll is not None:
                try:
                    v = float(mean_nll)
                    if not math.isnan(v):
                        is_notna = True
                except Exception:
                    is_notna = False
            if is_notna:
                out["n_mean_nll_notna"] += 1
            else:
                out["n_mean_nll_nan"] += 1

    out["n_index_valid"] = len(index_values)
    unique_idx = set(index_values)
    out["n_index_unique"] = len(unique_idx)
    out["n_index_dup"] = out["n_index_valid"] - out["n_index_unique"]

    if unique_idx:
        idx_min = min(unique_idx)
        idx_max = max(unique_idx)
        out["index_min"] = idx_min
        out["index_max"] = idx_max
        out["n_index_missing_between_min_max"] = (idx_max - idx_min + 1) - len(unique_idx)

    out["n_hidden_missing"] = out["n_hidden_path_nonempty"] - out["n_hidden_exists"]

    expected = out.get("expected_test_size")
    out["is_complete_by_count"] = (
        isinstance(expected, int) and out["n_records"] >= expected
    )

    return out


def compute_parse_stats(run_meta: dict) -> dict:
    out = {
        "n_parse_total": 0,
        "n_parse_success": 0,
        "n_parse_fail": 0,
        "parse_success_rate": None,
        "n_generated_text_missing": 0,
        "n_generated_text_empty": 0,
        "parsed_indexes": [],
        "failed_indexes": [],
    }

    records_path = Path(str(run_meta.get("records_path", "")))
    if not records_path.is_file():
        return out

    with records_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue

            out["n_parse_total"] += 1

            idx = row.get("index")
            try:
                idx_int = int(idx)
            except Exception:
                idx_int = None

            text = row.get("generated_text")
            if text is None:
                out["n_generated_text_missing"] += 1
                out["n_parse_fail"] += 1
                if idx_int is not None:
                    out["failed_indexes"].append(idx_int)
                continue

            text_str = str(text)
            if not text_str.strip():
                out["n_generated_text_empty"] += 1
                out["n_parse_fail"] += 1
                if idx_int is not None:
                    out["failed_indexes"].append(idx_int)
                continue

            pred = F.extract_property_value(text_str)
            if pred is None:
                out["n_parse_fail"] += 1
                if idx_int is not None:
                    out["failed_indexes"].append(idx_int)
                continue

            out["n_parse_success"] += 1
            if idx_int is not None:
                out["parsed_indexes"].append(idx_int)

    if out["n_parse_total"] > 0:
        out["parse_success_rate"] = out["n_parse_success"] / out["n_parse_total"]

    return out


def compute_parse_bias_ks(
    run_meta: dict,
    y_true_df,
    parse_stats: dict | None = None,
    alpha: float = 0.05,
) -> dict:
    out = {
        "n_parsed_y": 0,
        "n_failed_y": 0,
        "ks_stat": None,
        "ks_pvalue": None,
        "parse_bias_flag": None,
        "median_parsed": None,
        "median_failed": None,
        "iqr_parsed": None,
        "iqr_failed": None,
        "median_gap": None,
    }

    property_name = run_meta.get("property_name")
    if property_name not in ("formation_energy_per_atom", "band_gap"):
        return out
    if y_true_df is None or property_name not in y_true_df.columns:
        return out

    stats = parse_stats if parse_stats is not None else compute_parse_stats(run_meta)
    parsed_indexes = stats.get("parsed_indexes", [])
    failed_indexes = stats.get("failed_indexes", [])

    y_col = y_true_df[property_name].to_numpy()
    n = len(y_col)

    parsed_vals = []
    for idx in parsed_indexes:
        if isinstance(idx, int) and 0 <= idx < n:
            v = y_col[idx]
            if v is not None and not np.isnan(v):
                parsed_vals.append(float(v))

    failed_vals = []
    for idx in failed_indexes:
        if isinstance(idx, int) and 0 <= idx < n:
            v = y_col[idx]
            if v is not None and not np.isnan(v):
                failed_vals.append(float(v))

    out["n_parsed_y"] = len(parsed_vals)
    out["n_failed_y"] = len(failed_vals)
    if len(parsed_vals) == 0 or len(failed_vals) == 0:
        return out

    p_arr = np.asarray(parsed_vals, dtype=float)
    f_arr = np.asarray(failed_vals, dtype=float)

    out["median_parsed"] = float(np.median(p_arr))
    out["median_failed"] = float(np.median(f_arr))
    out["iqr_parsed"] = float(np.percentile(p_arr, 75) - np.percentile(p_arr, 25))
    out["iqr_failed"] = float(np.percentile(f_arr, 75) - np.percentile(f_arr, 25))
    out["median_gap"] = float(out["median_parsed"] - out["median_failed"])

    if len(parsed_vals) >= 2 and len(failed_vals) >= 2:
        ks = ks_2samp(p_arr, f_arr)
        out["ks_stat"] = float(ks.statistic)
        out["ks_pvalue"] = float(ks.pvalue)
        out["parse_bias_flag"] = bool(ks.pvalue < alpha)

    return out


def classify_status(
    row: dict,
    parse_rate_warn: float = 0.90,
    ks_alpha: float = 0.05,
) -> dict:
    out = dict(row)
    fail_reasons: list[str] = []
    warn_reasons: list[str] = []

    if int(out.get("n_index_dup", 0)) > 0:
        fail_reasons.append("index_dup")
    if int(out.get("n_hidden_missing", 0)) > 0:
        fail_reasons.append("hidden_missing")

    if not bool(out.get("is_complete_by_count", False)):
        warn_reasons.append("incomplete_count")

    rate = out.get("parse_success_rate")
    if rate is None:
        warn_reasons.append("parse_rate_none")
    else:
        try:
            if float(rate) < parse_rate_warn:
                warn_reasons.append("low_parse_rate")
        except Exception:
            warn_reasons.append("parse_rate_invalid")

    pval = out.get("ks_pvalue")
    if pval is not None:
        try:
            if float(pval) < ks_alpha:
                warn_reasons.append("parse_bias_ks")
        except Exception:
            pass

    if fail_reasons:
        status = "FAIL"
        reasons = fail_reasons + warn_reasons
    elif warn_reasons:
        status = "WARN"
        reasons = warn_reasons
    else:
        status = "OK"
        reasons = []

    out["status"] = status
    out["status_reasons"] = ";".join(reasons)
    return out


def write_outputs(df_audit, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "audit_table.csv"

    df_out = df_audit.copy()
    for col in ("parsed_indexes", "failed_indexes"):
        if col in df_out.columns:
            df_out = df_out.drop(columns=[col])

    status_rank = {"FAIL": 0, "WARN": 1, "OK": 2}
    if "status" in df_out.columns:
        df_out["status_rank"] = df_out["status"].map(status_rank).fillna(9).astype(int)
        df_out = df_out.sort_values(["status_rank", "run_id"]).drop(columns=["status_rank"])
    else:
        df_out = df_out.sort_values(["run_id"])

    df_out.to_csv(csv_path, index=False)
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Check parse integrity for all discovered runs")
    parser.add_argument("--runs_root", type=Path, required=True, help="Root directory containing inference run outputs")
    parser.add_argument("--test_csv", type=Path, required=True, help="Test CSV with ground-truth property values")
    parser.add_argument("--out_dir", type=Path, required=True, help="Directory to write audit_table.csv")
    args = parser.parse_args()

    y_true_df = pd.read_csv(args.test_csv)
    runs = discover_runs(runs_root=args.runs_root, split=TARGET_SPLIT, require_records=True)
    print(f"[INFO] discovered {len(runs)} runs")

    audited = []
    for r in runs:
        row = check_basic_integrity(r)
        row.update(compute_parse_stats(r))
        row.update(compute_parse_bias_ks(r, y_true_df, parse_stats=row))
        row = classify_status(row, parse_rate_warn=0.90, ks_alpha=0.05)
        audited.append(row)

    n_fail = sum(1 for r in audited if r["status"] == "FAIL")
    n_warn = sum(1 for r in audited if r["status"] == "WARN")
    n_ok   = sum(1 for r in audited if r["status"] == "OK")
    print(f"[INFO] FAIL={n_fail} WARN={n_warn} OK={n_ok}")

    df_audit = pd.DataFrame(audited)
    csv_path = write_outputs(df_audit, args.out_dir)
    print(f"[INFO] wrote: {csv_path}")


if __name__ == "__main__":
    main()
