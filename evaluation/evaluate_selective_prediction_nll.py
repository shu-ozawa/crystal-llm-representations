import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

import evaluation.evaluation_utils as F
from evaluation.evaluate_parse_integrity import discover_runs

TARGET_SPLIT = "test"
NLL_QUANTILES = (0.10, 0.25, 0.50)


def build_run_eval_df(
    run_meta: dict,
    y_true_arr: np.ndarray,
    tokenizer_cache: dict,
) -> pd.DataFrame:
    """
    Build per-record evaluation dataframe for one run.

    Columns: index, y_true, y_pred, abs_error, mean_nll, json_mean_nll, parse_ok, json_nll_ok
    """
    records_path = Path(run_meta["records_path"])
    run_id = run_meta["run_id"]

    # load tokenizer from meta.json
    tokenizer = None
    meta_path = Path(run_meta["run_dir"]) / TARGET_SPLIT / "meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            base_model_id = meta.get("base_model_id", "")
            if base_model_id:
                if base_model_id not in tokenizer_cache:
                    tokenizer_cache[base_model_id] = AutoTokenizer.from_pretrained(
                        base_model_id, use_fast=True, trust_remote_code=True, local_files_only=True,
                    )
                tokenizer = tokenizer_cache[base_model_id]
        except Exception:
            pass

    def first_braced_span(text: str) -> tuple[int, int] | None:
        start = text.find("{")
        if start < 0:
            return None
        end = text.find("}", start + 1)
        return (start, end + 1) if end >= 0 else None

    def token_span_from_char_span(frags: list[str], span: tuple[int, int]) -> tuple[int, int] | None:
        s_char, e_char = span
        pos, s_tok, e_tok = 0, None, None
        for i, frag in enumerate(frags):
            nxt = pos + len(frag)
            if (nxt > s_char) and (pos < e_char):
                if s_tok is None:
                    s_tok = i
                e_tok = i + 1
            pos = nxt
        return (s_tok, e_tok) if s_tok is not None else None

    rows = []
    with records_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            try:
                idx_int = int(rec.get("index"))
            except Exception:
                continue
            if not (0 <= idx_int < len(y_true_arr)):
                continue

            text = str(rec.get("generated_text") or "")
            y_pred = F.extract_property_value(text)
            parse_ok = y_pred is not None

            y_true = y_true_arr[idx_int]
            abs_error = abs(float(y_pred) - float(y_true)) if parse_ok and np.isfinite(y_true) else np.nan

            try:
                mean_nll = float(rec.get("mean_nll"))
                if np.isnan(mean_nll):
                    mean_nll = np.nan
            except Exception:
                mean_nll = np.nan

            # json_mean_nll: NLL averaged over tokens in the {value} span
            json_mean_nll = np.nan
            json_nll_ok = False
            span = first_braced_span(text)
            token_ids = rec.get("generated_token_ids")
            token_logprobs = rec.get("token_logprobs")

            if tokenizer is not None and span is not None and isinstance(token_ids, list) and isinstance(token_logprobs, list):
                n_tok = min(len(token_ids), len(token_logprobs))
                frags = [tokenizer.decode([int(tid)], skip_special_tokens=True, clean_up_tokenization_spaces=False) for tid in token_ids[:n_tok]]
                tok_span = token_span_from_char_span(frags, span)
                if tok_span is not None:
                    s_tok, e_tok = tok_span
                    vals = [float(lp) for lp in token_logprobs[s_tok:e_tok] if np.isfinite(float(lp))]
                    if vals:
                        json_mean_nll = float(-np.mean(vals))
                        json_nll_ok = True

            rows.append({
                "run_id":        run_id,
                "index":         idx_int,
                "y_true":        float(y_true) if np.isfinite(y_true) else np.nan,
                "y_pred":        float(y_pred) if parse_ok else np.nan,
                "abs_error":     abs_error,
                "mean_nll":      mean_nll,
                "json_mean_nll": json_mean_nll,
                "parse_ok":      parse_ok,
                "json_nll_ok":   json_nll_ok,
            })

    return pd.DataFrame(rows)


def compute_selective_metrics(
    df_run: pd.DataFrame,
    score_col: str,
    quantiles: tuple[float, ...] = NLL_QUANTILES,
    coverage_grid: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Compute risk-coverage curve and selective MAE summary for one run × score_col.

    Returns (curve_df, summary_dict).
    """
    run_id = str(df_run["run_id"].iloc[0]) if len(df_run) > 0 else "unknown"

    empty = {"run_id": run_id, "score_col": score_col, "n_total": 0,
             "baseline_mae": np.nan, "aurc": np.nan}
    for q in quantiles:
        tag = int(round(q * 100))
        empty[f"q{tag}_selective_mae"] = np.nan
        empty[f"q{tag}_improve_pct"] = np.nan

    d = df_run[df_run["abs_error"].notna() & df_run[score_col].notna()].copy()
    if len(d) < 30:
        return pd.DataFrame(), empty

    d = d.sort_values(score_col, ascending=True).reset_index(drop=True)
    errs = d["abs_error"].to_numpy(dtype=float)
    scores = d[score_col].to_numpy(dtype=float)
    n = len(d)
    baseline_mae = float(np.mean(errs))

    if coverage_grid is None:
        coverage_grid = np.unique(np.append(np.arange(0.05, 1.0, 0.05), 1.0))

    curve_rows = []
    for cov in coverage_grid:
        k = max(1, int(round(cov * n)))
        risk_mae = float(np.mean(errs[:k]))
        curve_rows.append({
            "run_id": run_id, "score_col": score_col,
            "coverage": float(cov), "n_selected": k, "n_total": n,
            "risk_mae": risk_mae, "baseline_mae": baseline_mae,
            "risk_reduction_pct": float(100.0 * (1.0 - risk_mae / baseline_mae)) if baseline_mae > 0 else np.nan,
        })

    curve_df = pd.DataFrame(curve_rows)
    aurc = float(np.trapezoid(curve_df["risk_mae"].to_numpy(), curve_df["coverage"].to_numpy()))

    summary = {"run_id": run_id, "score_col": score_col, "n_total": n,
               "baseline_mae": baseline_mae, "aurc": aurc}
    for q in quantiles:
        tag = int(round(q * 100))
        thr = float(np.quantile(scores, q))
        mask = scores <= thr
        sel_mae = float(np.mean(errs[mask])) if mask.sum() > 0 else np.nan
        improve = float(100.0 * (1.0 - sel_mae / baseline_mae)) if baseline_mae > 0 and np.isfinite(sel_mae) else np.nan
        summary[f"q{tag}_selective_mae"] = sel_mae
        summary[f"q{tag}_improve_pct"] = improve

    return curve_df, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute selective prediction metrics using mean NLL")
    parser.add_argument("--runs_root", type=Path, required=True, help="Root directory containing inference run outputs")
    parser.add_argument("--test_csv", type=Path, required=True, help="Test CSV with ground-truth property values")
    parser.add_argument("--out_dir", type=Path, required=True, help="Directory to write output CSVs")
    args = parser.parse_args()

    runs = discover_runs(runs_root=args.runs_root, split=TARGET_SPLIT, require_records=True)
    print(f"[INFO] {len(runs)} runs discovered")

    y_true_df_cache: dict = {}
    tokenizer_cache: dict = {}
    all_curves = []
    all_summaries = []

    for i, run_meta in enumerate(runs, 1):
        run_id = run_meta["run_id"]
        property_name = run_meta["property_name"]
        print(f"[INFO] ({i}/{len(runs)}) {run_id}")

        if property_name not in y_true_df_cache:
            y_true_df_cache[property_name] = pd.read_csv(args.test_csv)
        y_true_arr = y_true_df_cache[property_name][property_name].to_numpy(dtype=float)

        df_run = build_run_eval_df(run_meta, y_true_arr, tokenizer_cache)
        if df_run.empty:
            continue

        for score_col in ("mean_nll", "json_mean_nll"):
            curve_df, summary = compute_selective_metrics(df_run, score_col=score_col)
            summary.update({k: run_meta[k] for k in ("seed", "property_name", "modality", "variant", "model_size")})
            all_summaries.append(summary)
            if not curve_df.empty:
                for k in ("seed", "property_name", "modality", "variant", "model_size"):
                    curve_df[k] = run_meta[k]
                all_curves.append(curve_df)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = args.out_dir / "selective_summary.csv"
    pd.DataFrame(all_summaries).to_csv(summary_path, index=False)
    print(f"[INFO] wrote: {summary_path}")

    if all_curves:
        curves_path = args.out_dir / "risk_coverage_points.csv"
        pd.concat(all_curves, ignore_index=True).to_csv(curves_path, index=False)
        print(f"[INFO] wrote: {curves_path}")


if __name__ == "__main__":
    main()
