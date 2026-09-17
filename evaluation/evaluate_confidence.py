"""How well an NLL ranks the errors of the fine-tuned runs.

For every fine-tuned run at the adopted rank, the scored predictions
(load_run) are joined with the per-span NLL of the generation (load_run_nll),
and each score is evaluated by:

- rho          Spearman correlation between the score and the absolute error,
               with tied values given their average rank.
- naurc        normalized AURC, (AURC - AURC_oracle) / (MAE - AURC_oracle).
               Keeping materials in random order gives 1, keeping the smallest
               errors first gives 0.
- mae80_ratio  MAE of the 80% of materials with the lowest score divided by the
               MAE of all materials; mae50_ratio likewise.

Materials are kept from the lowest score upward. Materials that share a score
are taken in random order and the risk-coverage curve is its expectation, so
the result does not depend on how ties are broken.

The scores are integer_nll (the integer-part mean NLL, the score of the
paper), number_nll (the full-numeric mean NLL), decimal_nll and mean_nll. They
are evaluated for each seed and then averaged over the seeds of a condition.
As a baseline that needs several models, the standard deviation of the
predictions over the seeds is compared with the seed mean of integer_nll, both
ranking the error of the mean prediction.

The materials are those of the MAE table, the first occurrence of each
material with a label. A material whose value cannot be parsed or located in
the generation of any seed is dropped from every seed of that condition, so
that the seeds are compared on the same materials. Band gap is reported for
all materials and separately for metals (true value 0) and non-metals.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from evaluation.evaluate_metrics import load_run, load_run_nll, resolve_id_column
from evaluation.evaluate_parse_integrity import discover_runs

# LoRA rank adopted for each model size. Base models and runs at other ranks
# are skipped.
ADOPTED_VARIANT = {"1B": "r256", "3B": "r256", "8B": "r128"}
SCORES = ("integer_nll", "number_nll", "decimal_nll", "mean_nll")
COVERAGES = (0.8, 0.5)
METRICS = ["mae", "rho", "naurc"] + [f"mae{round(c * 100)}_ratio" for c in COVERAGES]
CONDITION = ["subset", "model_size", "modality"]


def risk_curve(score: np.ndarray, error: np.ndarray) -> np.ndarray:
    """MAE of the k materials with the lowest score, for k = 1..n.

    Within a block of tied scores every order is equally likely, so each
    material in the block contributes the block's mean error.
    """
    order = np.argsort(score, kind="stable")
    sorted_score, sorted_error = score[order], error[order]
    n = len(sorted_error)
    starts = np.flatnonzero(np.r_[True, sorted_score[1:] != sorted_score[:-1]])
    sizes = np.diff(np.r_[starts, n])
    block_sum = np.add.reduceat(sorted_error, starts)
    before = np.repeat(np.cumsum(block_sum) - block_sum, sizes)
    position = np.arange(n) - np.repeat(starts, sizes) + 1
    return (before + position * np.repeat(block_sum / sizes, sizes)) / np.arange(1, n + 1)


def ranking_metrics(score: np.ndarray, error: np.ndarray) -> dict:
    curve = risk_curve(score, error)
    mae, aurc = curve[-1], curve.mean()
    oracle = risk_curve(error, error).mean()
    out = {
        "mae": mae,
        "rho": spearmanr(score, error).statistic,
        "naurc": (aurc - oracle) / (mae - oracle),
    }
    for c in COVERAGES:
        out[f"mae{round(c * 100)}_ratio"] = curve[round(c * len(error)) - 1] / mae
    return out


def load_samples(runs: list[tuple[str, str]]) -> pd.DataFrame:
    """One row per (run, material): prediction, error and the NLL scores."""
    frames = []
    for runs_root, test_csv in runs:
        test_df = pd.read_csv(test_csv, na_filter=False, low_memory=False)
        id_col = resolve_id_column(test_df)
        for meta in discover_runs(Path(runs_root)):
            if meta["variant"] != ADOPTED_VARIANT.get(meta["model_size"]):
                continue
            records = Path(meta["records_path"])
            run = load_run(records, test_df, meta["property_name"], id_col=id_col).join(
                load_run_nll(records).set_index("row"), on="row"
            )
            run = run[run["is_first"] & run["is_labeled"]]
            frames.append(pd.DataFrame({
                "property":    meta["property_name"],
                "model_size":  meta["model_size"],
                "modality":    meta["modality"],
                "seed":        meta["seed"],
                "material_id": run[id_col].to_numpy(),
                "target":      run["target"].to_numpy(),
                "prediction":  run["prediction"].to_numpy(),
                "abs_error":   run["absolute_error"].to_numpy(),
                "usable":      (run["parse_success"] & run["span_ok"]).to_numpy(),
                **{score: run[score].to_numpy() for score in SCORES},
            }))
            print(f"  {meta['run_id']:40s} n={len(run)}", flush=True)
    return pd.concat(frames, ignore_index=True)


def evaluate(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Metrics for every seed and score, and for the seed-spread comparison."""
    seed_rows, spread_rows = [], []
    for (prop, size, modality), cond in samples.groupby(
        ["property", "model_size", "modality"], sort=True
    ):
        dropped = cond.loc[~cond["usable"], "material_id"].unique()
        cond = cond[~cond["material_id"].isin(dropped)]
        seeds = sorted(cond["seed"].unique())
        wide = cond.pivot(index="material_id", columns="seed")
        if wide["abs_error"].isna().to_numpy().any():
            raise ValueError(f"{prop}/{size}/{modality}: the seeds do not cover the same materials")

        column = {field: wide[field][seeds].to_numpy() for field in
                  ["target", "prediction", "abs_error", *SCORES]}
        subsets = [(prop, np.ones(len(wide), dtype=bool))]
        if prop == "band_gap":
            metal = column["target"][:, 0] == 0
            subsets += [("band_gap_metal", metal), ("band_gap_nonmetal", ~metal)]

        for subset, mask in subsets:
            key = {"subset": subset, "model_size": size, "modality": modality,
                   "n": int(mask.sum()), "n_dropped": len(dropped)}
            error = column["abs_error"][mask]
            for k, seed in enumerate(seeds):
                for score in SCORES:
                    seed_rows.append({**key, "score": score, "seed": seed,
                                      **ranking_metrics(column[score][mask, k], error[:, k])})

            if len(seeds) > 1:
                prediction = column["prediction"][mask]
                mean_error = np.abs(prediction.mean(axis=1) - column["target"][mask, 0])
                for score, values in [
                    ("integer_nll_seed_mean", column["integer_nll"][mask].mean(axis=1)),
                    ("prediction_seed_sd", prediction.std(axis=1, ddof=1)),
                ]:
                    spread_rows.append({**key, "score": score, **ranking_metrics(values, mean_error)})

    return pd.DataFrame(seed_rows), pd.DataFrame(spread_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank correlation and selective prediction of NLL scores")
    parser.add_argument(
        "--runs", nargs=2, action="append", required=True, metavar=("RUNS_ROOT", "TEST_CSV"),
        help="A runs root and the test CSV its runs are scored against; repeat for every root",
    )
    parser.add_argument("--out_dir", type=Path, required=True, help="Directory to write the output CSVs")
    args = parser.parse_args()

    samples = load_samples(args.runs)
    by_seed, seed_spread = evaluate(samples)

    summary = (
        by_seed.groupby([*CONDITION, "score", "n", "n_dropped"], sort=False)[METRICS]
        .agg(["mean", "std"])
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in [("confidence_by_seed.csv", by_seed),
                     ("confidence_summary.csv", summary),
                     ("confidence_seed_spread.csv", seed_spread)]:
        df.to_csv(args.out_dir / name, index=False)
        print(f"[INFO] wrote: {args.out_dir / name}")

    integer = summary[summary["score"] == "integer_nll"]
    print(integer[[*CONDITION, "n", "rho_mean", "naurc_mean", "mae80_ratio_mean"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
