"""Scoring for every table and figure in the paper.

load_run() joins one run's generations onto the ground truth; score_run()
aggregates that to the reported numbers. The CLI below goes through the same
two functions, so a number cannot drift between a notebook and a script.

Two decisions are baked in because the benchmark forces them:

- The MP test split repeats 53 materials 8 times each, 371 redundant rows of
  10,318. Scored as-is those materials carry 8x weight, so only the first
  occurrence of each material is kept by default.
- A generation that cannot be parsed and a material with no label are
  different failures, the model's and the benchmark's. The MP band gap
  column has 59 NaN labels. They are counted separately.
"""
import argparse
import json
import re
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from evaluation.evaluation_utils import CLAMP_MAX, CLAMP_MIN, classify_output
from evaluation.evaluate_parse_integrity import discover_runs

TARGET_SPLIT = "test"

# Candidate identifier columns, in priority order: MP then JARVIS.
ID_COLUMNS = ("material_id", "jarvis_id")

_DECODER = json.JSONDecoder()
_INDEX_RE = re.compile(rb'"index":\s*(\d+)')
_TEXT_KEY = b'"generated_text": '


def _read_records(records_path: Path) -> tuple[list[int], list[str]]:
    """Read index and generated_text out of a records.jsonl.

    A record line carries a few thousand token log-probabilities, so the JSON
    is never parsed in full: only the two fields scoring needs are pulled out
    of the raw bytes.
    """
    indices: list[int] = []
    texts: list[str] = []

    with open(records_path, "rb") as f:
        for line in f:
            match = _INDEX_RE.search(line, 0, 200)
            if match is None:
                raise ValueError(f"{records_path}: record without an 'index' field")
            indices.append(int(match.group(1)))

            head = line.find(_TEXT_KEY)
            if head < 0:
                raise ValueError(
                    f"{records_path}: record without a 'generated_text' field"
                )
            text, _ = _DECODER.raw_decode(
                line[head + len(_TEXT_KEY):].decode("utf-8", "replace")
            )
            texts.append(text)

    return indices, texts


def resolve_id_column(test_df: pd.DataFrame) -> str:
    """Return the identifier column of a test split (MP or JARVIS)."""
    for column in ID_COLUMNS:
        if column in test_df.columns:
            return column
    raise KeyError(
        f"No identifier column in the test split; expected one of {ID_COLUMNS}"
    )


def load_run(
    records_path: Path,
    test_df: pd.DataFrame,
    target_col: str,
    *,
    id_col: Optional[str] = None,
    clamp_min: float = CLAMP_MIN,
    clamp_max: float = CLAMP_MAX,
) -> pd.DataFrame:
    """Join one run's generations onto the ground truth, one row per record.

    Raises if the record count does not match the test split, or if the record
    indices are not the row positions 0..n-1. Either would score generations
    against the wrong material without raising anything by itself.

    Columns: row, <id_col>, target, generated_text, parse_success,
    was_clamped, prediction, absolute_error, is_labeled, is_first.
    """
    if id_col is None:
        id_col = resolve_id_column(test_df)

    indices, texts = _read_records(records_path)

    if len(texts) != len(test_df):
        raise ValueError(
            f"{records_path}: {len(texts)} records for a test split of "
            f"{len(test_df)} rows"
        )
    if indices != list(range(len(indices))):
        raise ValueError(
            f"{records_path}: record indices are not the row positions 0..n-1"
        )

    parsed = pd.DataFrame(
        [classify_output(t, clamp_min=clamp_min, clamp_max=clamp_max) for t in texts]
    )
    target = pd.to_numeric(test_df[target_col], errors="coerce").to_numpy(dtype=float)
    identifier = test_df[id_col].to_numpy()

    df = pd.DataFrame(
        {
            "row": np.arange(len(texts)),
            id_col: identifier,
            "target": target,
            "generated_text": texts,
            "parse_success": parsed["parse_success"].to_numpy(dtype=bool),
            "was_clamped": parsed["was_clamped"].to_numpy(dtype=bool),
            "prediction": pd.to_numeric(parsed["prediction"], errors="coerce"),
        }
    )
    df["absolute_error"] = (df["prediction"] - df["target"]).abs()
    df["is_labeled"] = np.isfinite(target)
    df["is_first"] = ~pd.Series(identifier).duplicated(keep="first").to_numpy()
    return df


def score_run(
    records_path: Path,
    test_df: pd.DataFrame,
    target_col: str,
    *,
    id_col: Optional[str] = None,
    drop_duplicates: bool = True,
    restrict_to: Optional[Iterable] = None,
    clamp_min: float = CLAMP_MIN,
    clamp_max: float = CLAMP_MAX,
) -> dict:
    """Aggregate one run to the numbers reported in the paper.

    drop_duplicates
        Keep only the first occurrence of each material. Pass False for
        row-wise scoring over the split as distributed.
    restrict_to
        Optional identifiers to keep, for comparing representations over an
        identical set of materials.

    mae and rmse are averaged over n_scored: materials that were kept, parsed
    successfully and carry a finite label. n_parse_failure and
    n_missing_label can overlap, since an unparseable generation for an
    unlabelled material counts in both.
    """
    if id_col is None:
        id_col = resolve_id_column(test_df)

    df = load_run(
        records_path,
        test_df,
        target_col,
        id_col=id_col,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
    )

    keep = np.ones(len(df), dtype=bool)
    if drop_duplicates:
        keep &= df["is_first"].to_numpy()
    if restrict_to is not None:
        keep &= df[id_col].isin(set(restrict_to)).to_numpy()

    kept = df[keep]
    scored = kept[kept["parse_success"] & kept["is_labeled"]]
    error = scored["absolute_error"].to_numpy()

    return {
        "n_material": int(len(kept)),
        "n_scored": int(len(scored)),
        "n_parse_failure": int((~kept["parse_success"]).sum()),
        "n_missing_label": int((~kept["is_labeled"]).sum()),
        "n_clamped": int(kept["was_clamped"].sum()),
        "parse_rate": float(kept["parse_success"].mean()) if len(kept) else float("nan"),
        "mae": float(error.mean()) if len(error) else float("nan"),
        "rmse": float(np.sqrt((error**2).mean())) if len(error) else float("nan"),
    }


# --- Per-span negative log-likelihood ------------------------------------
#
# Every generation has the fixed shape "{<property>: -2.7483 eV/atom}", so the
# token stream can be split into spans without calling a tokenizer per record.
# These are the token IDs of "{", "}", ":", "." and " e" in the 128k tokenizer
# shared by Llama-3.1 and Llama-3.2 (verified identical for the 1B, 3B and 8B
# checkpoints used here).
TOK_LBRACE, TOK_RBRACE, TOK_COLON, TOK_DOT, TOK_EV = 90, 92, 25, 13, 384


def _span_nll(token_ids: list[int], logprobs: list[float]) -> Optional[dict]:
    """Split one generation into integer / decimal / number spans.

    integer  sign, integer digits and the decimal point -- the tokens from
             just after ":" through ".". Always exactly three tokens, because
             the sign is absorbed into the leading token.
    decimal  the fractional digits, from just after "." to the end of the value.
    number   integer + decimal, i.e. the predicted value as a whole.

    The value is followed by a unit for some properties ("-2.746 eV/atom}") and
    by nothing for others ("{bulk_modulus_kv: 33.13}"), so it ends at whichever
    comes first: the unit token or the closing brace.

    Over 90% of a generation's NLL sits in the decimal span, but the training
    target is rounded to four decimal places, so the third and fourth digits
    are unpredictable in principle. That span is a floor rather than a signal,
    which is why the integer span is the one to compare conditions with.

    Returns None when the expected token sequence is absent, which happens for
    degenerate generations that also fail to parse.
    """
    try:
        ob = token_ids.index(TOK_LBRACE)
        cb = token_ids.index(TOK_RBRACE, ob + 1)
        colon = token_ids.index(TOK_COLON, ob + 1)
        dot = token_ids.index(TOK_DOT, colon + 1)
    except ValueError:
        return None
    if not (ob < colon < dot < cb):
        return None

    try:
        end = min(token_ids.index(TOK_EV, dot + 1), cb)
    except ValueError:
        end = cb
    if end <= dot:
        return None

    number = logprobs[colon + 1:end]
    integer = logprobs[colon + 1:dot + 1]
    decimal = logprobs[dot + 1:end]
    if not (number and integer and decimal):
        return None

    return {
        "integer_nll": -sum(integer) / len(integer),
        "decimal_nll": -sum(decimal) / len(decimal),
        "number_nll": -sum(number) / len(number),
        "n_integer_tok": len(integer),
    }


def load_run_nll(records_path: Path) -> pd.DataFrame:
    """Per-span NLL for one run, indexed by row so it joins onto load_run().

    Note that mean_nll, taken straight from the record, is the average over
    the whole generation. Its denominator changes with the number of decimal
    digits, so it moves with something unrelated to the prediction itself.
    It is kept here as a reference quantity, not as the primary one.

    Unlike load_run(), this parses each record in full: token_logprobs holds a
    few thousand values per record and cannot be skipped. Expect roughly an
    order of magnitude more time than load_run() on the same file.

    Columns: row, mean_nll, span_ok, integer_nll, decimal_nll, number_nll,
    n_integer_tok.
    """
    missing = {
        "integer_nll": float("nan"),
        "decimal_nll": float("nan"),
        "number_nll": float("nan"),
        "n_integer_tok": -1,
    }

    rows = []
    with open(records_path, "r", encoding="utf-8") as f:
        for position, line in enumerate(f):
            record = json.loads(line)
            span = _span_nll(record["generated_token_ids"], record["token_logprobs"])
            rows.append({
                "row": position,
                "mean_nll": record["mean_nll"],
                "span_ok": span is not None,
                **(span or missing),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score all discovered runs into metrics_table.csv"
    )
    parser.add_argument("--runs_root", type=Path, required=True, help="Root directory containing inference run outputs")
    parser.add_argument("--test_csv", type=Path, required=True, help="Test CSV with ground-truth property values")
    parser.add_argument("--out_dir", type=Path, required=True, help="Directory to write metrics_table.csv")
    parser.add_argument(
        "--keep_duplicates",
        action="store_true",
        help="Score every row instead of the first occurrence of each material",
    )
    args = parser.parse_args()

    test_df = pd.read_csv(args.test_csv)
    id_col = resolve_id_column(test_df)
    runs = discover_runs(runs_root=args.runs_root, split=TARGET_SPLIT, require_records=True)
    print(f"[INFO] {len(runs)} runs discovered, id column '{id_col}'")

    rows = []
    for run_meta in runs:
        metrics = score_run(
            Path(run_meta["records_path"]),
            test_df,
            run_meta["property_name"],
            id_col=id_col,
            drop_duplicates=not args.keep_duplicates,
        )
        rows.append({
            "run_id":        run_meta["run_id"],
            "seed":          run_meta["seed"],
            "property_name": run_meta["property_name"],
            "modality":      run_meta["modality"],
            "variant":       run_meta["variant"],
            "model_size":    run_meta["model_size"],
            **metrics,
        })
        print(f"  {run_meta['run_id']:45s}  mae={metrics['mae']:.4f}  n={metrics['n_scored']}")

    df = pd.DataFrame(rows)
    df = df.sort_values(["property_name", "modality", "model_size", "variant", "seed"]).reset_index(drop=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "metrics_table.csv"
    df.to_csv(out_path, index=False)
    print(f"[INFO] wrote: {out_path}")


if __name__ == "__main__":
    main()
