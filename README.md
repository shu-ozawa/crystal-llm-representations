# crystal-llm-representations

Code for the paper: **"Input Representation and Token-Level Confidence for LLMs in Materials Property Prediction"**

## Overview

This repository provides code for fine-tuning Llama 3 models with LoRA to predict properties of inorganic crystals from text, and for scoring the predictions and the negative log-likelihood (NLL) of the predicted values. The data come from [LLM4Mat-Bench](https://github.com/vertaix/LLM4Mat-Bench).

- **Input representations**: Composition, Crystal Summary, Local Environment, Full Description, CIF
- **Properties**: formation energy per atom and bandgap (Materials Project), bulk modulus (JARVIS-DFT)
- **Models**: Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct, each fine-tuned with seeds 43, 44, and 45

The fine-tuned LoRA adapters and training logs are available on Hugging Face: [ozashu/crystal-llm-representations](https://huggingface.co/ozashu/crystal-llm-representations)

## Requirements

```bash
pip install -r requirements.txt
```

The experiments were run with Python 3.11, torch 2.10.0, transformers 5.2.0, peft 0.18.1, datasets 4.6.1, pandas 3.0.1, and flash-attn 2.8.3. Training uses FlashAttention 2 by default; without flash-attn, pass `--attn_impl sdpa`.

Run all commands from the repository root. The evaluation scripts import the `evaluation` package, so run them with `python -m`.

## Usage

The commands below assume the LLM4Mat-Bench splits are placed in `data/raw/llm4mat_bench/mp/` and `data/raw/llm4mat_bench/jarvis_dft/`, each containing `train.csv`, `validation.csv`, and `test.csv`.

### 0. Prepare the CSV splits

**Materials Project.** The Materials Project `test.csv` in LLM4Mat-Bench repeats 53 materials eight times each in its metadata columns (10,318 rows for 9,947 materials), whereas its structure columns list the 9,947 structures once each followed by 371 empty rows. From the first repeated material onward, the CIF on a row belongs to a different material. This script pairs the two sides back up and writes one row per material, checking that the formula in every CIF matches `formula_pretty`. No material is dropped.

```bash
python scripts/fix_test_csv_cif.py \
    --test-csv data/raw/llm4mat_bench/mp/test.csv \
    --out-csv data/raw/llm4mat_bench/mp/test_fixed.csv
```

The CIF representation is built from `test_fixed.csv` (see step 1) and scored against it. The other four representations do not read the structure columns, so they use `test.csv` as distributed; the scoring keeps the first occurrence of each material, so every representation is scored on the same 9,947 materials.

**JARVIS-DFT.** The JARVIS-DFT splits name the formula column `formula`, and most of their materials have no bulk modulus. Rename the column to `formula_pretty` and keep only the 2,340 test materials with a bulk modulus. Materials without a label are dropped from the training and validation splits when the datasets are built in step 1.

```bash
python - <<'EOF'
from pathlib import Path
import pandas as pd

src = Path("data/raw/llm4mat_bench/jarvis_dft")
dst = Path("data/raw/llm4mat_bench/jarvis_dft_bulk")
dst.mkdir(parents=True, exist_ok=True)
for split in ["train", "validation", "test"]:
    df = pd.read_csv(src / f"{split}.csv", na_filter=False, low_memory=False)
    df = df.rename(columns={"formula": "formula_pretty"})
    if split == "test":
        df = df[df["bulk_modulus_kv"].str.strip().ne("")]
    df.to_csv(dst / f"{split}.csv", index=False)
EOF
```

### 1. Prepare datasets

Converts the CSV splits into chat-format Hugging Face datasets, saved as `{out_root}/{prop}/{input_type}/{split}/`. Materials with an empty input, and in the training and validation splits materials with an empty label, are skipped.

```bash
# Materials Project
python src/prepare_dataset_lora.py \
    --data_root data/raw/llm4mat_bench/mp \
    --out_root data/prepared/mp \
    --prop_names formation_energy_per_atom band_gap

# JARVIS-DFT
python src/prepare_dataset_lora.py \
    --data_root data/raw/llm4mat_bench/jarvis_dft_bulk \
    --out_root data/prepared/jarvis \
    --prop_names bulk_modulus_kv
```

**Arguments**
- `--data_root`: Directory containing `train.csv`, `validation.csv`, and `test.csv`; a split whose file is absent is skipped
- `--out_root`: Output directory
- `--prop_names`: Target property columns: `formation_energy_per_atom`, `band_gap`, `bulk_modulus_kv`
- `--input_types`: Input representations to process. Defaults to all five types
- `--max_words`: Optional word limit for `description` and `cif_structure` inputs. Not used in the paper, where over-long inputs are shortened at tokenization instead (steps 2 and 3)

**Input types**
| Key | Description |
|---|---|
| `formula_pretty` | Chemical formula only (Composition) |
| `descr_1st_sentence` | First sentence of natural-language description (Crystal Summary) |
| `descr_last` | Remaining sentences of description (Local Environment) |
| `description` | Full natural-language description (Full Description) |
| `cif_structure` | Raw CIF string |

**CIF test split for the Materials Project.** The CIF test split built above comes from the misaligned `test.csv` and should not be used. Build it from `test_fixed.csv` instead. `prepare_dataset_lora.py` reads `test.csv` from `--data_root`, so point it at a directory where `test.csv` is `test_fixed.csv`; with no `train.csv` or `validation.csv` there, only the test split is built.

```bash
mkdir -p data/raw/llm4mat_bench/mp_test_fixed
ln -s ../mp/test_fixed.csv data/raw/llm4mat_bench/mp_test_fixed/test.csv
python src/prepare_dataset_lora.py \
    --data_root data/raw/llm4mat_bench/mp_test_fixed \
    --out_root data/prepared/mp_test_fixed \
    --prop_names formation_energy_per_atom band_gap \
    --input_types cif_structure
```

### 2. Fine-tune with LoRA

```bash
python src/train_lora_model.py \
    --data_root data/prepared/mp \
    --prop_name formation_energy_per_atom \
    --input_type description \
    --model_id meta-llama/Llama-3.1-8B-Instruct \
    --lora_r 128 \
    --lora_alpha 256 \
    --seed 43 \
    --output_dir outputs/lora_train/formation_energy_per_atom/seed43/G8B-description-r128 \
    --save_adapter_dir outputs/lora_adapter/formation_energy_per_atom/seed43/G8B-description-r128
```

The defaults of the remaining arguments are the settings used in the paper: LoRA on the query and value projections with dropout 0.05, 6 epochs, a learning rate of 1e-4 with a cosine schedule and a warmup ratio of 0.08, AdamW with a weight decay of 0.01, a batch size of 1 with 32 gradient accumulation steps, and the loss computed over the assistant response only (`--loss_mode completion`). The rank must be given explicitly, since the default (`--lora_r 32`) is not the one adopted in the paper: `--lora_r 256 --lora_alpha 512` for 1B and 3B, and `--lora_r 128 --lora_alpha 256` for 8B. Every condition is trained with `--seed 43`, `44`, and `45`.

The model is evaluated on the validation split after every epoch, and the checkpoint with the lowest validation loss is saved to `--save_adapter_dir`. A checkpoint is kept for every epoch in `--output_dir`. A sequence longer than `--max_length` (8,192 tokens) is shortened by trimming the end of the user message, which keeps the system prompt and the response intact. The date in the chat template is pinned with `--date_string` (default `26 Jul 2024`).

### 3. Run inference

```bash
# Fine-tuned model (LoRA adapter)
python src/run_inference_with_hidden.py \
    --dataset_dir data/prepared/mp/formation_energy_per_atom/description \
    --adapter_dir outputs/lora_adapter/formation_energy_per_atom/seed43/G8B-description-r128 \
    --output_dir outputs/mp/eval_description/G8B-description-r128-seed43/test \
    --max_new_tokens 512 \
    --max_input_length 8192 \
    --date_string "26 Jul 2024"

# Base model (no fine-tuning)
python src/run_inference_with_hidden.py \
    --dataset_dir data/prepared/mp/formation_energy_per_atom/description \
    --model_id meta-llama/Llama-3.1-8B-Instruct \
    --output_dir outputs/mp/eval_description/G8B-description-base-seed42/test \
    --max_new_tokens 1024 \
    --stop_on_json_close \
    --max_input_length 8192 \
    --date_string "26 Jul 2024"
```

Decoding is greedy. The base models tend to go on generating properties they were not asked for, so their generation stops once the JSON closes. As in training, a prompt longer than `--max_input_length` is shortened by trimming the end of the user message. For CIF on the Materials Project, use the dataset built from `test_fixed.csv` (`data/prepared/mp_test_fixed/{prop}/cif_structure`).

**Output**
- `records.jsonl`: One record per test row, in dataset order, including `index`, `generated_text`, `generated_token_ids`, `token_logprobs`, `mean_nll` (over the whole generation), `stop_reason`, `prompt_hash`, and `prompt_fitted`
- `hidden_last_token/`: Last-token hidden states of every layer, one `.pt` file per record (not used in the paper)
- `meta.json`: Run configuration

**Run layout.** The evaluation scripts discover runs by their directory names and infer the property from where a run is placed:

| Property | Path under `--runs_root` | Run name |
|---|---|---|
| Formation energy | `{run}/test/records.jsonl` | `G{size}-{modality}-{variant}-seed{seed}` |
| Bandgap | `band_gap/{run}/test/records.jsonl` | `Bg{size}-{modality}-{variant}-seed{seed}` |
| Bulk modulus | `bulk_modulus_kv/{run}/test/records.jsonl` | `Bk{size}-{modality}-{variant}-seed{seed}` |

`{modality}` is one of `formula`, `descr1st`, `descr_last`, `description`, and `cif`, and `{variant}` labels the configuration, e.g. `r128` or `base`. Every run under `--runs_root` is scored against one `--test_csv`, so runs that need different test CSVs go under different roots:

| Runs | `--runs_root` | `--test_csv` |
|---|---|---|
| Materials Project, except CIF | `outputs/mp/eval_{modality}` | `data/raw/llm4mat_bench/mp/test.csv` |
| Materials Project, CIF | `outputs/mp/eval_cif` | `data/raw/llm4mat_bench/mp/test_fixed.csv` |
| JARVIS-DFT | `outputs/jarvis/eval_{modality}` | `data/raw/llm4mat_bench/jarvis_dft_bulk/test.csv` |

### 4. Evaluate

**Parse integrity check**
```bash
python -m evaluation.evaluate_parse_integrity \
    --runs_root outputs/mp/eval_description \
    --test_csv data/raw/llm4mat_bench/mp/test.csv \
    --out_dir results/mp/eval_description
```
Outputs `audit_table.csv` with, for each run, the record count against the expected test size, the parse success rate, a Kolmogorov-Smirnov test of whether parse failures depend on the true value, and a status of OK, WARN, or FAIL.

**MAE / RMSE**
```bash
python -m evaluation.evaluate_metrics \
    --runs_root outputs/mp/eval_description \
    --test_csv data/raw/llm4mat_bench/mp/test.csv \
    --out_dir results/mp/eval_description
```
Outputs `metrics_table.csv` with the MAE, RMSE, and parse rate of each run, and the numbers of parse failures, materials without a label, and clamped predictions. Only the first occurrence of each material is scored (`--keep_duplicates` scores every row). Parse failures and missing labels are counted separately, e.g. the 59 bandgap materials without a label.

**Integer-part mean NLL**

The confidence score in the paper is the mean NLL over the integer-part tokens of the predicted value, i.e. the sign, the integer digits, and the decimal point. `load_run_nll` computes it for every record (`integer_nll`), together with the full-numeric mean NLL over all digits (`number_nll`) and the mean NLL over the fractional digits (`decimal_nll`), and joins onto the scored predictions from `load_run` by row:

```python
from pathlib import Path

import pandas as pd

from evaluation.evaluate_metrics import load_run, load_run_nll

records = Path("outputs/mp/eval_description/G8B-description-r128-seed43/test/records.jsonl")
test_df = pd.read_csv("data/raw/llm4mat_bench/mp/test.csv")

run = load_run(records, test_df, "formation_energy_per_atom").join(
    load_run_nll(records).set_index("row"), on="row"
)
run = run[run["is_first"] & run["is_labeled"] & run["parse_success"]]
print(run[["material_id", "target", "prediction", "absolute_error", "integer_nll", "number_nll"]])
```

The Spearman correlations and normalized AURC reported in the paper are computed from these columns.

**Selective prediction by whole-generation NLL**
```bash
python -m evaluation.evaluate_selective_prediction_nll \
    --runs_root outputs/mp/eval_description \
    --test_csv data/raw/llm4mat_bench/mp/test.csv \
    --out_dir results/mp/eval_description
```
Outputs `selective_summary.csv` and `risk_coverage_points.csv`, ranking predictions by the mean NLL over the whole generation and over the JSON span. This script predates the integer-part score and scores every row without collapsing the repeated materials, so it does not reproduce the numbers in the paper.

---
