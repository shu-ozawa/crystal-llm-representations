# crystal-llm-representations

Code for the paper: **"Scale-Dependent Input Representation and Confidence Estimation for LLMs in Materials Property Prediction"**

## Overview

This repository provides code for fine-tuning large language models (LLMs) with LoRA on inorganic crystal property prediction tasks, using data from [LLM4Mat-Bench](https://github.com/vertaix/LLM4Mat-Bench). We evaluate five input representations (Composition, Crystal Summary, Local Environment, Full Description, CIF) across two target properties (formation energy per atom, band gap) with Llama-3.2-1B-Instruct and Llama-3.1-8B-Instruct.

Trained LoRA adapters are available on Hugging Face: [ozashu/crystal-llm-representations](https://huggingface.co/ozashu/crystal-llm-representations)

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

### 0. Fix CIF test data (CIF modality only)

The LLM4Mat-Bench test split contains inconsistencies between material IDs and CIF structural data. These scripts correct the CIF column by re-fetching crystal structures from the Materials Project.

```bash
# Step 1: replace cif_structure column using cached .cif files
python scripts/fix_test_csv_cif.py \
    --test-csv /path/to/test.csv \
    --cif-cache /path/to/cif_cache/

# Step 2: merge corrected CIF into the full 10,318-row test split
python scripts/create_unified_test_csv.py \
    --llm4mat-csv /path/to/test_llm4mat.csv \
    --cif-csv /path/to/test.csv \
    --out-csv /path/to/unified_test.csv

# Step 3: remap record indices in existing CIF inference outputs
python scripts/remap_cif_record_indices.py \
    --backup-csv /path/to/unified_test_cif_backup.csv \
    --new-csv /path/to/unified_test.csv \
    --runs-root /path/to/runs
```

### 1. Prepare dataset

Converts raw CSV splits from LLM4Mat-Bench into chat-format HuggingFace datasets.

```bash
python src/prepare_dataset_lora.py \
    --data_root /path/to/llm4mat_bench/formation_energy_per_atom \
    --out_root /path/to/prepared_data \
    --prop_names formation_energy_per_atom \
    --input_types formula_pretty descr_1st_sentence descr_last description cif_structure
```

**Arguments**
- `--data_root`: Directory containing `train.csv`, `validation.csv`, `test.csv` from LLM4Mat-Bench
- `--out_root`: Output directory (datasets saved as `{out_root}/{prop}/{input_type}/{split}/`)
- `--prop_names`: Target property column name(s). Supported: `formation_energy_per_atom`, `band_gap`
- `--input_types`: Input representations to process. Defaults to all five types
- `--max_words`: Optional word limit for long inputs (description, cif_structure)

**Input types**
| Key | Description |
|---|---|
| `formula_pretty` | Chemical formula only (Composition) |
| `descr_1st_sentence` | First sentence of natural-language description (Crystal Summary) |
| `descr_last` | Remaining sentences of description (Local Environment) |
| `description` | Full natural-language description (Full Description) |
| `cif_structure` | Raw CIF string |

**Note on CIF inputs**: The LLM4Mat-Bench test split contains some inconsistencies between material IDs and structural data. For CIF experiments, crystal structures were re-fetched from the Materials Project. See `scripts/fix_test_csv_cif.py` and `scripts/remap_cif_record_indices.py`.

### 2. Fine-tune with LoRA

```bash
python src/train_lora_model.py \
    --data_root /path/to/prepared_data \
    --prop_name formation_energy_per_atom \
    --input_type descr_1st_sentence \
    --model_id meta-llama/Llama-3.1-8B-Instruct \
    --output_dir /path/to/output \
    --seed 42
```

### 3. Run inference

```bash
# Fine-tuned model (LoRA adapter)
python src/run_inference_with_hidden.py \
    --dataset_dir /path/to/prepared_data/formation_energy_per_atom/descr_1st_sentence \
    --output_dir /path/to/output/run_name/test \
    --adapter_dir /path/to/lora_adapter

# Base model (no fine-tuning)
python src/run_inference_with_hidden.py \
    --dataset_dir /path/to/prepared_data/formation_energy_per_atom/descr_1st_sentence \
    --output_dir /path/to/output/run_name/test \
    --model_id meta-llama/Llama-3.1-8B-Instruct
```

**Output**
- `records.jsonl`: Per-sample results including `generated_text`, `token_logprobs`, `mean_nll`, `parse_rate`
- `hidden_last_token/`: Per-layer last-token hidden states saved as `.pt` files (one per sample)
- `meta.json`: Run configuration

### 4. Evaluate

All evaluation scripts take `--runs_root` (inference output directory), `--test_csv`, and `--out_dir`.

**Parse integrity check**
```bash
python evaluation/evaluate_parse_integrity.py \
    --runs_root /path/to/runs \
    --test_csv /path/to/test.csv \
    --out_dir /path/to/results
```
Outputs `audit_table.csv` with per-run integrity status (OK/WARN/FAIL) and parse success rates.

**MAE / RMSE**
```bash
python evaluation/evaluate_metrics.py \
    --runs_root /path/to/runs \
    --test_csv /path/to/test.csv \
    --out_dir /path/to/results
```
Outputs `metrics_table.csv`.

**Selective prediction (NLL uncertainty)**
```bash
python evaluation/evaluate_selective_prediction_nll.py \
    --runs_root /path/to/runs \
    --test_csv /path/to/test.csv \
    --out_dir /path/to/results
```
Outputs `selective_summary.csv` and `risk_coverage_points.csv`.

---
