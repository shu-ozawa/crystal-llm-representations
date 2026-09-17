from __future__ import annotations
from typing import Optional, List, Dict, Any
import argparse
from pathlib import Path
import pandas as pd
from datasets import Dataset, DatasetDict

system_prompt_formula = """
You are a material scientist. 
Look at the chemical formula of the given crystalline material and predict its property.
The output must be in a json format. For example: {property_name:predicted_property_value}.
Answer as precise as possible and in few words as possible.
"""

system_prompt_struct = """
You are a material scientist. 
Look at the cif structure information of the given crystalline material and predict its property.
The output must be in a json format. For example: {property_name:predicted_property_value}.
Answer as precise as possible and in few words as possible.
"""

system_prompt_descr = """
You are a material scientist. 
Look at the structure description of the given crystalline material and predict its property.
The output must be in a json format. For example: {property_name:predicted_property_value}.
Answer as precise as possible and in few words as possible.
"""

def trim_words(text: str, max_words: Optional[int]) -> str:
    if max_words is None:
        return str(text)
    return " ".join(str(text).split()[:max_words])

def user_prompt_formula(sentence, property_name=''):
    user_prompt =  f"""chemical formula: {sentence}.\nproperty name: {property_name}."""
    return user_prompt

def user_prompt_descr(sentence, property_name='', max_words: Optional[int] = None):
    sentence = trim_words(sentence, max_words)
    user_prompt =  f"""description: {sentence}.\nproperty name: {property_name}."""
    return user_prompt

def user_prompt_1st_descr(sentence, property_name=''):
    first_sentence = sentence.split(".")[0] + "."
    user_prompt =  f"""description: {first_sentence}\nproperty name: {property_name}."""
    return user_prompt

def user_prompt_descr_last(sentence, property_name=''):
    remaining = ".".join(sentence.split(".")[1:]).strip()
    user_prompt =  f"""description: {remaining}\nproperty name: {property_name}."""
    return user_prompt

def user_prompt_cif(sentence, property_name='', max_words: Optional[int] = None):
    sentence = trim_words(sentence, max_words)
    user_prompt =  f"""cif structure: {sentence}.\nproperty name: {property_name}."""
    return user_prompt

PROMPTS = {
"formula_pretty": (system_prompt_formula, user_prompt_formula),
"descr_1st_sentence": (system_prompt_descr, user_prompt_1st_descr),
"descr_last": (system_prompt_descr, user_prompt_descr_last),
"description": (system_prompt_descr, user_prompt_descr),
"cif_structure": (system_prompt_struct, user_prompt_cif),
}

INPUT_TYPES = ["formula_pretty", "descr_1st_sentence", "descr_last", "description", "cif_structure"]

# A property missing from this table gets a target with no unit at all, which
# makes its output format differ from the other properties. Add an entry when
# introducing a property.
UNIT_BY_PROP = {
    "formation_energy_per_atom": "eV/atom",
    "band_gap": "eV",
    "bulk_modulus_kv": "GPa",
}


def format_assistant_content(prop_name: str, value: Any) -> str:
    unit = UNIT_BY_PROP.get(prop_name)
    if unit:
        return f"{{{prop_name}: {value} {unit}}}"
    return f"{{{prop_name}: {value}}}"

def build_split(
        csv_path: str,
        prop_name: str,
        input_type: str,
        max_words: Optional[int] = None,
        is_test: bool = False,
    ) -> Dataset:
    # na_filter=False keeps the composition "NaN" (sodium nitride) as a string
    # instead of turning it into a missing value.
    df = pd.read_csv(csv_path, na_filter=False)

    # Special handling for descr_1st_sentence and descr_last: use description column
    actual_input_col = "description" if input_type in {"descr_1st_sentence", "descr_last"} else input_type

    if actual_input_col not in df.columns:
        raise KeyError(f"Column '{actual_input_col}' not found in {csv_path}. Available: {list(df.columns)[:12]}...")
    if not is_test and prop_name not in df.columns:
        raise KeyError(f"Property column '{prop_name}' not found in {csv_path}.")

    use_cols = [actual_input_col] + ([] if is_test else [prop_name])
    sub = df[use_cols]
    # Filter on empty strings rather than dropna(): with na_filter=False the
    # missing entries become "", so dropna() would not catch them. This keeps
    # the "NaN" composition while dropping genuinely empty inputs (e.g. the
    # 371 materials without a CIF) from that representation only.
    mask = sub[actual_input_col].astype(str).str.strip().ne("")
    if not is_test:
        mask &= sub[prop_name].astype(str).str.strip().ne("")
    df = sub[mask].reset_index(drop=True)

    sys_prompt, user_fn = PROMPTS[input_type]

    rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        # Build user message
        if input_type in ["description", "cif_structure"]:
            user_prompt = user_fn(row[actual_input_col], property_name=prop_name, max_words=max_words)
        else:
            user_prompt = user_fn(row[actual_input_col], property_name=prop_name)

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]

        record: Dict[str, Any] = {"messages": messages, "prop": prop_name, "input_type": input_type}

        if not is_test:
            val = row[prop_name]
            try:
                val_num = pd.to_numeric(val)
                if pd.notna(val_num):
                    val = round(float(val_num), 4)
            except Exception:
                pass
            messages.append({"role": "assistant", "content": format_assistant_content(prop_name, val)})
            record["value"] = val
        rows.append(record)
    return Dataset.from_list(rows)


def save_split(ds: Dataset, base_out: Path, split_name: str) -> None:
    out_dir = base_out / split_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_dir))
    print(f"Saved {split_name} -> {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, required=True, help="Dir containing train.csv, validation.csv, test.csv")
    parser.add_argument("--out_root", type=Path, required=True, help="Output root directory")
    parser.add_argument("--prop_names", type=str, nargs="+", required=True, help="One or more target property columns")
    parser.add_argument("--input_types", type=str, nargs="*", default=INPUT_TYPES, choices=INPUT_TYPES, help="Input types to process")
    parser.add_argument("--max_words", type=int, default=None)
    args = parser.parse_args()


    train_csv = args.data_root / "train.csv"
    val_csv = args.data_root / "validation.csv"
    test_csv = args.data_root / "test.csv"

    print(f"Processing {len(args.prop_names)} properties: {args.prop_names}")
    print(f"Processing {len(args.input_types)} input types: {args.input_types}")
    print("-" * 50)

    for i, prop in enumerate(args.prop_names, 1):
        print(f"[{i}/{len(args.prop_names)}] Processing property: '{prop}'")
        
        for j, input_type in enumerate(args.input_types, 1):
            print(f"  [{j}/{len(args.input_types)}] Processing input type: '{input_type}'")
            base_out = args.out_root / prop / input_type

            if train_csv.exists():
                print(f"    Building train split...")
                ds_tr = build_split(train_csv, prop, input_type, args.max_words, is_test=False)
                save_split(ds_tr, base_out, "train")
            if val_csv.exists():
                print(f"    Building validation split...")
                ds_va = build_split(val_csv, prop, input_type, args.max_words, is_test=False)
                save_split(ds_va, base_out, "validation")
            if test_csv.exists():
                print(f"    Building test split...")
                ds_te = build_split(test_csv, prop, input_type, args.max_words, is_test=True)
                save_split(ds_te, base_out, "test")
            print(f"  Completed input type: '{input_type}'")
        
        print(f"Completed property: '{prop}'")
        print("-" * 30)
    
    print("All processing completed successfully!")

if __name__ == "__main__":
    main()
