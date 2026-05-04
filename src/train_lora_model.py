from __future__ import annotations
import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, cast

import torch
from datasets import Dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model
)


def setup_logging(log_file: Path | None) -> logging.Logger:
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("run_finetuning")

def detect_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16

def get_num_proc(num_proc: int | None = None) -> int:
    """Get optimal number of processes for data processing"""
    if num_proc is not None:
        return max(1, num_proc)
    
    # Auto-detect: use 3/4 of available CPUs, but cap at 32 for memory efficiency
    cpu_count = os.cpu_count() or 1
    return min(32, max(1, int(cpu_count * 0.75)))

def ensure_pad_token(tokenizer):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

def sanitize_model_name(model_id: str) -> str:
    """Convert model_id to a safe directory name"""
    return model_id.replace("/", "_").replace(":", "_")

def create_experiment_name(args) -> str:
    """Create a unique experiment name based on LoRA settings"""
    # Target modules shorthand
    target_short = args.lora_targets.replace("_proj", "").replace(",", "")
    
    # Create experiment identifier
    exp_name = f"r{args.lora_r}_targets-{target_short}_drop{args.lora_dropout}"
    return exp_name

def map_to_text(batch, tokenizer):
    # Use the model's chat template to serialize multi-turn samples
    return {
        "text": tokenizer.apply_chat_template(
            batch["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
    }

def tokenize_fn(batch, tokenizer, max_length: int):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding=False,  # dynamic padding by collator
        max_length=max_length,
    )

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a language model with LoRA")
    # Data
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--prop_name", type=str, required=True)
    parser.add_argument("--input_type", type=str, required=True,
                        choices=["formula_pretty", "descr_1st_sentence", "descr_last","description", "cif_structure"])

    # Model / LoRA
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_targets", type=str, default="q_proj,v_proj",
                        help="Comma-separated target modules, e.g. 'q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj'")
    parser.add_argument("--attn_impl", type=str, default="flash_attention_2",
                        choices=["flash_attention_2", "sdpa", "eager"])

    # Tokenization
    parser.add_argument("--max_length", type=int, default=8192)

    # Optimization
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.08)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=32)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    # System
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--eval_strategy", type=str, default="epoch", choices=["epoch", "steps", "no"])
    parser.add_argument("--save_strategy", type=str, default="epoch", choices=["epoch", "steps", "no"])
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--load_best_model_at_end", default=True)
    parser.add_argument("--optim", type=str, default="adamw_torch_fused",
                        choices=["adamw_torch_fused", "adamw_torch", "adamw_hf"])
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Base name for logging/W&B tracking")
    parser.add_argument("--log_file", type=Path, default=None,
                        help="Optional path to append textual logs")
    
    # Data processing
    parser.add_argument("--num_proc", type=int, default=None,
                        help="Number of processes for data preprocessing (default: auto-detect 3/4 of CPUs, max 32)")

    # Export
    parser.add_argument("--save_adapter_dir", type=Path, default=None,
                        help="If set, saves LoRA adapter here at end (default: <output_dir>/lora_adapter/{prop_name}/{input_type}/{model_name})")

    args = parser.parse_args()

    logger = setup_logging(args.log_file)
    run_name = None
    if args.experiment_name:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"{args.experiment_name}-{timestamp}"
        os.environ.setdefault("WANDB_RUN_NAME", run_name)
        os.environ.setdefault("WANDB_NAME", run_name)
        logger.info("Using experiment name %s (W&B run: %s)", args.experiment_name, run_name)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    dtype = detect_dtype()
    
    # Configure number of processes for data processing
    num_proc = get_num_proc(args.num_proc)
    logger.info("Using %s processes for data preprocessing", num_proc)

    base = args.data_root / args.prop_name / args.input_type
    train_ds = cast(Dataset, load_from_disk(base / "train"))
    valid_ds = cast(Dataset, load_from_disk(base / "validation"))

    logger.info("Loading model %s", args.model_id)
    logger.info("Using random seed %d", int(args.seed))
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        dtype=dtype,
        attn_implementation=args.attn_impl,
    )
    # KV cache is not needed during training and consumes significant memory
    if getattr(model, "config", None) is not None:
        try:
            model.config.use_cache = False
        except Exception:
            pass
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    except Exception as e:
        logger.info("Fast tokenizer unavailable for %s, falling back to slow tokenizer (%s)", args.model_id, e)
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=False)
    ensure_pad_token(tokenizer)

    target_modules: List[str] = args.lora_targets.split(",")
    peft_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    if hasattr(model, "enable_input_require_grads"):
        # LoRA + gradient checkpointing needs inputs to carry gradients
        model.enable_input_require_grads()

    train_txt = train_ds.map(lambda b: map_to_text(b, tokenizer),
                             batched=True, remove_columns=[c for c in train_ds.column_names if c != "messages"],
                             num_proc=num_proc)
    valid_txt = valid_ds.map(lambda b: map_to_text(b, tokenizer),
                             batched=True, remove_columns=[c for c in valid_ds.column_names if c != "messages"],
                             num_proc=num_proc)
    
    train_tok = train_txt.map(lambda b: tokenize_fn(b, tokenizer, args.max_length),
                              batched=True, remove_columns=[c for c in train_txt.column_names if c != "text"],
                              num_proc=num_proc)
    valid_tok = valid_txt.map(lambda b: tokenize_fn(b, tokenizer, args.max_length),
                              batched=True, remove_columns=[c for c in valid_txt.column_names if c != "text"],
                              num_proc=num_proc)
    
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        seed=int(args.seed),
        data_seed=int(args.seed),
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,

        per_device_eval_batch_size=args.per_device_eval_batch_size,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,

        num_train_epochs=args.epochs,
        optim=args.optim,

        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        tf32=True,

        # Reduces memory at the cost of extra recomputation during backprop
        gradient_checkpointing=True,

        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end,

        logging_steps=args.logging_steps,
        run_name=run_name,
    )
    
    trainer = Trainer(
        model=model,
        train_dataset=train_tok,
        eval_dataset=valid_tok,
        args=training_args,
        data_collator=collator,
    )

    # Explicitly enable gradient checkpointing for compatibility across transformers versions
    try:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass

    trainer.train()

    # Save the final LoRA adapter with structured path
    if args.save_adapter_dir:
        adapter_dir = args.save_adapter_dir
    else:
        # Create structured directory: lora_adapter/{prop_name}/{input_type}/{model_name}/{experiment_name}
        model_name_safe = sanitize_model_name(args.model_id)
        exp_name = create_experiment_name(args)
        adapter_dir = args.output_dir / "lora_adapter" / args.prop_name / args.input_type / model_name_safe / exp_name
    
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    logger.info("Saved adapter to %s", adapter_dir)

if __name__ == "__main__":
    main()
