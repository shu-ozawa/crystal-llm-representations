from __future__ import annotations
import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, cast

import torch
from datasets import Dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
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
    """Create a unique experiment name based on training settings."""
    target_short = (
        args.lora_targets
        .replace("_proj", "")
        .replace(",", "-")
        .replace(" ", "")
    )

    exp_name = (
        f"loss-{args.loss_mode}"
        f"_r{args.lora_r}"
        f"_alpha{args.lora_alpha}"
        f"_targets-{target_short}"
        f"_drop{args.lora_dropout}"
        f"_lr{args.learning_rate}"
        f"_seed{args.seed}"
    )
    return exp_name

DEFAULT_DATE_STRING = "26 Jul 2024"


def build_chat_template_kwargs(date_string: str | None) -> Dict[str, Any]:
    """
    Build the extra kwargs passed to the chat template.

    Llama-3.2 templates inject the current date into the prompt via
    strftime_now, so unless date_string is pinned the training inputs
    change from one run date to the next.
    Only when it is None is the kwarg omitted (template default behaviour).
    """
    if date_string is None:
        return {}
    return {"date_string": date_string}


def apply_chat_template_ids(
    tokenizer,
    messages: List[Dict[str, Any]],
    *,
    add_generation_prompt: bool,
    date_string: str | None = None,
) -> List[int]:
    """Apply the model chat template and return a flat token-ID list."""
    token_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        # transformers 5.x returns a BatchEncoding by default, so this must be
        # set explicitly to get a flat list of token IDs.
        return_dict=False,
        **build_chat_template_kwargs(date_string),
    )

    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()

    # return_tensors is not set, so the result is normally 1-D; unwrap defensively.
    if token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise ValueError("Unexpected batched token IDs.")
        token_ids = token_ids[0]

    return list(token_ids)


def apply_chat_template_ids_batch(
    tokenizer,
    conversations: List[List[Dict[str, Any]]],
    *,
    add_generation_prompt: bool,
    date_string: str | None = None,
) -> List[List[int]]:
    """
    Batched variant. Tokenization after template application runs as one
    batch, which is substantially faster than calling per example.
    """
    batch_ids = tokenizer.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_dict=False,
        **build_chat_template_kwargs(date_string),
    )
    return [list(ids) for ids in batch_ids]


def fit_messages_to_max_length(
    messages: List[Dict[str, Any]],
    tokenizer,
    max_length: int,
    date_string: str | None = None,
    add_generation_prompt: bool = False,
) -> tuple[List[Dict[str, Any]], bool]:
    """
    Shorten only the last user message when the sequence exceeds max_length.

    The assistant completion is always kept in full, as are the chat
    template's role and end-of-turn tokens.

    Inference calls this with add_generation_prompt=True on a conversation
    that ends with the user turn: the alternative, letting the tokenizer
    truncate the rendered prompt from the left, drops the system prompt and
    with it the instruction to answer in JSON
    (src/run_inference_with_hidden.py).
    """
    fitted_messages = [dict(message) for message in messages]

    full_ids = apply_chat_template_ids(
        tokenizer,
        fitted_messages,
        add_generation_prompt=add_generation_prompt,
        date_string=date_string,
    )

    if len(full_ids) <= max_length:
        return fitted_messages, False

    # Find the last user message. A trailing assistant message is the
    # completion and is kept whole, so the search stops before it.
    searchable = (
        fitted_messages[:-1]
        if fitted_messages and fitted_messages[-1].get("role") == "assistant"
        else fitted_messages
    )
    user_indices = [
        index
        for index, message in enumerate(searchable)
        if message.get("role") == "user"
    ]

    if not user_indices:
        raise ValueError(
            "Sequence exceeds max_length, but no user message can be truncated."
        )

    user_index = user_indices[-1]
    user_content = fitted_messages[user_index].get("content")

    if not isinstance(user_content, str):
        raise TypeError(
            "The last user message content must be a string."
        )

    user_content_ids = tokenizer(
        user_content,
        add_special_tokens=False,
    )["input_ids"]

    # Binary-search the longest user content that still fits within max_length.
    lower = 0
    upper = len(user_content_ids)

    best_messages = None

    while lower <= upper:
        middle = (lower + upper) // 2

        truncated_content = tokenizer.decode(
            user_content_ids[:middle],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        candidate_messages = [
            dict(message)
            for message in fitted_messages
        ]
        candidate_messages[user_index]["content"] = truncated_content

        candidate_ids = apply_chat_template_ids(
            tokenizer,
            candidate_messages,
            add_generation_prompt=add_generation_prompt,
            date_string=date_string,
        )

        if len(candidate_ids) <= max_length:
            best_messages = candidate_messages
            lower = middle + 1
        else:
            upper = middle - 1

    if best_messages is None:
        raise ValueError(
            "The system prompt, chat-template tokens, and assistant "
            f"completion alone exceed max_length={max_length}."
        )

    return best_messages, True

def log_tokenization_stats(
    dataset: Dataset,
    dataset_name: str,
    logger: logging.Logger,
) -> None:
    total = len(dataset)
    truncated = int(sum(dataset["was_truncated"]))

    logger.info(
        "%s: total=%d, truncated=%d (%.3f%%), "
        "max_sequence_length=%d, "
        "min_completion_length=%d, "
        "max_completion_length=%d, "
        "min_loss_tokens=%d",
        dataset_name,
        total,
        truncated,
        100.0 * truncated / max(total, 1),
        max(dataset["sequence_length"]),
        min(dataset["completion_length"]),
        max(dataset["completion_length"]),
        min(dataset["num_loss_tokens"]),
    )


def tokenize_for_causal_lm(
    example: Dict[str, Any],
    tokenizer,
    max_length: int,
    loss_mode: str,
    date_string: str | None = None,
    precomputed_full_ids: List[int] | None = None,
    precomputed_prompt_ids: List[int] | None = None,
) -> Dict[str, Any]:
    """
    Switch only the labels while keeping input_ids identical.

    full:
        Compute the loss over the prompt and the assistant completion.

    completion:
        Mask the prompt with -100 and compute the loss only over the
        assistant completion.

    precomputed_*_ids are reused from the batched preprocessing pass. They
    are discarded and recomputed after truncation if the sequence exceeded
    max_length.
    """
    messages = example["messages"]

    if not messages:
        raise ValueError("messages is empty.")

    if messages[-1].get("role") != "assistant":
        raise ValueError(
            "The final message must be an assistant message. "
            f"Got role={messages[-1].get('role')!r}"
        )

    # The complete conversation fed to the model during training.
    full_ids = precomputed_full_ids
    if full_ids is None:
        full_ids = apply_chat_template_ids(
            tokenizer,
            messages,
            add_generation_prompt=False,
            date_string=date_string,
        )

    prompt_ids = precomputed_prompt_ids
    was_truncated = False

    if len(full_ids) > max_length:
        # Rare case only: shorten the user message and reapply the template.
        fitted_messages, was_truncated = fit_messages_to_max_length(
            messages=messages,
            tokenizer=tokenizer,
            max_length=max_length,
            date_string=date_string,
        )

        full_ids = apply_chat_template_ids(
            tokenizer,
            fitted_messages,
            add_generation_prompt=False,
            date_string=date_string,
        )

        # Everything up to the point where the assistant reply begins.
        prompt_ids = apply_chat_template_ids(
            tokenizer,
            fitted_messages[:-1],
            add_generation_prompt=True,
            date_string=date_string,
        )

    elif prompt_ids is None:
        # Everything up to the point where the assistant reply begins.
        prompt_ids = apply_chat_template_ids(
            tokenizer,
            messages[:-1],
            add_generation_prompt=True,
            date_string=date_string,
        )

    if len(full_ids) > max_length:
        raise RuntimeError(
            f"Sequence remains too long: {len(full_ids)} > {max_length}"
        )

    # Verify that prompt_ids is a prefix of full_ids.
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise RuntimeError(
            "Prompt tokens are not a prefix of the full conversation. "
            "The model's chat template may require model-specific handling."
        )

    completion_ids = full_ids[len(prompt_ids):]

    if not completion_ids:
        raise ValueError(
            "No assistant completion tokens were found."
        )

    if loss_mode == "full":
        labels = full_ids.copy()

    elif loss_mode == "completion":
        labels = (
            [-100] * len(prompt_ids)
            + completion_ids.copy()
        )

    else:
        raise ValueError(f"Unknown loss_mode: {loss_mode}")

    if len(labels) != len(full_ids):
        raise RuntimeError(
            "input_ids and labels have different lengths."
        )

    num_loss_tokens = sum(label != -100 for label in labels)

    if num_loss_tokens == 0:
        raise ValueError(
            "This sample has no tokens contributing to the loss."
        )

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,

        # Diagnostics only; removed before training starts.
        "sequence_length": len(full_ids),
        "prompt_length": len(prompt_ids),
        "completion_length": len(completion_ids),
        "num_loss_tokens": num_loss_tokens,
        "was_truncated": was_truncated,
    }


def tokenize_batch_for_causal_lm(
    batch: Dict[str, List[Any]],
    tokenizer,
    max_length: int,
    loss_mode: str,
    date_string: str | None = None,
) -> Dict[str, List[Any]]:
    """
    Batched preprocessing.

    Template application and tokenization run once per batch for both the
    full and the prompt sequences; only the (rare) samples that exceed
    max_length fall through to the per-example truncation path inside
    tokenize_for_causal_lm. Label construction and validation are identical
    to tokenize_for_causal_lm.
    """
    conversations = batch["messages"]

    full_ids_batch = apply_chat_template_ids_batch(
        tokenizer,
        conversations,
        add_generation_prompt=False,
        date_string=date_string,
    )
    prompt_ids_batch = apply_chat_template_ids_batch(
        tokenizer,
        [conversation[:-1] for conversation in conversations],
        add_generation_prompt=True,
        date_string=date_string,
    )

    columns: Dict[str, List[Any]] = {}

    for conversation, full_ids, prompt_ids in zip(
        conversations,
        full_ids_batch,
        prompt_ids_batch,
    ):
        record = tokenize_for_causal_lm(
            {"messages": conversation},
            tokenizer,
            max_length=max_length,
            loss_mode=loss_mode,
            date_string=date_string,
            precomputed_full_ids=full_ids,
            precomputed_prompt_ids=prompt_ids,
        )

        for key, value in record.items():
            columns.setdefault(key, []).append(value)

    return columns


@dataclass
class CausalLMDataCollator:
    tokenizer: Any
    pad_to_multiple_of: int | None = 8

    def __call__(
        self,
        features: List[Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        """
        Dynamically pad input_ids and attention_mask, filling the padded
        positions of labels with -100.
        """
        model_features = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            }
            for feature in features
        ]

        batch = self.tokenizer.pad(
            model_features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        padded_length = batch["input_ids"].shape[1]
        padded_labels = []

        for feature in features:
            labels = list(feature["labels"])
            padding_length = padded_length - len(labels)

            if padding_length < 0:
                raise ValueError(
                    "A label sequence is longer than the padded input."
                )

            if self.tokenizer.padding_side == "right":
                labels = labels + [-100] * padding_length
            else:
                labels = [-100] * padding_length + labels

            padded_labels.append(labels)

        batch["labels"] = torch.tensor(
            padded_labels,
            dtype=torch.long,
        )

        return batch    

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

    parser.add_argument(
        "--loss_mode",
        type=str,
        default="completion",
        choices=["full", "completion"],
        help=(
            "'full': compute loss on prompt and assistant tokens; "
            "'completion': compute loss only on assistant completion tokens"
        ),
    )

    parser.add_argument(
        "--date_string",
        type=str,
        default=DEFAULT_DATE_STRING,
        help=(
            "Fixed 'Today Date' passed to the chat template. "
            "Llama-3.2 templates inject the current date via strftime_now, "
            "so pinning keeps training inputs independent of the run date. "
            f"Default '{DEFAULT_DATE_STRING}' matches the Llama-3.1 template's "
            "built-in value. Pass an empty string to disable pinning."
        ),
    )

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
    # Default None keeps a checkpoint for every epoch. Independently of the
    # best-by-eval-loss selection, this allows re-selecting the checkpoint
    # with the lowest test MAE afterwards.
    parser.add_argument("--save_total_limit", type=int, default=None)
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
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help=("Resume training from a checkpoint. Pass 'auto' to use the "
                              "latest checkpoint in --output_dir, or an explicit "
                              "checkpoint-<step> path. Optimizer/scheduler/RNG state are "
                              "restored, so only the remaining steps are run."))
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
    tokenizer.padding_side = "right"

    target_modules: List[str] = [
        module.strip()
        for module in args.lora_targets.split(",")
        if module.strip()
    ]
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

    date_string = args.date_string.strip() if args.date_string else ""
    date_string = date_string if date_string else None
    if date_string is None:
        logger.info("Chat template date_string pinning is disabled")
    else:
        logger.info("Pinning chat template date_string to %r", date_string)

    tokenize_fn_kwargs = {
        "tokenizer": tokenizer,
        "max_length": args.max_length,
        "loss_mode": args.loss_mode,
        "date_string": date_string,
    }

    train_tok = train_ds.map(
        tokenize_batch_for_causal_lm,
        fn_kwargs=tokenize_fn_kwargs,
        batched=True,
        # Conservative value chosen for the memory cost of long CIF samples.
        batch_size=128,
        remove_columns=train_ds.column_names,
        num_proc=num_proc,
        desc=f"Tokenizing train data: loss_mode={args.loss_mode}",
    )

    valid_tok = valid_ds.map(
        tokenize_batch_for_causal_lm,
        fn_kwargs=tokenize_fn_kwargs,
        batched=True,
        batch_size=128,
        remove_columns=valid_ds.column_names,
        num_proc=num_proc,
        desc=f"Tokenizing validation data: loss_mode={args.loss_mode}",
    )

    log_tokenization_stats(train_tok, "train", logger)
    log_tokenization_stats(valid_tok, "validation", logger)

    # Inspect the loss-target tokens of a few samples.
    for sample_index in range(min(3, len(train_tok))):
        sample = train_tok[sample_index]

        loss_token_ids = [
            token_id
            for token_id, label in zip(
                sample["input_ids"],
                sample["labels"],
            )
            if label != -100
        ]

        logger.info(
            "Sample %d, loss_mode=%s, loss_tokens=%d",
            sample_index,
            args.loss_mode,
            len(loss_token_ids),
        )
        logger.info(
            "Decoded loss-target tokens: %r",
            tokenizer.decode(
                loss_token_ids,
                skip_special_tokens=False,
            ),
        )

    metadata_columns = [
        "sequence_length",
        "prompt_length",
        "completion_length",
        "num_loss_tokens",
        "was_truncated",
    ]

    train_tok = train_tok.remove_columns(metadata_columns)
    valid_tok = valid_tok.remove_columns(metadata_columns)

    collator = CausalLMDataCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    )
    
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
        load_best_model_at_end=True,

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

    # An interrupted run can resume from a checkpoint, with optimizer,
    # scheduler and RNG state restored. A run that had reached epoch 5 of 6
    # was once lost to a SLURM time limit; resuming means only the remaining
    # epoch has to be rerun.
    resume: bool | str | None = None
    if args.resume_from_checkpoint:
        resume = args.resume_from_checkpoint
        if str(resume).lower() == "auto":
            resume = True          # Trainer picks the latest checkpoint in output_dir
        logger.info("Resuming from checkpoint: %s", resume)

    train_result = trainer.train(resume_from_checkpoint=resume)

    # Record training metrics such as wall-clock time.
    logger.info("Train metrics: %s", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

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
