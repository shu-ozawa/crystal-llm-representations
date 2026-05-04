#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from datasets import load_from_disk
from peft import AutoPeftModelForCausalLM, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

ATTN_IMPLEMENTATION = "sdpa"


def detect_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def ensure_pad_token(tokenizer) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _should_stop_on_json_close(text_so_far: str) -> bool:
    open_idx = text_so_far.find("{")
    if open_idx < 0:
        return False
    close_idx = text_so_far.find("}", open_idx + 1)
    return close_idx >= 0


def get_model_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "model") and hasattr(model.model.model, "layers"):
        return model.model.model.layers
    if hasattr(model, "base_model") and hasattr(model.base_model, "model") and hasattr(model.base_model.model, "layers"):
        return model.base_model.model.layers
    raise ValueError(f"Could not locate model layers in {type(model)}.")


@dataclass(frozen=True)
class RunMeta:
    dataset_dir: str
    split: str
    adapter_dir: Optional[str]
    model_id: Optional[str]
    base_model_id: str
    max_new_tokens: int
    max_input_length: int
    stop_on_json_close: bool
    strip_trailing_assistant: bool
    dtype: str
    device: str
    created_at_unix: float



def load_dataset(dataset_dir: str, split: str):
    split_dir = os.path.join(dataset_dir, split)
    if os.path.isdir(split_dir):
        return load_from_disk(split_dir)
    ds = load_from_disk(dataset_dir)
    if hasattr(ds, "keys"):
        if split in ds:
            return ds[split]
        first_split = list(ds.keys())[0]
        print(f"[INFO] DatasetDict detected. Using split: {first_split}")
        return ds[first_split]
    return ds


def load_model_and_tokenizer(*, adapter_dir: Optional[str], model_id: Optional[str], dtype: torch.dtype):
    if bool(adapter_dir) == bool(model_id):
        raise ValueError("Provide exactly one of adapter_dir or model_id.")

    if adapter_dir:
        peft_cfg = PeftConfig.from_pretrained(adapter_dir)
        base_model_id = peft_cfg.base_model_name_or_path
        print(f"[INFO] Loading base model (with LoRA) from {base_model_id}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True, trust_remote_code=True)
        except Exception as e:
            print(f"[INFO] Fast tokenizer unavailable for {base_model_id}, falling back to slow one. Reason: {e}")
            tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=False, trust_remote_code=True)
        ensure_pad_token(tokenizer)
        model = AutoPeftModelForCausalLM.from_pretrained(
            adapter_dir,
            device_map="auto",
            torch_dtype=dtype,
            attn_implementation=ATTN_IMPLEMENTATION,
            trust_remote_code=True,
        )
        model.eval()
        return model, tokenizer, base_model_id

    assert model_id is not None
    base_model_id = model_id
    print(f"[INFO] Loading model from {model_id}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    except Exception as e:
        print(f"[INFO] Fast tokenizer unavailable for {model_id}, falling back to slow one. Reason: {e}")
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)
    ensure_pad_token(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=dtype,
        attn_implementation=ATTN_IMPLEMENTATION,
        trust_remote_code=True,
    )
    model.eval()

    return model, tokenizer, base_model_id


def capture_block_last_token_hidden_states(model, *, n_layers: int, hidden_dtype: torch.dtype) -> Tuple[Dict[int, torch.Tensor], List[torch.utils.hooks.RemovableHandle]]:
    cache: Dict[int, torch.Tensor] = {}

    def hook_fn(layer_idx: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            vec = hidden[:, -1, :].detach().to("cpu", dtype=hidden_dtype).clone()
            cache[layer_idx] = vec

        return hook

    layers = get_model_layers(model)
    if len(layers) != n_layers:
        n_layers = len(layers)

    handles: List[torch.utils.hooks.RemovableHandle] = []
    for i in range(n_layers):
        handles.append(layers[i].register_forward_hook(hook_fn(i)))
    return cache, handles


@torch.inference_mode()
def infer_one(*, model, tokenizer, prompt: str, max_new_tokens: int, max_input_length: int, stop_on_json_close: bool, hidden_dtype: torch.dtype) -> Dict[str, Any]:
    enc = tokenizer(prompt, return_tensors="pt", padding=False, truncation=True, max_length=max_input_length, add_special_tokens=False)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    input_ids: torch.Tensor = enc["input_ids"]
    attention_mask: Optional[torch.Tensor] = enc.get("attention_mask")
    n_layers = len(get_model_layers(model))
    hidden_cache, handles = capture_block_last_token_hidden_states(model, n_layers=n_layers, hidden_dtype=hidden_dtype)
    try:
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    finally:
        for h in handles:
            h.remove()
    past = out.past_key_values
    next_logits = out.logits[0, -1, :].float()
    hidden_layers: List[torch.Tensor] = []
    for i in range(n_layers):
        if i not in hidden_cache:
            raise RuntimeError(f"Missing hidden state for layer {i}.")
        v = hidden_cache[i]
        if v.ndim == 2 and v.shape[0] == 1:
            v = v[0]
        hidden_layers.append(v)
    hidden_last_token = torch.stack(hidden_layers, dim=0)

    generated_token_ids: List[int] = []
    token_logprobs: List[float] = []
    text_so_far = ""
    stop_reason = "max_new_tokens"
    eos_id = tokenizer.eos_token_id

    for _step in range(int(max_new_tokens)):
        token_id = int(torch.argmax(next_logits).item())
        if eos_id is not None and token_id == eos_id:
            stop_reason = "eos"
            break

        log_probs = torch.log_softmax(next_logits, dim=-1)
        token_logprob = float(log_probs[token_id].item())
        generated_token_ids.append(token_id)
        token_logprobs.append(token_logprob)
        frag = tokenizer.decode([token_id], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        text_so_far += frag
        if stop_on_json_close and _should_stop_on_json_close(text_so_far):
            stop_reason = "json_close"
            break

        token_tensor = torch.tensor([[token_id]], device=model.device, dtype=input_ids.dtype)
        out = model(input_ids=token_tensor, use_cache=True, past_key_values=past)
        past = out.past_key_values
        next_logits = out.logits[0, -1, :].float()

    n_scored = len(token_logprobs)
    pll_sum = float(sum(token_logprobs))
    mean_nll = float((-pll_sum) / n_scored) if n_scored > 0 else None
    ppl = float(torch.exp(torch.tensor(mean_nll)).item()) if mean_nll is not None else None
    return {
        "generated_text": text_so_far,
        "generated_token_ids": generated_token_ids,
        "token_logprobs": token_logprobs,
        "pll_sum": pll_sum,
        "mean_nll": mean_nll,
        "ppl": ppl,
        "n_tokens_scored": n_scored,
        "stop_reason": stop_reason,
        "hidden_last_token": hidden_last_token,
        "prompt_len_tokens": int(input_ids.shape[1]),
    }


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_runtime_args(args: argparse.Namespace) -> tuple[str, str, Optional[str], Optional[str]]:
    if args.dataset_dir is None:
        raise ValueError("--dataset_dir is required.")
    if args.output_dir is None:
        raise ValueError("--output_dir is required.")
    if bool(args.adapter_dir) == bool(args.model_id):
        raise ValueError("Provide exactly one of --adapter_dir or --model_id.")
    return args.dataset_dir, args.output_dir, args.adapter_dir, args.model_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference while saving per-layer last-token hidden states and per-step log p.")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Path to prepared dataset directory")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to write records.jsonl and hidden states")
    parser.add_argument("--adapter_dir", type=str, default=None, help="Path to LoRA adapter (mutually exclusive with --model_id)")
    parser.add_argument("--model_id", type=str, default=None, help="HuggingFace model ID for base model inference (mutually exclusive with --adapter_dir)")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_input_length", type=int, default=8192)
    parser.add_argument("--stop_on_json_close", action="store_true")
    parser.add_argument("--no_strip_trailing_assistant", action="store_true", help="Disable stripping the trailing assistant message in dataset 'messages'. By default, if messages[-1].role == 'assistant', it is dropped from the prompt to avoid including ground-truth answers during generation.")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--hidden_dtype", type=str, default="float16", choices=["float16", "float32"], help="Hidden state dtype for saving (on CPU).")
    parser.add_argument("--save_prompt_text", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir, output_dir, adapter_dir, model_id = resolve_runtime_args(args)
    if not Path(dataset_dir).is_dir():
        raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")

    print(f"[INFO] dataset_dir={dataset_dir}")
    print(f"[INFO] output_dir={output_dir}")
    print(f"[INFO] model_id={model_id or '<none>'}")
    print(f"[INFO] adapter_dir={adapter_dir or '<none>'}")

    dtype = detect_dtype()
    device_info = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device_info} dtype={dtype}")
    model, tokenizer, base_model_id = load_model_and_tokenizer(adapter_dir=adapter_dir, model_id=model_id, dtype=dtype)
    hidden_dtype = torch.float16 if args.hidden_dtype == "float16" else torch.float32
    ds = load_dataset(dataset_dir, args.split)
    messages_list = ds["messages"]
    total = len(messages_list)
    if args.max_samples is not None:
        total = min(total, int(args.max_samples))
    out_dir = Path(output_dir)
    hidden_dir = out_dir / "hidden_last_token"
    records_path = out_dir / "records.jsonl"
    meta_path = out_dir / "meta.json"
    strip_trailing_assistant = not bool(args.no_strip_trailing_assistant)
    meta = RunMeta(dataset_dir=dataset_dir, split=args.split, adapter_dir=adapter_dir, model_id=model_id, base_model_id=base_model_id, max_new_tokens=int(args.max_new_tokens), max_input_length=int(args.max_input_length), stop_on_json_close=bool(args.stop_on_json_close), strip_trailing_assistant=strip_trailing_assistant, dtype=str(dtype).replace("torch.", ""), device=device_info, created_at_unix=time.time())
    write_json(meta_path, asdict(meta))
    t0 = time.time()

    for idx in range(total):
        if (idx + 1) % 10 == 0 or idx == 0 or idx + 1 == total:
            elapsed = time.time() - t0
            speed = (idx + 1) / elapsed if elapsed > 0 else 0.0
            remaining = total - (idx + 1)
            eta = remaining / speed if speed > 0 else 0.0
            print(f"[PROGRESS] {idx+1}/{total} samples ({speed:.2f} samples/s, ETA ~{eta/60:.1f} min)", flush=True)

        messages = messages_list[idx]
        dropped_gt_assistant = False
        prompt_messages = messages
        try:
            if strip_trailing_assistant and isinstance(messages, list) and len(messages) > 0 and isinstance(messages[-1], dict) and messages[-1].get("role") == "assistant":
                prompt_messages = messages[:-1]
                dropped_gt_assistant = True

        except Exception:
            prompt_messages = messages
            dropped_gt_assistant = False
        prompt = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        res = infer_one(model=model, tokenizer=tokenizer, prompt=prompt, max_new_tokens=int(args.max_new_tokens), max_input_length=int(args.max_input_length), stop_on_json_close=bool(args.stop_on_json_close), hidden_dtype=hidden_dtype)
        hs: torch.Tensor = res.pop("hidden_last_token")
        hs_path = hidden_dir / f"{idx:08d}.pt"
        hs_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(hs, hs_path)
        record: Dict[str, Any] = {
            "index": idx,
            "split": args.split,
            "prompt_hash": sha256_text(prompt),
            "dropped_gt_assistant": dropped_gt_assistant,
            "n_messages_prompt": len(prompt_messages) if isinstance(prompt_messages, list) else None,
            "hidden_last_token_path": str(hs_path),
            "hidden_last_token_shape": list(hs.shape),
            "hidden_last_token_dtype": args.hidden_dtype,
            **res,
        }
        if args.save_prompt_text:
            record["prompt"] = prompt
        append_jsonl(records_path, [record])
    t1 = time.time()
    print(f"[INFO] Done. wall_time={t1-t0:.2f}s output_dir={out_dir}")


if __name__ == "__main__":
    main()
