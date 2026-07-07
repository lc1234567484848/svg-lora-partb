from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reward import evaluate_svg
from student_kit.utils import build_gemma3_prompt, clean_svg_output, load_jsonl, prompt_from_row, summarize_scores


def _normalize_adapter_config(adapter: str | Path) -> None:
    config_path = Path(adapter) / "adapter_config.json"
    if not config_path.exists():
        return
    data = json.loads(config_path.read_text(encoding="utf-8"))
    config_path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")


def _chunks(items: List[Dict[str, Any]], size: int):
    for start in range(0, len(items), size):
        yield start, items[start : start + size]


def _write_result(path: Path, result: Dict[str, Any]) -> None:
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def _generate_with_vllm(
    llm,
    tokenizer,
    rows: List[Dict[str, Any]],
    sampling_params,
    out_path: Path,
    result: Dict[str, Any],
    section: str,
    label: str,
    batch_size: int,
    lora_request=None,
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    result[section] = {"summary": summarize_scores(items), "items": items}
    _write_result(out_path, result)

    for start, batch_rows in _chunks(rows, batch_size):
        prompts = [
            {"prompt_token_ids": tokenizer(build_gemma3_prompt(row), add_special_tokens=False)["input_ids"]}
            for row in batch_rows
        ]
        started = time.perf_counter()
        outputs = llm.generate(
            prompts,
            sampling_params,
            lora_request=lora_request,
            use_tqdm=False,
        )
        elapsed = time.perf_counter() - started
        for offset, (row, output) in enumerate(zip(batch_rows, outputs)):
            idx = start + offset
            raw_output = output.outputs[0].text
            cleaned = clean_svg_output(raw_output)
            prompt_text = prompt_from_row(row)
            reward = evaluate_svg(prompt_text, cleaned)
            token_count = len(output.outputs[0].token_ids or [])
            item = {
                "index": idx,
                "prompt": prompt_text,
                "raw_output": raw_output,
                "svg": cleaned,
                "reward": reward,
            }
            items.append(item)
            result[section] = {"summary": summarize_scores(items), "items": items}
            if "base" in result and "finetuned" in result:
                base_summary = result["base"]["summary"]
                tuned_summary = result["finetuned"]["summary"]
                result["delta"] = {
                    "mean_score": round(tuned_summary["mean_score"] - base_summary["mean_score"], 6),
                    "valid_rate": round(tuned_summary["valid_rate"] - base_summary["valid_rate"], 6),
                }
            _write_result(out_path, result)
            print(
                f"[{label}] {idx + 1}/{len(rows)} tokens={token_count} "
                f"score={reward['score']:.4f} valid={reward['valid_svg']} "
                f"batch_time={elapsed:.1f}s",
                flush=True,
            )
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("valid_jsonl")
    parser.add_argument("--base-model", default="pretrained_model/gemma-3-270m")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--out", default="results.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=3072)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--max-model-len", type=int, default=4096)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    rows = load_jsonl(args.valid_jsonl, args.limit)
    out_path = Path(args.out)
    adapter_path = Path(args.adapter) if args.adapter else None
    enable_lora = bool(adapter_path and adapter_path.exists())
    if enable_lora:
        _normalize_adapter_config(adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    result: Dict[str, Any] = {
        "dataset": args.valid_jsonl,
        "limit": args.limit,
        "backend": "vllm",
        "base_model": args.base_model,
        "adapter": args.adapter,
        "decode": {
            "temperature": 0.0,
            "max_new_tokens": args.max_new_tokens,
            "stop": ["<end_of_turn>"],
            "batch_size": args.batch_size,
        },
    }
    _write_result(out_path, result)

    llm = LLM(
        model=args.base_model,
        tokenizer=args.base_model,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1,
        enable_lora=enable_lora,
        max_lora_rank=8,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        stop=["<end_of_turn>"],
        include_stop_str_in_output=True,
        skip_special_tokens=False,
    )

    print(f"Evaluating base model with vLLM on {len(rows)} rows...")
    _generate_with_vllm(llm, tokenizer, rows, sampling_params, out_path, result, "base", "base", args.batch_size)

    if enable_lora:
        print(f"Evaluating adapter with vLLM: {adapter_path}")
        lora_request = LoRARequest("logo_adapter", 1, str(adapter_path))
        _generate_with_vllm(
            llm,
            tokenizer,
            rows,
            sampling_params,
            out_path,
            result,
            "finetuned",
            "adapter",
            args.batch_size,
            lora_request=lora_request,
        )
    elif args.adapter:
        print(f"Adapter path not found, skipping finetuned eval: {adapter_path}")

    _write_result(out_path, result)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
