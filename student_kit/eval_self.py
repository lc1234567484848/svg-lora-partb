from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reward import evaluate_svg  
from student_kit.utils import (  
    build_gemma3_prompt,
    clean_svg_output,
    load_jsonl,
    prompt_from_row,
    summarize_scores,
    target_from_row,
)


def _normalize_adapter_config(adapter: str | Path) -> None:
    config_path = Path(adapter) / "adapter_config.json"
    if not config_path.exists():
        return
    data = json.loads(config_path.read_text(encoding="utf-8"))
    # PEFT opens this file with the Windows locale encoding. ASCII escaping
    # keeps Chinese workspace paths readable under GBK without changing values.
    config_path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")


def _load_model(base_model: str, adapter: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if adapter:
        from peft import PeftModel

        _normalize_adapter_config(adapter)
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tokenizer, model


def _stop_sequences(tokenizer) -> List[List[int]]:
    sequences: List[List[int]] = []
    for text in ("</svg>", "</svg>\n", "<end_of_turn>"):
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if ids:
            sequences.append(ids)
    return sequences


def _make_stopper(stop_sequences: List[List[int]]):
    import torch
    from transformers import StoppingCriteria

    class StopOnTokenSequences(StoppingCriteria):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            for sequence in stop_sequences:
                if len(sequence) <= input_ids.shape[1] and input_ids[0, -len(sequence) :].tolist() == sequence:
                    return True
            return False

    return StopOnTokenSequences()


def _generate_rows(
    rows: List[Dict[str, Any]],
    base_model: str,
    adapter: str | None,
    max_new_tokens: int,
    label: str,
    on_update: Callable[[List[Dict[str, Any]]], None] | None = None,
) -> List[Dict[str, Any]]:
    import torch
    from transformers import StoppingCriteriaList

    tokenizer, model = _load_model(base_model, adapter)
    end_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    eos_ids = [tokenizer.eos_token_id]
    if isinstance(end_turn_id, int) and end_turn_id >= 0:
        eos_ids.append(end_turn_id)
    stopping = StoppingCriteriaList([_make_stopper(_stop_sequences(tokenizer))])

    outputs: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        started = time.perf_counter()
        prompt_text = prompt_from_row(row)
        encoded = tokenizer(build_gemma3_prompt(row), return_tensors="pt", add_special_tokens=False)
        encoded = {k: v.to(model.device) for k, v in encoded.items()}
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                eos_token_id=eos_ids,
                pad_token_id=tokenizer.pad_token_id,
                stopping_criteria=stopping,
            )
        new_tokens = generated[0, encoded["input_ids"].shape[1] :]
        raw_output = tokenizer.decode(new_tokens, skip_special_tokens=False)
        cleaned = clean_svg_output(raw_output)
        reward = evaluate_svg(prompt_text, cleaned)
        elapsed = time.perf_counter() - started
        print(
            f"[{label}] {idx + 1}/{len(rows)} tokens={len(new_tokens)} "
            f"score={reward['score']:.4f} valid={reward['valid_svg']} time={elapsed:.1f}s",
            flush=True,
        )
        outputs.append(
            {
                "index": idx,
                "prompt": prompt_text,
                "target_svg": target_from_row(row),
                "raw_output": raw_output,
                "svg": cleaned,
                "reward": reward,
            }
        )
        if on_update is not None:
            on_update(outputs)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("valid_jsonl")
    parser.add_argument("--base-model", default="pretrained_model/gemma-3-270m")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--out", default="results.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=3072)
    args = parser.parse_args()

    rows = load_jsonl(args.valid_jsonl, args.limit)
    out = Path(args.out)
    result: Dict[str, Any] = {
        "dataset": args.valid_jsonl,
        "limit": args.limit,
        "base_model": args.base_model,
        "adapter": args.adapter,
        "decode": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
    }

    def write_result() -> None:
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    def update_base(items: List[Dict[str, Any]]) -> None:
        result["base"] = {"summary": summarize_scores(items), "items": items}
        write_result()

    def update_adapter(items: List[Dict[str, Any]]) -> None:
        tuned_summary = summarize_scores(items)
        base_summary = result["base"]["summary"]
        result["finetuned"] = {"summary": tuned_summary, "items": items}
        result["delta"] = {
            "mean_score": round(tuned_summary["mean_score"] - base_summary["mean_score"], 6),
            "valid_rate": round(tuned_summary["valid_rate"] - base_summary["valid_rate"], 6),
        }
        write_result()

    print(f"Evaluating base model on {len(rows)} rows...")
    base_items = _generate_rows(rows, args.base_model, None, args.max_new_tokens, "base", update_base)
    result["base"] = {"summary": summarize_scores(base_items), "items": base_items}
    write_result()
    print(f"Wrote interim base results to {out}")

    adapter_path = Path(args.adapter) if args.adapter else None
    if adapter_path and adapter_path.exists():
        print(f"Evaluating adapter: {adapter_path}")
        tuned_items = _generate_rows(
            rows,
            args.base_model,
            str(adapter_path),
            args.max_new_tokens,
            "adapter",
            update_adapter,
        )
        tuned_summary = summarize_scores(tuned_items)
        base_summary = result["base"]["summary"]
        result["finetuned"] = {"summary": tuned_summary, "items": tuned_items}
        result["delta"] = {
            "mean_score": round(tuned_summary["mean_score"] - base_summary["mean_score"], 6),
            "valid_rate": round(tuned_summary["valid_rate"] - base_summary["valid_rate"], 6),
        }
    elif args.adapter:
        print(f"Adapter path not found, skipping finetuned eval: {adapter_path}")

    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
