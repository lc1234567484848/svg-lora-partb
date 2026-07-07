from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _checkpoint_step(path: Path) -> int:
    try:
        return int(path.name.rsplit("-", 1)[-1])
    except ValueError:
        return -1


def find_checkpoint(output_dir: Path) -> Path:
    best_candidates = []
    for state_path in output_dir.rglob("trainer_state.json"):
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        best = state.get("best_model_checkpoint")
        if best:
            best_path = Path(best)
            if not best_path.is_absolute():
                best_path = (state_path.parent / best_path).resolve()
            if best_path.exists():
                best_candidates.append(best_path)
    for candidate in best_candidates:
        if (candidate / "adapter_config.json").exists() and (candidate / "adapter_model.safetensors").exists():
            return candidate

    checkpoints = [
        p
        for p in output_dir.rglob("checkpoint-*")
        if p.is_dir() and (p / "adapter_config.json").exists() and (p / "adapter_model.safetensors").exists()
    ]
    if not checkpoints:
        raise FileNotFoundError(f"No LoRA checkpoint found under {output_dir}")
    return sorted(checkpoints, key=lambda p: (_checkpoint_step(p), p.stat().st_mtime))[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="output/gemma-3-270m/logo_svg_lora")
    parser.add_argument("--adapter-dir", default="adapter")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    adapter_dir = Path(args.adapter_dir)
    checkpoint = find_checkpoint(output_dir)

    adapter_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint / "adapter_model.safetensors", adapter_dir / "adapter_model.safetensors")
    config = json.loads((checkpoint / "adapter_config.json").read_text(encoding="utf-8"))
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(config, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    print(f"Copied adapter from {checkpoint} to {adapter_dir}")


if __name__ == "__main__":
    main()
