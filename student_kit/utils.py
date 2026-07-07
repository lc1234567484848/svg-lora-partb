from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List


SVG_RE = re.compile(r"<svg\b[\s\S]*?</svg>", re.IGNORECASE)


def load_jsonl(path: str | Path, limit: int | None = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def prompt_from_row(row: Dict[str, Any]) -> str:
    messages = row["messages"]
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    user = next((m["content"] for m in messages if m["role"] == "user"), "")
    return f"{system}\n\n{user}".strip() if system else user


def target_from_row(row: Dict[str, Any]) -> str:
    return next((m["content"] for m in row["messages"] if m["role"] == "assistant"), "")


def build_gemma3_prompt(row: Dict[str, Any]) -> str:
    prompt = prompt_from_row(row)
    return f"<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"


def clean_svg_output(text: str) -> str:
    text = (text or "").strip()
    matches = [m.group(0).strip() for m in SVG_RE.finditer(text)]
    if not matches:
        return text
    real_matches = [m for m in matches if "..." not in m and len(m) > 80]
    if real_matches:
        return max(real_matches, key=len)
    return matches[0]


def summarize_scores(items: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(items)
    if not rows:
        return {"count": 0, "mean_score": 0.0, "valid_rate": 0.0, "component_means": {}}
    scores = [float(r["reward"]["score"]) for r in rows]
    valid = [1.0 if r["reward"].get("valid_svg") else 0.0 for r in rows]
    component_names = sorted({k for r in rows for k in r["reward"].get("components", {})})
    component_means = {
        name: round(mean(float(r["reward"]["components"].get(name, 0.0)) for r in rows), 6)
        for name in component_names
    }
    return {
        "count": len(rows),
        "mean_score": round(mean(scores), 6),
        "valid_rate": round(mean(valid), 6),
        "component_means": component_means,
    }
