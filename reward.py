"""Programmatic SVG-logo reward for the Part B assignment.

The score is intentionally conservative: it rewards valid, simple, centered SVG
logos and penalizes unsafe markup or degenerate text.  It is a training proxy,
not a visual-quality oracle.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple


ALLOWED_TAGS = {
    "svg",
    "defs",
    "g",
    "path",
    "circle",
    "ellipse",
    "rect",
    "polygon",
    "line",
    "linearGradient",
    "radialGradient",
    "stop",
}
FORBIDDEN_TAGS = {"script", "image", "foreignObject", "iframe", "video", "audio", "canvas", "style"}
GRAPHIC_TAGS = {"path", "circle", "ellipse", "rect", "polygon", "line"}
COLOR_RE = re.compile(r"#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?\b")
NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")

SHAPE_KEYWORDS = {
    "circle": {"circle", "circular", "round", "badge", "ring", "dot", "coin", "seal"},
    "rect": {"square", "rectangle", "rectangular", "box", "frame", "tile"},
    "ellipse": {"ellipse", "oval", "almond"},
    "path": {"curve", "leaf", "wave", "swirl", "stem", "note", "flame", "drop", "mountain", "star"},
    "line": {"line", "staff", "tick", "stroke"},
    "polygon": {"triangle", "polygon", "star", "diamond"},
}
COLOR_WORDS = {
    "red",
    "orange",
    "yellow",
    "gold",
    "golden",
    "green",
    "teal",
    "blue",
    "navy",
    "purple",
    "pink",
    "brown",
    "cream",
    "white",
    "black",
    "gray",
    "grey",
}


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _extract_single_svg(text: str) -> Tuple[str, bool]:
    if not isinstance(text, str):
        return "", False
    matches = list(re.finditer(r"<svg\b[\s\S]*?</svg>", text, flags=re.IGNORECASE))
    if len(matches) != 1:
        return text.strip(), False
    candidate = matches[0].group(0).strip()
    only_svg = text[: matches[0].start()].strip() == "" and text[matches[0].end() :].strip() == ""
    return candidate, only_svg


def _parse_svg(svg: str) -> Tuple[ET.Element | None, str | None]:
    try:
        return ET.fromstring(svg), None
    except ET.ParseError as exc:
        return None, str(exc)


def _iter_elements(root: ET.Element | None) -> Iterable[ET.Element]:
    if root is None:
        return []
    return root.iter()


def _all_numbers(root: ET.Element | None) -> List[float]:
    nums: List[float] = []
    for el in _iter_elements(root):
        for key, value in el.attrib.items():
            if key.lower() in {"id", "class", "fill", "stroke", "stop-color", "xmlns"}:
                continue
            for match in NUMBER_RE.findall(value):
                try:
                    nums.append(float(match))
                except ValueError:
                    pass
    return nums


def _collect_colors(svg: str, root: ET.Element | None) -> List[str]:
    colors = [c.lower() for c in COLOR_RE.findall(svg)]
    for el in _iter_elements(root):
        for key in ("fill", "stroke", "stop-color"):
            value = el.attrib.get(key, "").strip().lower()
            if value and value not in {"none", "transparent"} and not value.startswith("url("):
                colors.append(value)
    return colors


def _keyword_score(prompt: str, tag_counts: Counter, colors: List[str]) -> float:
    prompt_l = (prompt or "").lower()
    if not prompt_l:
        return 0.0

    wanted_shapes = 0
    matched_shapes = 0
    for tag, words in SHAPE_KEYWORDS.items():
        if any(word in prompt_l for word in words):
            wanted_shapes += 1
            if tag_counts.get(tag, 0) > 0:
                matched_shapes += 1

    prompt_color_count = sum(1 for word in COLOR_WORDS if word in prompt_l)
    has_svg_colors = len(set(colors)) > 0
    color_score = 1.0 if prompt_color_count == 0 else (1.0 if has_svg_colors else 0.0)
    shape_score = 1.0 if wanted_shapes == 0 else matched_shapes / wanted_shapes
    return 0.7 * shape_score + 0.3 * color_score


def evaluate_svg(prompt: str, svg: str) -> Dict[str, Any]:
    """Return a score dictionary with component scores in [0, 1]."""

    svg_text, single_root_text = _extract_single_svg(svg)
    root, parse_error = _parse_svg(svg_text)
    root_tag = _local_name(root.tag) if root is not None else ""
    parsed = root is not None and root_tag == "svg"

    elements = list(_iter_elements(root))
    tag_counts = Counter(_local_name(el.tag) for el in elements)
    tags = set(tag_counts)
    graphic_count = sum(tag_counts[tag] for tag in GRAPHIC_TAGS)
    forbidden_count = sum(tag_counts[tag] for tag in FORBIDDEN_TAGS)
    unknown_count = sum(count for tag, count in tag_counts.items() if tag not in ALLOWED_TAGS)

    viewbox = root.attrib.get("viewBox", "") if root is not None else ""
    xmlns = root.attrib.get("xmlns", "") if root is not None else ""
    has_xmlns = xmlns == "http://www.w3.org/2000/svg" or "xmlns=\"http://www.w3.org/2000/svg\"" in svg_text

    nums = _all_numbers(root)
    coord_like = [n for n in nums if math.isfinite(n)]
    if coord_like:
        in_bounds = sum(1 for n in coord_like if -32.0 <= n <= 288.0) / len(coord_like)
        extreme_penalty = 1.0 if max(abs(n) for n in coord_like) <= 10000 else 0.0
        bounds_score = in_bounds * extreme_penalty
    else:
        bounds_score = 0.0

    colors = _collect_colors(svg_text, root)
    unique_colors = {c for c in colors if c not in {"currentcolor"}}
    color_count = len(unique_colors)
    if color_count == 0:
        palette_score = 0.0
    elif 2 <= color_count <= 8:
        palette_score = 1.0
    elif color_count == 1 or 9 <= color_count <= 12:
        palette_score = 0.65
    else:
        palette_score = 0.25

    if 3 <= graphic_count <= 80:
        element_score = 1.0
    elif 1 <= graphic_count < 3 or 81 <= graphic_count <= 120:
        element_score = 0.55
    else:
        element_score = 0.0

    length = len(svg_text)
    if 300 <= length <= 6000:
        length_score = 1.0
    elif 120 <= length < 300 or 6000 < length <= 9000:
        length_score = 0.55
    else:
        length_score = 0.0

    structural_score = _clamp01(
        0.35 * float(parsed)
        + 0.15 * float(single_root_text)
        + 0.20 * float(viewbox.strip() == "0 0 256 256")
        + 0.10 * float(has_xmlns)
        + 0.20 * float(forbidden_count == 0 and unknown_count == 0)
    )
    keyword_score = _keyword_score(prompt, tag_counts, list(unique_colors)) if parsed else 0.0

    components = {
        "structure": structural_score,
        "elements": element_score,
        "palette": palette_score,
        "bounds": bounds_score,
        "length": length_score,
        "prompt_coverage": keyword_score,
    }
    total = (
        0.36 * components["structure"]
        + 0.18 * components["elements"]
        + 0.14 * components["palette"]
        + 0.14 * components["bounds"]
        + 0.08 * components["length"]
        + 0.10 * components["prompt_coverage"]
    )
    if not parsed or forbidden_count or unknown_count:
        total *= 0.55 if parsed else 0.0
    if graphic_count == 0:
        total *= 0.25

    return {
        "score": round(_clamp01(total), 6),
        "components": {k: round(_clamp01(v), 6) for k, v in components.items()},
        "valid_svg": bool(parsed and single_root_text and forbidden_count == 0 and unknown_count == 0),
        "graphic_elements": int(graphic_count),
        "unique_colors": sorted(unique_colors),
        "tag_counts": dict(tag_counts),
        "diagnostics": {
            "parse_error": parse_error,
            "single_svg_output": bool(single_root_text),
            "viewBox": viewbox,
            "has_xmlns": bool(has_xmlns),
            "forbidden_count": int(forbidden_count),
            "unknown_count": int(unknown_count),
            "coordinate_count": len(coord_like),
        },
    }


def score_svg(prompt: str, svg: str) -> float:
    """Return only the scalar reward score."""

    return float(evaluate_svg(prompt, svg)["score"])

