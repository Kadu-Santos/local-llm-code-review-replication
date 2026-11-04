"""utils.py

Shared utilities:
- load_data: read JSON/JSONL datasets
- construct_prompt: build prompts for {direct, direct_explain, full}
- parse_response: robust YES/NO extraction
- save_json: convenience writer

Note: Prompt templates are kept *compatible* with the original experiments.
If you change prompt wording, results may differ.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional


PromptMode = Literal["direct", "direct_explain", "full"]


def load_data(path: str) -> List[Dict[str, Any]]:
    """Load a dataset from JSON (.json) or JSON Lines (.jsonl)."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        if p.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in f if line.strip()]
        return json.load(f)


def construct_prompt(requirement: str, code: str, mode: PromptMode) -> str:
    """Construct the judge prompt.

    The three prompt modes correspond to the paper:
    - direct: verdict only
    - direct_explain: verdict + rationale
    - full: verdict + rationale + fix (if NO)
    """

    base = (
        "You are a senior software engineer specializing in code review and requirement conformance.\n\n"
        "Below is a requirement describing what a Python function is expected to do:\n\n"
        f"{requirement}\n\n"
        "And here is the implementation of the function:\n\n"
        f"```python\n{code}\n```\n\n"
    )

    if mode == "direct":
        return base + (
            "Please ignore any indentation error and lack of type hinting! Does the code satisfy the requirement?\n\n"
            "Answer only with YES or NO."
        )

    if mode == "direct_explain":
        return base + (
            "Step 1: Conformance Check\n\n"
            "Please ignore any indentation error and lack of type hinting! Does the code satisfy the requirement?\n\n"
            "Answer only with YES or NO.\n\n"
            "Step 2: Justification\n\n"
            "If your answer was NO, explain why the code does not meet the requirement.\n\n"
            "Be specific and refer to particular lines, conditions, or behaviors."
        )

    if mode == "full":
        # Keep the original structure used in experiments (verdict -> justification -> fix).
        return base + (
            "Step 1: Conformance Check\n\n"
            "Please ignore any indentation error and lack of type hinting! Does the code satisfy the requirement?\n\n"
            "Answer only with YES or NO.\n\n"
            "Step 2: Justification\n\n"
            "If your answer was NO, explain why the code does not meet the requirement.\n\n"
            "Be specific and refer to particular lines, conditions, or behaviors.\n\n"
            "Step 3: Fix\n\n"
            "If your answer was NO, please fix the code so that it meets the requirement.\n"
            "Return the fixed code in a single Python code block."
        )

    raise ValueError(f"Unknown mode: {mode}")


def parse_response(response_text: str) -> str:
    """Extract only the final YES/NO verdict from a model response.

    Reasoning models often include a <think>...</think> block with internal
    analysis before the actual final answer. We ignore the thought trace and
    prioritize the verdict written in the final answer section. If the model
    does not label the verdict explicitly, we fall back to the last YES/NO token
    in the non-thinking text.
    """
    if not response_text:
        return ""

    text = response_text.strip()

    # Keep only the final text after the reasoning block. This prevents the
    # internal chain-of-thought from being mistaken for the actual verdict.
    if "<think>" in text.lower():
        text = re.split(r"(?is)</think>", text, maxsplit=1)[-1].strip()

    patterns = [
        r"\b(?:final\s+answer|answer|verdict)\s*[:\-]?\s*(YES|NO)\b",
        r"\b(?:therefore|so|thus)\s*,?\s*(YES|NO)\b",
        r"\b(YES|NO)\b",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if not matches:
            continue
        return matches[-1].upper()

    return ""


def save_json(obj: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
