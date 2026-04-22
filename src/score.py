# src/score.py
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple

from score_fc_checkers import score_format_checker


def try_parse_json(text: str) -> Tuple[int, Optional[Any]]:
    """
    Returns (valid_json_flag, parsed_obj_or_None).
    Tries strict parse, then tries parsing from the first '{' or '[' onward.
    """
    text = text.strip()

    try:
        return 1, json.loads(text)
    except Exception:
        pass

    m = re.search(r"[\{\[]", text)
    if not m:
        return 0, None

    candidate = text[m.start():]
    try:
        return 1, json.loads(candidate)
    except Exception:
        return 0, None


def schema_validate_minimal(obj: Any, schema: Any) -> int:
    """
    Minimal schema validation (enough for your project plots):
    - requires object/dict if schema is dict
    - checks 'required'
    - checks additionalProperties == False vs properties/required keys

    Returns:
      1 pass, 0 fail, -1 unknown schema format
    """
    if not isinstance(schema, dict):
        return -1
    if not isinstance(obj, dict):
        return 0

    required = schema.get("required", None)
    if isinstance(required, list):
        for k in required:
            if k not in obj:
                return 0

    if schema.get("additionalProperties") is False:
        props = schema.get("properties", {})
        if isinstance(props, dict):
            allowed = set(props.keys()) | set(required or [])
            for k in obj.keys():
                if k not in allowed:
                    return 0

    return 1


def count_bullets(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.strip().startswith(("-", "*")))


def score_one(task: Dict[str, Any], output_text: str) -> Dict[str, Any]:
    task_type = task.get("task_type", "unknown")
    dataset = task.get("dataset", "")
    constraints = task.get("constraints", []) or []

    # --- Generic constraint extraction (optional; extend later as needed) ---
    banned_words: list[str] = []
    exact_bullets: Optional[int] = None

    for c in constraints:
        m = re.search(r"Do not use:\s*(.+)$", c, flags=re.I)
        if m:
            banned_words += [w.strip() for w in m.group(1).split(",") if w.strip()]
        m = re.search(r"exactly\s+(\d+)\s+bullets", c, flags=re.I)
        if m:
            exact_bullets = int(m.group(1))

    low = output_text.lower()
    banned_ok = 1
    for w in banned_words:
        if w.lower() in low:
            banned_ok = 0
            break

    bullets_ok = 1
    if exact_bullets is not None:
        bullets_ok = 1 if count_bullets(output_text) == exact_bullets else 0

    # --- JSON parse + schema validation (for schema_json tasks) ---
    valid_json, parsed = try_parse_json(output_text)

    schema_ok = -1
    schema = task.get("schema", None)
    if schema is not None and valid_json == 1:
        # schema may be a JSON string
        if isinstance(schema, str):
            try:
                schema = json.loads(schema)
            except Exception:
                schema = None
        if schema is not None:
            schema_ok = schema_validate_minimal(parsed, schema)
        else:
            schema_ok = -1

    # --- IFEval-FC special: format checker (deterministic) ---
    format_ok = -1
    format_name = "none"
    if dataset == "ifeval-fc":
        meta = task.get("meta", {}) or {}
        if valid_json == 1:
            format_ok, format_name = score_format_checker(parsed, meta)
        else:
            # if not JSON, it's definitely failing the "JSON args" requirement
            checker = meta.get("format_checker", {})
            format_name = checker.get("name", "unknown")
            format_ok = 0

    # --- Overall pass ---
    if task_type == "schema_json":
        # If format_ok is unknown (-1), don't penalize; otherwise require pass (=1)
        fmt_gate = (format_ok in (1, -1))
        overall_pass = 1 if (valid_json == 1 and schema_ok == 1 and banned_ok == 1 and bullets_ok == 1 and fmt_gate) else 0
    else:
        # For plain IFEval instruction_following, we track generic constraints only
        overall_pass = 1 if (banned_ok == 1 and bullets_ok == 1) else 0

    return {
        "valid_json": valid_json,
        "schema_ok": schema_ok,
        "format_ok": format_ok,
        "format_name": format_name,
        "banned_ok": banned_ok,
        "bullets_ok": bullets_ok,
        "overall_pass": overall_pass,
    }