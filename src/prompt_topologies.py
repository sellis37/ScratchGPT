# src/prompt_topologies.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class Topology:
    topology_id: str
    name: str


TOPOLOGIES: List[Topology] = [
    Topology("C_E_T", "Constraints → Examples → Task"),
    Topology("E_C_T", "Examples → Constraints → Task"),
    Topology("T_C_E", "Task → Constraints → Examples"),
    Topology("Cdelim_E_T", "Delimited-bullets Constraints → Examples → Task"),
    Topology("Cinterleave_T", "Constraints interleaved with Examples → Task"),
    Topology("C_E_T_C", "Constraints → Examples → Task → Constraints repeated"),
]


def _render_constraints(constraints: List[str], delimited: bool) -> str:
    if not constraints:
        return ""
    bullets = "\n".join([f"- {c}" for c in constraints])
    if delimited:
        return f"---CONSTRAINTS---\n{bullets}\n---END CONSTRAINTS---\n"
    return f"Constraints:\n{bullets}\n"


def _render_examples(task: Dict[str, Any]) -> str:
    """
    Optional: if you store examples in task['meta']['examples'] as:
      [{"input": "...", "output": "..."}, ...]
    """
    meta = task.get("meta", {}) or {}
    examples = meta.get("examples", None)
    if not examples:
        return ""
    lines: List[str] = ["Examples:"]
    for i, e in enumerate(examples):
        lines.append(f"Example {i+1} Input:\n{e.get('input','')}")
        lines.append(f"Example {i+1} Output:\n{e.get('output','')}")
        lines.append("")  # spacer
    return "\n".join(lines).strip() + "\n"


def _render_task_block(task: Dict[str, Any]) -> str:
    instr = task.get("instruction", "")
    inp = task.get("input", "")

    if inp:
        return f"Task:\n{instr}\n\nInput:\n{inp}\n\nAnswer:"
    return f"Task:\n{instr}\n\nAnswer:"


def build_prompt(task: Dict[str, Any], topology_id: str) -> str:
    """
    Builds a prompt containing the same semantic content but different ordering/structure.
    Special-cases IFEval-FC tasks to strongly encourage JSON-only output.
    """
    constraints = task.get("constraints", []) or []
    examples = _render_examples(task)
    task_block = _render_task_block(task)

    if topology_id == "C_E_T":
        prompt = f"{_render_constraints(constraints, delimited=False)}\n{examples}\n{task_block}".strip()

    elif topology_id == "E_C_T":
        prompt = f"{examples}\n{_render_constraints(constraints, delimited=False)}\n{task_block}".strip()

    elif topology_id == "T_C_E":
        prompt = f"{task_block}\n\n{_render_constraints(constraints, delimited=False)}\n{examples}".strip()

    elif topology_id == "Cdelim_E_T":
        prompt = f"{_render_constraints(constraints, delimited=True)}\n{examples}\n{task_block}".strip()

    elif topology_id == "Cinterleave_T":
        # If no examples exist, interleaving doesn't make sense; fall back.
        if not examples:
            prompt = f"{_render_constraints(constraints, delimited=False)}\n{task_block}".strip()
        else:
            # Split constraints into two roughly equal chunks and place around examples
            c_lines = [f"- {c}" for c in constraints]
            if not c_lines:
                prompt = f"{examples}\n{task_block}".strip()
            else:
                mid = max(1, len(c_lines) // 2)
                c1 = "Constraints (part 1):\n" + "\n".join(c_lines[:mid]) + "\n"
                c2 = "Constraints (part 2):\n" + "\n".join(c_lines[mid:]) + "\n"
                prompt = f"{c1}\n{examples}\n{c2}\n{task_block}".strip()

    elif topology_id == "C_E_T_C":
        c = _render_constraints(constraints, delimited=False).strip()
        prompt = f"{c}\n\n{examples}\n{task_block}\n\nReminder:\n{c}".strip()

    else:
        raise ValueError(f"Unknown topology_id: {topology_id}")

    # Strong JSON-only reminder for IFEval-FC (schema_json tasks)
    if task.get("dataset") == "ifeval-fc":
        prompt += "\n\nIMPORTANT: Output ONLY the JSON arguments object. No extra text."

    return prompt