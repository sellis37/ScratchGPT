import json
from pathlib import Path
from typing import Any, Iterable
import pandas as pd
import numpy as np

def _to_jsonable(x: Any) -> Any:
    """Recursively convert pandas/numpy objects to JSON-serializable Python types."""
    if x is None:
        return None

    # pandas missing
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass

    if isinstance(x, np.generic):
        return x.item()

    if isinstance(x, np.ndarray):
        return [_to_jsonable(v) for v in x.tolist()]

    if isinstance(x, pd.Timestamp):
        return x.isoformat()

    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}

    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]

    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", errors="replace")

    return x

def _loads_if_json_string(x: Any) -> Any:
    """If x is a JSON string, parse it; else return x."""
    if isinstance(x, str):
        s = x.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                return json.loads(s)
            except Exception:
                return x
    return x

def convert(in_path: str | Path, out_path: str | Path) -> int:
    in_path = Path(in_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(in_path)
    print("IFEval-FC columns:", list(df.columns))
    print("Rows:", len(df))

    n_tasks = 0
    with out_path.open("w", encoding="utf-8") as f_out:
        for row_idx, r in df.iterrows():
            row = r.to_dict()

            # Parse JSON strings into dicts (if they are strings)
            fn_schema = _loads_if_json_string(row.get("fn_schema"))
            fmt_checker = _loads_if_json_string(row.get("format"))

            chosen_param = row.get("chosen_param", None)
            filename = row.get("filename", f"ifeval_fc_row{row_idx}")

            # user_queries is usually a list; sometimes it may come as ndarray
            user_queries = _to_jsonable(row.get("user_queries", []))
            if isinstance(user_queries, str):
                # if it's a stringified JSON list, try to parse
                user_queries = _loads_if_json_string(user_queries)

            if not isinstance(user_queries, list):
                # fallback: treat as single query
                user_queries = [str(user_queries)]

            # explode: one task per query
            for q_idx, query in enumerate(user_queries):
                task_id = f"{filename}__q{q_idx}"

                instruction = (
                    "You are given a user request and a function schema.\n"
                    "Return ONLY a JSON object of arguments for calling the function.\n"
                    "Do not include the function name or any extra text.\n\n"
                    f"User request:\n{query}\n\n"
                    f"Function schema:\n{json.dumps(fn_schema, ensure_ascii=False)}\n\n"
                    f"Additional format constraint (applies to parameter '{chosen_param}'):\n"
                    f"{json.dumps(fmt_checker, ensure_ascii=False)}\n\n"
                    "Return only the JSON arguments object:"
                )

                task = {
                    "task_id": task_id,
                    "dataset": "ifeval-fc",
                    "task_type": "schema_json",
                    "instruction": instruction,
                    "input": "",
                    "constraints": [
                        "Output must be valid JSON.",
                        "Output must match the function schema parameter types/required keys.",
                    ],
                    "schema": _to_jsonable(fn_schema.get("parameters") if isinstance(fn_schema, dict) else fn_schema),
                    "gold": None,  # not provided in this parquet
                    "meta": _to_jsonable({
                        "row_idx": row_idx,
                        "filename": filename,
                        "chosen_param": chosen_param,
                        "format_checker": fmt_checker,
                        "fn_schema": fn_schema,
                        "original_query_index": q_idx,
                    }),
                }

                f_out.write(json.dumps(task, ensure_ascii=False) + "\n")
                n_tasks += 1

    return n_tasks

def main():
    n = convert("data_raw/ifeval-fc_input_data.parquet", "data_processed/tasks_ifeval_fc.jsonl")
    print(f"Wrote {n} tasks -> data_processed/tasks_ifeval_fc.jsonl")

if __name__ == "__main__":
    main()