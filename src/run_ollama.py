# src/run_ollama.py
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

import requests

from prompt_topologies import TOPOLOGIES, build_prompt
from score import score_one

OLLAMA_URL = "http://localhost:11434/api/generate"


def iter_tasks(paths: List[Path], limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    n = 0
    for p in paths:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
                n += 1
                if limit is not None and n >= limit:
                    return


def load_done_keys(raw_path: Path) -> Set[Tuple[str, str, int, str]]:
    """
    Returns set of completed keys: (task_id, topology_id, repeat_id, model)
    """
    done: Set[Tuple[str, str, int, str]] = set()
    if not raw_path.exists():
        return done
    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                done.add((obj["task_id"], obj["topology_id"], int(obj["repeat_id"]), obj["model"]))
            except Exception:
                continue
    return done


def ollama_generate(
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    num_predict: int,
    seed: Optional[int],
    timeout_s: int = 600,
) -> str:
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": num_predict,
        },
    }
    if seed is not None:
        payload["options"]["seed"] = seed  # supported by some backends

    r = requests.post(OLLAMA_URL, json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--num_predict", type=int, default=512)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--limit_tasks", type=int, default=0, help="0 = no limit")
    ap.add_argument("--run_id", type=str, default="")
    args = ap.parse_args()

    task_paths = [
        Path("data_processed/tasks_ifeval.jsonl"),
        Path("data_processed/tasks_ifeval_fc.jsonl"),
    ]

    run_id = args.run_id.strip() or time.strftime("RUN_%Y%m%d_%H%M%S")
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_path = run_dir / "raw.jsonl"
    scores_path = run_dir / "scores.csv"

    done = load_done_keys(raw_path)
    print(f"Run dir: {run_dir}")
    print(f"Resume: found {len(done)} completed generations")

    # Prepare scores writer
    new_scores = not scores_path.exists()
    f_scores = scores_path.open("a", newline="", encoding="utf-8")
    fieldnames = [
        "task_id",
        "dataset",
        "task_type",
        "model",
        "topology_id",
        "repeat_id",
        "temperature",
        "top_p",
        "num_predict",
        "valid_json",
        "schema_ok",
        "format_ok",
        "format_name",
        "banned_ok",
        "bullets_ok",
        "overall_pass",
    ]
    writer = csv.DictWriter(f_scores, fieldnames=fieldnames)
    if new_scores:
        writer.writeheader()

    limit = None if args.limit_tasks <= 0 else args.limit_tasks

    with raw_path.open("a", encoding="utf-8") as f_raw:
        for task in iter_tasks(task_paths, limit=limit):
            task_id = task["task_id"]
            for topo in TOPOLOGIES:
                prompt = build_prompt(task, topo.topology_id)

                for r_i in range(args.repeats):
                    for model in args.models:
                        key = (task_id, topo.topology_id, r_i, model)
                        if key in done:
                            continue

                        # reproducible-ish seed (still okay if backend ignores it)
                        seed = (hash(task_id) ^ hash(topo.topology_id) ^ hash(model) ^ r_i) & 0x7FFFFFFF

                        try:
                            out_text = ollama_generate(
                                model=model,
                                prompt=prompt,
                                temperature=args.temperature,
                                top_p=args.top_p,
                                num_predict=args.num_predict,
                                seed=seed,
                            )
                        except Exception as e:
                            print(f"ERROR model={model} task={task_id} topo={topo.topology_id} r={r_i}: {e}")
                            time.sleep(5)
                            continue

                        raw_obj = {
                            "task_id": task_id,
                            "dataset": task.get("dataset", ""),
                            "task_type": task.get("task_type", ""),
                            "model": model,
                            "topology_id": topo.topology_id,
                            "repeat_id": r_i,
                            "params": {
                                "temperature": args.temperature,
                                "top_p": args.top_p,
                                "num_predict": args.num_predict,
                                "seed": seed,
                            },
                            "prompt": prompt,
                            "response": out_text,
                        }
                        f_raw.write(json.dumps(raw_obj, ensure_ascii=False) + "\n")
                        f_raw.flush()

                        s = score_one(task, out_text)

                        writer.writerow(
                            {
                                "task_id": task_id,
                                "dataset": task.get("dataset", ""),
                                "task_type": task.get("task_type", ""),
                                "model": model,
                                "topology_id": topo.topology_id,
                                "repeat_id": r_i,
                                "temperature": args.temperature,
                                "top_p": args.top_p,
                                "num_predict": args.num_predict,
                                **s,
                            }
                        )
                        f_scores.flush()

                        done.add(key)

                        if args.sleep > 0:
                            time.sleep(args.sleep)

    f_scores.close()
    print("Done.")


if __name__ == "__main__":
    main()