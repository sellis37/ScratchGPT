# src/experiment.py
from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

import requests

from prompt_topologies import TOPOLOGIES, build_prompt
from score import score_one

OLLAMA_URL = "http://localhost:11434/api/generate"


def load_jsonl(path: str | Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    path = Path(path)
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def load_done_keys(raw_path: Path) -> Set[Tuple[str, str, int, str]]:
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
    endpoint: str = OLLAMA_URL,
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
        payload["options"]["seed"] = seed

    r = requests.post(endpoint, json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")


def run_experiment(
    *,
    run_id: str,
    models: list[str],
    tasks: list[dict],
    repeats: int = 10,
    temperature: float = 0.7,
    top_p: float = 1.0,
    num_predict: int = 512,
    sleep_s: float = 0.0,
    max_tasks: int | None = None,
    progress_every: int = 100,
    endpoint: str = OLLAMA_URL,
) -> Path:
    """
    Runs the full experiment and writes:
      runs/<run_id>/raw.jsonl
      runs/<run_id>/scores.csv

    Resume-safe: re-running the same run_id skips completed generations.
    """
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    raw_path = run_dir / "raw.jsonl"
    scores_path = run_dir / "scores.csv"

    done = load_done_keys(raw_path)
    print(f"Run dir: {run_dir}")
    print(f"Resume: found {len(done)} completed generations")

    # Prepare CSV writer
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

    tasks_iter = tasks[:max_tasks] if max_tasks is not None else tasks

    # Precompute total planned generations for ETA/progress
    total_planned = len(tasks_iter) * len(TOPOLOGIES) * repeats * len(models)
    remaining_planned = max(0, total_planned - len(done))

    print(f"Planned generations this run_id: {total_planned}")
    print(f"Remaining to run now: {remaining_planned}")

    started = time.time()
    completed_now = 0  # how many generations this invocation actually completed (not resumed/skipped)

    with raw_path.open("a", encoding="utf-8") as f_raw:
        for t_i, task in enumerate(tasks_iter):
            task_id = task["task_id"]

            for topo in TOPOLOGIES:
                prompt = build_prompt(task, topo.topology_id)

                for r_i in range(repeats):
                    for model in models:
                        key = (task_id, topo.topology_id, r_i, model)
                        if key in done:
                            continue

                        seed = (hash(task_id) ^ hash(topo.topology_id) ^ hash(model) ^ r_i) & 0x7FFFFFFF

                        try:
                            out_text = ollama_generate(
                                model=model,
                                prompt=prompt,
                                temperature=temperature,
                                top_p=top_p,
                                num_predict=num_predict,
                                seed=seed,
                                endpoint=endpoint,
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
                                "temperature": temperature,
                                "top_p": top_p,
                                "num_predict": num_predict,
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
                                "temperature": temperature,
                                "top_p": top_p,
                                "num_predict": num_predict,
                                **s,
                            }
                        )
                        f_scores.flush()
                        done.add(key)
                        completed_now += 1

                        # NEW: periodic progress logging
                        if progress_every and (completed_now % progress_every == 0):
                            elapsed = time.time() - started
                            rate = completed_now / elapsed if elapsed > 0 else 0.0  # gens/sec
                            sec_per_gen = (elapsed / completed_now) if completed_now > 0 else float("nan")
                            remaining = max(0, remaining_planned - completed_now)
                            eta_sec = (remaining / rate) if rate > 0 else float("inf")
                            pct = (completed_now / remaining_planned * 100.0) if remaining_planned > 0 else 100.0

                            def _fmt_hms(sec: float) -> str:
                                if sec == float("inf"):
                                    return "inf"
                                sec = int(round(sec))
                                h = sec // 3600
                                m = (sec % 3600) // 60
                                s = sec % 60
                                return f"{h:02d}:{m:02d}:{s:02d}"

                            print(
                                f"[{completed_now}/{remaining_planned} | {pct:5.1f}%] "
                                f"elapsed={_fmt_hms(elapsed)} "
                                f"avg={sec_per_gen:.2f}s/gen "
                                f"eta={_fmt_hms(eta_sec)}"
                            )

                        if sleep_s > 0:
                            time.sleep(sleep_s)

    f_scores.close()

    total_elapsed = time.time() - started
    if completed_now > 0:
        print(f"Done. Completed {completed_now} new generations in {total_elapsed/3600:.2f} hours "
              f"({total_elapsed/completed_now:.2f} s/gen).")
    else:
        print("Done. No new generations were needed (everything already completed).")

    return run_dir