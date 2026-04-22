# src/analysis_tools.py
from __future__ import annotations

from pathlib import Path
import pandas as pd


def analyze_run(run_id: str) -> dict[str, pd.DataFrame]:
    run_dir = Path("runs") / run_id
    scores_path = run_dir / "scores.csv"
    if not scores_path.exists():
        raise FileNotFoundError(f"Missing {scores_path}")

    df = pd.read_csv(scores_path)

    summary_model_topology = (
        df.groupby(["model", "topology_id"], as_index=False)
        .agg(
            n=("overall_pass", "size"),
            pass_rate=("overall_pass", "mean"),
            pass_std=("overall_pass", "std"),
            json_rate=("valid_json", "mean"),
            schema_rate=("schema_ok", lambda x: (x == 1).mean() if (x != -1).any() else float("nan")),
            format_rate=("format_ok", lambda x: (x == 1).mean() if (x != -1).any() else float("nan")),
        )
        .sort_values(["model", "topology_id"])
    )

    summary_topology = (
        df.groupby(["topology_id"], as_index=False)
        .agg(
            n=("overall_pass", "size"),
            pass_rate=("overall_pass", "mean"),
            pass_std=("overall_pass", "std"),
            json_rate=("valid_json", "mean"),
        )
        .sort_values(["topology_id"])
    )

    summary_dataset_tasktype = (
        df.groupby(["dataset", "task_type", "model", "topology_id"], as_index=False)
        .agg(
            n=("overall_pass", "size"),
            pass_rate=("overall_pass", "mean"),
            pass_std=("overall_pass", "std"),
            json_rate=("valid_json", "mean"),
            schema_rate=("schema_ok", lambda x: (x == 1).mean() if (x != -1).any() else float("nan")),
            format_rate=("format_ok", lambda x: (x == 1).mean() if (x != -1).any() else float("nan")),
        )
        .sort_values(["dataset", "model", "topology_id"])
    )

    # Save CSVs for report convenience
    summary_model_topology.to_csv(run_dir / "summary_model_topology.csv", index=False)
    summary_topology.to_csv(run_dir / "summary_topology.csv", index=False)
    summary_dataset_tasktype.to_csv(run_dir / "summary_dataset_tasktype.csv", index=False)

    return {
        "scores": df,
        "summary_model_topology": summary_model_topology,
        "summary_topology": summary_topology,
        "summary_dataset_tasktype": summary_dataset_tasktype,
    }

def patch_scores_csv_for_analysis(run_id, fallback_model_name="scratch-gpt"):
    run_dir = Path("runs") / run_id
    scores_path = run_dir / "scores.csv"

    if not scores_path.exists():
        raise FileNotFoundError(f"Missing {scores_path}")

    df = pd.read_csv(scores_path)

    # If the run produced no rows, make that obvious early
    if df.empty:
        raise ValueError(f"{scores_path} exists but is empty.")

    # Patch missing columns expected by analyze_run()
    if "model" not in df.columns:
        df["model"] = fallback_model_name

    # Optional safety: some pipelines expect topology_id too
    if "topology_id" not in df.columns:
        if "topology" in df.columns:
            df["topology_id"] = df["topology"]
        else:
            df["topology_id"] = "default"

    df.to_csv(scores_path, index=False)
    print(f"Patched {scores_path} with columns: {list(df.columns)}")