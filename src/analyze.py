# src/analyze.py
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", type=str, required=True)
    args = ap.parse_args()

    run_dir = Path("runs") / args.run_id
    scores_path = run_dir / "scores.csv"
    if not scores_path.exists():
        raise FileNotFoundError(f"Missing {scores_path}")

    df = pd.read_csv(scores_path)

    # Summary: per (model, topology)
    s1 = (
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
    s1.to_csv(run_dir / "summary_model_topology.csv", index=False)

    # Summary: topology pooled across models
    s2 = (
        df.groupby(["topology_id"], as_index=False)
        .agg(
            n=("overall_pass", "size"),
            pass_rate=("overall_pass", "mean"),
            pass_std=("overall_pass", "std"),
            json_rate=("valid_json", "mean"),
        )
        .sort_values(["topology_id"])
    )
    s2.to_csv(run_dir / "summary_topology.csv", index=False)

    # Summary: per dataset/task_type
    s3 = (
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
    s3.to_csv(run_dir / "summary_dataset_tasktype.csv", index=False)

    print("Wrote:")
    print(" -", run_dir / "summary_model_topology.csv")
    print(" -", run_dir / "summary_topology.csv")
    print(" -", run_dir / "summary_dataset_tasktype.csv")


if __name__ == "__main__":
    main()