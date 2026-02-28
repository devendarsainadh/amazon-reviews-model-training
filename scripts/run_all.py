from __future__ import annotations

import argparse
import subprocess


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full PySpark data engineering + ML pipeline")
    p.add_argument("--max-rows", type=int, default=300000)
    return p.parse_args()


def run(cmd: str) -> None:
    print(f"\n>>> {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def main() -> None:
    args = parse_args()
    run(f"python3 scripts/ingest_amazon_reviews.py --streaming --max-rows {args.max_rows}")
    run("python3 scripts/train_distributed_models.py")
    run("python3 scripts/evaluate_models.py")
    run("python3 scripts/scalability_analysis.py")
    run("python3 scripts/export_tableau_data.py")


if __name__ == "__main__":
    main()
