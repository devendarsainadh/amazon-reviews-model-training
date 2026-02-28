from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from pyspark.sql import functions as F

from common import build_spark, ensure_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Tableau-ready datasets")
    p.add_argument("--silver-path", default="outputs/silver_reviews")
    p.add_argument("--metrics-json", default="outputs/metrics/model_metrics.json")
    p.add_argument("--scaling-csv", default="outputs/metrics/scalability_metrics.csv")
    p.add_argument("--out-dir", default="outputs/tableau")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    ensure_dir(out)

    spark = build_spark("amazon-reviews-tableau-export")
    spark.sparkContext.setLogLevel("WARN")

    silver = spark.read.parquet(args.silver_path).cache()

    # Dashboard 1: Data quality & pipeline monitoring.
    dq = (
        silver.select(
            F.count("*").alias("rows_total"),
            F.sum(F.when(F.col("review_body").isNull() | (F.col("review_body") == ""), 1).otherwise(0)).alias("null_review_body"),
            F.sum(F.when(F.col("stars").isNull(), 1).otherwise(0)).alias("null_stars"),
        )
        .toPandas()
    )
    dq.to_csv(out / "dashboard1_data_quality.csv", index=False)

    daily = silver.groupBy(F.to_date("review_date").alias("review_day")).agg(F.count("*").alias("reviews_count"))
    daily.toPandas().to_csv(out / "dashboard1_daily_volume.csv", index=False)

    # Dashboard 2: Model performance & feature importance.
    with open(args.metrics_json, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    pd.DataFrame(metrics.get("models", [])).to_csv(out / "dashboard2_model_performance.csv", index=False)

    # Dashboard 3: Business insights.
    biz = (
        silver.groupBy("product_category", F.to_date("review_date").alias("review_day"))
        .agg(
            F.count("*").alias("reviews"),
            F.avg("stars").alias("avg_stars"),
            F.avg("is_positive").alias("positive_rate"),
        )
        .orderBy("review_day")
    )
    biz.toPandas().to_csv(out / "dashboard3_business_insights.csv", index=False)

    # Dashboard 4: Scalability & cost.
    if Path(args.scaling_csv).exists():
        pd.read_csv(args.scaling_csv).to_csv(out / "dashboard4_scalability.csv", index=False)

    silver.unpersist()
    spark.stop()


if __name__ == "__main__":
    main()
