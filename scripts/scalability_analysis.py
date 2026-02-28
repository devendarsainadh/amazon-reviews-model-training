from __future__ import annotations

import argparse
import time
from typing import Dict, List

import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import HashingTF, IDF, StopWordsRemover, Tokenizer
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from common import build_spark, ensure_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run strong/weak scaling experiments")
    p.add_argument("--bronze-path", default="outputs/bronze_reviews")
    p.add_argument("--output-csv", default="outputs/metrics/scalability_metrics.csv")
    p.add_argument("--partitions", default="4,8,16")
    p.add_argument("--base-fraction", type=float, default=0.05)
    return p.parse_args()


def prepare(df: DataFrame) -> DataFrame:
    return (
        df.select(F.concat_ws(" ", F.coalesce("review_title", F.lit("")), F.coalesce("review_body", F.lit(""))).alias("text"), "stars")
        .withColumn("label", F.when(F.col("stars") >= 4, F.lit(1.0)).otherwise(F.lit(0.0)))
        .dropna()
    )


def train_once(df: DataFrame) -> float:
    tok = Tokenizer(inputCol="text", outputCol="tokens")
    rem = StopWordsRemover(inputCol="tokens", outputCol="clean")
    tf = HashingTF(inputCol="clean", outputCol="tf", numFeatures=1 << 16)
    idf = IDF(inputCol="tf", outputCol="features")
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=20)
    pipe = Pipeline(stages=[tok, rem, tf, idf, lr])
    start = time.time()
    pipe.fit(df)
    return time.time() - start


def main() -> None:
    args = parse_args()
    spark = build_spark("amazon-reviews-scalability")
    spark.sparkContext.setLogLevel("WARN")

    part_values = [int(x.strip()) for x in args.partitions.split(",") if x.strip()]
    base = prepare(spark.read.parquet(args.bronze_path))

    records: List[Dict] = []

    # Strong scaling: fixed data size, increasing partitions/resources.
    fixed = base.sample(False, args.base_fraction, seed=42).cache()
    fixed.count()
    for p in part_values:
        t = train_once(fixed.repartition(p))
        records.append({"experiment": "strong", "partitions": p, "fraction": args.base_fraction, "train_seconds": round(t, 2)})
    fixed.unpersist()

    # Weak scaling: increase data size with partitions.
    for i, p in enumerate(part_values, start=1):
        frac = min(args.base_fraction * i, 1.0)
        d = base.sample(False, frac, seed=42).repartition(p).cache()
        d.count()
        t = train_once(d)
        records.append({"experiment": "weak", "partitions": p, "fraction": frac, "train_seconds": round(t, 2)})
        d.unpersist()

    out = pd.DataFrame(records)
    out["cost_proxy"] = out["partitions"] * out["train_seconds"]
    ensure_dir("outputs/metrics")
    out.to_csv(args.output_csv, index=False)
    spark.stop()


if __name__ == "__main__":
    main()
