from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

from datasets import load_dataset
from pyspark.sql import Row
from pyspark.sql import functions as F
from pyspark.sql import types as T

from common import append_jsonl, build_spark, ensure_dir, git_commit_hash, write_json


SCHEMA = T.StructType(
    [
        T.StructField("review_id", T.StringType(), False),
        T.StructField("product_id", T.StringType(), True),
        T.StructField("reviewer_id", T.StringType(), True),
        T.StructField("stars", T.IntegerType(), True),
        T.StructField("review_body", T.StringType(), True),
        T.StructField("review_title", T.StringType(), True),
        T.StructField("language", T.StringType(), True),
        T.StructField("product_category", T.StringType(), True),
        T.StructField("review_date", T.StringType(), True),
        T.StructField("verified_purchase", T.StringType(), True),
        T.StructField("vine", T.StringType(), True),
    ]
)


REQUIRED = {"review_id", "stars", "review_body", "review_date", "language", "product_category"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest Amazon reviews from Hugging Face to partitioned Parquet")
    p.add_argument("--dataset-name", default="McAuley-Lab/Amazon-Reviews-2023")
    p.add_argument("--dataset-config", default="raw_review_Electronics")
    p.add_argument("--split", default="full")
    p.add_argument("--streaming", action="store_true", default=True)
    p.add_argument("--batch-size", type=int, default=50000)
    p.add_argument("--max-rows", type=int, default=0, help="0 for all")
    p.add_argument("--bronze-path", default="outputs/bronze_reviews")
    p.add_argument("--audit-log", default="outputs/logs/ingestion_rejects.jsonl")
    p.add_argument("--metrics-json", default="outputs/metrics/ingestion_metrics.json")
    return p.parse_args()


def validated_rows(stream: Iterable[Dict], reject_log_path: str) -> Iterator[Row]:
    for idx, item in enumerate(stream):
        raw = normalize_record(item)
        missing = [k for k in REQUIRED if raw.get(k) in (None, "")]
        if missing:
            append_jsonl(reject_log_path, {"idx": idx, "reason": "missing_required", "fields": missing})
            continue

        try:
            stars = int(raw.get("stars")) if raw.get("stars") is not None else None
            if stars is not None and not (1 <= stars <= 5):
                raise ValueError("stars_out_of_range")
            datetime.strptime(raw["review_date"], "%Y-%m-%d")
        except Exception as exc:
            append_jsonl(reject_log_path, {"idx": idx, "reason": "validation_error", "error": str(exc)})
            continue

        yield Row(
            review_id=str(raw.get("review_id")),
            product_id=raw.get("product_id"),
            reviewer_id=raw.get("reviewer_id"),
            stars=stars,
            review_body=raw.get("review_body"),
            review_title=raw.get("review_title"),
            language=raw.get("language"),
            product_category=raw.get("product_category"),
            review_date=raw.get("review_date"),
            verified_purchase=raw.get("verified_purchase"),
            vine=raw.get("vine"),
        )


def normalize_record(item: Dict) -> Dict:
    if "review_id" in item:
        return item

    ts = item.get("timestamp")
    if ts is None:
        review_date = None
    else:
        review_date = datetime.utcfromtimestamp(int(ts) / 1000).strftime("%Y-%m-%d")

    category = item.get("_dataset_config", "unknown")
    if category.startswith("raw_review_"):
        category = category.replace("raw_review_", "", 1)

    return {
        "review_id": f"{item.get('user_id', 'na')}_{item.get('asin', 'na')}_{item.get('timestamp', 'na')}",
        "product_id": item.get("asin"),
        "reviewer_id": item.get("user_id"),
        "stars": item.get("rating"),
        "review_body": item.get("text"),
        "review_title": item.get("title"),
        "language": "en",
        "product_category": category,
        "review_date": review_date,
        "verified_purchase": str(item.get("verified_purchase")) if item.get("verified_purchase") is not None else None,
        "vine": None,
    }


def batched(rows: Iterator[Row], n: int) -> Iterator[List[Row]]:
    buf: List[Row] = []
    for row in rows:
        buf.append(row)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf


def main() -> None:
    args = parse_args()
    ensure_dir(Path(args.bronze_path))
    spark = build_spark("amazon-reviews-ingestion")
    spark.sparkContext.setLogLevel("WARN")

    # Data lineage metadata for reproducibility and auditability.
    lineage = {
        "dataset": args.dataset_name,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "streaming": args.streaming,
        "git_commit": git_commit_hash(),
    }

    ds = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.split,
        streaming=args.streaming,
        trust_remote_code=True,
    )
    ds = ds.map(lambda x: {**x, "_dataset_config": args.dataset_config})
    valid_iter = validated_rows(ds, args.audit_log)

    total_written = 0
    for chunk_idx, chunk in enumerate(batched(valid_iter, args.batch_size)):
        if args.max_rows > 0 and total_written >= args.max_rows:
            break
        if args.max_rows > 0 and total_written + len(chunk) > args.max_rows:
            chunk = chunk[: args.max_rows - total_written]

        sdf = spark.createDataFrame(chunk, schema=SCHEMA)
        sdf = (
            sdf.withColumn("review_ts", F.to_timestamp("review_date"))
            .withColumn("review_year", F.year("review_ts"))
            .withColumn("review_month", F.month("review_ts"))
            .drop("review_ts")
        )

        (
            sdf.repartition("language", "review_year", "review_month")
            .write.mode("append")
            .partitionBy("language", "review_year", "review_month")
            .parquet(args.bronze_path)
        )
        total_written += len(chunk)

        append_jsonl(
            "outputs/logs/ingestion_batches.jsonl",
            {"batch": chunk_idx, "rows": len(chunk), "total_written": total_written},
        )

    metrics = {
        **lineage,
        "rows_written": total_written,
        "bronze_path": args.bronze_path,
        "audit_log": args.audit_log,
    }
    write_json(args.metrics_json, metrics)
    spark.stop()


if __name__ == "__main__":
    main()
