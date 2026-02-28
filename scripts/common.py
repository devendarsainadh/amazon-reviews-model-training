from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from pyspark.sql import SparkSession


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def git_commit_hash() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def build_spark(app_name: str, extra_conf: Dict[str, str] | None = None) -> SparkSession:
    conf = {
        "spark.sql.adaptive.enabled": "true",
        "spark.sql.adaptive.coalescePartitions.enabled": "true",
        "spark.sql.shuffle.partitions": os.getenv("SPARK_SHUFFLE_PARTITIONS", "200"),
        "spark.sql.files.maxPartitionBytes": "134217728",
        "spark.serializer": "org.apache.spark.serializer.KryoSerializer",
        "spark.driver.maxResultSize": "2g",
        "spark.sql.parquet.compression.codec": "snappy",
        "spark.sql.execution.arrow.pyspark.enabled": "true",
    }
    if extra_conf:
        conf.update(extra_conf)

    builder = SparkSession.builder.appName(app_name)
    for k, v in conf.items():
        builder = builder.config(k, v)
    return builder.getOrCreate()


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def append_jsonl(path: str | Path, payload: Dict[str, Any]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
