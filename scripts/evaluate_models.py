from __future__ import annotations

import argparse
from typing import Dict

import numpy as np
import pandas as pd
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from common import build_spark, write_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate saved model predictions with bootstrap CIs and business metrics")
    p.add_argument("--predictions-path", default="outputs/predictions")
    p.add_argument("--output-json", default="outputs/metrics/evaluation_metrics.json")
    p.add_argument("--bootstrap-iters", type=int, default=200)
    p.add_argument("--profit-tp", type=float, default=3.0)
    p.add_argument("--cost-fp", type=float, default=1.0)
    p.add_argument("--cost-fn", type=float, default=2.0)
    return p.parse_args()


def bootstrap_ci(pdf: pd.DataFrame, metric_col: str, label_col: str, iters: int) -> Dict[str, float]:
    vals = []
    n = len(pdf)
    if n == 0:
        return {"low": 0.0, "high": 0.0}
    for _ in range(iters):
        sample = pdf.sample(n=n, replace=True)
        vals.append((sample[metric_col] == sample[label_col]).mean())
    low, high = np.percentile(vals, [2.5, 97.5])
    return {"low": float(low), "high": float(high)}


def business_value(df: DataFrame, tp_profit: float, fp_cost: float, fn_cost: float) -> Dict[str, float]:
    agg = (
        df.select(
            F.sum(F.when((F.col("label") == 1.0) & (F.col("prediction") == 1.0), 1).otherwise(0)).alias("tp"),
            F.sum(F.when((F.col("label") == 0.0) & (F.col("prediction") == 1.0), 1).otherwise(0)).alias("fp"),
            F.sum(F.when((F.col("label") == 1.0) & (F.col("prediction") == 0.0), 1).otherwise(0)).alias("fn"),
        )
        .collect()[0]
        .asDict()
    )
    expected_profit = agg["tp"] * tp_profit - agg["fp"] * fp_cost - agg["fn"] * fn_cost
    return {"expected_profit": float(expected_profit), **agg}


def main() -> None:
    args = parse_args()
    spark = build_spark("amazon-reviews-evaluation")
    spark.sparkContext.setLogLevel("WARN")

    pred = spark.read.parquet(args.predictions_path).select("label", "prediction", "probability", "rawPrediction")
    auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC").evaluate(pred)
    f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1").evaluate(pred)

    pdf = pred.select("label", "prediction").toPandas()
    ci = bootstrap_ci(pdf, metric_col="prediction", label_col="label", iters=args.bootstrap_iters)
    biz = business_value(pred, args.profit_tp, args.cost_fp, args.cost_fn)

    payload = {
        "auc": auc,
        "f1": f1,
        "bootstrap_accuracy_ci": ci,
        "business_metrics": biz,
        "notes": "Use temporal splits upstream to avoid leakage; for imbalanced subsets, use class-weighting or threshold tuning.",
    }
    write_json(args.output_json, payload)
    spark.stop()


if __name__ == "__main__":
    main()
