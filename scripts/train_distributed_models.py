from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.classification import LinearSVC, LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import (
    HashingTF,
    IDF,
    SQLTransformer,
    StopWordsRemover,
    StringIndexer,
    Tokenizer,
    VectorAssembler,
)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from common import build_spark, ensure_dir, utc_now_iso, write_json
from custom_transformers import TextStatsTransformer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train distributed PySpark MLlib models")
    p.add_argument("--bronze-path", default="outputs/bronze_reviews")
    p.add_argument("--silver-path", default="outputs/silver_reviews")
    p.add_argument("--model-dir", default="outputs/models")
    p.add_argument("--metrics-json", default="outputs/metrics/model_metrics.json")
    p.add_argument("--predictions-path", default="outputs/predictions")
    p.add_argument("--sample-fraction", type=float, default=0.20)
    p.add_argument("--cv-folds", type=int, default=3)
    p.add_argument("--cv-parallelism", type=int, default=2)
    p.add_argument("--fast-mode", action="store_true", help="Use low-memory training settings for local runs")
    return p.parse_args()


def build_features(df: DataFrame) -> DataFrame:
    category_dim = (
        df.select("product_category")
        .where(F.col("product_category").isNotNull())
        .distinct()
        .withColumn("category_name_len", F.length("product_category"))
    )
    enriched = df.join(F.broadcast(category_dim), on="product_category", how="left")

    staged = (
        enriched.select(
            "review_id",
            "review_date",
            "stars",
            "product_category",
            "category_name_len",
            F.coalesce("review_title", F.lit(""),).alias("review_title"),
            F.coalesce("review_body", F.lit(""),).alias("review_body"),
        )
        .withColumn("review_text", F.concat_ws(" ", F.col("review_title"), F.col("review_body")))
        .withColumn("label", F.when(F.col("stars") >= 4, F.lit(1.0)).otherwise(F.lit(0.0)))
        .withColumn("category_name_len", F.coalesce(F.col("category_name_len"), F.lit(0)))
        .withColumn("review_dt", F.to_date("review_date"))
    )
    return staged


def temporal_split(df: DataFrame) -> Tuple[DataFrame, DataFrame, DataFrame]:
    q = df.approxQuantile("unix_time", [0.70, 0.85], 0.01)
    train_end, valid_end = q
    train = df.filter(F.col("unix_time") <= train_end)
    valid = df.filter((F.col("unix_time") > train_end) & (F.col("unix_time") <= valid_end))
    test = df.filter(F.col("unix_time") > valid_end)
    return train, valid, test


def fit_cv_pipeline(train_df: DataFrame, estimator, grid, folds: int, parallelism: int):
    evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    cv = CrossValidator(
        estimator=estimator,
        estimatorParamMaps=grid,
        evaluator=evaluator,
        numFolds=folds,
        parallelism=parallelism,
    )
    return cv.fit(train_df)


def summarize_predictions(name: str, preds: DataFrame) -> Dict:
    auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC").evaluate(preds)
    f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1").evaluate(preds)
    acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy").evaluate(preds)
    return {"model": name, "auc": auc, "f1": f1, "accuracy": acc}


def sklearn_baseline(train_pdf: pd.DataFrame, test_pdf: pd.DataFrame) -> Dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=5)
    x_train = vec.fit_transform(train_pdf["review_text"])
    x_test = vec.transform(test_pdf["review_text"])

    y_train = train_pdf["label"].astype(int)
    y_test = test_pdf["label"].astype(int)

    clf = LogisticRegression(max_iter=200, n_jobs=1)
    clf.fit(x_train, y_train)
    prob = clf.predict_proba(x_test)[:, 1]
    pred = (prob >= 0.5).astype(int)

    return {
        "model": "sklearn_logreg",
        "auc": float(roc_auc_score(y_test, prob)),
        "f1": float(f1_score(y_test, pred)),
        "accuracy": float(accuracy_score(y_test, pred)),
    }


def main() -> None:
    args = parse_args()
    spark = build_spark("amazon-reviews-ml")
    spark.sparkContext.setLogLevel("WARN")

    ensure_dir(args.model_dir)
    ensure_dir(Path(args.metrics_json).parent)

    base = spark.read.parquet(args.bronze_path)
    feat = build_features(base).withColumn("unix_time", F.unix_timestamp("review_dt"))
    feat = feat.dropna(subset=["review_text", "label", "unix_time"])

    if args.fast_mode and args.sample_fraction >= 0.01:
        args.sample_fraction = 0.005
    if args.fast_mode:
        args.cv_folds = min(args.cv_folds, 2)
        args.cv_parallelism = 1

    if args.sample_fraction < 1.0:
        feat = feat.sample(withReplacement=False, fraction=args.sample_fraction, seed=42)
    if args.fast_mode:
        feat = feat.repartition(8)

    # Persist once before repeated CV fits and multi-model scoring.
    feat.persist()
    train, valid, test = temporal_split(feat)

    tokenizer = Tokenizer(inputCol="review_text", outputCol="tokens")
    remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens")
    num_features = 1 << 12 if args.fast_mode else 1 << 18
    hashing = HashingTF(inputCol="filtered_tokens", outputCol="tf", numFeatures=num_features)
    idf = IDF(inputCol="tf", outputCol="tfidf")
    txt_stats = TextStatsTransformer(inputCol="review_text", outputPrefix="txt_")
    cat_idx = StringIndexer(inputCol="product_category", outputCol="product_category_idx", handleInvalid="keep")
    vec = VectorAssembler(
        inputCols=[
            "tfidf",
            "txt_token_count",
            "txt_char_count",
            "txt_exclaim_count",
            "txt_upper_ratio",
            "category_name_len",
            "product_category_idx",
        ],
        outputCol="features",
    )

    base_stages: List = [tokenizer, remover, hashing, idf, txt_stats, cat_idx, vec]

    models = {}
    metrics = []
    predictions = {}

    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10 if args.fast_mode else 50)
    lr_pipe = Pipeline(stages=base_stages + [lr])
    if args.fast_mode:
        lr_grid = ParamGridBuilder().addGrid(lr.regParam, [0.05]).addGrid(lr.elasticNetParam, [0.0]).build()
    else:
        lr_grid = (
            ParamGridBuilder()
            .addGrid(lr.regParam, [0.01, 0.1])
            .addGrid(lr.elasticNetParam, [0.0, 0.5])
            .build()
        )

    rf = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=20 if args.fast_mode else 100)
    rf_pipe = Pipeline(stages=base_stages + [rf])
    if args.fast_mode:
        rf_grid = ParamGridBuilder().addGrid(rf.maxDepth, [5]).addGrid(rf.maxBins, [32]).build()
    else:
        rf_grid = ParamGridBuilder().addGrid(rf.maxDepth, [8, 12]).addGrid(rf.maxBins, [64]).build()

    svc = LinearSVC(featuresCol="features", labelCol="label", maxIter=10 if args.fast_mode else 30)
    svc_pipe = Pipeline(stages=base_stages + [svc])
    if args.fast_mode:
        svc_grid = ParamGridBuilder().addGrid(svc.regParam, [0.1]).build()
    else:
        svc_grid = ParamGridBuilder().addGrid(svc.regParam, [0.01, 0.1]).build()

    training_jobs = [("logreg", lr_pipe, lr_grid), ("random_forest", rf_pipe, rf_grid), ("linear_svc", svc_pipe, svc_grid)]

    for name, pipe, grid in training_jobs:
        start = time.time()
        spark.sparkContext.setJobGroup(f"train_{name}", f"Cross-validation training for {name}")
        fitted = fit_cv_pipeline(train, pipe, grid, folds=args.cv_folds, parallelism=args.cv_parallelism)
        models[name] = fitted

        pred = fitted.transform(test)
        predictions[name] = pred
        metric = summarize_predictions(name, pred)
        metric["train_seconds"] = round(time.time() - start, 2)
        metrics.append(metric)

        model_path = f"{args.model_dir}/{name}"
        fitted.bestModel.write().overwrite().save(model_path)
        with open(f"{args.model_dir}/{name}_meta.pkl", "wb") as f:
            pickle.dump(metric, f)

    # Single-node sklearn baseline for distributed vs local comparison.
    sk_train_pdf = train.select("review_text", "label").limit(20000 if args.fast_mode else 200000).toPandas()
    sk_test_pdf = test.select("review_text", "label").limit(5000 if args.fast_mode else 50000).toPandas()
    sk_metrics = sklearn_baseline(sk_train_pdf, sk_test_pdf)
    metrics.append(sk_metrics)

    feat.unpersist()

    payload = {
        "timestamp_utc": utc_now_iso(),
        "sample_fraction": args.sample_fraction,
        "cv_folds": args.cv_folds,
        "cv_parallelism": args.cv_parallelism,
        "models": metrics,
    }
    write_json(args.metrics_json, payload)

    best_name = sorted(metrics[:3], key=lambda x: x["auc"], reverse=True)[0]["model"]
    predictions[best_name].select("label", "prediction", "probability", "rawPrediction").write.mode("overwrite").parquet(
        args.predictions_path
    )

    # Silver table for analytics and downstream Tableau datasets.
    silver = (
        base.withColumn("review_dt", F.to_date("review_date"))
        .withColumn("is_positive", F.when(F.col("stars") >= 4, 1).otherwise(0))
        .withColumn("review_month_start", F.date_trunc("month", F.col("review_dt")))
    )
    silver.write.mode("overwrite").parquet(args.silver_path)

    spark.stop()


if __name__ == "__main__":
    main()
