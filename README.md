# Amazon Reviews PySpark Data Engineering + Distributed ML

This project uses Hugging Face `amazon_reviews_multi` (English) directly in code (streaming supported) and builds a full PySpark pipeline for ingestion, ML training, scalability analysis, and Tableau-ready exports.

## 1. Data Engineering

### a) Ingestion and storage design
- SparkSession tuning in `scripts/common.py`:
  - adaptive execution on
  - shuffle partitions configurable (`SPARK_SHUFFLE_PARTITIONS`)
  - Kryo serializer
  - Snappy Parquet compression
- Ingestion in `scripts/ingest_amazon_reviews.py`:
  - Hugging Face streaming read (`load_dataset(..., streaming=True)`)
  - Validation for required fields, star range, and date parsing
  - Reject logging to `outputs/logs/ingestion_rejects.jsonl`
  - Partitioned Parquet write by `language`, `review_year`, `review_month`
- Storage format: Parquet chosen for column pruning, predicate pushdown, and efficient Spark scans.

### b) Distributed processing pipeline
- Broadcast join implemented in `scripts/train_distributed_models.py` (`broadcast(category_dim)`).
- Persist/unpersist strategy:
  - feature table cached before repeated CV model training
  - unpersisted after training
- Error handling and lineage:
  - ingestion reject logs
  - batch logs (`outputs/logs/ingestion_batches.jsonl`)
  - metrics metadata with git hash and runtime config

### c) Performance optimization
- DataFrame API used as default for Catalyst/Tungsten optimization; RDD avoided unless custom low-level processing is needed.
- Shuffle/partition tuning:
  - configurable partition count in Spark conf
  - explicit repartition before writes/training experiments
- Spark UI evidence:
  - run jobs and capture screenshots from stages/jobs tabs for report inclusion.

## 2. Scalability and Distributed ML

### a) MLlib implementation
- 3 MLlib algorithms:
  - Logistic Regression
  - Random Forest Classifier
  - Linear SVC
- sklearn single-node baseline in same script.
- Custom transformer: `TextStatsTransformer` in `scripts/custom_transformers.py`.
- Model serialization:
  - MLlib `save()` for best models
  - Pickle metadata for quick model summaries

### b) Distributed training and tuning
- `CrossValidator` with configurable `parallelism`.
- Hyperparameter grids scoped to computational limits.
- Checkpoint-style persistence through saved best models and metrics artifacts.
- Resource allocation is externally configurable via Spark submit args and shuffle partition env var.

### c) Scalability analysis
- `scripts/scalability_analysis.py`:
  - Strong scaling: fixed fraction, increasing partitions
  - Weak scaling: data fraction increases with partitions
  - Cost proxy generated (`partitions * train_seconds`)
- Bottleneck detection supported through runtime metrics + Spark UI.

## 3. Tableau Visualization
- Export script: `scripts/export_tableau_data.py`
- Dashboard datasets:
  - `dashboard1_data_quality.csv`
  - `dashboard1_daily_volume.csv`
  - `dashboard2_model_performance.csv`
  - `dashboard3_business_insights.csv`
  - `dashboard4_scalability.csv`
- Detailed dashboard strategy: `tableau/README.md`

## 4. Model Evaluation
- Temporal split strategy in training script (quantile-based time split).
- CV-based distributed evaluation with AUC/F1/Accuracy.
- Bootstrap confidence intervals and business profit metric in `scripts/evaluate_models.py`.

## Quick Start

1. Install dependencies:
```bash
python3 -m pip install -r requirements.txt
```

2. Run full pipeline (example at 300k rows):
```bash
python3 scripts/run_all.py --max-rows 300000
```

3. Run each stage manually:
```bash
python3 scripts/ingest_amazon_reviews.py --streaming --max-rows 300000
python3 scripts/train_distributed_models.py
python3 scripts/evaluate_models.py
python3 scripts/scalability_analysis.py
python3 scripts/export_tableau_data.py
```

## Output structure
- Bronze data: `outputs/bronze_reviews`
- Silver data: `outputs/silver_reviews`
- Models: `outputs/models`
- Metrics: `outputs/metrics`
- Logs: `outputs/logs`
- Tableau-ready files: `outputs/tableau`

## Notes
- For >=1GB ingestion, set `--max-rows 0` in ingestion and run full split(s).
- You can extend to multiple languages by looping `--lang` values and unioning downstream.
