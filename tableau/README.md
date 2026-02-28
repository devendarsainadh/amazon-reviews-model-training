# Tableau Dashboard Plan

## Dashboard 1: Data quality and pipeline monitoring
- File: `outputs/tableau/dashboard1_data_quality.csv`
- File: `outputs/tableau/dashboard1_daily_volume.csv`
- Views: null-rate KPI cards, daily ingestion trend, reject counts from ingestion logs.

## Dashboard 2: Model performance and feature importance
- File: `outputs/tableau/dashboard2_model_performance.csv`
- Views: AUC/F1/accuracy by model, training time comparison, CV settings as context.

## Dashboard 3: Business insights and recommendations
- File: `outputs/tableau/dashboard3_business_insights.csv`
- Views: category-level average rating trend, positive-rate heatmap, seasonal peaks.

## Dashboard 4: Scalability and cost analysis
- File: `outputs/tableau/dashboard4_scalability.csv`
- Views: strong vs weak scaling runtime, cost proxy, bottleneck annotation.

## Best practices checklist
- Use Tableau extracts for large CSVs.
- Add parameter controls for model/date/category filtering.
- Use LOD for category-level benchmarks.
- Configure phone layout for mobile responsiveness.
- Add dashboard actions and annotations for key findings.
