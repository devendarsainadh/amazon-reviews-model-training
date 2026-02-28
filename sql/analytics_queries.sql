-- Category-level sentiment trend
SELECT
  product_category,
  DATE_TRUNC('month', review_dt) AS month_start,
  COUNT(*) AS reviews,
  AVG(stars) AS avg_stars,
  AVG(CASE WHEN stars >= 4 THEN 1 ELSE 0 END) AS positive_rate
FROM silver_reviews
GROUP BY product_category, DATE_TRUNC('month', review_dt)
ORDER BY month_start, product_category;

-- Data quality checks
SELECT
  COUNT(*) AS rows_total,
  SUM(CASE WHEN review_body IS NULL OR review_body = '' THEN 1 ELSE 0 END) AS null_body,
  SUM(CASE WHEN stars IS NULL OR stars NOT BETWEEN 1 AND 5 THEN 1 ELSE 0 END) AS invalid_stars
FROM silver_reviews;
