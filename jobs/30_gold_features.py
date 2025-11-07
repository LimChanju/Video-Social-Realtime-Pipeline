"""Snippet: HLL approx uniques + CDF/PDF thresholds -> gold.
Run with:
  spark-submit --packages io.delta:delta-spark_2.12:3.2.0 jobs/30_gold_features.py
"""
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import approx_count_distinct as acd, col
from pyspark.sql.functions import expr
from delta import configure_spark_with_delta_pip

SILVER  = os.getenv("SILVER_DIR", "data/silver")
GOLD    = os.getenv("GOLD_DIR", "data/gold")
TOP_PCT = float(os.getenv("TOP_PCT", "0.9"))

builder = (
    SparkSession.builder.appName("gold_features_snippet")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
)

spark = configure_spark_with_delta_pip(builder).getOrCreate()

# ========================================================================================================================
# 1. Silver Layer 로드 (social metrics)
# ========================================================================================================================
metrics = spark.read.format("delta").load(f"{SILVER}/social_metrics")

# ========================================================================================================================
# 2. Engagement 집계 (피처 엔지니어링)
# ========================================================================================================================
# mock: use 'count' as engagement metric (sum over last window already)
eng = metrics.groupby("video_id").agg(expr("sum(count) as engagement_24h"))

# ========================================================================================================================
# D. Flajolet-Martin / HLL++ Approx Uniques Counts
# ========================================================================================================================
# Approx uniques example (here we only have video_id; in real data use author_id)
uniq_est = metrics.groupBy("video_id").agg(acd("author_id").alias("uniq_users_est"))

# Join engagement + unique features
joined = eng.join(uniq_est, "video_id", "left")

# ========================================================================================================================
# E. 경험적 CDF / PDF 기반 임계치 계산
# ========================================================================================================================
cut = joined.approxQuantile("engagement_24h", [TOP_PCT], 0.001)[0]

# 히스토그램 기반 PDF 근사값 추가
hist_bins = 20
hist_df = joined.select("engagement_24h").rdd.flatMap(lambda x: x).histogram(hist_bins) # hist_df: (bin_edges, counts)

# ========================================================================================================================
# 3. Labeling (상위 TCP_PCT 이상이면 1)
# ========================================================================================================================
labeled = joined.withColumn("label", (col("engagement_24h") >= cut).cast("int"))

# ========================================================================================================================
# 4. Gold Layer Delta Format으로 저장
# ========================================================================================================================
(labeled.write.format("delta").mode("overwrite").save(f"{GOLD}/features"))

print(f"✅ Gold features written. P{TOP_PCT*100:.0f} cut = {cut}")
