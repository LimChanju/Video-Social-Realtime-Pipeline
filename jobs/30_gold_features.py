"""Snippet: HLL approx uniques + CDF/PDF thresholds -> gold.
Run with:
  spark-submit --packages io.delta:delta-spark_2.12:3.2.0 jobs/30_gold_features.py
"""
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import approx_count_distinct as acd, col

SILVER  = os.getenv("SILVER_DIR", "data/silver")
GOLD    = os.getenv("GOLD_DIR", "data/gold")
TOP_PCT = float(os.getenv("TOP_PCT", "0.9"))

spark = (SparkSession.builder.appName("gold_features_snippet")
    .config("spark.sql.extensions","io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog","org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.sql.warehouse.dir","warehouse").getOrCreate())

metrics = spark.read.format("delta").load(f"{SILVER}/social_metrics")
# mock: use 'count' as engagement metric (sum over last window already)
eng = metrics.select("video_id","count").groupBy("video_id").sum("count").withColumnRenamed("sum(count)","engagement_24h")

# Approx uniques example (here we only have video_id; in real data use author_id)
uniq_est = metrics.groupBy("video_id").agg(acd("video_id").alias("uniq_users_est"))

joined = eng.join(uniq_est, "video_id", "left")
cut = joined.approxQuantile("engagement_24h", [TOP_PCT], 0.001)[0]
labeled = joined.withColumn("label", (col("engagement_24h") >= cut).cast("int"))

(labeled.write.format("delta").mode("overwrite").save(f"{GOLD}/features"))
print("Gold features written. P{:.0f} cut = {}".format(TOP_PCT*100, cut))
