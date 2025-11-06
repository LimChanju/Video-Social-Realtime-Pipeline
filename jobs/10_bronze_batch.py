"""Snippet: batch ingest landing -> bronze (Delta), with Bloom pre-filter.
Run with:
  spark-submit --packages io.delta:delta-spark_2.12:3.2.0 jobs/10_bronze_batch.py
"""
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, current_date, lit

LANDING = os.getenv("LANDING_DIR", "data/landing")
BRONZE  = os.getenv("BRONZE_DIR", "data/bronze")

spark = (SparkSession.builder.appName("bronze_batch_snippet")
    .config("spark.sql.extensions","io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog","org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.sql.warehouse.dir","warehouse").getOrCreate())

# Read all json lines in landing
df = spark.read.json(LANDING)
incoming = df.selectExpr("post_id","video_id","author_id","text","ts")              .withColumn("ingest_date", current_date())              .withColumn("source", lit("mock"))

# Create Bloom from recent bronze (if exists)
from pathlib import Path
bf = None
if Path(BRONZE).exists():
    try:
        recent_ids = spark.read.format("delta").load(BRONZE).select("post_id").na.drop()
        bf = recent_ids.agg(expr("bloom_filter(post_id, 100000, 0.01) as bf")).collect()[0]["bf"]
    except Exception as e:
        bf = None

filtered = incoming if bf is None else incoming.filter(~expr(f"might_contain('{bf}', post_id)"))

(filtered.write.format("delta").mode("append")
    .partitionBy("ingest_date","source").save(BRONZE))

print("Bronze appended. Rows:", filtered.count())
