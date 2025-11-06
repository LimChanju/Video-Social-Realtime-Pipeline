"""Snippet: micro-batch 'prediction' appender (toy)."""
import os, time
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp

GOLD = os.getenv("GOLD_DIR","data/gold")
spark = (SparkSession.builder.appName("predict_append_snippet")
    .config("spark.sql.extensions","io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog","org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.sql.warehouse.dir","warehouse").getOrCreate())

features = spark.read.format("delta").load(f"{GOLD}/features")
pred = features.selectExpr("video_id","engagement_24h as score")                .withColumn("event_time", current_timestamp())

(pred.write.format("delta").mode("append").save(f"{GOLD}/predictions"))
print("Appended predictions (toy).")
