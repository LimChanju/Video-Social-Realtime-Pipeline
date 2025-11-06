"""Snippet: file streaming source -> sliding window + watermark -> silver.
Run with:
  spark-submit --packages io.delta:delta-spark_2.12:3.2.0 jobs/20_silver_stream.py
Then drop files into data/landing (see scripts/make_sample_data.sh)
"""
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, window

LANDING = os.getenv("LANDING_DIR", "data/landing")
SILVER  = os.getenv("SILVER_DIR", "data/silver")
WATERMARK = os.getenv("WATERMARK", "10 minutes")
WIN     = os.getenv("WINDOW_SIZE", "1 hour")
SLIDE   = os.getenv("WINDOW_SLIDE","5 minutes")

spark = (SparkSession.builder.appName("silver_stream_snippet")
    .config("spark.sql.extensions","io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog","org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.sql.warehouse.dir","warehouse").getOrCreate())

schema = "post_id string, text string, lang string, ts long, author_id string, video_id string"

raw = (spark.readStream
    .format("json")
    .schema(schema)
    .load(LANDING))

clean = (raw
  .filter((col("post_id").isNotNull()) & (col("lang")=="en"))
  .withColumn("event_time", to_timestamp((col("ts")/1000).cast("timestamp")))
  .withWatermark("event_time", WATERMARK)
  .dropDuplicates(["post_id"]))

win = (clean
  .groupBy(window(col("event_time"), WIN, SLIDE), col("video_id"))
  .count())

q = (win.writeStream
  .format("delta")
  .option("checkpointLocation", "chk/silver")
  .outputMode("append")
  .start(f"{SILVER}/social_metrics"))

q.awaitTermination()
