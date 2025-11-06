"""Snippet: Streamlit view with CDF/PDF and Bloom presence check (toy).
Run with: streamlit run app/app.py
"""
import os
import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.functions import expr, col

GOLD = os.getenv("GOLD_DIR","data/gold")
SILVER = os.getenv("SILVER_DIR","data/silver")

st.set_page_config(page_title="Realtime Video Insights (Snippets)", layout="wide")
query = st.text_input("Search by keyword or video_id", "")

spark = (SparkSession.builder
    .config("spark.sql.extensions","io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog","org.apache.spark.sql.delta.catalog.DeltaCatalog")
    .config("spark.sql.warehouse.dir","warehouse").getOrCreate())

try:
    features = spark.read.format("delta").load(f"{GOLD}/features")
    metrics = spark.read.format("delta").load(f"{SILVER}/social_metrics")
except Exception as e:
    st.warning("No data yet. Run the jobs first."); st.stop()

# CDF cut @ 90%
cut = features.approxQuantile("engagement_24h", [0.90], 0.001)[0]
st.metric("Top 10% cut (CDF 0.90)", f"{cut:,.2f}")

# PDF histogram (bucketed)
hist = (features.selectExpr("width_bucket(engagement_24h, 0, 10000, 30) as bkt")
        .groupBy("bkt").count()).toPandas()
if not hist.empty:
    st.bar_chart(hist.set_index("bkt"))

# Bloom filter presence check for recent video_ids (toy)
recent = features.select("video_id")
bf = recent.agg(expr("bloom_filter(video_id, 100000, 0.01) as bf")).collect()[0]["bf"]
if query:
    exists = spark.createDataFrame([(query,)], "video_id string")                   .select(expr(f"might_contain('{bf}', video_id)").alias("maybe")).collect()[0][0]
    st.info(f"Bloom says existence possible: {exists}")

df = features if not query else features.filter((col("video_id")==query) | (col("video_id").contains(query)))
st.dataframe(df.limit(200).toPandas(), use_container_width=True)
