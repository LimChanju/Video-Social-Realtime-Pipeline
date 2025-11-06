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
# ================================================================================
# 1. Landing 데이터 읽기 (Read all json lines in landing)
# ================================================================================
df = spark.read.json(LANDING) # NDJSON을 읽어서
print(f"Landing Data 로드 완료 (rows: {df.count()})")

# ================================================================================
# A. Reservoir Sampling으로 스트리밍 전 단계의 대표 샘플 확보
# ================================================================================
# 폭증 시 incoming 전체를 다 저장하지 않고, 각 쿼리/키워드 별로 대표 샘플(k개)만 남기거나 
def reservoir_per_keyword(iterator, k=64):
    """Vitter's Algorithm R - uniform random sample of size k"""
    reservoir, n = [], 0 # 샘플 저장소와 전체 개수
    for row in iterator:
        n += 1
        if len(reservoir) < k:
            reservoir.append(row)
        else:
            j = spark._jvm.java.util.concurrent.ThreadLocalRandom.current().nextInt(n)
            if j < k:
                reservoir[j] = row
    return iter(reservoir)

# Spark RDD로 변환 후 Reservoir 샘플링 적용
rdd = df.rdd.map(lambda row: row.asDict())
sampled_rdd = rdd.mapPartitions(lambda it: reservoir_per_keyword(it, k=64))
sample_df = spark.createDataaFrame(sampled_rdd, schema=df.schema)
print(f"Reservoir Sampling 완료. 샘플 크기: {sample_df.count()}")

# ================================================================================
# 1-2. 스키마 고정 및 메타컬럼 추가 (ingest_date, source)
# ================================================================================
incoming = df.selectExpr("post_id","video_id","author_id","text","ts")\
           .withColumn("ingest_date", current_date())\
           .withColumn("source", lit("mock"))
# bronze 레이어로 넣기 좋게 스키마 고정 + 메타컬럼 추가(ingest_date, source)

# ================================================================================
# 2. Bloom Filter로 중복 필터링 (Create Bloom from recent bronze (if exists))
# ================================================================================
from pathlib import Path
bf = None
if Path(BRONZE).exists():
    try: # post_id를 기준으로 Bloom 필터 생성한 후에 새로 들어온 것 중 중복 post_id인 행은 제외
        recent_ids = spark.read.format("delta").load(BRONZE).select("post_id").na.drop()
        bf = recent_ids.agg(expr("bloom_filter(post_id, 100000, 0.01) as bf")).collect()[0]["bf"]
    except Exception as e:
        bf = None

filtered = incoming if bf is None else incoming.filter(~expr(f"might_contain('{bf}', post_id)"))
# 3. Bronze Delta에 Append (증분 수집)
# 스키마를 고정한 채 Delta Lake 포맷으로 증분 저장
(filtered.write.format("delta").mode("append")
    .partitionBy("ingest_date","source").save(BRONZE))

print("Bronze appended. Rows:", filtered.count())