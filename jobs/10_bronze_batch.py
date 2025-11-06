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
# 2. 앞으로 들어올 데이터들은 스키마 고정 및 메타컬럼 추가 (ingest_date, source)
# bronze 레이어로 넣기 좋게 스키마 고정 + 메타컬럼 추가(ingest_date, source)
# ================================================================================
incoming = df.selectExpr("post_id","video_id","author_id","text","ts")\
           .withColumn("ingest_date", current_date())\
           .withColumn("source", lit("mock"))

# ================================================================================
# B. Bloom Filter로 중복 필터링 (Create Bloom from recent bronze (if exists))
# 목적: 최근 7일 간의 post_id / video_id 중복 방지
# ================================================================================
from pathlib import Path
from pyspark.sql.functions import date_sub

bf_post, bf_video = None, None

if Path(BRONZE).exists():
    try:# 최근 7일치 데이터만 필터링
        bronze_df = spark.read.format("delta").load(BRONZE)
        bronze_recent = bronze_df.filter(col("ingest_date") >= date_sub(current_date(), 7))
        
        # post_id 기준 Bloom 필터 생성
        bf_post = bronze_recent.select("post_id").na.drop().agg(
            expr("bloom_filter(post_id, 100000, 0.01) as bf_post")
        ).collect()[0]["bf_post"]
        
        # video_id 기준 Bloom 필터 생성
        bf_post = bronze_recent.select("video_id").na.drop().agg(
            expr("bloom_filter(video_id, 100000, 0.01) as bf_post")
        ).collect()[0]["bf_video"]
        
        print("최근 7일 Bronze 데이터로 Bloom 필터 생성 완료.")
        
    except Exception as e:
        print("Bloom 필터 생성 중 오류 발생:", e)
        bf_post, bf_video = None, None

# incoming 데이터를 Bloom 필터로 빠른 중복 제거
if bf_post or bf_video:
    conds = []
    if bf_post:
        conds.append(f"might_contain('{bf_post}', post_id)")
    if bf_video:
        conds.append(f"might_contain('{bf_video}', video_id)")
        
    condition = " OR ".join(conds)
    filtered = incoming.filter(~expr(condition))
    print(f"Bloom 필터 적용 완료. 중복 제거 후 rows: {filtered.count}")
else:
    filtered = incoming
    print("Bloom 필터가 없어 중복 제거를 건너뜀.")

# 3. Bronze Delta에 Append (증분 수집)
# 스키마를 고정한 채 Delta Lake 포맷으로 증분 저장
(filtered.write.format("delta").mode("append")
    .partitionBy("ingest_date","source").save(BRONZE))

print("Bronze appended. Rows:", filtered.count())