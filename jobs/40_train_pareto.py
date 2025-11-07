"""Snippet: multi-model training metrics + Pareto front selection.
Run with:
  spark-submit --packages io.delta:delta-spark_2.12:3.2.0 jobs/40_train_pareto.py
"""
import os, time
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip
from pyspark.ml import Pipeline
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

GOLD = os.getenv("GOLD_DIR","data/gold")

builder = (
    SparkSession.builder.appName("gold_features_snippet")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
)

spark = configure_spark_with_delta_pip(builder).getOrCreate()

# =============================================
# 1. Load Data
# =============================================
df = spark.read.format("delta").load(f"{GOLD}/features").fillna(0)

# =============================================
# 2. Feature 구성 (하나의 벡터(Features)로 합침)
# =============================================
# Minimal features (no text here). In your real pipeline, include text features.
feat_cols = ["engagement_24h","uniq_users_est"]
si = StringIndexer(inputCol="label", outputCol="label_idx")
va = VectorAssembler(inputCols=feat_cols, outputCol="features")

# =============================================
# 3. 후보 모델 목록 (로지스틱 회귀, 랜덤 포레스트, GBT)
# =============================================
candidates = [
  ("lr", LogisticRegression(labelCol="label_idx", maxIter=50)),
  ("rf", RandomForestClassifier(labelCol="label_idx", numTrees=200, maxDepth=10)),
  ("gbt", GBTClassifier(labelCol="label_idx", maxIter=80, maxDepth=6)),
]

# =============================================
# 4. 평가 지표 설정
# =============================================
evaluator = BinaryClassificationEvaluator(labelCol="label_idx", metricName="areaUnderPR")

# =============================================
# 5. 각 모델 학습 및 평가 (AUC-PR, 추론 지연 시간, 특징 개수)
# =============================================
results = []
for name, clf in candidates:
    pipe = Pipeline(stages=[si, va, clf]).fit(df)
    t0 = time.time(); pred = pipe.transform(df)
    latency_ms = (time.time()-t0)*1000
    aucpr = evaluator.evaluate(pred)
    feat_count = len(feat_cols)
    results.append({"name":name,"aucpr":aucpr,"latency_ms":latency_ms,"feat_count":feat_count})

# =============================================
# 6. Pareto 지배(dominance) 판정 함수 정의
# =============================================
def dominates(a,b):
    return (a["aucpr"]>=b["aucpr"] and a["latency_ms"]<=b["latency_ms"] and a["feat_count"]<=b["feat_count"]) and \
           (a["aucpr"]>b["aucpr"] or a["latency_ms"]<b["latency_ms"] or a["feat_count"]<b["feat_count"])

# =============================================
# 7. Pareto front 계산
# =============================================
pareto = []
for r in results:
    if not any(dominates(o,r) for o in results if o is not r):
        pareto.append(r)

print("\n=== All Candidates ===")
for r in results:
    print(f"{r['name']}: AUC={r['aucpr']:.4f}, latency={r['latency_ms']:.1f} ms, features={r['feat_count']}")

print("\n=== Pareto Front (Best Trade-offs) ===")
for r in pareto:
    print(f"✅ {r['name']}: AUC={r['aucpr']:.4f}, latency={r['latency_ms']:.1f} ms, features={r['feat_count']}")
