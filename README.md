# Video/Social Realtime Pipeline — Snippet Pack (Local FS, No Kafka/MinIO)

This is a **snippet-only** pack for coursework. Files contain *concise* code blocks
to illustrate key ideas (reservoir sampling, sliding window + watermark, Bloom filter,
Flajolet–Martin/HLL via approx_count_distinct, CDF/PDF labeling, Pareto-front selection,
and a Streamlit view).

> These are *not* full apps. They’re short, copy-ready excerpts to paste into your own codebases.
> Paths assume local Delta tables under `./data/*`. Adjust as needed.

## Layout
- `jobs/00_fetch_to_landing.py` – mock API fetch to `data/landing` (NDJSON), optional reservoir sampling dump
- `jobs/10_bronze_batch.py` – batch ingest + Bloom filter pre-check
- `jobs/20_silver_stream.py` – file streaming source + sliding window + watermark
- `jobs/30_gold_features.py` – approx unique users (HLL) + CDF/PDF thresholds
- `jobs/40_train_pareto.py` – multi-model metrics + Pareto front selection
- `jobs/50_predict_stream.py` – tiny prediction appender (optional)
- `app/app.py` – Streamlit snippet (CDF/PDF chart + Bloom existence check)

Quick start:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python jobs/00_fetch_to_landing.py
# Submit other snippets with spark-submit (Delta package required)
```

