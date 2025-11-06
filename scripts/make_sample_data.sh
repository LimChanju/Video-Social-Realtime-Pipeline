#!/usr/bin/env bash
set -e
mkdir -p data/landing
for i in {1..5}; do
  ts=$(date +%s%3N)
  echo "{\"post_id\":\"p$i\",\"text\":\"sample like $i\",\"lang\":\"en\",\"ts\":$ts,\"author_id\":\"u$((RANDOM%5))\",\"video_id\":\"v$((RANDOM%3))\"}" >> data/landing/events_$i.json
  sleep 1
done
echo "Wrote sample JSON lines to data/landing/*.json"
