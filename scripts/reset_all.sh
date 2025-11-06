#!/usr/bin/env bash
rm -rf data/* mlruns/* chk/* metastore_db/* warehouse/*
mkdir -p data/{landing,bronze,silver,gold} chk/{silver,gold}
echo "Local state reset."
