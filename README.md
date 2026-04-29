# Market Toolkit Screener

This repository is the read-only screening project.

It consumes local OHLCV snapshots published by the collector repository and produces rankings, tables, and eventual web-facing outputs.

It does not download or repair data.

## Layout

- `app/` contains screener code, configs, docs, tests, and universe lists
- `workspace/` contains screener outputs, caches, and local runtime artifacts

## Primary command

- `python3 app/scripts/run_screener.py`

## Operating rule

This repo should only read validated local data. Any collector changes belong in the collector repository.
