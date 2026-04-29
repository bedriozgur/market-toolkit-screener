# Project Baseline

## Boundary

This repository owns the screener application.

It reads published local OHLCV snapshots and produces ranking output, tables, and future web-facing views.

It does not download or repair OHLCV data.

## Runtime layout

- local data root: `workspace/data`
- screener outputs: `workspace/outputs/screeners`

## Operating rule

This repo should only depend on local snapshot files and the screener code path. Collector changes belong in the collector repository.
