# App

This directory contains the screener application layer.

## What belongs here

- screener scripts and ranking logic
- indicator and feature code required by the screener
- screener configs
- screener tests and docs
- universe definitions used for local screening

## Important rule

Runtime-generated files should not be written inside `app/`.

Local OHLCV inputs are expected under `../workspace/data/`.

## Main entry points

- `app/scripts/run_screener.py`
- `app/scripts/screener/engine.py`
- `app/scripts/screener/feature_builder.py`

## Notes

- Universe definition files live under `app/universes/`
- The repository root `README.md` explains the screener boundary
- `workspace/README.md` explains the runtime artifact area
