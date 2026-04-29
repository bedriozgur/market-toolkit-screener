# Project Baseline

## Boundary

The project is split into two operational areas:

- **Collector**: downloads, cleans, repairs, validates, and stores OHLCV data
- **Screener**: reads validated local OHLCV data and produces rankings, reports, and web-friendly outputs

The screener must never depend on live data fetching. It should consume the latest local snapshot only.

## Supported data sources

The collector is limited to these source families:

- `yfinance` for the primary OHLCV download path
- `twelvedata` for the alternate Twelve Data path
- `borsapy` for BIST TradingView-based cross-check and repair

No other external data sources are part of the supported baseline.

## Runtime layout

Tracked project assets live under `app/`.

Runtime artifacts live under `workspace/`, including:

- downloaded OHLCV files
- locks
- logs
- validation outputs
- pipeline summaries
- screener exports

## Default paths

- collector data root: `workspace/data`
- pipeline summaries: `workspace/outputs/reports`
- screener outputs: `workspace/outputs/screeners`
- locks: `workspace/locks`

## Operating rule

The collector can be scheduled independently on Windows now and Linux later.
Telegram alerts are for failure conditions only, not green or minor-warning runs.
