# run_screener.py

Run score-based screeners on stored OHLCV files under `data/{universe}/ohlcv`.

## Built-in screeners

- trend_leader
- pullback
- breakout_candidate
- fresh_flip
- alphatrend
- technical_rank

## Usage

```bash
python3 app/scripts/run_screener.py \
  --universe bist100 \
  --intervals 1d \
  --config config/screener_bist.json \
  --screeners trend_leader,pullback,breakout_candidate,fresh_flip,alphatrend,technical_rank \
  --summary-json
```

Add `--enrichment-refresh` when you want to bypass cached Yahoo/TradingView enrichment and fetch fresh market data.

Telegram notifications use the same environment variables as the collector:
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

Pass `--telegram` to force sending, or `--no-telegram` to suppress it. If neither flag is set, the screener sends when both Telegram env vars are present.

## Required arguments

- `--universe`
- `--intervals`
- `--config`

## Outputs

Written under `$MARKET_TOOLKIT_WORKSPACE/outputs/screeners/`:
- screener CSV files
- `*_universe_rank_*.csv` full universe score table
- `*_top10_*.csv` top 10 ranked tickers for the universe/interval
- optional summary JSON

The universe rank table combines technical indicators, fresh flips, fundamentals, and optional Reddit/Twitter sentiment fields when they are present in the loaded data or universe metadata.

Optional market-data enrichment can also add Yahoo Finance analyst recommendations, analyst price targets, Yahoo Finance news sentiment, upgrades/downgrades, and TradingView technical ratings when `market_data_enrichment.enabled` is turned on in the config. Results are cached under the workspace cache directory.

TradingView routing is now auto-selected from the universe name: BIST universes use `turkey`/`BIST`, and everything else falls back to `america`/`NASDAQ` unless you override it explicitly in config.

## Notes

- Reads existing OHLCV CSV or Parquet files
- Uses the screener package under `app/scripts/screener/`
- Does not download or refresh data
