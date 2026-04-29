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

## Required arguments

- `--universe`
- `--intervals`
- `--config`

## Outputs

Written under `workspace/outputs/screeners/`:
- screener CSV files
- optional summary JSON

## Notes

- Reads existing OHLCV CSV or Parquet files
- Uses the screener package under `app/scripts/screener/`
- Does not download or refresh data
