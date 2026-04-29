# OHLCV Operating Model

## Recommended job classes

- fast: `15m,30m`
- medium: `1h,2h,4h`
- daily: `1d`
- weekly: `1wk`
- monthly: `1mo`

## Recommended universes

- `bist30`
- `bist50`
- `bist100`
- `bist500`
- `nasdaq100`
- `sp500`

## Operating principles

- run fast jobs separately and distribute them in time
- use incremental mode for standard operation
- allow grouped daily / weekly / monthly jobs
- rely on universe-level locks for concurrency safety
- use Telegram alerts as the primary monitoring channel

## Notes

A full `all` + all-interval run is useful as a benchmark, but normal operation should use job-specific intervals only.
