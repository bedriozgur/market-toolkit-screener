# Workspace

This directory is reserved for local runtime artifacts.

Typical contents include:

- downloaded OHLCV data
- logs
- cache files
- lock files
- validation outputs
- pipeline summaries
- screener exports
- reports

Typical subdirectories include:

- `data/`
- `logs/`
- `locks/`
- `cache/`
- `outputs/`

This structure is kept in Git only as a lightweight skeleton.

Generated contents under `workspace/` are ignored by Git and should not be committed.
