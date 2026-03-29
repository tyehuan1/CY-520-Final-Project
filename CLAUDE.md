# Malware Family Classification Project

## Quick Reference
- **Python 3.11** on Windows 11
- Run tests: `python -m pytest tests/ -v` from project root
- All constants live in `config.py` — no magic numbers in source files
- Random seed: 42 (set in config.py)

## Conventions
- Type hints on all function signatures
- Docstrings on all public functions (Google style: Args/Returns/Raises)
- Use `src.utils.get_logger(__name__)` for logging
- Cache intermediate results to `cache/` to avoid recomputation
- Tests mirror source layout: `src/foo.py` → `tests/test_foo.py`

## Data
- Primary dataset (Mal-API-2019): `data/Mal API.txt` + `data/Mal API Labels.csv`
- Secondary dataset (MalbehavD-V1): `data/MalBehavD-V1-dataset.csv`
- Both loaded via `src.data_loader` into `List[Dict]` with keys `sequence`, `label`
