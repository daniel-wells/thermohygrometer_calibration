# Thermo-Hygrometer Calibration

This project is set up for a simulation-first workflow that mirrors your real ingestion path:

1. Simulate SwitchBot-style exports only.
2. Analyze by reformatting exports into processed long-format data.
3. Visualize temperature and humidity over time with plotnine.
4. Render a basic Quarto report.

The default setup matches your experiment:
- 7 devices total
- 3 `ip65`, 4 `standard`
- two lines with a 4/3 arrangement

## Nix + uv environment

```bash
nix develop
uv sync
```

The environment is stored outside Dropbox:

```bash
UV_PROJECT_ENVIRONMENT=$HOME/.cache/uv/venvs/thermohygrometer_calibration
```

## 1) Simulate SwitchBot exports only

```bash
uv run python -m thermohygrometer_calibration.simulate \
  --layout data/layout.csv \
  --output-dir data/simulated \
  --start "2026-07-01 00:00:00" \
  --periods 672 \
  --freq 15min \
  --seed 42
```

Output:
- `data/simulated/<device_id>_data.csv` files only (SwitchBot schema)

## 2) Analyze (reformat to processed long data)

```bash
uv run python -m thermohygrometer_calibration.analyze \
  --input-dir data/simulated \
  --output-dir data/processed
```

Outputs:
- `data/processed/measurements_long.csv`
- `data/processed/device_summary.csv`

## 3) Visualize with plotnine

```bash
uv run python -m thermohygrometer_calibration.visualize \
  --input data/processed/measurements_long.csv \
  --output-dir results/figures
```

Outputs:
- `results/figures/temperature_by_device.png` (all devices on one plot)
- `results/figures/humidity_by_device.png` (all devices on one plot)

## 4) Render Quarto report

```bash
quarto render report.qmd
```

Output:
- `report.html`

## Notes for next step (Bayesian hierarchical model)

The simulator already includes:
- shared structured time-series latent environment
- type-level (`ip65` vs `standard`) effects
- line-level effects
- device-level random effects and drift
- correlated residual structure across devices

So you can now prototype posterior recovery by fitting your Bayesian hierarchical model to `data/processed/measurements_long.csv`.
