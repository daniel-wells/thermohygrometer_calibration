# Thermo-Hygrometer Calibration

## Nix + uv environment

```bash
nix develop
uv sync
```

The environment is stored outside Dropbox:

```bash
UV_PROJECT_ENVIRONMENT=$HOME/.cache/uv/venvs/thermohygrometer_calibration
```

## 1) Simulate SwitchBot exports

```bash
uv run python -m thermohygrometer_calibration.simulate \
  --layout data/layout.csv \
  --output-dir data/simulated \
  --truth-output-dir data/simulated_truth \
  --start "2026-07-01 00:00:00" \
  --periods 672 \
  --freq 15min \
  --seed 42
```

Output:
- `data/simulated/<device_id>_data.csv`
- `data/simulated_truth/device_effects.csv`
- `data/simulated_truth/type_effects.csv`
- `data/simulated_truth/line_effects.csv`

## 2) Analyse

```bash
uv run python -m thermohygrometer_calibration.analyze \
  --input-dir data/simulated \
  --output-dir data/processed
```

Outputs:
- `data/processed/measurements_long.csv`
- `data/processed/device_summary.csv`

## 3) Visualise

```bash
uv run python -m thermohygrometer_calibration.visualize \
  --input data/processed/measurements_long.csv \
  --output-dir results/figures
```

Outputs:
- `results/figures/temperature_by_device.png` (all devices on one plot)
- `results/figures/humidity_by_device.png` (all devices on one plot)

## 4) Render report

```bash
quarto render index.qmd
```

Output:
- `report.html`

## Notes for next step (Bayesian hierarchical model)

The simulator already includes:
- shared structured time-series latent environment
- type-level (`ip65` vs `standard`) effects
- line-level effects
- device-level random effects
- correlated residual structure across devices

So you can now prototype posterior recovery by fitting your Bayesian hierarchical model to `data/processed/measurements_long.csv`.
