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
  --layout data/layout_simulated.csv \
  --output-dir data/simulated \
  --truth-output-dir data/simulated_truth \
  --start "2026-07-05 00:00:00" \
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
  --dataset simulated
```

```bash
uv run python -m thermohygrometer_calibration.analyze \
  --dataset actual
```

Outputs:
- `data/processed/simulated/measurements_long.csv`
- `data/processed/simulated/device_summary.csv`
- `data/processed/actual/measurements_long.csv`
- `data/processed/actual/device_summary.csv`


## 3) Visualise

```bash
uv run python -m thermohygrometer_calibration.visualize \
  --dataset simulated
```

```bash
uv run python -m thermohygrometer_calibration.visualize \
  --dataset actual
```

Outputs:
- `results/simulated/figures/temperature_by_device.png`
- `results/simulated/figures/humidity_by_device.png`
- `results/actual/figures/temperature_by_device.png`
- `results/actual/figures/humidity_by_device.png`

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
