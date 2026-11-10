# HydraFlowNet forecast client

Command-line tool that downloads historical forecasts from the HydraFlowNet API and saves them to a CSV file.

## Installation

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh                                  # Linux / macOS
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows
```

uv installs a suitable Python version and the dependencies automatically.

## Usage

Run from the project root. On the first run, uv creates `.venv` and installs the dependencies:

```bash
uv run hydraflownet-download --api-key <API_KEY> --model 5d
uv run hydraflownet-download --api-key <API_KEY> --model 42d --from 2026-02-01T00:00:00+00:00 --to 2026-06-01T00:00:00+00:00 --output 42d.csv
```

To install the command globally, so that it can be run from any directory without `uv run`:

```bash
uv tool install .
hydraflownet-download --api-key <API_KEY> --model 5d
```

### Arguments

| Argument       | Required | Default                     | Description                                        |
|----------------|----------|-----------------------------|----------------------------------------------------|
| `--api-key`    | yes      |                             | Your API key                                       |
| `--model`      | yes      |                             | `5d`, `42d`, `210d`, or a model UUID               |
| `--from`       | no       | `2026-02-01T00:00:00+00:00` | Start of the period (ISO datetime)                 |
| `--to`         | no       | now (UTC)                   | End of the period (ISO datetime)                   |
| `--output`     | no       | `forecasts_<model>.csv`     | Output CSV path                                    |
| `--url`        | no       | HydraFlowNet API endpoint   | API endpoint                                       |
| `--chunk-days` | no       | `30`                        | Days requested per API call                        |

## Output

CSV in long format, one row per (forecast_time, target_time), sorted by both:

| Column            | Type                   | Description                                   |
|-------------------|------------------------|-----------------------------------------------|
| `forecast_time`   | ISO datetime with zone | Time at which the forecast was issued         |
| `target_time`     | ISO datetime with zone | Time the predicted value refers to            |
| `predicted_value` | float                  | Value predicted by the model for target_time  |

Loading in pandas:

```python
import pandas as pd

df = pd.read_csv("forecasts_5d.csv", parse_dates=["forecast_time", "target_time"])
# One forecast as a Series indexed by target_time:
first = df["forecast_time"].min()
series = df[df["forecast_time"] == first].set_index("target_time")["predicted_value"]
```
