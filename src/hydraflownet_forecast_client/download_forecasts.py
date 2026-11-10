"""
Download historical forecasts from the HydraFlowNet API and save them to a local CSV file.

Requires only ``requests``, ``pandas`` and ``tqdm``.

Usage (from the project root, with uv)
-----
    uv run hydraflownet-download --api-key <API_KEY> --model 5d
    uv run hydraflownet-download --api-key <API_KEY> --model 42d \\
        --from 2026-02-01T00:00:00+00:00 --to 2026-06-01T00:00:00+00:00 --output 42d.csv
    uv run hydraflownet-download --api-key <API_KEY> --model <MODEL_UUID>

Or install the command globally with ``uv tool install .`` and run ``hydraflownet-download ...``.

Arguments
---------
    --api-key     API key (required).
    --model       Model: one of 5d, 42d, 210d, or a raw model UUID (required).
    --from        Start of the period (ISO datetime). Default: 2026-02-01T00:00:00+00:00.
    --to          End of the period (ISO datetime). Default: now (UTC).
    --output      Output CSV path. Default: forecasts_<model>.csv.
    --url         API endpoint. Default: https://hydraflownet-api.test.medius.si/v1/forecast/historical
    --chunk-days  Length (in days) of the time window requested per API call. Default: 30.

Saved object
------------
A CSV file at ``--output`` (long format, one row per (forecast_time, target_time) pair,
sorted by forecast_time, then target_time). Columns:
    - forecast_time   (str, ISO 8601 datetime with timezone): time at which the forecast was issued.
    - target_time     (str, ISO 8601 datetime with timezone): time the predicted value refers to.
    - predicted_value (float): value predicted by the model for target_time, as returned by the API.

To load it back in pandas:
    df = pd.read_csv("forecasts_5d.csv", parse_dates=["forecast_time", "target_time"])
    # one forecast as a Series indexed by target_time:
    df[df.forecast_time == some_time].set_index("target_time")["predicted_value"]
"""

import argparse
import logging

import pandas as pd
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("download_forecasts")

DEFAULT_URL = "https://hydraflownet-api.test.medius.si/v1/forecast/historical"
DEFAULT_FROM = "2026-02-01T00:00:00+00:00"

# Model name -> model UUID on the API.
MODEL_UUIDS = {
    "5d": "478fb99e-bba3-4bac-9609-bd4ea1a4e133",
    "42d": "39e33f76-7a37-4c7e-a70c-9112f01a53e1",
    "210d": "59c01b34-df06-4170-930e-1447520bfe99",
}

COLUMNS = ["forecast_time", "target_time", "predicted_value"]


def resolve_model_uuid(model: str) -> str:
    """
    Map a model name to its UUID. Values that are not known names are treated as raw UUIDs.

    :param model: str - model name ("5d", "42d", "210d") or a model UUID.
    :return: str - model UUID.
    """
    if model in MODEL_UUIDS:
        return MODEL_UUIDS[model]
    return model


def make_time_windows(from_time: pd.Timestamp, to_time: pd.Timestamp,
                      chunk_days: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Split the period [from_time, to_time] into consecutive windows of at most chunk_days days.

    :param from_time: pd.Timestamp - start of the period (timezone aware).
    :param to_time: pd.Timestamp - end of the period (timezone aware).
    :param chunk_days: int - maximal window length in days.
    :return: list[tuple[pd.Timestamp, pd.Timestamp]] - list of (window_start, window_end) pairs.
    """
    if from_time >= to_time:
        raise ValueError(f"--from ({from_time}) must be before --to ({to_time})")

    windows = []
    start = from_time
    while start < to_time:
        end = min(start + pd.Timedelta(days=chunk_days), to_time)
        windows.append((start, end))
        start = end
    return windows


def fetch_window(url: str, api_key: str, model_uuid: str, from_time: str, to_time: str) -> pd.DataFrame:
    """
    Fetch historical forecasts for one time window from the API.

    :param url: str - API endpoint.
    :param api_key: str - API key.
    :param model_uuid: str - model UUID.
    :param from_time: str - window start (ISO datetime).
    :param to_time: str - window end (ISO datetime).
    :return: pd.DataFrame - long format with columns forecast_time (datetime), target_time (datetime),
        predicted_value (float).
    """
    params = {
        "apiKey": api_key,
        "from": from_time,
        "to": to_time,
        "modelUuid": model_uuid,
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    # Response structure: {"forecasts": {forecast_time: {model_id: [{"targetTime": ..., "predictedValue": ...}]}}}
    rows = []
    for forecast_time, model_dict in data["forecasts"].items():
        for model_id, forecast_list in model_dict.items():
            for item in forecast_list:
                rows.append((forecast_time, item["targetTime"], item["predictedValue"]))

    df = pd.DataFrame(rows, columns=COLUMNS)
    df["forecast_time"] = pd.to_datetime(df["forecast_time"], utc=True)
    df["target_time"] = pd.to_datetime(df["target_time"], utc=True)
    df["predicted_value"] = df["predicted_value"].astype(float)
    return df


def download_forecasts(url: str, api_key: str, model_uuid: str, from_time: pd.Timestamp,
                       to_time: pd.Timestamp, chunk_days: int) -> pd.DataFrame:
    """
    Download all historical forecasts in [from_time, to_time], window by window.

    :param url: str - API endpoint.
    :param api_key: str - API key.
    :param model_uuid: str - model UUID.
    :param from_time: pd.Timestamp - start of the period (timezone aware).
    :param to_time: pd.Timestamp - end of the period (timezone aware).
    :param chunk_days: int - window length in days for a single API call.
    :return: pd.DataFrame - long format with columns forecast_time, target_time, predicted_value,
        deduplicated and sorted by (forecast_time, target_time).
    """
    windows = make_time_windows(from_time, to_time, chunk_days)
    logger.info(f"Downloading forecasts from {from_time} to {to_time} in {len(windows)} window(s)")

    frames = []
    for start, end in tqdm(windows, desc="Downloading"):
        frames.append(fetch_window(url, api_key, model_uuid, start.isoformat(), end.isoformat()))

    df = pd.concat(frames, ignore_index=True)
    # Forecasts at window boundaries may be returned twice.
    df = df.drop_duplicates(subset=["forecast_time", "target_time"], keep="last")
    df = df.sort_values(["forecast_time", "target_time"]).reset_index(drop=True)
    return df


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    :return: argparse.Namespace - parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Download historical forecasts from the HydraFlowNet API to CSV.")
    parser.add_argument("--api-key", required=True, help="API key.")
    parser.add_argument("--model", required=True,
                        help=f"Model name ({', '.join(MODEL_UUIDS)}) or a model UUID.")
    parser.add_argument("--from", dest="from_time", default=DEFAULT_FROM,
                        help=f"Start of the period, ISO datetime. Default: {DEFAULT_FROM}.")
    parser.add_argument("--to", dest="to_time", default=None,
                        help="End of the period, ISO datetime. Default: now (UTC).")
    parser.add_argument("--output", default=None, help="Output CSV path. Default: forecasts_<model>.csv.")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"API endpoint. Default: {DEFAULT_URL}.")
    parser.add_argument("--chunk-days", type=int, default=30,
                        help="Window length in days for a single API call. Default: 30.")
    return parser.parse_args()


def to_utc_timestamp(value: str) -> pd.Timestamp:
    """
    Parse an ISO datetime string to a UTC timestamp (naive values are assumed to be UTC).

    :param value: str - ISO datetime.
    :return: pd.Timestamp - timezone-aware timestamp in UTC.
    """
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def main() -> None:
    """
    Entry point: parse arguments, download forecasts and save them to CSV.

    :return: None
    """
    args = parse_args()

    model_uuid = resolve_model_uuid(args.model)
    from_time = to_utc_timestamp(args.from_time)
    to_time = pd.Timestamp.now(tz="UTC") if args.to_time is None else to_utc_timestamp(args.to_time)
    output = args.output if args.output is not None else f"forecasts_{args.model}.csv"

    logger.info(f"Model: {args.model} (UUID: {model_uuid})")
    df = download_forecasts(args.url, args.api_key, model_uuid, from_time, to_time, args.chunk_days)

    df.to_csv(output, index=False, date_format="%Y-%m-%dT%H:%M:%S%z")
    logger.info(f"Saved {len(df)} rows ({df['forecast_time'].nunique()} forecasts) to {output}")


if __name__ == "__main__":
    main()
