"""
Tests for hydraflownet_forecast_client.download_forecasts.

The API is never called - ``requests.get`` is replaced with a fake that returns prepared JSON.
"""

import sys

import pandas as pd
import pytest
import requests

from hydraflownet_forecast_client import download_forecasts as df_mod


class FakeResponse:
    """Minimal stand-in for requests.Response."""

    def __init__(self, payload: dict, status_code: int = 200):
        """
        :param payload: dict - JSON body returned by .json().
        :param status_code: int - HTTP status code; >= 400 makes raise_for_status raise.
        """
        self.payload = payload
        self.status_code = status_code

    def json(self) -> dict:
        """
        :return: dict - the prepared payload.
        """
        return self.payload

    def raise_for_status(self) -> None:
        """
        Raise requests.HTTPError for error status codes, like the real response.

        :return: None
        """
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


def make_payload(forecasts: dict[str, list[tuple[str, float]]], model_id: str = "4") -> dict:
    """
    Build an API payload in the format returned by /v1/forecast/historical.

    :param forecasts: dict[str, list[tuple[str, float]]] - forecast_time -> list of (target_time, predicted_value).
    :param model_id: str - model id key used inside each forecast.
    :return: dict - {"forecasts": {forecast_time: {model_id: [{"targetTime": ..., "predictedValue": ...}]}}}
    """
    return {
        "forecasts": {
            forecast_time: {
                model_id: [{"targetTime": t, "predictedValue": v} for t, v in values]
            }
            for forecast_time, values in forecasts.items()
        }
    }


# --- resolve_model_uuid ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["5d", "42d", "210d"])
def test_resolve_model_uuid_known_names(name: str) -> None:
    """Known model names map to their UUIDs."""
    assert df_mod.resolve_model_uuid(name) == df_mod.MODEL_UUIDS[name]


def test_resolve_model_uuid_raw_uuid_passthrough() -> None:
    """Unknown values are returned unchanged (treated as raw UUIDs)."""
    uuid = "00000000-0000-0000-0000-000000000000"
    assert df_mod.resolve_model_uuid(uuid) == uuid


# --- make_time_windows ----------------------------------------------------------------------------

def test_make_time_windows_single_window() -> None:
    """A period shorter than chunk_days gives exactly one window equal to the period."""
    start = pd.Timestamp("2026-02-01", tz="UTC")
    end = pd.Timestamp("2026-02-10", tz="UTC")
    assert df_mod.make_time_windows(start, end, 30) == [(start, end)]


def test_make_time_windows_multiple_windows_cover_period() -> None:
    """Windows are contiguous, cover the whole period, and the last one is truncated."""
    start = pd.Timestamp("2026-02-01", tz="UTC")
    end = pd.Timestamp("2026-02-26", tz="UTC")
    windows = df_mod.make_time_windows(start, end, 10)

    assert len(windows) == 3
    assert windows[0][0] == start
    assert windows[-1][1] == end
    # Each window starts where the previous one ended.
    for (_, prev_end), (next_start, _) in zip(windows[:-1], windows[1:]):
        assert prev_end == next_start
    assert windows[-1][1] - windows[-1][0] == pd.Timedelta(days=5)


@pytest.mark.parametrize("end", ["2026-02-01", "2026-01-01"])
def test_make_time_windows_invalid_period(end: str) -> None:
    """from_time >= to_time raises ValueError."""
    with pytest.raises(ValueError):
        df_mod.make_time_windows(pd.Timestamp("2026-02-01", tz="UTC"), pd.Timestamp(end, tz="UTC"), 30)


# --- to_utc_timestamp -----------------------------------------------------------------------------

def test_to_utc_timestamp_naive_is_assumed_utc() -> None:
    """Naive datetimes are localized to UTC."""
    assert df_mod.to_utc_timestamp("2026-02-01T00:00:00") == pd.Timestamp("2026-02-01T00:00:00", tz="UTC")


def test_to_utc_timestamp_converts_offset() -> None:
    """Datetimes with an offset are converted to UTC."""
    ts = df_mod.to_utc_timestamp("2026-02-01T02:00:00+02:00")
    assert ts == pd.Timestamp("2026-02-01T00:00:00", tz="UTC")
    assert str(ts.tz) == "UTC"


# --- fetch_window ---------------------------------------------------------------------------------

def test_fetch_window_parses_response_and_sends_params(monkeypatch: pytest.MonkeyPatch) -> None:
    """Response is parsed to long format with correct dtypes, and the right query params are sent."""
    calls = []
    payload = make_payload({
        "2026-02-01T00:00:00Z": [("2026-02-02T00:00:00Z", 1.5), ("2026-02-03T00:00:00Z", 2.5)],
        "2026-02-02T00:00:00Z": [("2026-02-03T00:00:00Z", 3.0)],
    })

    def fake_get(url: str, params: dict) -> FakeResponse:
        calls.append((url, params))
        return FakeResponse(payload)

    monkeypatch.setattr(df_mod.requests, "get", fake_get)

    result = df_mod.fetch_window("http://api", "KEY", "UUID", "2026-02-01T00:00:00+00:00", "2026-02-05T00:00:00+00:00")

    assert calls == [("http://api", {
        "apiKey": "KEY",
        "from": "2026-02-01T00:00:00+00:00",
        "to": "2026-02-05T00:00:00+00:00",
        "modelUuid": "UUID",
    })]
    assert list(result.columns) == df_mod.COLUMNS
    assert len(result) == 3
    assert str(result["forecast_time"].dt.tz) == "UTC"
    assert str(result["target_time"].dt.tz) == "UTC"
    assert result["predicted_value"].dtype == float
    assert result["predicted_value"].tolist() == [1.5, 2.5, 3.0]


def test_fetch_window_empty_forecasts(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty 'forecasts' dict gives an empty DataFrame with the expected columns."""
    monkeypatch.setattr(df_mod.requests, "get", lambda url, params: FakeResponse({"forecasts": {}}))

    result = df_mod.fetch_window("http://api", "KEY", "UUID", "a", "b")

    assert result.empty
    assert list(result.columns) == df_mod.COLUMNS


def test_fetch_window_http_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """HTTP errors are not swallowed."""
    monkeypatch.setattr(df_mod.requests, "get", lambda url, params: FakeResponse({}, status_code=401))

    with pytest.raises(requests.HTTPError):
        df_mod.fetch_window("http://api", "BAD_KEY", "UUID", "a", "b")


def test_fetch_window_missing_forecasts_key_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A malformed response without 'forecasts' raises KeyError."""
    monkeypatch.setattr(df_mod.requests, "get", lambda url, params: FakeResponse({"something": 1}))

    with pytest.raises(KeyError):
        df_mod.fetch_window("http://api", "KEY", "UUID", "a", "b")


# --- download_forecasts ---------------------------------------------------------------------------

def test_download_forecasts_deduplicates_and_sorts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rows duplicated across windows are dropped and the result is sorted."""
    # Window 1 and window 2 both return the boundary forecast (2026-02-11).
    responses = [
        make_payload({
            "2026-02-11T00:00:00Z": [("2026-02-12T00:00:00Z", 10.0)],
            "2026-02-01T00:00:00Z": [("2026-02-03T00:00:00Z", 2.0), ("2026-02-02T00:00:00Z", 1.0)],
        }),
        make_payload({
            "2026-02-11T00:00:00Z": [("2026-02-12T00:00:00Z", 10.0)],
            "2026-02-15T00:00:00Z": [("2026-02-16T00:00:00Z", 20.0)],
        }),
    ]
    sent_windows = []

    def fake_get(url: str, params: dict) -> FakeResponse:
        sent_windows.append((params["from"], params["to"]))
        return FakeResponse(responses[len(sent_windows) - 1])

    monkeypatch.setattr(df_mod.requests, "get", fake_get)

    result = df_mod.download_forecasts(
        "http://api", "KEY", "UUID",
        pd.Timestamp("2026-02-01", tz="UTC"), pd.Timestamp("2026-02-21", tz="UTC"), chunk_days=10,
    )

    assert len(sent_windows) == 2
    assert len(result) == 4
    assert not result.duplicated(subset=["forecast_time", "target_time"]).any()
    assert result["predicted_value"].tolist() == [1.0, 2.0, 10.0, 20.0]
    assert result.index.tolist() == list(range(4))


# --- main (CLI) -----------------------------------------------------------------------------------

def test_main_writes_csv(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """End-to-end CLI run: model name resolves to UUID and the CSV is written with the expected content."""
    output = tmp_path / "out.csv"
    sent_params = []
    payload = make_payload({
        "2026-02-01T00:00:00Z": [("2026-02-02T00:00:00Z", 1.5), ("2026-02-03T00:00:00Z", 2.5)],
    })

    def fake_get(url: str, params: dict) -> FakeResponse:
        sent_params.append(params)
        return FakeResponse(payload)

    monkeypatch.setattr(df_mod.requests, "get", fake_get)
    monkeypatch.setattr(sys, "argv", [
        "hydraflownet-download",
        "--api-key", "KEY",
        "--model", "42d",
        "--from", "2026-02-01T00:00:00+00:00",
        "--to", "2026-02-05T00:00:00+00:00",
        "--output", str(output),
    ])

    df_mod.main()

    assert len(sent_params) == 1
    assert sent_params[0]["modelUuid"] == df_mod.MODEL_UUIDS["42d"]
    assert sent_params[0]["apiKey"] == "KEY"

    saved = pd.read_csv(output)
    assert list(saved.columns) == df_mod.COLUMNS
    assert pd.to_datetime(saved["forecast_time"], utc=True).tolist() == [pd.Timestamp("2026-02-01", tz="UTC")] * 2
    assert pd.to_datetime(saved["target_time"], utc=True).tolist() == [
        pd.Timestamp("2026-02-02", tz="UTC"), pd.Timestamp("2026-02-03", tz="UTC"),
    ]
    assert saved["predicted_value"].tolist() == [1.5, 2.5]


def test_main_missing_required_args_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing --api-key / --model makes argparse exit."""
    monkeypatch.setattr(sys, "argv", ["hydraflownet-download"])

    with pytest.raises(SystemExit):
        df_mod.main()
