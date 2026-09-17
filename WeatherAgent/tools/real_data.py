"""Real meteorological data for WeatherAgent.

Replaces the synthetic `data/sample_weather.csv` with:
  * real hourly surface observations at Nanjing Lukou (ZSNJ) from the Iowa
    Environmental Mesonet (ASOS/METAR archive);
  * real archived numerical-model forecasts (GFS / ECMWF IFS) from the
    Open-Meteo historical-forecast and previous-runs APIs.

All times are UTC.  Temperatures in degrees Celsius, pressure in hPa, wind in
metres per second, precipitation in millimetres.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import pandas as pd
import requests

UA = {"User-Agent": "WeatherAgent/2.0 (student research prototype)"}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")


@dataclass(frozen=True)
class Station:
    """A surface station used as verification truth."""

    code: str
    name: str
    lat: float
    lon: float
    elevation_m: float


STATIONS = {
    # Nanjing Lukou International Airport (ZSNJ), the station closest to NUIST.
    "ZSNJ": Station("ZSNJ", "南京禄口", 31.74, 118.86, 15.0),
    # Second stations, used for cross-station (transferability) tests.
    "ZSSS": Station("ZSSS", "上海虹桥", 31.20, 121.34, 3.0),
    "ZSPD": Station("ZSPD", "上海浦东", 31.14, 121.81, 4.0),
}


def _cache_path(name: str) -> str:
    os.makedirs(RAW_DIR, exist_ok=True)
    return os.path.join(RAW_DIR, name)


def _get_json(url: str, cache_name: str | None = None, retries: int = 3) -> dict:
    """GET a JSON document, transparently handling gzip and an on-disk cache."""
    if cache_name:
        path = _cache_path(cache_name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=UA, timeout=120)
            resp.raise_for_status()
            raw = resp.content
            if raw[:2] == b"\x1f\x8b":
                raw = gzip.decompress(raw)
            payload = json.loads(raw.decode("utf-8"))
            if "error" in payload:
                raise ValueError(payload.get("reason", payload["error"]))
            if cache_name:
                with open(_cache_path(cache_name), "w", encoding="utf-8") as fh:
                    json.dump(payload, fh)
            return payload
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"request failed: {url}") from last_error


# --------------------------------------------------------------------------- #
# observations
# --------------------------------------------------------------------------- #
OBS_FIELDS = {
    "tmpf": "t_obs",       # 2 m air temperature, degF -> degC
    "dwpf": "td_obs",      # 2 m dew point,      degF -> degC
    "relh": "rh_obs",      # relative humidity,  %
    "sknt": "wind_obs",    # wind speed,         kt -> m/s
    "alti": "p_obs",       # altimeter setting,  inHg -> hPa
    "p01i": "prcp_obs",    # 1 h precipitation,  in -> mm
}


def fetch_station_obs(
    station: str,
    start: str | date,
    end: str | date,
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Download hourly surface observations (IEM ASOS/METAR archive)."""
    start = pd.Timestamp(start).date()
    end = pd.Timestamp(end).date()
    frames = []

    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=120), end)
        # version suffix: bump when the requested field list changes
        cache_name = f"obs_v2_{station}_{cursor:%Y%m%d}_{chunk_end:%Y%m%d}.csv"
        path = _cache_path(cache_name)
        if not (use_cache and os.path.exists(path)):
            url = (
                "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
                f"?station={station}"
                f"&data={','.join(OBS_FIELDS)}"
                f"&year1={cursor:%Y}&month1={cursor:%m}&day1={cursor:%d}"
                f"&year2={chunk_end:%Y}&month2={chunk_end:%m}&day2={chunk_end:%d}"
                "&tz=UTC&format=onlycomma&latlon=no&missing=M&trace=T"
                "&direct=no&report_type=3"
            )
            text = ""
            for attempt in range(4):
                try:
                    resp = requests.get(url, headers=UA, timeout=180)
                    resp.raise_for_status()
                    text = resp.text
                    break
                except Exception:  # noqa: BLE001
                    if attempt == 3:
                        raise
                    time.sleep(3 * (attempt + 1))
            text = "\n".join(
                line for line in text.splitlines() if not line.startswith("#")
            )
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        frames.append(pd.read_csv(path))
        cursor = chunk_end + timedelta(days=1)

    obs = pd.concat(frames, ignore_index=True)
    obs = obs.rename(columns=OBS_FIELDS)
    obs = obs.loc[:, ["valid", *OBS_FIELDS.values()]]
    obs["time"] = pd.to_datetime(obs["valid"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    obs = obs.drop(columns=["valid"])

    for column in ("t_obs", "td_obs", "rh_obs", "wind_obs", "p_obs", "prcp_obs"):
        obs[column] = pd.to_numeric(obs[column], errors="coerce")

    obs["t_obs"] = (obs["t_obs"] - 32.0) * 5.0 / 9.0
    obs["td_obs"] = (obs["td_obs"] - 32.0) * 5.0 / 9.0
    obs["wind_obs"] = obs["wind_obs"] * 0.514444
    obs["p_obs"] = obs["p_obs"] * 33.8639
    obs["prcp_obs"] = obs["prcp_obs"] * 25.4

    obs = obs.dropna(subset=["t_obs"])
    obs = obs.sort_values("time").drop_duplicates(subset=["time"], keep="first")
    return obs.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# forecasts
# --------------------------------------------------------------------------- #
FC_VARS = (
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "wind_speed_10m",
    "precipitation",
)


def fetch_model_archive(
    lat: float,
    lon: float,
    start: str | date,
    end: str | date,
    *,
    models: tuple[str, ...] = ("gfs_seamless",),
    use_cache: bool = True,
) -> pd.DataFrame:
    """Archived single run-to-run forecast series (one value per valid time)."""
    payload = _get_json(
        "https://historical-forecast-api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={pd.Timestamp(start):%Y-%m-%d}&end_date={pd.Timestamp(end):%Y-%m-%d}"
        f"&hourly={','.join(FC_VARS)}"
        f"&models={','.join(models)}&wind_speed_unit=ms&timezone=UTC",
        cache_name=(
            f"archive_{lat:.2f}_{lon:.2f}_{pd.Timestamp(start):%Y%m%d}"
            f"_{pd.Timestamp(end):%Y%m%d}_{'-'.join(models)}.json"
        )
        if use_cache
        else None,
    )
    hourly = payload["hourly"]
    frame = pd.DataFrame({"time": pd.to_datetime(hourly["time"])})
    for model in models:
        for var in FC_VARS:
            key = var if len(models) == 1 else f"{var}_{model}"
            if key in hourly:
                frame[f"{var}__{model}"] = hourly[key]
    frame["grid_elevation_m"] = payload.get("elevation")
    return frame


def fetch_forecast_runs(
    lat: float,
    lon: float,
    *,
    leads: tuple[int, ...] = (1, 2, 3, 5, 7),
    past_days: int = 92,
    model: str = "gfs_seamless",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Archived model runs at fixed lead times, from Open-Meteo previous-runs API.

    Column `temperature_2m_lead3` is the forecast issued 3 days before `time`.
    """
    variables = []
    for lead in leads:
        variables.append(f"temperature_2m_previous_day{lead}")
        variables.append(f"relative_humidity_2m_previous_day{lead}")
        variables.append(f"wind_speed_10m_previous_day{lead}")
        variables.append(f"pressure_msl_previous_day{lead}")

    payload = _get_json(
        "https://previous-runs-api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&hourly={','.join(variables)}"
        f"&past_days={past_days}&forecast_days=1"
        f"&models={model}&wind_speed_unit=ms&timezone=UTC",
        cache_name=f"runs_{lat:.2f}_{lon:.2f}_{past_days}_{'-'.join(map(str, leads))}_{model}.json"
        if use_cache
        else None,
    )
    hourly = payload["hourly"]
    frame = pd.DataFrame({"time": pd.to_datetime(hourly["time"])})
    for lead in leads:
        for var, suffix in (
            ("temperature_2m", "t"),
            ("relative_humidity_2m", "rh"),
            ("wind_speed_10m", "wind"),
            ("pressure_msl", "p"),
        ):
            key = f"{var}_previous_day{lead}"
            if key in hourly:
                frame[f"{suffix}_lead{lead}"] = hourly[key]
    frame["grid_elevation_m"] = payload.get("elevation")
    return frame


# --------------------------------------------------------------------------- #
# assembled datasets
# --------------------------------------------------------------------------- #
def build_dropin_dataset(
    station: str = "ZSNJ",
    start: str | date = "2024-01-01",
    end: str | date = "2026-09-16",
    model: str = "gfs_seamless",
) -> pd.DataFrame:
    """DataFrame with the same columns as `data/sample_weather.csv`.

    columns: time, forecast_temperature, observed_temperature, humidity,
             pressure, wind_speed, hour, month, station, model, grid_elevation_m
    """
    site = STATIONS[station]
    obs = fetch_station_obs(station, start, end)
    fcst = fetch_model_archive(site.lat, site.lon, start, end, models=(model,))

    merged = obs.merge(fcst, on="time", how="inner")
    out = pd.DataFrame(
        {
            "time": merged["time"],
            "forecast_temperature": merged[f"temperature_2m__{model}"],
            "observed_temperature": merged["t_obs"],
            "humidity": merged[f"relative_humidity_2m__{model}"],
            "pressure": merged[f"pressure_msl__{model}"],
            "wind_speed": merged[f"wind_speed_10m__{model}"],
            "hour": merged["time"].dt.hour,
            "month": merged["time"].dt.month,
            "station": station,
            "model": model,
            "grid_elevation_m": merged["grid_elevation_m"],
        }
    )
    out["observed_humidity"] = merged["rh_obs"]
    out["observed_wind_speed"] = merged["wind_obs"]
    out["observed_pressure"] = merged["p_obs"]
    out["forecast_precipitation"] = merged[f"precipitation__{model}"]
    out["observed_precipitation"] = merged["prcp_obs"]
    out = out.dropna(subset=["forecast_temperature", "observed_temperature"])
    return out.sort_values("time").reset_index(drop=True)


def build_leadtime_dataset(
    station: str = "ZSNJ",
    *,
    leads: tuple[int, ...] = (1, 2, 3, 5, 7),
    past_days: int = 92,
    model: str = "gfs_seamless",
) -> pd.DataFrame:
    """Tidy long table: one row per (valid time, lead time)."""
    site = STATIONS[station]
    runs = fetch_forecast_runs(
        site.lat, site.lon, leads=leads, past_days=past_days, model=model
    )
    start = runs["time"].min().date()
    obs = fetch_station_obs(station, start, runs["time"].max().date())

    merged = obs.merge(runs, on="time", how="inner")
    records = []
    for lead in leads:
        block = pd.DataFrame(
            {
                "time": merged["time"],
                "lead_day": lead,
                "forecast_temperature": merged[f"t_lead{lead}"],
                "observed_temperature": merged["t_obs"],
                "humidity": merged[f"rh_lead{lead}"],
                "wind_speed": merged[f"wind_lead{lead}"],
                "pressure": merged[f"p_lead{lead}"],
                "hour": merged["time"].dt.hour,
                "month": merged["time"].dt.month,
                "station": station,
                "model": model,
                "grid_elevation_m": merged["grid_elevation_m"],
            }
        )
        block["valid_date"] = merged["time"].dt.normalize()
        block["init_time"] = merged["time"] - pd.to_timedelta(lead, unit="D")
        records.append(block)

    out = pd.concat(records, ignore_index=True)
    out = out.dropna(subset=["forecast_temperature", "observed_temperature"])
    return out.sort_values(["lead_day", "time"]).reset_index(drop=True)


def _cli() -> None:
    out_dir = os.path.join(ROOT, "data")
    os.makedirs(out_dir, exist_ok=True)

    dropin = build_dropin_dataset("ZSNJ")
    dropin_path = os.path.join(out_dir, "sample_weather_real.csv")
    dropin.to_csv(dropin_path, index=False, encoding="utf-8-sig")
    print(f"drop-in dataset : {dropin_path}  ({len(dropin)} rows)")

    shanghai = build_dropin_dataset("ZSSS")
    shanghai_path = os.path.join(out_dir, "zsss_shanghai_real.csv")
    shanghai.to_csv(shanghai_path, index=False, encoding="utf-8-sig")
    print(f"second station  : {shanghai_path}  ({len(shanghai)} rows)")

    lead = build_leadtime_dataset()
    lead_path = os.path.join(out_dir, "gfs_leadtime.csv")
    lead.to_csv(lead_path, index=False, encoding="utf-8-sig")
    print(f"lead-time dataset: {lead_path}  ({len(lead)} rows)")


if __name__ == "__main__":
    _cli()
