"""资料同化实验：多站 2 米气温的**真正**最优插值（Optimal Interpolation）。

与 `legacy/oi_correction.py` 的区别
-----------------------------------
原实现把 OI 权重乘在「训练集平均偏差」上（`analysis = forecast + w * mean_bias`），
那相当于把常数去偏打了九折，并不构成 OI。真正的 OI 权重作用在**空间观测新息**上：

    x_a = x_b + B Hᵀ (H B Hᵀ + R)⁻¹ (y − H x_b)

其中
  * x_b 背景场（这里是 GFS 2 米气温预报，在各站位置取值）
  * y   观测（各站 METAR 气温）
  * B   背景场误差协方差，用均匀各向同性高斯模型：B_ij = σ_b² · exp(−d_ij² / (2L²))
  * R   观测误差协方差，取对角 σ_o²
  * L   背景误差相关长度（km）——本项目要通过试验来标定它

因为站点位置固定，权重矩阵与时间无关，可以预先算好再一次性作用到整条时间序列上，
所以整个实验是向量化的，非常快。

检验方式采用**留一站交叉检验（leave-one-station-out）**：
对每个站，只用其余 5 个站的观测做分析，再与该站真实观测比较。
这样得到的分析误差是诚实的样本外误差。
"""

from __future__ import annotations

import gzip
import io
import json
import os
import time
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

UA = {"User-Agent": "WeatherAgent/2.0 (student research prototype)"}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")


@dataclass(frozen=True)
class Site:
    code: str
    name: str
    lat: float
    lon: float


# 长三角 6 个有人工观测的机场站（均为 IEM METAR 归档中有连续记录的站）
SITES: dict[str, Site] = {
    "ZSNJ": Site("ZSNJ", "南京禄口", 31.74, 118.86),
    "ZSOF": Site("ZSOF", "合肥新桥", 31.78, 116.98),
    "ZSSS": Site("ZSSS", "上海虹桥", 31.20, 121.34),
    "ZSPD": Site("ZSPD", "上海浦东", 31.14, 121.81),
    "ZSHC": Site("ZSHC", "杭州萧山", 30.23, 120.43),
    "ZSNB": Site("ZSNB", "宁波栎社", 29.83, 121.46),
}

EARTH_RADIUS_KM = 6371.0


# --------------------------------------------------------------------------- #
# 数据获取
# --------------------------------------------------------------------------- #
def _cache(name: str) -> str:
    os.makedirs(RAW_DIR, exist_ok=True)
    return os.path.join(RAW_DIR, name)


def fetch_station_temperature(code: str, start, end) -> pd.DataFrame:
    """逐小时 2 米气温（由 METAR 的 ℉ 换算为 ℃）。"""
    start = pd.Timestamp(start).date()
    end = pd.Timestamp(end).date()
    frames = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=120), end)
        name = f"da_obs_v1_{code}_{cursor:%Y%m%d}_{chunk_end:%Y%m%d}.csv"
        path = _cache(name)
        if not os.path.exists(path):
            url = (
                "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
                f"?station={code}&data=tmpf"
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
            text = "\n".join(line for line in text.splitlines() if not line.startswith("#"))
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        frames.append(pd.read_csv(path))
        cursor = chunk_end + timedelta(days=1)

    obs = pd.concat(frames, ignore_index=True)
    obs["time"] = pd.to_datetime(obs["valid"], utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
    obs["t_obs"] = (pd.to_numeric(obs["tmpf"], errors="coerce") - 32.0) * 5.0 / 9.0
    obs = obs.dropna(subset=["t_obs"])
    obs["station"] = code
    obs = obs[["time", "station", "t_obs"]]
    obs = obs.sort_values("time").drop_duplicates(subset=["time"], keep="first")
    return obs.reset_index(drop=True)


def fetch_background_points(latitudes: list[float], longitudes: list[float], start, end) -> pd.DataFrame:
    """GFS 2 米气温背景场，在给定的一组坐标上取值。"""
    lats = ",".join(f"{value:.4f}" for value in latitudes)
    lons = ",".join(f"{value:.4f}" for value in longitudes)
    name = f"da_bg_v1_{len(latitudes)}_{pd.Timestamp(start):%Y%m%d}_{pd.Timestamp(end):%Y%m%d}.json"
    path = _cache(name)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
    else:
        url = (
            "https://historical-forecast-api.open-meteo.com/v1/forecast"
            f"?latitude={lats}&longitude={lons}"
            f"&start_date={pd.Timestamp(start):%Y-%m-%d}&end_date={pd.Timestamp(end):%Y-%m-%d}"
            "&hourly=temperature_2m&models=gfs_seamless&timezone=UTC"
        )
        resp = requests.get(url, headers=UA, timeout=180)
        resp.raise_for_status()
        raw = resp.content
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        payload = json.loads(raw.decode("utf-8"))
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        time.sleep(1)

    if not isinstance(payload, list):
        payload = [payload]

    records = []
    for item in payload:
        frame = pd.DataFrame(
            {
                "time": pd.to_datetime(item["hourly"]["time"]),
                "lat": item["latitude"],
                "lon": item["longitude"],
                "t_bg": item["hourly"]["temperature_2m"],
                "grid_elevation_m": item.get("elevation"),
            }
        )
        records.append(frame)
    return pd.concat(records, ignore_index=True)


def build_assimilation_dataset(
    start: str = "2024-01-01",
    end: str = "2026-09-15",
    codes: list[str] | None = None,
) -> pd.DataFrame:
    """把 6 个站的观测与对应位置的背景场合并成长表。"""
    codes = codes or list(SITES)
    sites = [SITES[code] for code in codes]

    obs = pd.concat(
        [fetch_station_temperature(site.code, start, end) for site in sites],
        ignore_index=True,
    )

    background = fetch_background_points(
        [site.lat for site in sites], [site.lon for site in sites], start, end
    )
    background["station"] = [
        _nearest_station(row, sites) for row in background[["lat", "lon"]].itertuples(index=False)
    ]

    merged = obs.merge(
        background[["time", "station", "t_bg", "grid_elevation_m"]],
        on=["time", "station"],
        how="inner",
    )
    merged["lat"] = merged["station"].map({site.code: site.lat for site in sites})
    merged["lon"] = merged["station"].map({site.code: site.lon for site in sites})
    return merged.dropna(subset=["t_obs", "t_bg"]).sort_values(["time", "station"]).reset_index(drop=True)


def _nearest_station(row, sites: list[Site]) -> str:
    distances = [
        (row.lat - site.lat) ** 2 + (row.lon - site.lon) ** 2 for site in sites
    ]
    return sites[int(np.argmin(distances))].code


# --------------------------------------------------------------------------- #
# 最优插值
# --------------------------------------------------------------------------- #
def haversine_matrix(latitudes, longitudes) -> np.ndarray:
    """站点间大圆距离矩阵（km）。"""
    lat = np.radians(np.asarray(latitudes, dtype=float))[:, None]
    lon = np.radians(np.asarray(longitudes, dtype=float))[:, None]
    dlat = lat - lat.T
    dlon = lon - lon.T
    a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(lat.T) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def cross_distance(lat_a, lon_a, lat_b, lon_b) -> np.ndarray:
    """两组点之间的两两距离矩阵（km），形状 (len(a), len(b))。"""
    lat_a = np.radians(np.asarray(lat_a, dtype=float))[:, None]
    lon_a = np.radians(np.asarray(lon_a, dtype=float))[:, None]
    lat_b = np.radians(np.asarray(lat_b, dtype=float))[None, :]
    lon_b = np.radians(np.asarray(lon_b, dtype=float))[None, :]
    dlat = lat_a - lat_b
    dlon = lon_a - lon_b
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_a) * np.cos(lat_b) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def gain_matrix(distance: np.ndarray, length_scale_km: float, sigma_b: float, sigma_o: float) -> np.ndarray:
    """OI 增益矩阵 A = B (B + R)⁻¹（对角观测误差）。"""
    background = sigma_b**2 * np.exp(-(distance**2) / (2.0 * length_scale_km**2))
    observation = np.eye(distance.shape[0]) * sigma_o**2
    return background @ np.linalg.inv(background + observation)


def weights_excluding(
    distance: np.ndarray,
    index: int,
    length_scale_km: float,
    sigma_b: float,
    sigma_o: float,
) -> np.ndarray:
    """留一站检验用的权重：把第 index 站**整个从方程组中剔除**后求解。

    正确的形式是  w = B_{i,O} (B_{OO} + R_{OO})⁻¹
    其中 O 表示其余站点。不能拿完整 6×6 增益矩阵的一行来用——那样
    权重是按“包含本站观测”的系统算出来的，直接套到子集上不再是最优解。
    """
    others = [j for j in range(distance.shape[0]) if j != index]
    sub = distance[np.ix_(others, others)]
    background_oo = sigma_b**2 * np.exp(-(sub**2) / (2.0 * length_scale_km**2))
    background_io = sigma_b**2 * np.exp(
        -(distance[index, others] ** 2) / (2.0 * length_scale_km**2)
    )
    observation_oo = np.eye(len(others)) * sigma_o**2
    return background_io @ np.linalg.inv(background_oo + observation_oo)


def wide_matrices(dataset: pd.DataFrame) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    """把长表转成 (站点列表, 背景场宽表, 观测宽表)，行=时间、列=站点。"""
    codes = sorted(dataset["station"].unique())
    background = dataset.pivot(index="time", columns="station", values="t_bg")[codes]
    observed = dataset.pivot(index="time", columns="station", values="t_obs")[codes]
    return codes, background, observed


def leave_one_out(
    dataset: pd.DataFrame,
    *,
    length_scale_km: float,
    sigma_o: float,
    total_variance: float | None = None,
    sample_every: int = 1,
) -> dict:
    """留一站交叉检验：每个站只用其他站做分析，再与该站观测比较。

    关于 σb 与 σo 的口径（这一点很容易错）
    --------------------------------------
    从数据里能直接估计的只有**新息总方差**

        Var(bg − obs) = σb² + σo²

    即背景场误差与观测误差之和。如果把 `std(bg − obs)` 直接当成 σb，再在 R 里
    额外加上 σo²，就等于把观测误差算了两次，会让分析权重被高估。
    因此这里采用标准做法：由用户指定观测误差 σo，再反解

        σb² = max(Var(bg − obs) − σo², 小量)

    真正决定权重的是比值 σo²/σb²，所以对 σo 做敏感性试验是有意义的。
    """
    codes, background, observed = wide_matrices(dataset)
    if sample_every > 1:
        background = background.iloc[::sample_every]
        observed = observed.iloc[::sample_every]

    latitudes = [SITES[code].lat for code in codes]
    longitudes = [SITES[code].lon for code in codes]
    distance = haversine_matrix(latitudes, longitudes)

    innovation = (observed - background).to_numpy()  # d = y − H x_b
    innovation = np.nan_to_num(innovation, nan=0.0)  # 缺测站不提供信息
    bg_values = background.to_numpy()
    obs_values = observed.to_numpy()

    if total_variance is None:
        total_variance = float(np.nanvar((background - observed).to_numpy()))
    sigma_b = float(np.sqrt(max(total_variance - sigma_o**2, 0.05)))

    analysis = bg_values.copy()
    per_station = []
    for index, code in enumerate(codes):
        others = [j for j in range(len(codes)) if j != index]
        weights = weights_excluding(distance, index, length_scale_km, sigma_b, sigma_o)
        analysis[:, index] = bg_values[:, index] + innovation[:, others] @ weights

        valid = ~np.isnan(obs_values[:, index])
        background_error = bg_values[valid, index] - obs_values[valid, index]
        analysis_error = analysis[valid, index] - obs_values[valid, index]
        per_station.append(
            {
                "station": code,
                "name": SITES[code].name,
                "n": int(valid.sum()),
                "bg_MAE": float(np.abs(background_error).mean()),
                "bg_RMSE": float(np.sqrt((background_error**2).mean())),
                "bg_bias": float(background_error.mean()),
                "oi_MAE": float(np.abs(analysis_error).mean()),
                "oi_RMSE": float(np.sqrt((analysis_error**2).mean())),
                "oi_bias": float(analysis_error.mean()),
                "RMSE_improvement_pct": float(
                    100
                    * (np.sqrt((background_error**2).mean()) - np.sqrt((analysis_error**2).mean()))
                    / np.sqrt((background_error**2).mean())
                ),
            }
        )

    valid_all = ~np.isnan(obs_values)
    background_error = bg_values[valid_all] - obs_values[valid_all]
    analysis_error = analysis[valid_all] - obs_values[valid_all]
    overall = {
        "length_scale_km": length_scale_km,
        "sigma_o": sigma_o,
        "sigma_b": sigma_b,
        "total_std": float(np.sqrt(total_variance)),
        "n": int(valid_all.sum()),
        "bg_MAE": float(np.abs(background_error).mean()),
        "bg_RMSE": float(np.sqrt((background_error**2).mean())),
        "bg_bias": float(background_error.mean()),
        "oi_MAE": float(np.abs(analysis_error).mean()),
        "oi_RMSE": float(np.sqrt((analysis_error**2).mean())),
        "oi_bias": float(analysis_error.mean()),
        "RMSE_improvement_pct": float(
            100
            * (np.sqrt((background_error**2).mean()) - np.sqrt((analysis_error**2).mean()))
            / np.sqrt((background_error**2).mean())
        ),
    }
    return {
        "overall": overall,
        "per_station": pd.DataFrame(per_station),
        "analysis": pd.DataFrame(analysis, index=background.index, columns=codes),
        "background": background,
        "observed": observed,
    }


def sensitivity(
    dataset: pd.DataFrame,
    *,
    length_scales: tuple[float, ...] = (50, 100, 150, 200, 300, 400, 600, 800, 1200),
    sigma_o_values: tuple[float, ...] = (0.5, 1.0, 2.0),
    total_variance: float | None = None,
    sample_every: int = 6,
) -> pd.DataFrame:
    """相关长度与观测误差对分析效果的影响——OI 的核心权衡。"""
    if total_variance is None:
        total_variance = float(
            np.nanvar((dataset["t_bg"] - dataset["t_obs"]).to_numpy())
        )
    rows = []
    for sigma_o in sigma_o_values:
        for length_scale in length_scales:
            result = leave_one_out(
                dataset,
                length_scale_km=length_scale,
                sigma_o=sigma_o,
                total_variance=total_variance,
                sample_every=sample_every,
            )
            row = dict(result["overall"])
            row["sigma_o"] = sigma_o
            rows.append(row)
    return pd.DataFrame(rows)


def innovation_correlation(dataset: pd.DataFrame) -> pd.DataFrame:
    """站点之间观测新息（obs − bg）的相关系数矩阵。

    这一项用来判断「误差的空间相关性」是否真的存在——如果新息在站点之间
    高度相关，说明误差以空间上大尺度、共同变化的成分为主，OI 才可能通过
    邻站信息改进本站的分析。
    """
    codes, background, observed = wide_matrices(dataset)
    innovation = (observed - background).dropna(how="all")
    matrix = innovation.corr(min_periods=200)
    matrix.index.name = "station"
    return matrix


def evaluate_methods(
    dataset: pd.DataFrame,
    *,
    length_scale_km: float,
    sigma_o: float,
    split_date: str,
    sample_every: int = 1,
) -> dict:
    """在**同一评估时段**上比较三种做法，且全部只用标定期数据标定。

    方法：
      1. 背景场            —— GFS 原始预报
      2. 本站气候态偏差订正 —— 用标定期该站平均偏差减去（传统后处理的代表）
      3. 最优插值 OI       —— 同一时刻邻站观测新息的空间加权

    评估时段与标定时段严格分离，因此三个方法都不存在数据泄漏。
    OI 的优势只能来自「同一时刻的邻站信息」，这正是资料同化的本质。
    """
    split = pd.Timestamp(split_date)
    calibration = dataset[dataset["time"] < split]
    evaluation = dataset[dataset["time"] >= split]
    if calibration.empty or evaluation.empty:
        raise ValueError("split_date 必须把数据分成非空的两段")

    total_variance = float(np.nanvar((calibration["t_bg"] - calibration["t_obs"]).to_numpy()))
    station_bias = (
        (calibration["t_bg"] - calibration["t_obs"]).groupby(calibration["station"]).mean()
    )

    result = leave_one_out(
        evaluation,
        length_scale_km=length_scale_km,
        sigma_o=sigma_o,
        total_variance=total_variance,
        sample_every=sample_every,
    )
    codes, background, observed = wide_matrices(evaluation)
    debiased = background - pd.Series(
        {code: station_bias.get(code, 0.0) for code in codes}
    )

    def _scores(estimate: pd.DataFrame, label: str) -> dict:
        error = (estimate - observed).to_numpy()
        error = error[~np.isnan(error)]
        return {
            "方法": label,
            "n": int(error.size),
            "bias": float(error.mean()),
            "MAE": float(np.abs(error).mean()),
            "RMSE": float(np.sqrt((error**2).mean())),
        }

    table = pd.DataFrame(
        [
            _scores(background, "① 背景场（GFS 原始预报）"),
            _scores(debiased, "② 本站气候态偏差订正"),
            _scores(result["analysis"], "③ 最优插值 OI（邻站同刻观测）"),
        ]
    )
    raw_rmse = float(table.loc[0, "RMSE"])
    table["相对背景场改善 %"] = 100 * (raw_rmse - table["RMSE"]) / raw_rmse

    calibration_counts = calibration.groupby("station").size().to_dict()
    result.update(
        {
            "comparison": table,
            "station_bias": station_bias.rename("calibration_bias"),
            "total_variance": total_variance,
            "split_date": split,
            "calibration_size": int(len(calibration)),
            "evaluation_size": int(len(evaluation)),
            "calibration_counts": calibration_counts,
        }
    )
    return result


# --------------------------------------------------------------------------- #
# 空间分析场（用于画“背景场 → 分析场”的二维对比图）
# --------------------------------------------------------------------------- #
def grid_points(
    lat_min: float, lat_max: float, lon_min: float, lon_max: float, step: float
) -> list[tuple[float, float]]:
    latitudes = np.arange(lat_min, lat_max + 1e-9, step)
    longitudes = np.arange(lon_min, lon_max + 1e-9, step)
    return [(float(lat), float(lon)) for lat in latitudes for lon in longitudes]


def fetch_grid_background(
    points: list[tuple[float, float]], start, end, *, chunk: int = 60
) -> pd.DataFrame:
    """在规则网格上取 GFS 2 米气温背景场（一次请求多个坐标，分批以免 URL 过长）。"""
    frames = []
    for offset in range(0, len(points), chunk):
        block = points[offset : offset + chunk]
        lats = ",".join(f"{lat:.4f}" for lat, _ in block)
        lons = ",".join(f"{lon:.4f}" for _, lon in block)
        name = (
            f"da_grid_v1_{offset}_{len(block)}_"
            f"{pd.Timestamp(start):%Y%m%d%H}_{pd.Timestamp(end):%Y%m%d%H}.json"
        )
        path = _cache(name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                payload = json.load(fh)
        else:
            url = (
                "https://historical-forecast-api.open-meteo.com/v1/forecast"
                f"?latitude={lats}&longitude={lons}"
                f"&start_date={pd.Timestamp(start):%Y-%m-%d}&end_date={pd.Timestamp(end):%Y-%m-%d}"
                "&hourly=temperature_2m&models=gfs_seamless&timezone=UTC"
            )
            resp = requests.get(url, headers=UA, timeout=180)
            resp.raise_for_status()
            raw = resp.content
            if raw[:2] == b"\x1f\x8b":
                raw = gzip.decompress(raw)
            payload = json.loads(raw.decode("utf-8"))
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            time.sleep(1)
        if not isinstance(payload, list):
            payload = [payload]
        # 注意：接口会把坐标吸附到它自己的网格上，返回的 lat/lon 并不整齐。
        # 为了保证随后能重排成规则网格，这里用**请求时的坐标**，返回顺序与请求顺序一致。
        for (requested_lat, requested_lon), item in zip(block, payload):
            frames.append(
                pd.DataFrame(
                    {
                        "time": pd.to_datetime(item["hourly"]["time"]),
                        "lat": round(float(requested_lat), 4),
                        "lon": round(float(requested_lon), 4),
                        "t_bg": item["hourly"]["temperature_2m"],
                        "grid_elevation_m": item.get("elevation"),
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def grid_analysis(
    grid_background: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    length_scale_km: float,
    sigma_o: float,
    total_variance: float,
) -> pd.DataFrame:
    """把某一时刻的站点观测融合进网格背景场，得到二维分析场。

    `observations` 需要包含 station / lat / lon / t_obs 四列（同一时刻）。
    """
    sigma_b = float(np.sqrt(max(total_variance - sigma_o**2, 0.05)))

    station_lat = observations["lat"].to_numpy()
    station_lon = observations["lon"].to_numpy()
    station_distance = haversine_matrix(station_lat, station_lon)
    station_covariance = sigma_b**2 * np.exp(
        -(station_distance**2) / (2.0 * length_scale_km**2)
    )
    observation_covariance = np.eye(len(observations)) * sigma_o**2

    innovation = (observations["t_obs"] - observations["t_bg"]).to_numpy()
    innovation = np.where(np.isnan(innovation), 0.0, innovation)

    output = grid_background.copy()
    grid_to_station = cross_distance(
        grid_background["lat"].to_numpy(), grid_background["lon"].to_numpy(),
        station_lat, station_lon,
    )
    cross_covariance = sigma_b**2 * np.exp(
        -(grid_to_station**2) / (2.0 * length_scale_km**2)
    )
    # 正确的 OI 形式：x_a = x_b + B_gs (B_ss + R)⁻¹ d
    # 注意不要再乘一次 B_ss，否则会把增量放大若干倍。
    weights = cross_covariance @ np.linalg.inv(
        station_covariance + observation_covariance
    )
    increment = weights @ innovation
    output["t_analysis"] = output["t_bg"] + increment
    output["increment"] = increment
    return output
