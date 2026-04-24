from __future__ import annotations

import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import Point
from shapely.prepared import prep

from helpers import DATA_DIR, RANDOM_STATE

# Константы
NE_BASE = "https://naciscdn.org/naturalearth/50m/physical"
LAND_ZIP = f"{NE_BASE}/ne_50m_land.zip"
COAST_ZIP = f"{NE_BASE}/ne_50m_coastline.zip"


def _download(url: str, path: Path) -> Path:
    """Загружает слои из Natural Earth.

    Args:
        url: URL для загрузки.
        path: Путь для сохранения.
        
    Returns:
        Путь к загруженному файлу.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    path.write_bytes(r.content)
    return path


def _load_ne_layers(cache_dir: Path) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Загружает слои из Natural Earth.

    Args:
        cache_dir: Путь к директории для кэширования.

    Returns:
        Кортеж из двух GeoDataFrame: land и coast.
    """

    land_zip = _download(LAND_ZIP, cache_dir / "ne_50m_land.zip")
    coast_zip = _download(COAST_ZIP, cache_dir / "ne_50m_coastline.zip")
    land = gpd.read_file(f"zip://{land_zip}!ne_50m_land.shp")
    coast = gpd.read_file(f"zip://{coast_zip}!ne_50m_coastline.shp")
    return land, coast


def _distances_km(lon: np.ndarray, lat: np.ndarray, coast_3857: gpd.GeoDataFrame) -> np.ndarray:
    """Вычисляет расстояния до берега.

    Args:
        lon: Массив долгот.
        lat: Массив широт.
        coast_3857: GeoDataFrame с берегом в проекции 3857.

    Returns:
        Массив расстояний до берега в километрах.
    """

    pts = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(lon, lat), crs="EPSG:4326"
    ).to_crs(3857)
    j = pts.sjoin_nearest(coast_3857, how="left", distance_col="dist_m")
    return (j["dist_m"].to_numpy(dtype=np.float64) / 1000.0).clip(0, 1e6)


def _ocean_candidates(rng: np.random.Generator, prep_land, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Генерирует океанические кандидаты.

    Args:
        rng: Генератор случайных чисел.
        prep_land: Prepared land.
        n: Количество кандидатов.

    Returns:
        Кортеж из двух массивов: lon и lat.
    """

    lon_acc, lat_acc = [], []
    attempts = 0
    while len(lon_acc) < n and attempts < n * 500:
        attempts += 1
        lon = rng.uniform(-180.0, 180.0)
        lat = math.degrees(math.asin(rng.uniform(-1.0, 1.0)))
        if lat < -60 or lat > 75:
            continue
        if prep_land.contains(Point(lon, lat)):
            continue
        lon_acc.append(lon)
        lat_acc.append(lat)
    if len(lon_acc) < n:
        raise RuntimeError("Не удалось набрать океанических кандидатов.")
    return np.asarray(lon_acc, dtype=np.float64), np.asarray(lat_acc, dtype=np.float64)


def _land_candidates(rng: np.random.Generator, prep_land, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Генерирует сушевые кандидаты.

    Args:
        rng: Генератор случайных чисел.
        prep_land: Prepared land.
        n: Количество кандидатов.

    Returns:
        Кортеж из двух массивов: lon и lat.
    """

    lon_acc, lat_acc = [], []
    attempts = 0
    while len(lon_acc) < n and attempts < n * 8000:
        attempts += 1
        lon = rng.uniform(-180.0, 180.0)
        lat = rng.uniform(-55.0, 70.0)
        if prep_land.contains(Point(lon, lat)):
            lon_acc.append(lon)
            lat_acc.append(lat)
    if len(lon_acc) < n:
        raise RuntimeError("Не удалось набрать сушевых кандидатов.")
    return np.asarray(lon_acc, dtype=np.float64), np.asarray(lat_acc, dtype=np.float64)


def build_global_dataset(
    n_per_class: int = 2500,
    cache_dir: Path | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Сборка глобального датасета.

    Args:
        n_per_class: Количество точек на класс.
        cache_dir: Путь к директории для кэширования.
        rng: Генератор случайных чисел.

    Returns:
        DataFrame с глобальным датасетом.
    """

    cache_dir = cache_dir or (DATA_DIR / "natural_earth_cache")
    rng = rng or np.random.default_rng(RANDOM_STATE)
    land, coast = _load_ne_layers(cache_dir)
    land_union = land.geometry.union_all()
    prep_land = prep(land_union)
    coast_gdf = gpd.GeoDataFrame(geometry=coast.geometry, crs=coast.crs).to_crs(3857)

    rows: list[dict] = []

    def append_rows(lon, lat, labels):
        """Добавляет строки в DataFrame.

        Args:
            lon: Массив долгот.
            lat: Массив широт.
            labels: Массив меток классов.
        """

        rad_lat = np.radians(lat)
        rad_lon = np.radians(lon)
        d = _distances_km(lon, lat, coast_gdf)
        for i in range(len(lon)):
            lab = labels[i]
            rows.append(
                {
                    "latitude": float(lat[i]),
                    "longitude": float(lon[i]),
                    "sin_lat": float(math.sin(rad_lat[i])),
                    "cos_lat": float(math.cos(rad_lat[i])),
                    "sin_lon": float(math.sin(rad_lon[i])),
                    "cos_lon": float(math.cos(rad_lon[i])),
                    "distance_to_coast_km": float(d[i]),
                    "is_land": float(lab in ("NEAR_COAST", "COASTLINE")),
                    "target_class": lab,
                }
            )

    def take_ocean(pred_open: bool) -> None:
        """Генерирует океанические кандидаты.

        Args:
            pred_open: True, если предсказывается открытое море.
        """
        lon_sel, lat_sel, lab_sel = [], [], []
        batch = max(500, n_per_class // 2)
        guard = 0
        while len(lon_sel) < n_per_class and guard < 5000:
            guard += 1
            lo, la = _ocean_candidates(rng, prep_land, batch)
            dist = _distances_km(lo, la, coast_gdf)
            open_mask = dist > 20.0
            if pred_open:
                m = open_mask
                lab = np.full(m.sum(), "OPEN_SEA", dtype=object)
            else:
                m = ~open_mask
                lab = np.full(m.sum(), "COASTAL_SEA", dtype=object)
            lo, la = lo[m], la[m]
            for i in range(len(lo)):
                lon_sel.append(lo[i])
                lat_sel.append(la[i])
                lab_sel.append(str(lab[i]))
                if len(lon_sel) >= n_per_class:
                    break
        append_rows(
            np.asarray(lon_sel[:n_per_class], dtype=np.float64),
            np.asarray(lat_sel[:n_per_class], dtype=np.float64),
            lab_sel[:n_per_class],
        )

    def take_land(coastline_only: bool) -> None:
        """Генерирует сушевые кандидаты.
        Args:
            coastline_only: True, если предсказывается только береговая линия.
        """

        lon_sel, lat_sel, lab_sel = [], [], []
        batch = max(2000, n_per_class * 4)
        guard = 0
        while len(lon_sel) < n_per_class and guard < 8000:
            guard += 1
            lo, la = _land_candidates(rng, prep_land, batch)
            dist = _distances_km(lo, la, coast_gdf)
            if coastline_only:
                m = dist <= 1.0
                lab_arr = np.full(m.sum(), "COASTLINE", dtype=object)
            else:
                m = (dist > 1.0) & (dist <= 5.0)
                lab_arr = np.full(m.sum(), "NEAR_COAST", dtype=object)
            lo, la = lo[m], la[m]
            for i in range(len(lo)):
                lon_sel.append(lo[i])
                lat_sel.append(la[i])
                lab_sel.append(str(lab_arr[i]))
                if len(lon_sel) >= n_per_class:
                    break
        if len(lon_sel) < n_per_class:
            raise RuntimeError(
                "Недостаточно сушевых точек в полосе 0–1 км или 1–5 км. "
                "Уменьшите n_per_class или увеличьте число попыток в data_builder."
            )
        append_rows(
            np.asarray(lon_sel[:n_per_class], dtype=np.float64),
            np.asarray(lat_sel[:n_per_class], dtype=np.float64),
            lab_sel[:n_per_class],
        )

    take_ocean(pred_open=True)
    take_ocean(pred_open=False)
    take_land(coastline_only=True)
    take_land(coastline_only=False)

    return pd.DataFrame(rows)
