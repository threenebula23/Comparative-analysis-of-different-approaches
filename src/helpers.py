from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

# Константы
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PLOT_DIR = PROJECT_ROOT / "plot"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

RANDOM_STATE = 42
GLOBAL_DATASET_PATH = DATA_DIR / "global_coastal_points.parquet"
PROCESSED_PATH = DATA_DIR / "processed_splits.npz"
METRICS_JSON_PATH = DATA_DIR / "metrics_all_models.json"

CLASS_NAMES = ["OPEN_SEA", "COASTAL_SEA", "NEAR_COAST", "COASTLINE"]


def init_notebook_path() -> Path:
    """Вызывать из первой ячейки ноутбука: добавляет каталог `src/` в sys.path.

    Returns:
        Путь к каталогу проекта.
    """

    for p in (Path.cwd(), Path.cwd().parent):
        hp = p / "src" / "helpers.py"
        if hp.is_file():
            sd = str((p / "src").resolve())
            if sd not in sys.path:
                sys.path.insert(0, sd)
            return p.resolve()
    raise RuntimeError("Не найден каталог проекта (ожидается src/helpers.py рядом с notebooks/).")


def ensure_dirs() -> None:
    """Создает директории для данных, графиков и ноутбуков."""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


def classification_metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Вычисляет метрики для классификации.

    Args:
        y_true: Истинные метки классов.
        y_pred: Предсказанные метки классов.

    Returns:
        Метрики для классификации.
        - accuracy: Точность.
        - precision_macro: Точность (макро-среднее).
        - recall_macro: Полнота (макро-среднее).
        - f1_macro: F1-мера (макро-среднее).
        - precision_micro: Точность (микро-среднее).
        - recall_micro: Полнота (микро-среднее).
        - f1_micro: F1-мера (микро-среднее).
    """

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_micro": float(
            precision_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "recall_micro": float(
            recall_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
    }


def print_report(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Выводит отчет о классификации.

    Args:
        y_true: Истинные метки классов.
        y_pred: Предсказанные метки классов.
    """

    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))


def append_metrics_store(model_name: str, metrics: dict, test_y_pred: np.ndarray | None = None) -> None:
    """Добавляет строку в JSON-хранилище метрик (перезапись файла целиком при необходимости объединения).

    Args:
        model_name: Имя модели.
        metrics: Метрики для модели.
        test_y_pred: Предсказанные метки классов для тестового набора.
    """

    ensure_dirs()
    store: dict = {}
    if METRICS_JSON_PATH.exists():
        store = json.loads(METRICS_JSON_PATH.read_text(encoding="utf-8"))
    entry = {"metrics": metrics}
    if test_y_pred is not None:
        entry["pred_shape"] = list(test_y_pred.shape)
    store[model_name] = entry
    METRICS_JSON_PATH.write_text(json.dumps(store, indent=2, ensure_ascii=False), encoding="utf-8")


def load_metrics_store() -> dict:
    """Загружает хранилище метрик из JSON-файла.

    Returns:
        Хранилище метрик.
        - model_name: Имя модели.
        - metrics: Метрики для модели.
        - pred_shape: Форма предсказанных меток классов для тестового набора.
    """

    if not METRICS_JSON_PATH.exists():
        return {}
    return json.loads(METRICS_JSON_PATH.read_text(encoding="utf-8"))


def load_xy_from_processed() -> tuple:
    """Загружает данные из обработанных данных.

    Returns:
        - X_train: Матрица признаков для обучающего набора.
        - X_val: Матрица признаков для валидационного набора.
        - X_test: Матрица признаков для тестового набора.
        - y_train: Истинные метки классов для обучающего набора.
        - y_val: Истинные метки классов для валидационного набора.
        - y_test: Истинные метки классов для тестового набора.
        - fc: Список признаков.
    """

    z = np.load(PROCESSED_PATH, allow_pickle=True)
    fc = None
    if "feature_columns" in z.files:
        arr = z["feature_columns"]
        fc = arr.tolist() if getattr(arr, "ndim", 0) > 0 else [str(arr.item())]
    return (
        z["X_train"],
        z["X_val"],
        z["X_test"],
        z["y_train"],
        z["y_val"],
        z["y_test"],
        fc,
    )


def feature_columns_default() -> list[str]:
    """Возвращает список признаков по умолчанию.
    
    Returns:
        Список признаков по умолчанию.
    """
    return [
        "latitude",
        "longitude",
        "sin_lat",
        "cos_lat",
        "sin_lon",
        "cos_lon",
        "distance_to_coast_km",
        "is_land",
    ]
