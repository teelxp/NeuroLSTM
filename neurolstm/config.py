"""Конфигурация эксперимента. Неопределенные в диссертации числа не маскируются."""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path
from typing import Any

DEFAULT: dict[str, Any] = {
    "profile": "explicit_configuration",
    "data": {
        "source": "csv", "path": None, "query_file": None,
        "dsn_environment": "NEUROLSTM_DSN", "column_map": {},
        "provenance": None,
        "pressure_unit": "bar", "pressure_reference": "absolute",
        "atmospheric_pressure_pa": None, "temperature_unit": "C",
        "flow_unit": "t/h", "humidity_description": None,
        "sample_period_seconds": None, "period_tolerance": 0.01,
        "window": 30, "horizons": [1], "stride": 1,
        "train_fraction": 0.7, "validation_fraction": 0.1,
        "gap_rows": 0, "max_rows": None,
        "interval_quadrature": "trapezoid",
        "labels": "column", "low_pressure_pa": None, "high_pressure_pa": None,
    },
    "physics": {
        "volume_m3": None, "molar_mass_kg_mol": None,
        "compressibility": None, "gas_constant": 8.31446261815324,
        "residual_scale_kg_s": None, "residual_epsilon_kg_s": 1e-12,
        "mass_weight": 1.0, "pressure_limit_weight": 0.0,
        "pressure_min_pa": None, "pressure_max_pa": None,
        "consistency_tolerance_kg_s": None,
    },
    "model": {
        "hidden_size": 100, "layers": 3,
        "sensor_mode": "three_base_features",
        "predict_flow": True, "classify": True,
        "dropout": 0.0, "layer_norm": False,
    },
    "training": {
        "epochs": 25000, "seed": 42, "device": "auto",
        "threads": 1, "deterministic": True,
        "optimizer": "AdamW", "lr": 0.0005, "eta_min": 0.000001,
        "cosine_steps": 100, "schedule": "cosine_clamped",
        "weight_decay": 0.0, "l2_weight": 0.0,
        "regression_loss": "mse", "pressure_weight": 1.0,
        "flow_weight": 1.0, "classification_weight": 1.0,
        "gradient_mode": "dissertation", "projection_epsilon": 1e-12,
        "beta_phys_max": 1.0, "tau_phys": 1000.0,
        "beta_sem_max": 1.0, "tau_sem": 1000.0,
        "batch_mode": "full", "micro_batch_size": 64,
        "gradient_clip_norm": None, "validation_every": 1,
        "checkpoint_every": 100, "select_by": "pressure_mae",
    },
    "metrics": {
        "pressure_relative_tolerance": 0.1,
        "pressure_denominator_floor_pa": 1.0,
        "classification_threshold": 0.5,
    },
    "semantics": {
        "path": None, "embedding_size": 0, "state_components": 0,
        "weights": {"reconstruction": 0.0, "cause": 0.0, "phase": 0.0,
                    "graph": 0.0, "entropy": 1.0, "wave": 0.0},
        "time_gamma": 0.1, "memory_tau": 5.0,
        "minkowski_alpha": 1.0, "spatial_weights": [],
        "lambda_separation": 1.0, "graph_threshold": 0.5,
        "graph_target": 0.8, "graph_key_fraction": 0.1,
        "kl_mode": "log_softmax", "epsilon": 1e-8,
        "wave_speed": 1.0, "wave_damping": 0.0,
        "wave_dt": 1.0, "wave_dx": 1.0,
        "wave_intensity_weight": 1.0,
    },
    "clustering": {
        "enabled": False, "clusters": 3, "pca_components": 8,
        "weights": [0.5, 0.0, 0.5], "restarts": 10,
        "epsilon": 1e-8,
    },
}


def _merge(base: dict, update: dict, prefix: str = "") -> dict:
    out = copy.deepcopy(base)
    for key, value in update.items():
        if key not in out:
            raise ValueError(f"Неизвестный параметр: {prefix}{key}")
        if isinstance(out[key], dict) and key not in {"column_map"}:
            if not isinstance(value, dict):
                raise ValueError(f"{prefix}{key}: требуется объект JSON")
            out[key] = _merge(out[key], value, prefix + key + ".")
        else:
            out[key] = value
    return out


def validate(cfg: dict) -> dict:
    # Конфигурация должна сохраняться без NaN/Infinity и скрытых Python-объектов.
    json.dumps(cfg, allow_nan=False)
    d, p, m, t, s = (cfg[k] for k in ("data", "physics", "model", "training", "semantics"))
    if d["source"] not in {"csv", "postgres"}:
        raise ValueError("data.source: csv или postgres")
    if d["pressure_reference"] not in {"absolute", "gauge"}:
        raise ValueError("Укажите absolute или gauge для давления")
    if d["pressure_reference"] == "gauge" and not d["atmospheric_pressure_pa"]:
        raise ValueError("Для избыточного давления требуется atmospheric_pressure_pa")
    if d["pressure_unit"] not in {"Pa", "bar", "MPa"} or d["temperature_unit"] not in {"K", "C"}:
        raise ValueError("Неподдерживаемые единицы давления/температуры")
    if d["flow_unit"] not in {"kg/s", "t/h"}:
        raise ValueError("flow_unit: kg/s или t/h")
    if not 0 < d["train_fraction"] < 1 or not 0 <= d["validation_fraction"] < 1:
        raise ValueError("Некорректные доли разбиения")
    if d["train_fraction"] + d["validation_fraction"] >= 1:
        raise ValueError("Необходимо оставить независимый тестовый интервал")
    if d["labels"] not in {"column", "pressure_thresholds"}:
        raise ValueError("labels: column или pressure_thresholds")
    if d["labels"] == "pressure_thresholds" and all(d[k] is None for k in ("low_pressure_pa", "high_pressure_pa")):
        raise ValueError("Пороговые метки требуют явно заданных физических порогов")
    hs = d["horizons"]
    if not hs or hs != sorted(set(hs)) or any(not isinstance(h, int) or isinstance(h, bool) or h < 1 for h in hs):
        raise ValueError("horizons: возрастающий список положительных целых чисел")
    for key in ("window", "stride"):
        if not isinstance(d[key], int) or d[key] < 1:
            raise ValueError(f"{key}: требуется положительное целое число")
    if d["sample_period_seconds"] is not None and d["sample_period_seconds"] <= 0:
        raise ValueError("sample_period_seconds должен быть положительным")
    if not isinstance(d["gap_rows"], int) or d["gap_rows"] < 0 or not 0 <= d["period_tolerance"] < 1:
        raise ValueError("Некорректные gap_rows или period_tolerance")
    if d["interval_quadrature"] not in {"trapezoid", "left", "right"}:
        raise ValueError("Неизвестная квадратурная формула среднего расхода")
    for key in ("volume_m3", "molar_mass_kg_mol", "compressibility", "residual_scale_kg_s", "consistency_tolerance_kg_s"):
        if p[key] is not None and (not math.isfinite(p[key]) or p[key] <= 0):
            raise ValueError(f"physics.{key} должен быть положительным")
    if p["gas_constant"] <= 0 or p["residual_epsilon_kg_s"] < 0:
        raise ValueError("Некорректные константы физического функционала")
    if p["pressure_limit_weight"] > 0:
        if p["pressure_min_pa"] is None or p["pressure_max_pa"] is None or p["pressure_min_pa"] >= p["pressure_max_pa"]:
            raise ValueError("Штраф давления требует согласованных физических границ")
    if m["sensor_mode"] not in {"three_base_features", "seven_telemetry_features"}:
        raise ValueError("Неизвестный sensor_mode")
    if m["hidden_size"] < 1 or m["layers"] < 1 or not 0 <= m["dropout"] < 1:
        raise ValueError("Некорректная архитектура")
    if t["optimizer"] not in {"AdamW", "SGD"}:
        raise ValueError("optimizer: AdamW или SGD")
    if t["schedule"] not in {"cosine_clamped", "cosine_restarts", "constant"}:
        raise ValueError("Неизвестный планировщик")
    if t["gradient_mode"] not in {"dissertation", "sum"}:
        raise ValueError("gradient_mode: dissertation или sum")
    if t["batch_mode"] not in {"full", "mini"}:
        raise ValueError("batch_mode: full или mini")
    if t["regression_loss"] not in {"mse", "mae"}:
        raise ValueError("regression_loss: mse или mae")
    for key in ("epochs", "micro_batch_size", "cosine_steps", "validation_every", "checkpoint_every", "threads"):
        if not isinstance(t[key], int) or t[key] < 1:
            raise ValueError(f"training.{key}: требуется положительное целое число")
    if not 0 <= t["eta_min"] <= t["lr"] or t["lr"] <= 0:
        raise ValueError("Необходимо 0 <= eta_min <= lr и lr > 0")
    if t["tau_phys"] <= 0 or t["tau_sem"] <= 0 or t["projection_epsilon"] <= 0:
        raise ValueError("Параметры расписания и epsilon должны быть положительными")
    if t["weight_decay"] > 0 and t["l2_weight"] > 0:
        raise ValueError("Выберите явный L2 или decoupled weight decay, чтобы не удвоить регуляризацию")
    if t["select_by"] not in {"pressure_mae", "fixed_weight_loss"}:
        raise ValueError("Некорректный select_by")
    if s["path"] is not None and (s["embedding_size"] < 1 or s["state_components"] < 1):
        raise ValueError("Семантический режим требует размерностей embedding_size и state_components")
    if s["path"] is None and s["embedding_size"] != 0:
        raise ValueError("embedding_size задан, но отсутствует семантический набор")
    if s["kl_mode"] not in {"log_softmax", "dissertation_epsilon"}:
        raise ValueError("Неизвестный вариант KL")
    if not 0 <= s["graph_threshold"] < 1 or not 0 < s["graph_key_fraction"] <= 1:
        raise ValueError("Для неотрицательных степеней нужен graph_threshold >= 0")
    if cfg["clustering"]["enabled"] and t["batch_mode"] == "full":
        raise ValueError("Кластерные мини-батчи не меняют полный градиент; выберите batch_mode=mini")
    for section, keys in [(p,["mass_weight", "pressure_limit_weight"]),
                          (t,["pressure_weight", "flow_weight", "classification_weight", "beta_phys_max", "beta_sem_max", "l2_weight", "weight_decay"]),
                          (s["weights"], list(s["weights"]))]:
        if any(not math.isfinite(section[k]) or section[k] < 0 for k in keys):
            raise ValueError("Все веса должны быть конечными и неотрицательными")
    if cfg["metrics"]["pressure_denominator_floor_pa"] <= 0 or cfg["metrics"]["pressure_relative_tolerance"] <= 0:
        raise ValueError("Порог относительной ошибки и масштаб знаменателя должны быть положительными")
    if t["gradient_clip_norm"] is not None and t["gradient_clip_norm"] <= 0:
        raise ValueError("gradient_clip_norm должен быть положительным")
    if s["epsilon"] <= 0 or s["memory_tau"] <= 0 or s["time_gamma"] < 0 or s["lambda_separation"] < 0:
        raise ValueError("Некорректные параметры семантической модели")
    if not 0 <= s["graph_target"] <= 1:
        raise ValueError("graph_target должен находиться в [0,1]")
    if not 0 <= cfg["metrics"]["classification_threshold"] <= 1:
        raise ValueError("Порог классификации должен находиться в [0,1]")
    return cfg


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8-sig") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("Конфигурация должна быть объектом JSON")
    return validate(_merge(DEFAULT, raw))
