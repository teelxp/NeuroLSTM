"""Чтение CSV/PostgreSQL, единицы SI, хронологическое разбиение, ленивые окна.

Ни масштабирование, ни PCA не подгоняются по validation/test. Окна разных
выборок не пересекаются даже входными строками. Строки разных объектов и
временные разрывы не соединяются. Исходные измерения не синтезируются.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset

FEATURES = ["pressure", "temperature", "flow_rate", "external_temp", "humidity", "mass_in", "mass_out"]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def load_frame(cfg: dict, *, inference: bool = False) -> tuple[pd.DataFrame, dict]:
    d, phys = cfg["data"], cfg["physics"]
    if d["source"] == "csv":
        if not d["path"]:
            raise ValueError("Укажите data.path с исходными измерениями")
        frame = pd.read_csv(d["path"], nrows=d["max_rows"])
        source_hash = sha256_file(d["path"])
    else:
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError('Для PostgreSQL установите pip install "psycopg[binary]>=3.2,<4"') from exc
        dsn = os.environ.get(d["dsn_environment"])
        if not dsn or not d["query_file"]:
            raise ValueError("Нужны DSN в переменной окружения и data.query_file")
        query = Path(d["query_file"]).read_text(encoding="utf-8")
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute("SET TRANSACTION READ ONLY")
                cur.execute(query)
                frame = pd.DataFrame(cur.fetchall(), columns=[col.name for col in cur.description])
        source_hash = None  # хэш фактического результата рассчитывается ниже.
    frame = frame.rename(columns=d["column_map"]).copy()
    if frame.columns.duplicated().any():
        raise ValueError("column_map образует повторяющиеся имена столбцов")
    if "object_id" not in frame:
        frame["object_id"] = "single_object"
    if "timestamp" not in frame:
        raise ValueError("Отсутствует timestamp. Укажите переименование через column_map")
    if frame["object_id"].isna().any():
        raise ValueError("object_id содержит пустые значения")
    frame["object_id"] = frame["object_id"].astype(str)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    if frame["timestamp"].isna().any():
        raise ValueError("timestamp содержит пустые значения")
    if frame.duplicated(["object_id", "timestamp"]).any():
        raise ValueError("Обнаружены дубликаты object_id/timestamp; требуется явное правило разрешения")
    for name, key in [("volume_m3", "volume_m3"), ("molar_mass_kg_mol", "molar_mass_kg_mol"), ("compressibility", "compressibility")]:
        if name not in frame:
            if phys[key] is None:
                raise ValueError(f"Нет {name}: предоставьте столбец или физически обоснованное значение в physics.{key}")
            frame[name] = phys[key]
    required = FEATURES + ["volume_m3", "molar_mass_kg_mol", "compressibility"]
    missing = [col for col in required if col not in frame]
    if missing:
        raise ValueError("Отсутствуют измерения: " + ", ".join(missing) + ". mass_out не заменяется mass_in.")
    for col in required:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    if not np.isfinite(frame[required].to_numpy(dtype=float)).all():
        raise ValueError("NaN/Inf в обязательной телеметрии. Данные не заполняются будущими значениями: исправьте источник или выделите непрерывные полные интервалы")
    pressure_factor = {"Pa": 1., "bar": 1e5, "MPa": 1e6}[d["pressure_unit"]]
    frame["pressure"] *= pressure_factor
    if d["pressure_reference"] == "gauge":
        frame["pressure"] += d["atmospheric_pressure_pa"]
    if d["temperature_unit"] == "C":
        frame["temperature"] += 273.15
        frame["external_temp"] += 273.15
    if d["flow_unit"] == "t/h":
        frame[["mass_in", "mass_out"]] *= 1000./3600.
    for col in ["pressure", "temperature", "external_temp", "volume_m3", "molar_mass_kg_mol", "compressibility"]:
        if (frame[col] <= 0).any():
            raise ValueError(f"{col} должен быть положительным после перевода в SI")
    # flow_rate — самостоятельный объемный/режимный признак; он не используется
    # вместо массового расхода. Его единицы фиксируются в provenance пользователя.
    if cfg["model"]["classify"] and not inference:
        if d["labels"] == "column":
            if "accident" not in frame:
                raise ValueError("Требуется независимый столбец accident (0/1), либо явный labels=pressure_thresholds")
            frame["accident"] = pd.to_numeric(frame["accident"], errors="raise")
            if not frame["accident"].isin([0,1]).all():
                raise ValueError("accident должен содержать только 0/1")
        else:
            flag = np.zeros(len(frame), dtype=bool)
            if d["low_pressure_pa"] is not None:
                flag |= frame["pressure"].to_numpy() < d["low_pressure_pa"]
            if d["high_pressure_pa"] is not None:
                flag |= frame["pressure"].to_numpy() > d["high_pressure_pa"]
            frame["accident"] = flag.astype(float)
    if "accident" not in frame:
        frame["accident"] = 0.0  # не используется, когда классификационная голова выключена.
    period = d["sample_period_seconds"]
    if period is None or not np.isfinite(period) or period <= 0:
        raise ValueError("Требуется явно заданный data.sample_period_seconds")
    frame = frame.sort_values(["object_id", "timestamp"], kind="stable").reset_index(drop=True)
    frame["time_ns"] = frame["timestamp"].astype("int64")
    for oid, group in frame.groupby("object_id", sort=False):
        for key in ("volume_m3", "molar_mass_kg_mol"):
            if not np.allclose(group[key], group[key].iloc[0], rtol=1e-9, atol=0):
                raise ValueError(f"{oid}: {key} меняется. Формула (4.9) требует фиксированных параметров участка; разделите режимы")
    dt = frame.groupby("object_id", sort=False)["time_ns"].diff().to_numpy()/1e9
    same_period = np.isfinite(dt) & (np.abs(dt-period) <= period*d["period_tolerance"])
    frame["segment"] = (~same_period).cumsum()
    inventory = (frame["pressure"]*frame["molar_mass_kg_mol"]*frame["volume_m3"] /
                 (frame["compressibility"]*phys["gas_constant"]*frame["temperature"]))
    frame["inventory_kg"] = inventory
    frame["mass_rate"] = frame.groupby("segment", sort=False)["inventory_kg"].diff() / dt
    canonical_hash = hashlib.sha256(pd.util.hash_pandas_object(frame[required+["object_id", "time_ns", "accident"]], index=False).values.tobytes()).hexdigest()
    audit = {"rows": len(frame), "objects": int(frame["object_id"].nunique()),
             "segments": int(frame["segment"].nunique()), "source_sha256": source_hash,
             "canonical_data_sha256": canonical_hash, "provenance": d["provenance"],
             "units_internal": {"pressure":"Pa absolute", "temperature":"K", "mass_flow":"kg/s"},
             "labels": d["labels"] if cfg["model"]["classify"] else "not_used",
             "input_truncated_by_max_rows": d["max_rows"] is not None}
    if not len(frame):
        raise ValueError("Источник не содержит строк")
    return frame, audit


@dataclass
class Scaling:
    minimum: np.ndarray
    span: np.ndarray
    constant: np.ndarray
    mass_rate_scale: float
    residual_scale: float

    @classmethod
    def fit(cls, frame: pd.DataFrame, mask: np.ndarray, physics: dict) -> "Scaling":
        x = frame.loc[mask, FEATURES].to_numpy(dtype=np.float64)
        if len(x) < 2:
            raise ValueError("Недостаточно обучающих строк")
        scaler = MinMaxScaler().fit(x)
        span = scaler.data_range_.copy()
        const = span <= 1e-12
        # Постоянный вход отображается в 0 (4.13), но цель должна иметь диапазон.
        if const[0]:
            raise ValueError("Давление постоянно на train: требуется отдельный физический диапазон/другая постановка")
        rate = frame.loc[mask, "mass_rate"].to_numpy(float)
        rate = rate[np.isfinite(rate)]
        scale_rate = max(float(np.sqrt(np.mean(rate**2))) if len(rate) else 0.0, 1e-12)
        # Выбор масштаба — явно документированная инженерная конкретизация PtL.
        given = physics["residual_scale_kg_s"]
        rscale = float(given) if given is not None else max(float(np.sqrt(np.mean(x[:,5:7]**2))), 1e-12)
        return cls(scaler.data_min_, span, const, scale_rate, rscale)

    def transform(self, x: np.ndarray) -> np.ndarray:
        out = (x-self.minimum)/np.where(self.constant, 1.0, self.span)
        out[...,self.constant] = 0
        return out.astype(np.float32)

    def to_dict(self) -> dict:
        return {"minimum":self.minimum.tolist(), "span":self.span.tolist(),
                "constant":self.constant.tolist(), "mass_rate_scale":self.mass_rate_scale,
                "residual_scale":self.residual_scale, "feature_order":FEATURES}

    @classmethod
    def from_dict(cls, value: dict) -> "Scaling":
        if value["feature_order"] != FEATURES:
            raise ValueError("Порядок признаков checkpoint несовместим")
        return cls(np.array(value["minimum"]), np.array(value["span"]),
                   np.array(value["constant"], dtype=bool), value["mass_rate_scale"], value["residual_scale"])

    def inverse_tensor(self, value: torch.Tensor, feature: int) -> torch.Tensor:
        # Для target_outflow при постоянном train отказ проверяется отдельно.
        return value * float(self.span[feature]) + float(self.minimum[feature])


def split_rows(frame: pd.DataFrame, cfg: dict) -> tuple[np.ndarray, dict]:
    times = np.sort(frame["time_ns"].unique())
    n = len(times)
    first = int(n*cfg["train_fraction"])
    second = int(n*(cfg["train_fraction"]+cfg["validation_fraction"]))
    if first < 1 or second >= n:
        raise ValueError("Недостаточно временных точек для заданного разбиения")
    labels = np.searchsorted([times[first], times[second]], frame["time_ns"].to_numpy(), side="right")
    # При validation_fraction=0 обе границы совпадают: train=0, test=2.
    return labels, {"train_end_exclusive": pd.Timestamp(times[first], tz="UTC").isoformat(),
                    "test_start_inclusive": pd.Timestamp(times[second], tz="UTC").isoformat(),
                    "raw_row_counts": {name:int((labels==i).sum()) for i,name in enumerate(["train","validation","test"])}}


def group_tensor(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros((len(x),3,3), dtype=np.float32)
    mask = np.zeros_like(out)
    out[:,0,:] = x[:,:3]
    out[:,1,:2] = x[:,3:5]
    out[:,2,:2] = x[:,5:7]
    mask[:,0,:] = 1
    mask[:,1,:2] = 1
    mask[:,2,:2] = 1
    return out, mask


class GasWindows(Dataset):
    def __init__(self, frame: pd.DataFrame, cfg: dict, scaling: Scaling,
                 row_labels: np.ndarray, split: int, semantic_store: Any = None):
        self.frame, self.cfg, self.scaling = frame, cfg, scaling
        self.semantic_store = semantic_store
        self.raw = frame[FEATURES].to_numpy(dtype=np.float64)
        self.normalized = scaling.transform(self.raw)
        self.ns = frame["time_ns"].to_numpy(dtype=np.int64)
        self.inventory = frame["inventory_kg"].to_numpy(float)
        self.rates = (frame["mass_rate"].to_numpy(float)/scaling.mass_rate_scale).astype(np.float32)
        self.z = frame["compressibility"].to_numpy(float)
        self.vol = frame["volume_m3"].to_numpy(float)
        self.molar = frame["molar_mass_kg_mol"].to_numpy(float)
        self.accident = frame["accident"].to_numpy(dtype=np.float32)
        self.ids = frame["object_id"].to_numpy(str)
        self.horizons = np.array(cfg["horizons"], dtype=int)
        self.origins: list[int] = []
        self.cumulative = np.zeros((len(frame),2), dtype=np.float64)
        # Каждая независимая сумма интеграла начинается заново на сегменте.
        for _, part in frame.groupby("segment", sort=False):
            idx = part.index.to_numpy()
            f = self.raw[idx,5:7]
            dt = np.diff(self.ns[idx]).astype(float)/1e9
            quadrature = cfg["interval_quadrature"]
            rates = .5*(f[:-1]+f[1:]) if quadrature == "trapezoid" else f[:-1] if quadrature == "left" else f[1:]
            self.cumulative[idx[1:]] = np.cumsum(rates*dt[:,None], axis=0)
            eligible = idx[row_labels[idx]==split]
            gap = cfg["gap_rows"] if split > 0 else 0
            if gap:
                eligible = eligible[gap:]
            # Все признаки и target должны находиться в одной выборке.
            if len(eligible) <= cfg["window"]+self.horizons[-1]:
                continue
            start, end = eligible[0], eligible[-1]
            for origin in range(start+cfg["window"], end-self.horizons[-1]+1, cfg["stride"]):
                if semantic_store is None or semantic_store.has(self.ids[origin], self.ns[origin]):
                    self.origins.append(origin)
        self.origins_array = np.array(self.origins, dtype=int)

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, index: int) -> dict:
        i = self.origins[index]
        start = i-self.cfg["window"]+1
        targets = i+self.horizons
        x, mask = group_tensor(self.normalized[start:i+1])
        elapsed = (self.ns[targets]-self.ns[i]).astype(float)/1e9
        mean_flows = (self.cumulative[targets]-self.cumulative[i])/elapsed[:,None]
        meta = {"pressure_now": np.array([self.raw[i,0]]),
                "temperature_now": np.array([self.raw[i,1]]), "z_now": np.array([self.z[i]]),
                "temperature_future": self.raw[targets,1], "z_future":self.z[targets],
                "volume":np.array([self.vol[i]]), "molar_mass":np.array([self.molar[i]]),
                "mean_mass_in":mean_flows[:,0], "mean_mass_out":mean_flows[:,1],
                "elapsed_seconds":elapsed}
        result = {"x":torch.from_numpy(x), "mask":torch.from_numpy(mask),
                  "mass_rate":torch.from_numpy(self.rates[start:i+1,None]),
                  "target_pressure":torch.tensor(self.raw[targets,0],dtype=torch.float64),
                  "target_flow":torch.tensor(self.raw[targets,6],dtype=torch.float64),
                  "target_pressure_normalized":torch.from_numpy(self.normalized[targets,0]),
                  "target_flow_normalized":torch.from_numpy(self.normalized[targets,6]),
                  "accident":torch.from_numpy(self.accident[targets]),
                  "meta":{k:torch.tensor(v,dtype=torch.float64) for k,v in meta.items()},
                  "origin_index":torch.tensor(i), "origin_time_ns":torch.tensor(self.ns[i]),
                  "target_time_ns":torch.from_numpy(self.ns[targets]), "object_id":self.ids[i]}
        if self.semantic_store is not None:
            result["semantics"] = self.semantic_store.get(self.ids[i], self.ns[i])
        return result

    def used_rows(self) -> set[int]:
        rows: set[int] = set()
        for i in self.origins:
            rows.update(range(i-self.cfg["window"], i+int(self.horizons[-1])+1))
        return rows


def prepare(cfg: dict, scaling: Scaling | None = None) -> tuple[dict[str,GasWindows], Scaling, dict]:
    frame, audit = load_frame(cfg)
    labels, split_audit = split_rows(frame, cfg["data"])
    scaling = scaling or Scaling.fit(frame, labels==0, cfg["physics"])
    if cfg["model"]["predict_flow"] and scaling.constant[6]:
        raise ValueError("Выходной расход постоянен на train; отключите отдельную регрессию расхода или используйте данные с вариативностью")
    semantic_store = None
    if cfg["semantics"]["path"] is not None:
        from .semantics import SemanticStore
        semantic_store = SemanticStore(cfg["semantics"]["path"], cfg["semantics"])
        audit["semantic_data_sha256"] = sha256_file(cfg["semantics"]["path"])
    datasets = {name:GasWindows(frame, cfg["data"], scaling, labels, i, semantic_store)
                for i,name in enumerate(["train","validation","test"])}
    for name in ["train","test"]+(["validation"] if cfg["data"]["validation_fraction"] else []):
        if not len(datasets[name]):
            raise ValueError(f"Нет окон {name}: проверьте длину ряда, период, window/horizons и покрытие семантического файла")
    audit.update(split_audit)
    audit["windows"] = {key:len(ds) for key,ds in datasets.items()}
    audit["window"] = cfg["data"]["window"]
    audit["horizons"] = cfg["data"]["horizons"]
    audit["interval_quadrature"] = cfg["data"]["interval_quadrature"]
    audit["split_policy"] = "global_chronological_disjoint_complete_windows_with_past_derivative"
    return datasets, scaling, audit
