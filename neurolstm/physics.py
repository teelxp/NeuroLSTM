"""Physics-to-Loss: физические величины SI, без detach прогнозируемого давления.

Газовый баланс: (4.7)-(4.11). Входные/выходные расходы — наблюдения;
прогноз расхода дополнительной головы НЕ подменяет измеренную опору.
"""
from __future__ import annotations
from collections.abc import Mapping
import torch
from torch import Tensor


def gas_inventory(pressure_pa: Tensor, temperature_k: Tensor, z: Tensor,
                  volume_m3: Tensor, molar_mass: Tensor, gas_constant: float = 8.31446261815324) -> Tensor:
    """(4.8)-(4.9): запас газа, кг. Давление — абсолютное, температура — K."""
    # Положительность метаданных проверяется при чтении. Давление прогноза
    # не обрезаем: иначе нарушение скрывается и градиент в этой области исчезает.
    return pressure_pa * molar_mass * volume_m3 / (z * gas_constant * temperature_k)


def gas_mass_residual(prediction_pa: Tensor, meta: Mapping[str, Tensor],
                      gas_constant: float = 8.31446261815324) -> Tensor:
    """(4.10), результат [B,H] в кг/с; future-поля допустимы только в loss/eval."""
    current = gas_inventory(meta["pressure_now"], meta["temperature_now"], meta["z_now"],
                            meta["volume"], meta["molar_mass"], gas_constant)
    future = gas_inventory(prediction_pa, meta["temperature_future"], meta["z_future"],
                           meta["volume"], meta["molar_mass"], gas_constant)
    return meta["mean_mass_in"] - meta["mean_mass_out"] - (future - current) / meta["elapsed_seconds"]


def normalized_square(residual: Tensor, scale: float | Tensor, epsilon: float = 0.0) -> Tensor:
    """Шаги 3-4 PtL: нормированная среднеквадратичная невязка."""
    return (residual / (scale + epsilon)).square().mean()


def gas_loss(prediction_pa: Tensor, meta: Mapping[str, Tensor], cfg: dict,
             residual_scale: float) -> tuple[Tensor, Tensor]:
    residual = gas_mass_residual(prediction_pa, meta, cfg["gas_constant"])
    loss = cfg["mass_weight"] * normalized_square(residual, residual_scale, cfg["residual_epsilon_kg_s"])
    if cfg["pressure_limit_weight"]:
        lower, upper = cfg["pressure_min_pa"], cfg["pressure_max_pa"]
        # (4.40)-(4.41): именно мягкий штраф, не гарантированный инвариант.
        r = (torch.relu(prediction_pa - upper) + torch.relu(lower - prediction_pa)) / (upper - lower)
        loss = loss + cfg["pressure_limit_weight"] * r.square().mean()
    return loss, residual


def observed_mass_feature(p: Tensor, t: Tensor, z: Tensor, volume: Tensor,
                          molar: Tensor, elapsed: Tensor) -> Tensor:
    """Обратная разность dM/dt для (4.20), без использования будущих отсчетов."""
    inventory = gas_inventory(p, t, z, volume, molar)
    return (inventory[..., 1:] - inventory[..., :-1]) / elapsed


def hydro_balance(storage_now: Tensor, storage_next: Tensor, inflow: Tensor,
                  outflow: Tensor, dt_seconds: float | Tensor) -> Tensor:
    """(5.26): запас в м3, расходы м3/с, dt в секундах."""
    return storage_next - storage_now - dt_seconds * (inflow - outflow)


def muskingum_residual(out_next: Tensor, in_next: Tensor, in_now: Tensor,
                      out_now: Tensor, c1: float, c2: float, c3: float) -> Tensor:
    """(5.28); коэффициенты задаются предметной калибровкой."""
    if abs(c1+c2+c3-1.0) > 1e-6:
        raise ValueError("Коэффициенты Muskingum должны суммироваться в 1")
    return out_next - (c1*in_next + c2*in_now + c3*out_now)


def rating_curve(level_m: Tensor, c: float, h0: float, alpha: float) -> Tensor:
    """(5.11)/(5.25): расход из неотрицательной кривой расходов."""
    if c <= 0 or alpha <= 0:
        raise ValueError("C и alpha должны быть положительными")
    return c * torch.relu(level_m - h0).pow(alpha)


def active_power_residual(supply_w: Tensor, branch_w: Tensor, losses_w: Tensor) -> Tensor:
    """(5.43): независимая подведенная мощность минус нагрузки и потери."""
    return supply_w - branch_w.sum(dim=-1) - losses_w


def apparent_power_violation(active_w: Tensor, reactive_var: Tensor, limit_va: Tensor) -> Tensor:
    """(5.45)-(5.49): проверка режима, не обрезание прогноза спроса."""
    return torch.relu(torch.linalg.vector_norm(torch.stack([active_w, reactive_var], dim=-1), dim=-1) - limit_va)
