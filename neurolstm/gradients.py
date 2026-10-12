"""Раздельные градиенты и их согласование: (3.93)-(3.103).

Сохраняется исходная регуляризованная проекция с epsilon и приоритет data.
Это не PCGrad: положительные скалярные произведения также проецируются.
При полном батче сначала суммируются ИСХОДНЫЕ градиенты всех микробатчей,
и лишь затем однократно применяется (3.98)-(3.100).
"""
from __future__ import annotations
import math
from collections.abc import Sequence
import torch
from torch import Tensor, nn


def flattened_gradient(loss: Tensor, parameters: Sequence[nn.Parameter], *, retain_graph: bool) -> Tensor:
    if not loss.requires_grad:
        return torch.cat([torch.zeros_like(p).reshape(-1) for p in parameters])
    grads = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if g is None else g.detach()).reshape(-1)
                      for p, g in zip(parameters, grads)])


def dot(a: Tensor, b: Tensor) -> Tensor:
    # Более точная редукция, без изменения размерности и устройства параметров.
    return (a.double() * b.double()).sum().to(a.dtype)


def projection(v: Tensor, basis: Tensor, epsilon: float) -> Tensor:
    return dot(v, basis) / (dot(basis, basis) + epsilon) * basis


def combine_gradients(g_data: Tensor, g_phys: Tensor, g_sem: Tensor, g_reg: Tensor,
                      beta_phys: float, beta_sem: float, lambda_reg: float,
                      epsilon: float = 1e-12, mode: str = "dissertation") -> tuple[Tensor, dict]:
    if mode == "dissertation":
        gp = g_phys - projection(g_phys, g_data, epsilon)  # (3.98)
        gs = g_sem - projection(g_sem, g_data, epsilon) - projection(g_sem, gp, epsilon)  # (3.99)
    elif mode == "sum":
        gp, gs = g_phys, g_sem  # (3.97), контрольная абляция.
    else:
        raise ValueError("Неизвестный способ согласования градиентов")
    result = g_data + beta_phys*gp + beta_sem*gs + lambda_reg*g_reg
    stats = {
        "norm_data": float(g_data.norm()), "norm_phys": float(g_phys.norm()),
        "norm_sem": float(g_sem.norm()), "norm_phys_projected": float(gp.norm()),
        "norm_sem_projected": float(gs.norm()),
        "dot_data_phys_projected": float(dot(g_data, gp)),
        "dot_data_sem_projected": float(dot(g_data, gs)),
        "dot_phys_sem_projected": float(dot(gp, gs)),
    }
    return result, stats


def set_gradient(parameters: Sequence[nn.Parameter], gradient: Tensor) -> None:
    if not torch.isfinite(gradient).all():
        raise FloatingPointError("Итоговый градиент содержит NaN/Inf; обновление отменено")
    cursor = 0
    for p in parameters:
        n = p.numel()
        p.grad = gradient[cursor:cursor+n].view_as(p).clone()
        cursor += n
    if cursor != gradient.numel():
        raise ValueError("Размер градиента не совпадает с числом параметров")


def ramp(epoch_index: int, maximum: float, tau: float) -> float:
    """(3.101)-(3.102); q=0 -> beta=0."""
    return maximum * (-math.expm1(-epoch_index / tau))


def learning_rate(epoch_index: int, cfg: dict) -> float:
    """Явный планировщик. При cosine_clamped после T_max сохраняется eta_min."""
    if cfg["schedule"] == "constant":
        return cfg["lr"]
    phase = min(epoch_index, cfg["cosine_steps"])
    if cfg["schedule"] == "cosine_restarts":
        phase = epoch_index % cfg["cosine_steps"]
    return cfg["eta_min"] + .5*(cfg["lr"]-cfg["eta_min"])*(1+math.cos(math.pi*phase/cfg["cosine_steps"]))
