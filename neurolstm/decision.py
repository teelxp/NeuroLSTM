"""Расчетное ядро (4.35)-(4.57). Не осуществляет управление оборудованием.

Сценарии последствий, неопределенность и основания действий поступают извне.
Отсутствующие проверки не заменяются автоматически высокими оценками.
В (4.47) в PDF дублированы веса; используем отдельно отмеченное определение
Q_src из формулы (5.63). См. docs/IMPLEMENTATION_MAP.md.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


CRITERIA = ("stat", "phys", "sem")


def statistical_score(
    confirmed_errors: list[float],
    pressure_scale: float,
    uncertainty: float,
    data_quality: float,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """(4.35)-(4.37): только ошибки УЖЕ подтвержденных прошлых прогнозов."""
    if len(weights) != 3 or any(not math.isfinite(v) or v < 0 for v in weights):
        raise ValueError("Нужны три конечных неотрицательных веса")
    if not confirmed_errors:
        raise ValueError("Нет подтвержденных ретроспективных прогнозов")
    if (not math.isfinite(pressure_scale) or pressure_scale <= 0
            or not 0 <= uncertainty <= 1 or not 0 <= data_quality <= 1):
        raise ValueError("Некорректные исходные оценки")
    errors = np.asarray(confirmed_errors, dtype=float)
    if not np.isfinite(errors).all():
        raise ValueError("Некорректные ошибки")
    loss = (
        weights[0] * float(np.mean((errors / pressure_scale) ** 2))
        + weights[1] * uncertainty
        + weights[2] * (1 - data_quality)
    )
    return math.exp(-loss)


def consistency_assessment(
    scores: dict[str, float | None],
    quality: dict[str, float],
    omega: dict[str, float],
    thresholds: dict[str, float],
    uncertainty: float,
    gamma: float = 1.0,
    max_uncertainty: float = 0.2,
    previous_smoothed: float | None = None,
    smoothing_alpha: float = 1.0,
    required_complete: bool = True,
) -> dict:
    """(4.46), (4.48)-(4.53), Q_src по (5.63).

    required_complete — явно дополнительное, консервативное правило:
    неполнота обязательного основания блокирует режимную рекомендацию.
    Даже при отключении этого правила полное отсутствие данных блокирует допуск.
    """
    keys = CRITERIA
    if any(set(value) != set(keys) for value in (scores, quality, omega, thresholds)):
        raise ValueError("Нужны ровно ключи stat, phys, sem")
    values = (
        list(quality.values()) + list(omega.values()) + list(thresholds.values())
        + [uncertainty, gamma, max_uncertainty, smoothing_alpha]
        + [value for value in scores.values() if value is not None]
    )
    if any(not math.isfinite(value) for value in values):
        raise ValueError("NaN/Inf в показателях допуска")
    if not 0 <= max_uncertainty <= 1:
        raise ValueError("Некорректная максимальная неопределенность")
    if previous_smoothed is not None and not 0 <= previous_smoothed <= 1:
        raise ValueError("Некорректная предыдущая сглаженная оценка")
    if not 0 <= uncertainty <= 1 or gamma < 0 or not 0 < smoothing_alpha <= 1:
        raise ValueError("Некорректные параметры допуска")
    for key in keys:
        if omega[key] <= 0 or not 0 <= quality[key] <= 1 or not 0 <= thresholds[key] <= 1:
            raise ValueError("Некорректные веса/порог/полнота")
        if scores[key] is not None and not 0 <= scores[key] <= 1:
            raise ValueError("Оценки должны принадлежать [0,1]")
        if scores[key] is None and quality[key] > 0:
            raise ValueError("Отсутствующая оценка должна иметь q=0")

    total = sum(omega[key] * quality[key] for key in keys)
    q_source = total / sum(omega.values())
    reasons = []
    weights = {key: omega[key] * quality[key] / total if total else 0.0 for key in keys}
    values_c = {key: float(scores[key]) if scores[key] is not None else 0.0 for key in keys}
    mean = sum(weights[key] * values_c[key] for key in keys)
    disagreement = math.sqrt(sum(weights[key] * (values_c[key] - mean) ** 2 for key in keys))

    # Не вычисляем 0^0 и log(0). Отсутствующая компонента снижает Q_source.
    if not total or any(weights[key] > 0 and values_c[key] == 0 for key in keys):
        psi = 0.0
    else:
        log_product = sum(weights[key] * math.log(values_c[key]) for key in keys if weights[key] > 0)
        psi = q_source * math.exp(-gamma * disagreement + log_product)
    smoothed = psi if previous_smoothed is None else (
        smoothing_alpha * psi + (1 - smoothing_alpha) * previous_smoothed
    )
    if not total:
        reasons.append("no_source_information")
    for key in keys:
        if scores[key] is None:
            reasons.append("missing_" + key)
        elif values_c[key] < thresholds[key]:
            reasons.append("low_" + key)
    if required_complete and any(quality[key] < 1 for key in keys):
        reasons.append("mandatory_checks_incomplete")
    if uncertainty > max_uncertainty:
        reasons.append("excess_uncertainty")
    accepted = not reasons
    return {
        "scores": scores, "source_quality": quality, "weights": weights,
        "q_source": q_source, "mean_consistency": mean,
        "disagreement_std": disagreement, "psi": psi, "smoothed_psi": smoothed,
        "admitted": accepted,
        "information_quality": smoothed * (1 - uncertainty) if accepted else 0.0,
        "mode": "operator_review" if accepted else "diagnostic_only",
        "blocking_reasons": reasons,
    }


def action_utility(
    expected_risk_reduction: float,
    expected_information_gain: float,
    reversibility: float,
    normalized_time: float,
    normalized_cost: float,
    uncertainty: float,
    weights: list[float],
) -> float:
    """(4.56): J = beta1*dH + beta2*dQ + beta3*Rev - beta4*T - beta5*K - beta6*u.

    Ожидаемые изменения dH, dQ может вычислять только предметная сценарная модель.
    Эта функция не создает оценки эффекта действий из названий действий.
    """
    values = np.asarray([
        expected_risk_reduction, expected_information_gain, reversibility,
        normalized_time, normalized_cost, uncertainty,
    ], dtype=float)
    beta = np.asarray(weights, dtype=float)
    if beta.shape != (6,) or not np.isfinite(beta).all() or (beta < 0).any():
        raise ValueError("Нужны шесть конечных неотрицательных весов")
    if not np.isfinite(values).all() or (np.abs(values[:2]) > 1).any():
        raise ValueError("Изменения нормированных H и Q должны принадлежать [-1,1]")
    if (values[2:] < 0).any() or (values[2:] > 1).any():
        raise ValueError("Нормированные критерии должны принадлежать [0,1]")
    signs = np.asarray([1, 1, 1, -1, -1, -1])
    return float(beta @ (signs * values))


def rank_actions(
    actions: list[dict[str, Any]],
    scenario_weights: list[float],
    risk_aversion: float = 0.0,
    required_constraints: tuple[str, ...] = ("physical", "technological", "semantic"),
) -> list[dict]:
    """(4.55)-(4.57): фильтрация, среднее минус штраф стандартного отклонения.

    constraints — функции ДОПУСТИМОСТИ g<=0, не произвольные знакопеременные
    невязки равенств. Для каждой обязательной группы требуется явный результат.
    Групповое значение может быть максимумом всех обязательных g этой группы.
    При отсутствии допустимых альтернатив возвращается пустой список.
    """
    weights = np.asarray(scenario_weights, dtype=float)
    if (weights.ndim != 1 or len(weights) == 0 or not np.isfinite(weights).all()
            or (weights < 0).any() or not np.isclose(weights.sum(), 1)
            or not math.isfinite(risk_aversion) or risk_aversion < 0):
        raise ValueError("Нужны нормированные веса сценариев и lambda>=0")
    if not required_constraints:
        raise ValueError("Задайте хотя бы одну обязательную группу ограничений")
    result = []
    for action in actions:
        constraints = action.get("constraints", {})
        if not set(required_constraints).issubset(constraints):
            continue
        if any(value is None or not math.isfinite(value) or value > 0 for value in constraints.values()):
            continue
        utilities = np.asarray(action["scenario_utilities"], dtype=float)
        if utilities.shape != weights.shape or not np.isfinite(utilities).all():
            raise ValueError("Для каждого действия нужны полезности всех сценариев")
        mean = float(weights @ utilities)
        std = float(np.sqrt(weights @ ((utilities - mean) ** 2)))
        result.append({
            "id": action["id"], "expected_utility": mean, "scenario_std": std,
            "robust_utility": mean - risk_aversion * std, "constraints": constraints,
        })
    return sorted(result, key=lambda action: (-action["robust_utility"], str(action["id"])))
