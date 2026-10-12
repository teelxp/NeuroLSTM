"""Критерий выбора архитектуры Ψ (2.15) и отдельно экспериментальный proxy §6.1.1.

Эти функции не подставляют пропущенные показатели и не смешивают показатель
выбора архитектуры с динамической согласованностью решения из главы 4.
"""
from __future__ import annotations
import math


def architecture_psi(scores_5v:dict[str,float],physical_residual:float,
                     physical_limit:float,entropy_divergence:float,entropy_limit:float,
                     weights:tuple[float,float,float]=(1/3,1/3,1/3))->dict:
    expected={"Volume","Velocity","Variety","Veracity","Value"}
    if set(scores_5v)!=expected:
        raise ValueError("Для (2.15) нужны все пять заранее нормированных показателей 5V")
    if any(not math.isfinite(v) or not 0<=v<=1 for v in scores_5v.values()):
        raise ValueError("Нормированные показатели должны находиться в [0,1]")
    if any(not math.isfinite(v) for v in [physical_residual,physical_limit,entropy_divergence,entropy_limit,*weights]):
        raise ValueError("NaN/Inf в критерии архитектуры")
    if physical_residual<0 or entropy_divergence<0 or physical_limit<=0 or entropy_limit<=0:
        raise ValueError("Невязки должны быть неотрицательными, пределы положительными")
    if min(weights)<0 or not math.isclose(sum(weights),1):
        raise ValueError("Требуются неотрицательные веса с суммой 1")
    phi=sum(scores_5v.values())/5
    cphys=1-min(1,physical_residual/physical_limit)
    cent=1-min(1,entropy_divergence/entropy_limit)
    return {"criterion":"architecture_psi_equation_2_15","phi_5v":phi,"c_phys":cphys,
            "c_ent":cent,"psi":weights[0]*phi+weights[1]*cphys+weights[2]*cent}


def auxiliary_experiment_proxy(training_seconds:float,reference_min_seconds:float,
                               prediction_accuracy:float,classification_accuracy:float,
                               f1:float,error_fraction:float,physical_pass_fraction:float)->dict:
    """Частный расчет со стр.266. Все accuracy/f1 в долях, не в процентах.

    Bernoulli-распределения построены из агрегированных показателей; это не
    распределения выходных физических величин и не полная проверка 5V.
    """
    for v in [prediction_accuracy,classification_accuracy,f1,error_fraction,physical_pass_fraction]:
        if not math.isfinite(v) or not 0<=v<=1:raise ValueError("Оценки должны быть долями [0,1]")
    if not 0<reference_min_seconds<=training_seconds:
        raise ValueError("Некорректные времена обучения")
    cver=(prediction_accuracy+classification_accuracy+f1)/3
    phi=(reference_min_seconds/training_seconds+cver+1-error_fraction)/3
    p=physical_pass_fraction;q=cver
    kl=0.
    for mass,reference in [(p,q),(1-p,1-q)]:
        if mass==0:continue
        if reference==0:kl=float("inf");break
        kl+=mass*math.log(mass/reference)
    cent=math.exp(-kl)
    return {"criterion":"auxiliary_three_component_proxy_NOT_equation_2_15",
            "phi_3components":phi,"c_phys":p,"c_ent_proxy":cent,"psi_proxy":(phi+p+cent)/3,
            "all_5v_evaluated":False}
