"""Метрики из фактических прогнозов, без констант из экспериментальных таблиц."""
from __future__ import annotations
import numpy as np


class RegressionClassificationMetrics:
    def __init__(self,horizons:list[int],relative_tolerance:float=.1,
                 denominator_floor_pa:float=1.,classification_threshold:float=.5,
                 physics_tolerance:float|None=None):
        self.horizons=horizons
        h=len(horizons)
        self.n=np.zeros(h,dtype=np.int64)
        self.abs=np.zeros(h);self.sq=np.zeros(h);self.ape=np.zeros(h)
        self.correct_pressure=np.zeros(h,dtype=np.int64)
        self.cm=np.zeros((h,2,2),dtype=np.int64)
        self.phys_abs=np.zeros(h);self.phys_sq=np.zeros(h);self.phys_ok=np.zeros(h,dtype=np.int64)
        self.flow_abs=np.zeros(h);self.flow_sq=np.zeros(h)
        self.has_classification=False;self.has_flow=False
        self.relative_tolerance=relative_tolerance;self.floor=denominator_floor_pa
        self.threshold=classification_threshold;self.physics_tolerance=physics_tolerance
        if relative_tolerance<0 or denominator_floor_pa<=0 or not 0<=classification_threshold<=1:
            raise ValueError("Некорректные пороги метрик")

    def update(self,truth:np.ndarray,pred:np.ndarray,residual:np.ndarray,
               labels:np.ndarray|None=None,probability:np.ndarray|None=None,
               flow_truth:np.ndarray|None=None,flow_pred:np.ndarray|None=None)->None:
        if truth.shape!=pred.shape or residual.shape!=truth.shape or truth.shape[1]!=len(self.horizons):
            raise ValueError("Несогласованные размерности метрик")
        if not all(np.isfinite(a).all() for a in [truth,pred,residual]):
            raise FloatingPointError("Метрики содержат NaN/Inf")
        error=pred-truth
        relative=np.abs(error)/np.maximum(np.abs(truth),self.floor)
        self.n+=len(truth);self.abs+=np.abs(error).sum(0);self.sq+=(error**2).sum(0)
        self.ape+=relative.sum(0);self.correct_pressure+=(relative<self.relative_tolerance).sum(0)
        self.phys_abs+=np.abs(residual).sum(0);self.phys_sq+=(residual**2).sum(0)
        if self.physics_tolerance is not None:
            self.phys_ok+=(np.abs(residual)<=self.physics_tolerance).sum(0)
        if probability is not None:
            if labels is None or not np.isin(labels,[0,1]).all() or not np.isfinite(probability).all():
                raise ValueError("Некорректные метки/вероятности классификации")
            self.has_classification=True
            binary=probability>=self.threshold
            for k in range(len(self.horizons)):
                for a in (0,1):
                    for b in (0,1):
                        self.cm[k,a,b]+=int(((labels[:,k]==a)&(binary[:,k]==b)).sum())
        if flow_pred is not None:
            if flow_truth is None or not np.isfinite(flow_pred).all():
                raise ValueError("Нет корректного эталона расхода")
            self.has_flow=True
            e=flow_pred-flow_truth
            self.flow_abs+=np.abs(e).sum(0);self.flow_sq+=(e**2).sum(0)

    def result(self)->dict:
        rows=[]
        for k,h in enumerate(self.horizons):
            n=int(self.n[k])
            if not n:
                raise ValueError("Пустое оценочное множество")
            row={"horizon_steps":h,"samples":n,
                 "pressure_mae_pa":float(self.abs[k]/n),
                 "pressure_mae_mpa":float(self.abs[k]/n/1e6),
                 "pressure_rmse_pa":float(np.sqrt(self.sq[k]/n)),
                 "pressure_rmse_mpa":float(np.sqrt(self.sq[k]/n)/1e6),
                 "pressure_mape_percent":float(100*self.ape[k]/n),
                 "pressure_within_tolerance_percent":float(100*self.correct_pressure[k]/n),
                 "pressure_outside_tolerance_percent":float(100*(1-self.correct_pressure[k]/n)),
                 "mass_residual_mae_kg_s":float(self.phys_abs[k]/n),
                 "mass_residual_rmse_kg_s":float(np.sqrt(self.phys_sq[k]/n)),
                 "physics_consistency_percent":float(100*self.phys_ok[k]/n) if self.physics_tolerance is not None else None,
                 "physics_violation_percent":float(100*(1-self.phys_ok[k]/n)) if self.physics_tolerance is not None else None}
            if self.has_flow:
                row.update(flow_mae_kg_s=float(self.flow_abs[k]/n),flow_rmse_kg_s=float(np.sqrt(self.flow_sq[k]/n)))
            if self.has_classification:
                tn,fp,fn,tp=(int(x) for x in self.cm[k].ravel())
                row.update(accuracy_percent=100*(tp+tn)/n,
                           precision=tp/(tp+fp) if tp+fp else None,
                           recall=tp/(tp+fn) if tp+fn else None,
                           f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                           false_positive_rate=fp/(fp+tn) if fp+tn else None,
                           false_negative_rate=fn/(tp+fn) if tp+fn else None,
                           confusion_matrix={"tn":tn,"fp":fp,"fn":fn,"tp":tp})
            rows.append(row)
        return {"horizons":rows,"pressure_mae_pa_macro":float(np.mean([r["pressure_mae_pa"] for r in rows])),
                "definitions":{"prediction_accuracy":"100 * mean(abs(pred-true)/max(abs(true),floor) < relative_tolerance)",
                               "physics_consistency":"100 * mean(abs(mass_residual_kg_s) <= fixed_tolerance_kg_s)",
                               "physical_quantity":"integrated_inventory_mass_balance; NOT PINN PDE loss",
                               "undefined_class_metrics":"null when corresponding denominator is zero"},
                "thresholds":{"pressure_relative":self.relative_tolerance,"pressure_floor_pa":self.floor,
                              "classification_probability":self.threshold,"mass_residual_kg_s":self.physics_tolerance}}
