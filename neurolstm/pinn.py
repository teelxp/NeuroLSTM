"""Базовая газовая PINN по (6.4)-(6.7); отдельный формат пространственных данных.

Не выводит координаты из номера строки и не выдает временной CSV за пространственное
поле. Для честного сравнения с LSTM требуется общий независимый протокол оценки.
"""
from __future__ import annotations
import copy
import math
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from torch import Tensor,nn
from .data import sha256_file
from .training import write_json,device_for,environment,source_manifest


class GasPINN(nn.Module):
    def __init__(self,input_min:Tensor,input_span:Tensor,output_min:Tensor,output_span:Tensor,
                 width:int=64,layers:int=6):
        super().__init__()
        self.register_buffer("input_min",input_min)
        self.register_buffer("input_span",input_span)
        self.register_buffer("output_min",output_min)
        self.register_buffer("output_span",output_span)
        net=[];dim=len(input_min)
        for _ in range(layers):
            net += [nn.Linear(dim,width),nn.Tanh()];dim=width
        net.append(nn.Linear(dim,3))
        self.network=nn.Sequential(*net)

    def forward(self,physical_input:Tensor)->Tensor:
        normalized=(physical_input-self.input_min)/self.input_span
        raw=self.network(normalized)
        fields=raw[...,:2]*self.output_span+self.output_min
        return torch.cat([fields,raw[...,2:3]],dim=-1)


def gas_pde_residual(model:nn.Module,coordinates:Tensor,diameter:float,sound_speed:float,
                     friction:float,base_velocity:float,create_graph:bool=True)->tuple[Tensor,Tensor,Tensor]:
    """(6.6); x в м, t в с, давление Па, расход кг/с. Производные в SI.

    Линеаризованная модель для практически горизонтального участка, v_b>=0.
    Она не заменяет модель нелинейных волн/произвольного рельефа.
    """
    if diameter<=0 or sound_speed<=0 or friction<0 or base_velocity<0:
        raise ValueError("Некорректные параметры линеаризованной газовой модели")
    x=coordinates if coordinates.requires_grad else coordinates.detach().requires_grad_(True)
    out=model(x)
    dp=torch.autograd.grad(out[:,0].sum(),x,create_graph=create_graph,retain_graph=True)[0]
    dm=torch.autograd.grad(out[:,1].sum(),x,create_graph=create_graph,retain_graph=create_graph)[0]
    area=np.pi*diameter*diameter/4
    rp=dp[:,1]+sound_speed**2/area*dm[:,0]
    rm=dm[:,1]+area*dp[:,0]+friction*base_velocity/diameter*out[:,1]-friction*area*base_velocity**2/(2*sound_speed**2*diameter)*out[:,0]
    return rp,rm,out


def train_pinn(config_path:str|Path,output:str|Path)->dict:
    cfg=json.loads(Path(config_path).read_text(encoding="utf-8"))
    json.dumps(cfg,allow_nan=False)
    required=["data_path","diameter_m","sound_speed_m_s","friction_lambda","baseline_velocity_m_s",
              "adam_steps","lbfgs_steps","collocation_points","lr","width","layers","validation_every","seed","device"]
    if any(key not in cfg or cfg[key] is None for key in required):
        raise ValueError("PINN: заполните все необходимые физические и экспериментальные параметры конфигурации")
    if cfg["width"]<1 or cfg["layers"]<1 or cfg["lr"]<=0:
        raise ValueError("PINN: некорректная архитектура или скорость обучения")
    if cfg["adam_steps"]<1 or cfg["lbfgs_steps"]<0 or cfg["collocation_points"]<1 or cfg["validation_every"]<1:
        raise ValueError("PINN: некорректное число шагов/точек")
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("PINN: папка вывода не пуста")
    output.mkdir(parents=True,exist_ok=True)
    device=device_for(cfg["device"])
    torch.set_num_threads(1);torch.manual_seed(cfg["seed"])
    rng=np.random.default_rng(cfg["seed"])
    with np.load(cfg["data_path"],allow_pickle=False) as data:
        a={key:data[key] for key in data.files}
    for key in ("x_m","t_s","pressure_pa","mass_flow_kg_s","accident","split","is_initial","is_boundary"):
        if key not in a or a[key].ndim!=1:
            raise ValueError(f"PINN: нужен одномерный массив {key}")
    n=len(a["x_m"])
    if any(len(v)!=n for k,v in a.items() if k!="provenance") or not np.isin(a["split"],[0,1,2]).all():
        raise ValueError("PINN: массивы должны иметь согласованную длину, split=0/1/2")
    context=a.get("context",np.empty((n,0)))
    if context.ndim!=2 or not np.isin(a["accident"],[0,1]).all():
        raise ValueError("PINN: context должен иметь форму [N,D], accident содержит 0/1")
    inp=np.column_stack([a["x_m"],a["t_s"],context])
    targets=np.column_stack([a["pressure_pa"],a["mass_flow_kg_s"]])
    if not np.isfinite(inp).all() or not np.isfinite(targets).all():
        raise ValueError("PINN: NaN/Inf во входах или целях")
    tr=a["split"]==0;val=a["split"]==1;test=a["split"]==2
    if min(int(tr.sum()),int(val.sum()),int(test.sum()))<1:
        raise ValueError("PINN требует отдельные train, validation, test")
    if not (a["t_s"][tr].max()<a["t_s"][val].min() and a["t_s"][val].max()<a["t_s"][test].min()):
        raise ValueError("PINN: временные интервалы train/validation/test пересекаются")
    if not np.isin(a["is_initial"],[0,1]).all() or not np.isin(a["is_boundary"],[0,1]).all():
        raise ValueError("PINN: маски начальных и граничных условий должны содержать 0/1")
    initial=tr&a["is_initial"].astype(bool);boundary=tr&a["is_boundary"].astype(bool)
    if not initial.any() or not boundary.any():
        raise ValueError("Для (6.7) нужны начальные и граничные условия из обучающего интервала")
    minimum=inp[tr].min(0);span=np.ptp(inp[tr],axis=0)
    output_min=targets[tr].min(0);output_span=np.ptp(targets[tr],axis=0)
    if min(span[:2])<=0 or min(output_span)<=0:
        raise ValueError("PINN требует диапазоны координаты, времени, давления и расхода на train")
    span=np.where(span>1e-12,span,1.)
    tensor=lambda v:torch.as_tensor(v,dtype=torch.float32,device=device)
    model=GasPINN(tensor(minimum),tensor(span),tensor(output_min),tensor(output_span),cfg["width"],cfg["layers"]).to(device)
    inputs=tensor(inp);truth=tensor(targets);labels=tensor(a["accident"])
    masks={k:torch.as_tensor(v,device=device) for k,v in {"train":tr,"validation":val,"test":test,"initial":initial,"boundary":boundary}.items()}
    # Коллокационные координаты лежат только в обучающей области.
    col=np.zeros((cfg["collocation_points"],inp.shape[1]))
    col[:,:2]=rng.uniform(minimum[:2],minimum[:2]+span[:2],size=(len(col),2))
    if context.shape[1]:
        col[:,2:]=context[rng.choice(np.flatnonzero(tr),size=len(col),replace=True)]
    collocation=tensor(col)
    np.savez_compressed(output/"collocation_train_only.npz",coordinates=col)
    rscale_p=float(cfg.get("residual_scale_p",output_span[0]/span[1]))
    rscale_m=float(cfg.get("residual_scale_m",output_span[1]/span[1]))
    if rscale_p<=0 or rscale_m<=0:
        raise ValueError("PINN: масштабы невязок должны быть положительными")
    weight=cfg.get("weights",{"data":1.,"pde_p":1.,"pde_m":1.,"ic":1.,"bc":1.,"classification":1.})
    if set(weight)!={"data","pde_p","pde_m","ic","bc","classification"} or min(weight.values())<0:
        raise ValueError("PINN: нужны шесть неотрицательных весов (6.7)")
    def objective():
        out=model(inputs[masks["train"]])
        data=((out[:,:2]-truth[masks["train"]])/model.output_span).square().mean()
        cls=nn.functional.binary_cross_entropy_with_logits(out[:,2],labels[masks["train"]])
        ic=((model(inputs[masks["initial"]])[:,:2]-truth[masks["initial"]])/model.output_span).square().mean()
        bc=((model(inputs[masks["boundary"]])[:,:2]-truth[masks["boundary"]])/model.output_span).square().mean()
        rp,rm,_=gas_pde_residual(model,collocation,cfg["diameter_m"],cfg["sound_speed_m_s"],cfg["friction_lambda"],cfg["baseline_velocity_m_s"])
        terms={"data":data,"classification":cls,"ic":ic,"bc":bc,"pde_p":(rp/rscale_p).square().mean(),"pde_m":(rm/rscale_m).square().mean()}
        loss=sum(weight[k]*v for k,v in terms.items())
        if not torch.isfinite(loss):raise FloatingPointError("PINN: неконечная потеря")
        return loss,terms
    start=time.perf_counter();best=float("inf");best_state=None;best_step=0;closure_calls=0
    def select(step):
        nonlocal best,best_state,best_step
        model.eval()
        with torch.no_grad():
            score=float((model(inputs[masks["validation"]])[:,0]-truth[masks["validation"],0]).abs().mean())
        if score<best:
            best=score;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};best_step=step
        model.train()
        return score
    adam=torch.optim.Adam(model.parameters(),lr=cfg["lr"])
    lbfgs=torch.optim.LBFGS(model.parameters(),lr=1.,max_iter=1,history_size=50,line_search_fn="strong_wolfe")
    total=cfg["adam_steps"]+cfg["lbfgs_steps"]
    with (output/"history.jsonl").open("w",encoding="utf-8") as history:
        for step in range(total):
            if step<cfg["adam_steps"]:
                adam.zero_grad(set_to_none=True);loss,terms=objective();loss.backward();adam.step();closure_calls+=1
                scalar=float(loss.detach())
            else:
                def closure():
                    nonlocal closure_calls
                    lbfgs.zero_grad(set_to_none=True);loss,_=objective();loss.backward();closure_calls+=1
                    return loss
                scalar=float(lbfgs.step(closure).detach())
            record={"step":step+1,"phase":"Adam" if step<cfg["adam_steps"] else "LBFGS","loss":scalar}
            if (step+1)%cfg["validation_every"]==0 or step+1==total:
                record["validation_pressure_mae_pa"]=select(step+1)
            history.write(json.dumps(record,allow_nan=False)+"\n")
    seconds=time.perf_counter()-start
    assert best_state is not None
    model.load_state_dict(best_state);model.eval()
    torch.save({"format_version":"gas_pinn_1","model_state":best_state,"config":cfg,"best_step":best_step,
                "data_sha256":sha256_file(cfg["data_path"]),"source":source_manifest()},output/"best_pinn.pt")
    with torch.enable_grad():
        rp,rm,out=gas_pde_residual(model,inputs[masks["test"]],cfg["diameter_m"],cfg["sound_speed_m_s"],cfg["friction_lambda"],cfg["baseline_velocity_m_s"],False)
    pred=out.detach().cpu().numpy();y=targets[test];prob=1/(1+np.exp(-np.clip(pred[:,2],-80,80)))
    binary=prob>=cfg.get("classification_threshold",.5);truth_cls=a["accident"][test]
    tp=int(((binary==1)&(truth_cls==1)).sum());fp=int(((binary==1)&(truth_cls==0)).sum());fn=int(((binary==0)&(truth_cls==1)).sum())
    result={"pressure_mae_pa":float(np.abs(pred[:,0]-y[:,0]).mean()),
            "pressure_rmse_pa":float(np.sqrt(np.mean((pred[:,0]-y[:,0])**2))),
            "flow_mae_kg_s":float(np.abs(pred[:,1]-y[:,1]).mean()),
            "accuracy":float(np.mean(binary==truth_cls)),"f1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            "pde_pressure_residual_rms_pa_s":float(rp.detach().square().mean().sqrt()),
            "pde_momentum_residual_rms_kg_s2":float(rm.detach().square().mean().sqrt()),
            "training_seconds":seconds,"optimizer_steps":total,"closure_evaluations":closure_calls,
            "selected_step":best_step,"collocation_points":len(col),"parameter_count":sum(p.numel() for p in model.parameters()),
            "test_samples":int(test.sum()),"direct_comparison_to_lstm_physics_percent_valid":False,
            "data_sha256":sha256_file(cfg["data_path"]),"environment":environment(device),"source":source_manifest()}
    pd.DataFrame({"x_m":a["x_m"][test],"t_s":a["t_s"][test],"pressure_true_pa":y[:,0],"pressure_pred_pa":pred[:,0],
                  "mass_flow_true_kg_s":y[:,1],"mass_flow_pred_kg_s":pred[:,1],"accident_probability":prob}).to_csv(output/"test_predictions.csv",index=False)
    write_json(output/"metrics.json",result);write_json(output/"config.resolved.json",cfg)
    return result
