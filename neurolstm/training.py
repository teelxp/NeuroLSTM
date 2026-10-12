"""Обучение, независимая оценка и инференс. Реальные метрики и паспорт запуска."""
from __future__ import annotations
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import time
from typing import Any
import numpy as np
import pandas as pd
import sklearn
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from . import __version__
from .config import validate
from .data import prepare, load_frame, Scaling, group_tensor, FEATURES
from .model import AdaptiveLSTM
from .physics import gas_loss
from .gradients import flattened_gradient, combine_gradients, set_gradient, ramp, learning_rate
from .metrics import RegressionClassificationMetrics
from .semantics import semantic_loss


def write_json(path:str|Path,value:Any)->None:
    path=Path(path)
    temp=path.with_name(path.name+".tmp")
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    temp.replace(path)


def device_for(name:str)->torch.device:
    if name=="auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev=torch.device(name)
    if dev.type=="cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA недоступна; выберите device=cpu")
    return dev


def setup(cfg:dict)->torch.device:
    t=cfg["training"]
    random.seed(t["seed"]);np.random.seed(t["seed"]);torch.manual_seed(t["seed"])
    torch.set_num_threads(t["threads"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(t["seed"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    torch.use_deterministic_algorithms(t["deterministic"])
    return device_for(t["device"])


def move(batch:Any,device:torch.device)->Any:
    if isinstance(batch,Tensor):
        return batch.to(device)
    if isinstance(batch,dict):
        return {k:move(v,device) for k,v in batch.items()}
    return batch


def source_manifest()->dict:
    root=Path(__file__).resolve().parent.parent
    paths=list((root/"neurolstm").glob("*.py"))+[root/"Neuro.py"]
    result={}
    for p in sorted(paths):
        if p.exists():
            result[p.relative_to(root).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    try:
        commit=subprocess.run(["git","rev-parse","HEAD"],cwd=root,capture_output=True,text=True,timeout=3,check=True).stdout.strip()
    except (OSError,subprocess.SubprocessError):
        commit=None
    return {"version":__version__,"git_commit":commit,"files_sha256":result}


def environment(device:torch.device)->dict:
    return {"python":platform.python_version(),"platform":platform.platform(),"torch":str(torch.__version__),
            "numpy":np.__version__,"pandas":pd.__version__,"scikit_learn":sklearn.__version__,
            "device":str(device),"cuda_runtime":torch.version.cuda,
            "gpu":torch.cuda.get_device_name(device) if device.type=="cuda" else None}


def losses_for(model:AdaptiveLSTM,batch:dict,scaling:Scaling,cfg:dict)->tuple[dict,dict]:
    out=model(batch["x"],batch["mask"],batch["mass_rate"])
    t=cfg["training"]
    regression=torch.nn.functional.mse_loss if t["regression_loss"]=="mse" else torch.nn.functional.l1_loss
    data=t["pressure_weight"]*regression(out["pressure_normalized"],batch["target_pressure_normalized"])
    if cfg["model"]["predict_flow"]:
        data=data+t["flow_weight"]*regression(out["flow_normalized"],batch["target_flow_normalized"])
        out["flow_kg_s"]=scaling.inverse_tensor(out["flow_normalized"].double(),6)
    if cfg["model"]["classify"]:
        data=data+t["classification_weight"]*torch.nn.functional.binary_cross_entropy_with_logits(out["accident_logit"],batch["accident"])
    out["pressure_pa"]=scaling.inverse_tensor(out["pressure_normalized"].double(),0)
    phys,residual=gas_loss(out["pressure_pa"],batch["meta"],cfg["physics"],scaling.residual_scale)
    out["mass_residual_kg_s"]=residual
    sem=data.new_zeros(())
    components={}
    if model.semantic is not None:
        sem_pred=model.semantic(out["hidden"],batch["semantics"])
        sem,components=semantic_loss(sem_pred,batch["semantics"],cfg["semantics"])
    losses={"data":data,"phys":phys,"sem":sem}
    losses.update({"sem_"+key:value for key,value in components.items()})
    for name,value in losses.items():
        if not torch.isfinite(value):
            raise FloatingPointError(f"Потеря {name} содержит NaN/Inf")
    return losses,out


def save_checkpoint(path:Path,model:AdaptiveLSTM,optimizer:torch.optim.Optimizer,cfg:dict,scaling:Scaling,
                    epoch:int,best:float,audit:dict,selected_on:str,training_seconds:float)->None:
    np_state=np.random.get_state()
    checkpoint={"format_version":2,"model_state":model.state_dict(),"optimizer_state":optimizer.state_dict(),
                "config":cfg,"scaling":scaling.to_dict(),"epoch":epoch,"best_score":best,
                "audit":audit,"selected_on":selected_on,"training_seconds":training_seconds,
                "rng_torch":torch.get_rng_state(),"rng_python":random.getstate(),
                "rng_numpy":[np_state[0],np_state[1].tolist(),np_state[2],np_state[3],np_state[4]],
                "rng_cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "source":source_manifest()}
    temp=path.with_name(path.name+".tmp")
    torch.save(checkpoint,temp);temp.replace(path)


def load_checkpoint(path:str|Path,device:torch.device)->dict:
    # Только tensor/state-dict формат; прежний произвольный pickle не исполняется.
    cp=torch.load(path,map_location=device,weights_only=True)
    if not isinstance(cp,dict) or cp.get("format_version")!=2:
        raise ValueError("Старый .pth не совместим с новой моделью; требуется новое обучение")
    return cp


def evaluate_dataset(model:AdaptiveLSTM,dataset,scaling:Scaling,cfg:dict,device:torch.device,
                     predictions_path:Path|None=None)->dict:
    model.eval()
    params=cfg["metrics"]
    acc=RegressionClassificationMetrics(cfg["data"]["horizons"],params["pressure_relative_tolerance"],
                                       params["pressure_denominator_floor_pa"],params["classification_threshold"],
                                       cfg["physics"]["consistency_tolerance_kg_s"])
    totals={};n=0;elapsed=0.
    handle=None
    writer=None
    try:
        if predictions_path is not None:
            handle=predictions_path.open("w",newline="",encoding="utf-8")
        with torch.no_grad():
            for original in DataLoader(dataset,batch_size=cfg["training"]["micro_batch_size"],shuffle=False,num_workers=0):
                batch=move(original,device)
                if device.type=="cuda":torch.cuda.synchronize(device)
                start=time.perf_counter()
                losses,out=losses_for(model,batch,scaling,cfg)
                if device.type=="cuda":torch.cuda.synchronize(device)
                elapsed+=time.perf_counter()-start
                count=len(batch["x"]);n+=count
                for key,value in losses.items():totals[key]=totals.get(key,0.)+float(value)*count
                cpu=lambda v:v.detach().cpu().numpy()
                truth=cpu(batch["target_pressure"]);pred=cpu(out["pressure_pa"]);residual=cpu(out["mass_residual_kg_s"])
                prob=cpu(out["accident_probability"]) if cfg["model"]["classify"] else None
                flow=cpu(out["flow_kg_s"]) if cfg["model"]["predict_flow"] else None
                acc.update(truth,pred,residual,cpu(batch["accident"]) if prob is not None else None,prob,
                           cpu(batch["target_flow"]) if flow is not None else None,flow)
                if handle is not None:
                    for i in range(count):
                        for j,h in enumerate(cfg["data"]["horizons"]):
                            row={"object_id":original["object_id"][i],"origin_time_ns":int(original["origin_time_ns"][i]),
                                 "target_time_ns":int(original["target_time_ns"][i,j]),"horizon_steps":h,
                                 "true_pressure_pa":float(truth[i,j]),"pred_pressure_pa":float(pred[i,j]),
                                 "mass_residual_kg_s":float(residual[i,j]),
                                 "pred_mass_out_kg_s":float(flow[i,j]) if flow is not None else "",
                                 "accident_probability":float(prob[i,j]) if prob is not None else "",
                                 "true_accident":float(original["accident"][i,j]) if prob is not None else ""}
                            if writer is None:
                                writer=csv.DictWriter(handle,fieldnames=list(row));writer.writeheader()
                            writer.writerow(row)
        result=acc.result()
        result["losses"]={key:value/n for key,value in totals.items()}
        result["evaluation_seconds_including_losses"]=elapsed
        result["seconds_per_window_including_losses"]=elapsed/n
        result["semantics_enabled"]=model.semantic is not None
        result["semantic_loss_interpretation"]="computed" if model.semantic is not None else "not_evaluated"
        result["semantic_components_evaluated"]=[key for key in totals if key.startswith("sem_")]
        result["physics_context"]="retrospective_observed_future_flows_temperature_Z_only_in_loss"
        return result
    finally:
        if handle is not None:handle.close()


def _cluster_labels(dataset,cfg:dict)->tuple[np.ndarray,dict]:
    from .clustering import EVKMeans
    # Окно — объект кластеризации, не будущая целевая переменная.
    numeric=[];physical=[];modal=[]
    for i in dataset.origins:
        first=i-cfg["data"]["window"]+1
        numeric.append(dataset.normalized[first:i+1].reshape(-1))
        r=dataset.raw[first:i+1,5]-dataset.raw[first:i+1,6]-dataset.rates[first:i+1]*dataset.scaling.mass_rate_scale
        physical.append([float(np.mean(r))/dataset.scaling.residual_scale,float(np.sqrt(np.mean(r*r)))/dataset.scaling.residual_scale])
        if dataset.semantic_store is not None:
            modal.append(dataset.semantic_store.get(dataset.ids[i],dataset.ns[i])["input_embeddings"].numpy().mean(0))
    c=cfg["clustering"]
    cluster=EVKMeans(c["clusters"],c["pca_components"],c["weights"],c["epsilon"],cfg["training"]["seed"],c["restarts"])
    labels=cluster.fit_predict(np.array(numeric),np.array(physical),np.array(modal) if modal else None)
    return labels,{"cvr":cluster.training_cvr,"counts":np.bincount(labels).tolist(),
                   "one_cluster_scales":cluster.scales.tolist(),"source":"only_train_input_windows",
                   "numeric_distance":"PCA eigenvalue scaled, (2.26)","weights":c["weights"]}


def train(cfg:dict,output:str|Path,resume:str|Path|None=None)->dict:
    cfg=validate(cfg)
    device=setup(cfg)
    output=Path(output)
    if resume is not None and Path(resume).resolve().parent != output.resolve():
        raise ValueError("Resume: сохраняйте исходную папку запуска вместе с best.pt и журналом")
    if output.exists() and any(output.iterdir()) and resume is None:
        raise FileExistsError(f"{output}: результаты уже существуют. Выберите новую папку или --resume")
    output.mkdir(parents=True,exist_ok=True)
    datasets,scaling,audit=prepare(cfg)
    model=AdaptiveLSTM(cfg["model"],cfg["data"]["horizons"],cfg["semantics"]).to(device)
    t=cfg["training"]
    cls=torch.optim.AdamW if t["optimizer"]=="AdamW" else torch.optim.SGD
    optimizer=cls(model.parameters(),lr=t["lr"],weight_decay=t["weight_decay"])
    start_epoch=0;best=float("inf");previous_seconds=0.
    if resume is not None:
        cp=load_checkpoint(resume,device)
        # Возобновляется ТА ЖЕ постановка; разрешено изменить только число эпох,
        # физическое размещение и частоту записи контрольных точек.
        old=json.loads(json.dumps(cp["config"]));new=json.loads(json.dumps(cfg))
        for key in ["epochs","device","checkpoint_every"]:
            old["training"].pop(key);new["training"].pop(key)
        if old!=new or cp["audit"]["canonical_data_sha256"]!=audit["canonical_data_sha256"]:
            raise ValueError("Resume: конфигурация или данные изменились")
        if cp["source"]["files_sha256"]!=source_manifest()["files_sha256"]:
            raise ValueError("Resume: исходный код изменился, создайте новый запуск")
        if cp["audit"].get("semantic_data_sha256")!=audit.get("semantic_data_sha256"):
            raise ValueError("Resume: семантические данные изменились")
        model.load_state_dict(cp["model_state"]);optimizer.load_state_dict(cp["optimizer_state"])
        scaling=Scaling.from_dict(cp["scaling"])
        start_epoch=cp["epoch"]+1;best=cp["best_score"];previous_seconds=cp["training_seconds"]
        torch.set_rng_state(cp["rng_torch"].cpu());random.setstate(cp["rng_python"])
        ns=cp["rng_numpy"];np.random.set_state((ns[0],np.array(ns[1],np.uint32),ns[2],ns[3],ns[4]))
        if device.type=="cuda" and cp["rng_cuda"]:torch.cuda.set_rng_state_all(cp["rng_cuda"])
    if start_epoch>=t["epochs"]:
        raise ValueError("Число epochs не превосходит уже выполненное")
    metadata={"config":cfg,"data":audit,"environment":environment(device),"source":source_manifest(),
              "parameters":model.parameter_count(),"reported_dissertation_metrics_injected":False,
              "training_status":"running","test_used_for_selection":False,
              "semantics_enabled":model.semantic is not None,
              "selection":"validation" if len(datasets["validation"]) else "last_epoch_without_validation",
              "gradient_policy":"accumulate_raw_then_project_once" if t["batch_mode"]=="full" else "project_each_minibatch"}
    labels=None
    if cfg["clustering"]["enabled"]:
        labels,cluster_audit=_cluster_labels(datasets["train"],cfg)
        metadata["clustering"]=cluster_audit
        np.save(output/"train_cluster_labels.npy",labels)
    write_json(output/"run_manifest.json",metadata)
    write_json(output/"config.resolved.json",cfg)
    write_json(output/"scaling.json",scaling.to_dict())
    parameters=tuple(p for p in model.parameters() if p.requires_grad)
    zero=lambda:torch.cat([torch.zeros_like(p).reshape(-1) for p in parameters])
    start_run=time.perf_counter()
    selected_on=metadata["selection"]
    with (output/"history.jsonl").open("a" if resume else "w",encoding="utf-8") as history:
        for epoch in range(start_epoch,t["epochs"]):
            model.train()
            lr=learning_rate(epoch,t)
            for group in optimizer.param_groups:group["lr"]=lr
            bp=ramp(epoch,t["beta_phys_max"],t["tau_phys"])
            bs=ramp(epoch,t["beta_sem_max"],t["tau_sem"])
            ntrain=len(datasets["train"])
            if labels is None:
                order=np.random.default_rng(t["seed"]+epoch).permutation(ntrain).tolist() if t["batch_mode"]=="mini" else list(range(ntrain))
            else:
                from .clustering import clustered_order
                order=clustered_order(labels,t["seed"]+epoch)
            loader=DataLoader(datasets["train"],batch_size=t["micro_batch_size"],sampler=order,num_workers=0)
            buffers={key:zero() for key in ["data","phys","sem"]}
            summed={};steps=0;norm_sums={}
            for batch in loader:
                batch=move(batch,device)
                loss,out=losses_for(model,batch,scaling,cfg)
                b=len(batch["x"])
                gradients={"data":flattened_gradient(loss["data"],parameters,retain_graph=True),
                           "phys":flattened_gradient(loss["phys"],parameters,retain_graph=model.semantic is not None),
                           "sem":flattened_gradient(loss["sem"],parameters,retain_graph=False)}
                for name,value in loss.items():summed[name]=summed.get(name,0.)+float(value.detach())*b/ntrain
                if any(not torch.isfinite(g).all() for g in gradients.values()):
                    raise FloatingPointError("Неконечный компонентный градиент")
                if t["batch_mode"]=="full":
                    for key in buffers:buffers[key].add_(gradients[key],alpha=b/ntrain)
                else:
                    buffers=gradients
                last_batch=(steps+1)==len(loader)
                if t["batch_mode"]=="mini" or last_batch:
                    reg=torch.cat([2*p.detach().reshape(-1) for p in parameters]) if t["l2_weight"] else zero()
                    combined,norms=combine_gradients(buffers["data"],buffers["phys"],buffers["sem"],reg,bp,bs,t["l2_weight"],t["projection_epsilon"],t["gradient_mode"])
                    optimizer.zero_grad(set_to_none=True)
                    set_gradient(parameters,combined)
                    if t["gradient_clip_norm"] is not None:
                        torch.nn.utils.clip_grad_norm_(parameters,t["gradient_clip_norm"],error_if_nonfinite=True)
                    optimizer.step()
                    for key,value in norms.items():norm_sums[key]=norm_sums.get(key,0.)+value
                steps+=1
                del loss,out,gradients,batch
            updates=steps if t["batch_mode"]=="mini" else 1
            record={"epoch":epoch+1,"lr":lr,"beta_phys":bp,"beta_sem":bs,
                    "train_losses":summed,"optimizer_steps":updates,
                    "gradients":{k:v/updates for k,v in norm_sums.items()}}
            validation=None
            if len(datasets["validation"]) and ((epoch+1)%t["validation_every"]==0 or epoch+1==t["epochs"]):
                validation=evaluate_dataset(model,datasets["validation"],scaling,cfg,device)
                record["validation"]=validation
                score=validation["pressure_mae_pa_macro"] if t["select_by"]=="pressure_mae" else (validation["losses"]["data"]+t["beta_phys_max"]*validation["losses"]["phys"]+t["beta_sem_max"]*validation["losses"]["sem"])
                if score<best:
                    best=score
                    save_checkpoint(output/"best.pt",model,optimizer,cfg,scaling,epoch,best,audit,selected_on,previous_seconds+time.perf_counter()-start_run)
            if not len(datasets["validation"]) and epoch+1==t["epochs"]:
                best=float(summed["data"])
                save_checkpoint(output/"best.pt",model,optimizer,cfg,scaling,epoch,best,audit,selected_on,previous_seconds+time.perf_counter()-start_run)
            if (epoch+1)%t["checkpoint_every"]==0 or epoch+1==t["epochs"]:
                save_checkpoint(output/"last.pt",model,optimizer,cfg,scaling,epoch,best,audit,selected_on,previous_seconds+time.perf_counter()-start_run)
            history.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+"\n");history.flush()
            if epoch<3 or (epoch+1)%t["checkpoint_every"]==0 or epoch+1==t["epochs"]:
                print(f"epoch={epoch+1}/{t['epochs']} data={summed['data']:.6g} phys={summed['phys']:.6g} sem={summed['sem']:.6g} lr={lr:.4g}")
    metadata["training_status"]="complete"
    metadata["training_seconds"]=previous_seconds+time.perf_counter()-start_run
    metadata["completed_epochs"]=t["epochs"]
    write_json(output/"run_manifest.json",metadata)
    # Независимое тестирование намеренно вынесено в отдельную команду evaluate.
    return metadata


def evaluate(checkpoint:str|Path,output:str|Path,device_name:str="cpu")->dict:
    device=device_for(device_name)
    cp=load_checkpoint(checkpoint,device)
    cfg=cp["config"];scaling=Scaling.from_dict(cp["scaling"])
    torch.set_num_threads(cfg["training"]["threads"])
    datasets,_,audit=prepare(cfg,scaling)
    if audit["canonical_data_sha256"]!=cp["audit"]["canonical_data_sha256"] or audit.get("semantic_data_sha256")!=cp["audit"].get("semantic_data_sha256"):
        raise ValueError("Данные изменились после обучения. Нельзя выдавать их за исходную тестовую выборку")
    path=Path(output)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError("Папка оценки не пуста; выберите новую, чтобы сохранить историю")
    path.mkdir(parents=True,exist_ok=True)
    model=AdaptiveLSTM(cfg["model"],cfg["data"]["horizons"],cfg["semantics"]).to(device)
    model.load_state_dict(cp["model_state"])
    result=evaluate_dataset(model,datasets["test"],scaling,cfg,device,path/"predictions.csv")
    result.update(checkpoint_epoch=cp["epoch"]+1,selected_on=cp["selected_on"],dataset=audit,
                  environment=environment(device),source=source_manifest())
    write_json(path/"metrics.json",result)
    return result


def predict(checkpoint:str|Path,csv_path:str|Path,output:str|Path,device_name:str="cpu")->list[dict]:
    device=device_for(device_name);cp=load_checkpoint(checkpoint,device)
    cfg=cp["config"];scaling=Scaling.from_dict(cp["scaling"])
    cfg["data"]["source"]="csv";cfg["data"]["path"]=str(csv_path);cfg["data"]["max_rows"]=None
    frame,_=load_frame(cfg,inference=True)
    model=AdaptiveLSTM(cfg["model"],cfg["data"]["horizons"],cfg["semantics"]).to(device)
    model.load_state_dict(cp["model_state"]);model.eval()
    results=[];length=cfg["data"]["window"]
    torch.set_num_threads(cfg["training"]["threads"])
    with torch.no_grad():
        for obj,group in frame.groupby("object_id",sort=False):
            segment=group[group["segment"]==group["segment"].iloc[-1]]
            if len(segment)<length+1:
                raise ValueError(f"{obj}: для инференса нужны {length+1} непрерывных прошлых отсчетов")
            window=segment.tail(length)
            x,mask=group_tensor(scaling.transform(window[FEATURES].to_numpy()))
            rate=window["mass_rate"].to_numpy(dtype=np.float32)/scaling.mass_rate_scale
            out=model(torch.from_numpy(x[None]).to(device),torch.from_numpy(mask[None]).to(device),
                      torch.from_numpy(rate[None,:,None]).to(device))
            pressure=scaling.inverse_tensor(out["pressure_normalized"].double(),0)[0].cpu().tolist()
            flow=scaling.inverse_tensor(out["flow_normalized"].double(),6)[0].cpu().tolist() if cfg["model"]["predict_flow"] else None
            prob=out["accident_probability"][0].cpu().tolist() if cfg["model"]["classify"] else None
            origin=window["timestamp"].iloc[-1]
            for j,h in enumerate(cfg["data"]["horizons"]):
                p=pressure[j]
                row={"object_id":obj,"origin":origin.isoformat(),
                     "target_time":(origin+pd.Timedelta(seconds=cfg["data"]["sample_period_seconds"]*h)).isoformat(),
                     "horizon_steps":h,"pressure_pa":p,"pressure_bar":p/1e5,
                     "mass_out_kg_s":flow[j] if flow is not None else None,
                     "accident_probability":prob[j] if prob is not None else None,
                     "future_mass_balance_verified":False,
                     "semantic_explanation_verified":False,
                     "status":"forecast_requires_operator_and_physical_scenario_verification"}
                if prob is None:
                    lo,hi=cfg["data"]["low_pressure_pa"],cfg["data"]["high_pressure_pa"]
                    row["pressure_threshold_exceeded"]=(p<lo if lo is not None else False) or (p>hi if hi is not None else False)
                results.append(row)
    if Path(output).exists():
        raise FileExistsError("Файл прогноза уже существует")
    write_json(output,results)
    return results
