"""Семантические операторы главы 3 и дополнительная обучаемая компонента.

Формулы не заменены оценкой 'качества текста' по одному cosine. Входы —
явно подготовленные эмбеддинги, разметка пар и, при необходимости, поле/источник.
Устройство SemanticHead — инженерная параметризация не заданного в диссертации
отображения T_Theta. Оно не является LLM и не генерирует естественный язык.
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


def cosine_matrix(embeddings: Tensor, epsilon: float = 1e-8) -> Tensor:
    """(3.41), epsilon прибавлен к произведению норм, как в тексте."""
    norm = embeddings.norm(dim=-1)
    return (embeddings @ embeddings.transpose(-1,-2)) / (norm.unsqueeze(-1)*norm.unsqueeze(-2)+epsilon)


def semantic_time(embeddings: Tensor, gamma: float, tau: float, epsilon: float = 1e-8) -> Tensor:
    """(3.40)-(3.42); последняя ось событий. Никаких отрицательных шагов времени."""
    if gamma < 0 or tau <= 0:
        raise ValueError("Требуются gamma>=0 и tau>0")
    n = embeddings.shape[-2]
    if n < 1:
        raise ValueError("Нет смысловых событий")
    sim = (1+cosine_matrix(embeddings,epsilon))/2
    values = [torch.ones_like(embeddings[...,0,0])]
    for i in range(1,n):
        distance = torch.arange(i,0,-1,device=embeddings.device,dtype=embeddings.dtype)
        increment = 1+gamma*(torch.exp(-distance/tau)*sim[...,i,:i]).sum(dim=-1)
        values.append(values[-1]+increment)
    return torch.stack(values,dim=-1)


def causal_interval(positions: Tensor, times: Tensor, alpha: float, beta: Tensor) -> Tensor:
    """(3.46): отрицательное время, положительные пространственные вклады."""
    spatial = positions.unsqueeze(-2)-positions.unsqueeze(-3)
    temporal = times.unsqueeze(-1)-times.unsqueeze(-2)
    return (spatial.square()*beta).sum(dim=-1)-alpha*temporal.square()


def pair_average(values: Tensor, pairs: Tensor) -> Tensor:
    count = pairs.sum(dim=(-1,-2))
    if (count <= 0).any():
        raise ValueError("Положительный вес ограничения требует непустой разметки пар в каждом примере")
    return ((values*pairs).sum(dim=(-1,-2))/count).mean()


def pure_fidelity(states: Tensor) -> Tensor:
    """(3.72)-(3.74), нормированные комплексные состояния."""
    norm = states.abs().square().sum(-1,keepdim=True).sqrt().clamp_min(1e-12)
    states = states/norm
    return (states.conj() @ states.transpose(-1,-2)).abs().square().clamp(0,1)


def graph_coherence(embeddings: Tensor, threshold: float, key_fraction: float,
                    epsilon: float = 1e-8) -> Tensor:
    """(3.81)-(3.82): пороговый граф, взвешенные степени, НЕ тест связности.

    threshold>=0 исключает отрицательные ребра. Выбор ребер/top-k дискретный;
    производные считаются по весам при текущей топологии (кусочная гладкость).
    """
    if not 0 <= threshold < 1 or not 0 < key_fraction <= 1:
        raise ValueError("Некорректные параметры графа")
    sim = cosine_matrix(embeddings,epsilon)
    n = sim.shape[-1]
    eye = torch.eye(n,device=sim.device,dtype=torch.bool)
    adjacency = torch.where((sim>=threshold)&~eye,sim,torch.zeros_like(sim))
    degree = adjacency.sum(-1)
    k = max(1,math.ceil(key_fraction*n))
    relative_key = degree.topk(k,dim=-1).values.mean(-1)/(degree.max(-1).values+epsilon)
    mean = degree.mean(-1)
    std = degree.std(-1,unbiased=False)
    return relative_key*mean/(mean+std+epsilon)


def kl_from_embeddings(reference: Tensor, predicted: Tensor, weight: Tensor, bias: Tensor,
                       mode: str = "log_softmax", epsilon: float = 1e-8) -> Tensor:
    """(3.88)-(3.90). Фиксированная матрица W_p исключает ее схлопывание в 0."""
    ref_logits = torch.einsum("bed,bkd->bek",reference,weight)+bias[:,None,:]
    pred_logits = torch.einsum("bed,bkd->bek",predicted,weight)+bias[:,None,:]
    logp, logq = ref_logits.log_softmax(-1), pred_logits.log_softmax(-1)
    if mode == "dissertation_epsilon":
        # Буквальное epsilon-выражение из (3.90); отмечено как другой функционал.
        return (logp.exp()*((logp.exp()+epsilon).log()-(logq.exp()+epsilon).log())).sum(-1).mean()
    return (logp.exp()*(logp-logq)).sum(-1).mean()


def wave_residual(field: Tensor, source: Tensor, index: Tensor, dt: float,
                  dx: float, speed: float, damping: float) -> Tensor:
    """(3.57)-(3.64), комплексное поле [B,Nt,Nx], внутренние узлы.

    Это вычисление невязки на заданной сетке, а не явный временной интегратор.
    Коэффициент 1/n^2 на гранях — среднее двух соседних значений (явная
    инженерная конкретизация). Источник shape=[B,Nt-2,Nx-2].
    """
    if field.ndim != 3 or min(field.shape[-2:])<3 or field.shape != index.shape:
        raise ValueError("Поле и показатель среды должны иметь одинаковую форму [B,Nt>=3,Nx>=3]")
    if source.shape != field[:,1:-1,1:-1].shape:
        raise ValueError("Источник задается во внутренних узлах")
    if dt <= 0 or dx <= 0 or speed <= 0 or damping < 0 or (index<=0).any():
        raise ValueError("Некорректные параметры волнового оператора")
    u = field[:,1:-1,:]
    a = index[:,1:-1,:].pow(-2)
    faces = .5*(a[...,1:]+a[...,:-1])
    flux = faces*(u[...,1:]-u[...,:-1])/dx
    divergence = (flux[...,1:]-flux[...,:-1])/dx
    temporal2 = (field[:,2:,1:-1]-2*field[:,1:-1,1:-1]+field[:,:-2,1:-1])/(dt*dt)
    temporal1 = (field[:,1:-1,1:-1]-field[:,:-2,1:-1])/dt
    return temporal2-speed*speed*divergence+damping*temporal1-source


class SemanticHead(nn.Module):
    def __init__(self, hidden_size: int, cfg: dict):
        super().__init__()
        d, k = cfg["embedding_size"], cfg["state_components"]
        self.cfg = cfg
        self.decoder = nn.Sequential(nn.Linear(hidden_size+d,hidden_size),nn.Tanh(),nn.Linear(hidden_size,d))
        self.state_decoder = nn.Linear(d,2*k) if cfg["weights"]["phase"] else None
        self.field_decoder = nn.Sequential(nn.Linear(hidden_size+2,hidden_size),nn.Tanh(),nn.Linear(hidden_size,2)) if cfg["weights"]["wave"] else None

    def forward(self, hidden: Tensor, batch: dict[str,Tensor]) -> dict[str,Tensor]:
        inp = batch["input_embeddings"].to(hidden.dtype)
        expanded = hidden[:,None,:].expand(-1,inp.shape[1],-1)
        embedding = self.decoder(torch.cat([expanded,inp],dim=-1))
        result = {"embedding":embedding}
        if self.state_decoder is not None:
            amplitude, phase = self.state_decoder(embedding).chunk(2,dim=-1)
            result["states"] = torch.polar(amplitude.softmax(-1).sqrt(),phase)
        if self.field_decoder is not None:
            shape = batch["wave_intensity"].shape
            nt,nx = shape[-2:]
            tt,xx = torch.meshgrid(torch.arange(nt,device=hidden.device,dtype=hidden.dtype)*self.cfg["wave_dt"],
                                  torch.arange(nx,device=hidden.device,dtype=hidden.dtype)*self.cfg["wave_dx"],indexing="ij")
            coordinates = torch.stack([tt,xx],dim=-1)[None].expand(len(hidden),-1,-1,-1)
            h = hidden[:,None,None,:].expand(-1,nt,nx,-1)
            real_imag = self.field_decoder(torch.cat([h,coordinates],dim=-1))
            result["field"] = torch.complex(real_imag[...,0],real_imag[...,1])
        return result


def semantic_loss(prediction: dict, batch: dict, cfg: dict) -> tuple[Tensor,dict[str,Tensor]]:
    pred = prediction["embedding"]
    reference = batch["reference_embeddings"].to(pred.dtype)
    weights = cfg["weights"]
    components: dict[str,Tensor] = {}
    if weights["reconstruction"]:
        # Эмпирическая привязка декодера к эталону; это инженерная конкретизация,
        # а не новая формула диссертации и не автоматическое доказательство смысла.
        components["reconstruction"] = F.mse_loss(pred,reference)
    if weights["cause"]:
        projection = batch["projection"].to(pred.dtype)
        position = torch.einsum("bed,bdr->ber",pred,projection)
        times = semantic_time(pred,cfg["time_gamma"],cfg["memory_tau"],cfg["epsilon"])
        beta = pred.new_tensor(cfg["spatial_weights"])
        if beta.numel()!=position.shape[-1] or (beta<=0).any() or cfg["minkowski_alpha"]<=0:
            raise ValueError("spatial_weights должны соответствовать положительным весам координат")
        interval = causal_interval(position,times,cfg["minkowski_alpha"],beta)
        components["cause"] = pair_average(torch.relu(interval).square(),batch["cause_pairs"])
    if weights["phase"]:
        fidelity = pure_fidelity(prediction["states"])
        components["phase"] = pair_average((1-fidelity).square(),batch["coherent_pairs"])
        if cfg["lambda_separation"]:
            components["phase"] = components["phase"]+cfg["lambda_separation"]*pair_average(fidelity.square(),batch["separate_pairs"])
    if weights["graph"]:
        score = graph_coherence(pred,cfg["graph_threshold"],cfg["graph_key_fraction"],cfg["epsilon"])
        components["graph"] = torch.relu(cfg["graph_target"]-score).square().mean()
    if weights["entropy"]:
        components["entropy"] = kl_from_embeddings(reference,pred,batch["basis_weight"].to(pred.dtype),
                                                    batch["basis_bias"].to(pred.dtype),cfg["kl_mode"],cfg["epsilon"])
    if weights["wave"]:
        field = prediction["field"]
        residual = wave_residual(field,batch["wave_source"],batch["wave_index"],
                                 cfg["wave_dt"],cfg["wave_dx"],cfg["wave_speed"],cfg["wave_damping"])
        components["wave"] = residual.abs().square().mean()+cfg["wave_intensity_weight"]*(field.abs().square()-batch["wave_intensity"]).square().mean()
    if not components:
        raise ValueError("Все семантические веса нулевые")
    total = sum(weights[key]*value for key,value in components.items())
    return total,components


class SemanticStore:
    """NPZ без pickle. Все пары задаются разметкой, не генерируются из предсказаний."""
    def __init__(self, path: str | Path, cfg: dict):
        with np.load(path,allow_pickle=False) as data:
            self.arrays = {k:data[k] for k in data.files}
        a = self.arrays
        mandatory = ["object_id","origin_time_ns","available_time_ns","input_embeddings","reference_embeddings","basis_weight","basis_bias"]
        if cfg["weights"]["cause"]:
            mandatory += ["cause_pairs","projection"]
        if cfg["weights"]["phase"]:
            mandatory += ["coherent_pairs"]
            if cfg["lambda_separation"]:
                mandatory += ["separate_pairs"]
        if cfg["weights"]["wave"]:
            mandatory += ["wave_source","wave_index","wave_intensity"]
        missing = [k for k in mandatory if k not in a]
        if missing:
            raise ValueError("Семантический файл: отсутствуют "+", ".join(missing))
        n = len(a["origin_time_ns"])
        if a["input_embeddings"].ndim != 3 or a["reference_embeddings"].shape!=a["input_embeddings"].shape:
            raise ValueError("Ожидаются эмбеддинги [N,E,D]")
        _,e,d = a["input_embeddings"].shape
        if d!=cfg["embedding_size"] or a["basis_weight"].shape!=(cfg["state_components"],d):
            raise ValueError("Размерности семантического файла не соответствуют конфигурации")
        if a["basis_bias"].shape!=(cfg["state_components"],):
            raise ValueError("Некорректный basis_bias")
        if a["available_time_ns"].shape!=(n,e) or (a["available_time_ns"]>a["origin_time_ns"][:,None]).any():
            raise ValueError("Входное текстовое событие доступно позже момента прогноза: утечка будущей информации")
        self.global_keys = {"basis_weight","basis_bias","projection"}
        for key,value in a.items():
            if key not in self.global_keys and len(value)!=n:
                raise ValueError(f"Неверная первая размерность {key}")
            if np.issubdtype(value.dtype,np.number) and not np.isfinite(value).all():
                raise ValueError(f"NaN/Inf в {key}")
        for key in ("cause_pairs","coherent_pairs","separate_pairs"):
            if key in mandatory:
                if a[key].shape!=(n,e,e) or not np.isin(a[key],[0,1]).all() or np.any(np.diagonal(a[key],axis1=-2,axis2=-1)):
                    raise ValueError(f"{key}: нужны матрицы [N,E,E] без петель")
                if np.any(a[key].sum(axis=(-1,-2))<=0):
                    raise ValueError(f"{key}: пустая разметка недопустима при положительном весе")
        if "cause_pairs" in mandatory and np.tril(a["cause_pairs"]).any():
            raise ValueError("cause_pairs[i,j] допускается только при j>i")
        if "coherent_pairs" in mandatory and "separate_pairs" in mandatory and np.any(a["coherent_pairs"]*a["separate_pairs"]):
            raise ValueError("Одна пара одновременно объявлена согласованной и разделенной")
        if a["origin_time_ns"].dtype.kind not in "iu" or a["available_time_ns"].dtype.kind not in "iu":
            raise ValueError("Временные метки семантических данных должны быть целыми наносекундами")
        if "projection" in mandatory and (a["projection"].ndim != 2 or a["projection"].shape[0] != d):
            raise ValueError("projection должна иметь форму [D,r]")
        if "wave_source" in mandatory:
            shape = a["wave_intensity"].shape
            if len(shape) != 3 or min(shape[-2:]) < 3 or a["wave_index"].shape != shape:
                raise ValueError("wave_intensity и wave_index должны иметь форму [N,Nt>=3,Nx>=3]")
            if a["wave_source"].shape != (n,shape[1]-2,shape[2]-2):
                raise ValueError("wave_source задается во внутренних узлах [N,Nt-2,Nx-2]")
            if (a["wave_index"] <= 0).any() or (a["wave_intensity"] < 0).any():
                raise ValueError("Показатель среды положителен, интенсивность неотрицательна")
        self.index = {(str(obj),int(time)):i for i,(obj,time) in enumerate(zip(a["object_id"],a["origin_time_ns"]))}
        if len(self.index)!=n:
            raise ValueError("Повторяющиеся object_id/origin_time_ns в семантическом файле")
        self.value_keys = [key for key in mandatory if key not in {"object_id","origin_time_ns","available_time_ns"}]

    def has(self, obj: str, time: int) -> bool:
        return (obj,int(time)) in self.index

    def get(self, obj: str, time: int) -> dict[str,Tensor]:
        i = self.index[(obj,int(time))]
        result = {}
        for key in self.value_keys:
            value = self.arrays[key] if key in self.global_keys else self.arrays[key][i]
            dtype = torch.complex64 if np.iscomplexobj(value) else torch.float32
            result[key] = torch.as_tensor(np.array(value,copy=True),dtype=dtype)
        return result
