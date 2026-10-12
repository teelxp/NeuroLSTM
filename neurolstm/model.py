"""Адаптивная рекуррентная модель по (4.19)-(4.20), выходы по (6.2)."""
from __future__ import annotations
import torch
from torch import Tensor, nn


class AdaptiveLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, context_size: int = 2,
                 layer_norm: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        # Четыре затвора объединены в один Linear только для эффективности.
        # Порядок f, i, candidate, o. Ровно один bias на затвор.
        self.W = nn.Linear(input_size, 4*hidden_size, bias=False)
        self.U = nn.Linear(hidden_size, 4*hidden_size, bias=False)
        self.V = nn.Linear(context_size, 4*hidden_size, bias=False)
        self.M = nn.Linear(1, 4*hidden_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(4*hidden_size))
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(4)]) if layer_norm else None

    def forward(self, x: Tensor, h: Tensor, c: Tensor, z: Tensor,
                mass_rate: Tensor) -> tuple[Tensor, Tensor]:
        chunks = (self.W(x)+self.U(h)+self.V(z)+self.M(mass_rate)+self.bias).chunk(4, dim=-1)
        if self.norms is not None:
            chunks = tuple(norm(v) for norm, v in zip(self.norms, chunks))
        f, i, candidate, o = chunks
        c_new = torch.sigmoid(f)*c + torch.sigmoid(i)*torch.tanh(candidate)
        h_new = torch.sigmoid(o)*torch.tanh(c_new)
        return h_new, c_new


class AdaptiveLSTM(nn.Module):
    """4D: [B,T,3,3]. Маска отделяет padding от реальных нулевых значений.

    group0=[pressure, temperature, flow_rate]; group1=[external_temp,humidity,0];
    group2=[mass_in,mass_out,0]. mass_rate передается отдельным причинным входом.
    В режиме three_base_features W получает group0, V — 2 контекстных признака.
    В seven_telemetry_features W получает все 7 признаков, V — контекст, как в 4.20.
    """
    def __init__(self, cfg: dict, horizons: list[int], semantic_cfg: dict | None = None):
        super().__init__()
        self.cfg = dict(cfg)
        self.horizons = list(horizons)
        self.hidden_size = cfg["hidden_size"]
        self.layers = cfg["layers"]
        size = 3 if cfg["sensor_mode"] == "three_base_features" else 7
        self.cells = nn.ModuleList([AdaptiveLSTMCell(size if layer == 0 else self.hidden_size,
                                   self.hidden_size, layer_norm=cfg["layer_norm"])
                                   for layer in range(self.layers)])
        self.dropout = nn.Dropout(cfg["dropout"])
        self.channels = 1 + int(cfg["predict_flow"]) + int(cfg["classify"])
        self.output = nn.Linear(self.hidden_size, len(horizons)*self.channels)
        self.semantic = None
        if semantic_cfg and semantic_cfg["path"] is not None:
            from .semantics import SemanticHead
            self.semantic = SemanticHead(self.hidden_size, semantic_cfg)

    def forward(self, x: Tensor, mask: Tensor, mass_rate: Tensor) -> dict[str, Tensor]:
        if x.ndim != 4 or x.shape[-2:] != (3,3) or mask.shape != x.shape:
            raise ValueError("Ожидаются x и mask формы [B,T,3,3]")
        if mass_rate.shape != x.shape[:2]+(1,):
            raise ValueError("mass_rate должен иметь форму [B,T,1]")
        x = x * mask  # никаких будущих наблюдений в forward.
        batch, steps = x.shape[:2]
        h = [x.new_zeros(batch, self.hidden_size) for _ in range(self.layers)]
        c = [x.new_zeros(batch, self.hidden_size) for _ in range(self.layers)]
        for i in range(steps):
            sensor = x[:,i,0,:]
            context = x[:,i,1,:2]
            if self.cfg["sensor_mode"] == "seven_telemetry_features":
                sensor = torch.cat([sensor, context, x[:,i,2,:2]], dim=-1)
            hidden = sensor
            for layer, cell in enumerate(self.cells):
                h[layer], c[layer] = cell(hidden, h[layer], c[layer], context, mass_rate[:,i])
                hidden = self.dropout(h[layer])
        y = self.output(hidden).reshape(batch, len(self.horizons), self.channels)
        result = {"pressure_normalized": y[...,0], "hidden": hidden}
        pos = 1
        if self.cfg["predict_flow"]:
            result["flow_normalized"] = y[...,pos]
            pos += 1
        if self.cfg["classify"]:
            result["accident_logit"] = y[...,pos]
            result["accident_probability"] = torch.sigmoid(y[...,pos])
        return result

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
