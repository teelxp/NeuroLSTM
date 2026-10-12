import copy
import importlib
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader
from neurolstm.config import DEFAULT,validate,_merge
from neurolstm.data import prepare,load_frame,Scaling,split_rows,FEATURES,group_tensor
from neurolstm.model import AdaptiveLSTM,AdaptiveLSTMCell
from neurolstm.training import losses_for
from neurolstm.gradients import flattened_gradient,combine_gradients


def test_import_is_side_effect_free():
    mod=importlib.import_module("Neuro")
    assert callable(mod.main)


def test_disjoint_splits_and_chronological_targets(demo_cfg):
    sets,scale,audit=prepare(demo_cfg)
    assert sets["train"].used_rows().isdisjoint(sets["validation"].used_rows())
    assert sets["train"].used_rows().isdisjoint(sets["test"].used_rows())
    assert sets["validation"].used_rows().isdisjoint(sets["test"].used_rows())
    b=sets["train"][0]
    assert torch.all(b["target_time_ns"]>b["origin_time_ns"])
    assert b["x"].shape==(8,3,3)
    assert b["mass_rate"].shape==(8,1)
    assert torch.all(b["mask"][:,1,2]==0)


def test_train_only_scaling(demo_cfg):
    frame,_=load_frame(demo_cfg);labels,_=split_rows(frame,demo_cfg["data"])
    s1=Scaling.fit(frame,labels==0,demo_cfg["physics"])
    frame.loc[labels==2,"pressure"]*=1000
    s2=Scaling.fit(frame,labels==0,demo_cfg["physics"])
    assert np.array_equal(s1.span,s2.span) and np.array_equal(s1.minimum,s2.minimum)


def test_constant_feature_zero(demo_cfg):
    frame,_=load_frame(demo_cfg);frame["humidity"]=8.
    labels,_=split_rows(frame,demo_cfg["data"])
    scale=Scaling.fit(frame,labels==0,demo_cfg["physics"])
    assert (scale.transform(frame[FEATURES].to_numpy())[:,4]==0).all()


def test_measured_flows_not_copied_from_input(demo_cfg):
    df=pd.read_csv(demo_cfg["data"]["path"]);df=df.drop(columns="mass_out");df.to_csv(demo_cfg["data"]["path"],index=False)
    with pytest.raises(ValueError,match="mass_out"):prepare(demo_cfg)


def test_future_pressure_changes_not_input(demo_cfg):
    sets,scale,_=prepare(demo_cfg);a=sets["train"][0]
    m=AdaptiveLSTM(demo_cfg["model"],demo_cfg["data"]["horizons"]).eval()
    b={k:v[None] for k,v in a.items() if k in {"x","mask","mass_rate"}}
    out1=m(**b)["pressure_normalized"]
    a["target_pressure"]+=1e6;a["meta"]["temperature_future"]+=100
    out2=m(**b)["pressure_normalized"]
    assert torch.equal(out1,out2)


def test_synthetic_true_pressure_has_zero_balance(demo_cfg):
    sets,scale,_=prepare(demo_cfg)
    batch=next(iter(DataLoader(sets["train"],batch_size=10)))
    from neurolstm.physics import gas_mass_residual
    residual=gas_mass_residual(batch["target_pressure"],batch["meta"])
    assert residual.abs().max()<1e-9


def test_units_tonnes_per_hour_to_si(demo_cfg):
    source=pd.read_csv(demo_cfg["data"]["path"])
    frame,_=load_frame(demo_cfg)
    assert np.allclose(frame["mass_in"],source["mass_in"]/3.6)
    assert np.allclose(frame["pressure"],source["pressure"]*1e5)


def test_time_gap_does_not_cross_window(demo_cfg):
    df=pd.read_csv(demo_cfg["data"]["path"]);df=df.drop(index=40).reset_index(drop=True);df.to_csv(demo_cfg["data"]["path"],index=False)
    sets,_,_=prepare(demo_cfg)
    for ds in sets.values():
        for i in ds.origins:
            values=ds.frame.iloc[i-8:i+3]["segment"].unique()
            assert len(values)==1


def test_duplicate_timestamps_rejected(demo_cfg):
    df=pd.read_csv(demo_cfg["data"]["path"]);df.loc[1,"timestamp"]=df.loc[0,"timestamp"];df.to_csv(demo_cfg["data"]["path"],index=False)
    with pytest.raises(ValueError,match="дубликаты"):prepare(demo_cfg)


def test_nan_not_silently_imputed(demo_cfg):
    df=pd.read_csv(demo_cfg["data"]["path"]);df.loc[5,"temperature"]=np.nan;df.to_csv(demo_cfg["data"]["path"],index=False)
    with pytest.raises(ValueError,match="NaN"):prepare(demo_cfg)


def test_no_validation_profile(demo_cfg):
    demo_cfg["data"].update(train_fraction=.8,validation_fraction=0.)
    sets,_,_=prepare(demo_cfg)
    assert len(sets["validation"])==0 and len(sets["train"])>0 and len(sets["test"])>0


def test_5_base_neural_inputs_parameter_count():
    m=AdaptiveLSTM(DEFAULT["model"],[1])
    assert m.parameter_count()==206303


@pytest.mark.parametrize("sensor",["three_base_features","seven_telemetry_features"])
@pytest.mark.parametrize("flow,classification",[(False,False),(True,False),(False,True),(True,True)])
def test_model_output_variants(sensor,flow,classification):
    cfg=copy.deepcopy(DEFAULT["model"]);cfg.update(hidden_size=8,layers=2,sensor_mode=sensor,predict_flow=flow,classify=classification)
    m=AdaptiveLSTM(cfg,[1,3])
    out=m(torch.randn(2,5,3,3),torch.ones(2,5,3,3),torch.zeros(2,5,1))
    assert out["pressure_normalized"].shape==(2,2)
    assert ("flow_normalized" in out)==flow and ("accident_logit" in out)==classification


def test_cell_matches_equation_4_20():
    torch.manual_seed(4)
    cell=AdaptiveLSTMCell(3,4)
    x=torch.randn(2,3);h=torch.randn(2,4);c=torch.randn(2,4);z=torch.randn(2,2);mass=torch.randn(2,1)
    f,i,cand,o=(cell.W(x)+cell.U(h)+cell.V(z)+cell.M(mass)+cell.bias).chunk(4,-1)
    expected_c=f.sigmoid()*c+i.sigmoid()*cand.tanh();expected_h=o.sigmoid()*expected_c.tanh()
    actual_h,actual_c=cell(x,h,c,z,mass)
    assert torch.equal(actual_h,expected_h) and torch.equal(actual_c,expected_c)


def test_semantic_gradient_reaches_shared_lstm(semantic_cfg):
    sets,scale,_=prepare(semantic_cfg)
    batch=next(iter(DataLoader(sets["train"],batch_size=4)))
    model=AdaptiveLSTM(semantic_cfg["model"],semantic_cfg["data"]["horizons"],semantic_cfg["semantics"])
    loss,_=losses_for(model,batch,scale,semantic_cfg)
    g=torch.autograd.grad(loss["sem"],model.cells[0].W.weight)[0]
    assert g.abs().sum()>0 and torch.isfinite(g).all()
    assert {"sem_cause","sem_phase","sem_graph","sem_entropy","sem_wave"}.issubset(loss)


def test_global_gradient_accumulation_equivalence(demo_cfg):
    torch.manual_seed(42)
    sets,scale,_=prepare(demo_cfg)
    model=AdaptiveLSTM(demo_cfg["model"],demo_cfg["data"]["horizons"]).double()
    # Входы модели double; стандартный CLI использует float32.
    def cast(b):
        return {k:cast(v) if isinstance(v,dict) else v.double() if isinstance(v,torch.Tensor) and v.is_floating_point() else v for k,v in b.items()}
    params=tuple(model.parameters())
    all_batch=cast(next(iter(DataLoader(torch.utils.data.Subset(sets["train"],range(8)),batch_size=8))))
    loss,_=losses_for(model,all_batch,scale,demo_cfg)
    gd=flattened_gradient(loss["data"],params,retain_graph=True)
    gp=flattened_gradient(loss["phys"],params,retain_graph=False)
    dsum=torch.zeros_like(gd);psum=torch.zeros_like(gp)
    for b in DataLoader(torch.utils.data.Subset(sets["train"],range(8)),batch_size=3):
        b=cast(b);l,_=losses_for(model,b,scale,demo_cfg)
        dsum+=flattened_gradient(l["data"],params,retain_graph=True)*len(b["x"])/8
        psum+=flattened_gradient(l["phys"],params,retain_graph=False)*len(b["x"])/8
    assert torch.allclose(gd,dsum,atol=1e-10,rtol=1e-8)
    assert torch.allclose(gp,psum,atol=1e-9,rtol=1e-8)


@pytest.mark.parametrize("update",[
    {"training":{"lr":0.}}, {"training":{"eta_min":.01}},
    {"data":{"train_fraction":.9,"validation_fraction":.2}},
    {"data":{"pressure_reference":"gauge","atmospheric_pressure_pa":None}},
    {"training":{"weight_decay":.01,"l2_weight":.01}},
    {"clustering":{"enabled":True}},
    {"data":{"horizons":[2,1]}}, {"training":{"gradient_mode":"fake"}},
    {"semantics":{"embedding_size":5}},
])
def test_invalid_config_rejected(update):
    with pytest.raises(ValueError):validate(_merge(DEFAULT,update))


def test_unknown_config_rejected():
    with pytest.raises(ValueError):_merge(DEFAULT,{"training":{"epohs":50}})
