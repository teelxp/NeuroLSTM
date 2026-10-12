import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from neurolstm.training import train,evaluate,predict,load_checkpoint,write_json
from neurolstm.metrics import RegressionClassificationMetrics
from neurolstm.decision import statistical_score,consistency_assessment,rank_actions,action_utility
from neurolstm.semantics import SemanticStore
from neurolstm.pinn import gas_pde_residual,train_pinn


def test_train_evaluate_predict_and_artifacts(demo_cfg,tmp_path):
    out=tmp_path/"run"
    summary=train(demo_cfg,out)
    assert summary["training_status"]=="complete" and summary["test_used_for_selection"] is False
    assert not (out/"metrics.json").exists()
    cp=load_checkpoint(out/"best.pt",torch.device("cpu"))
    assert cp["selected_on"]=="validation"
    result=evaluate(out/"best.pt",tmp_path/"eval")
    assert len(result["horizons"])==2
    assert (tmp_path/"eval"/"predictions.csv").exists()
    pred=predict(out/"best.pt",demo_cfg["data"]["path"],tmp_path/"forecast.json")
    assert len(pred)==2 and pred[0]["future_mass_balance_verified"] is False
    assert "sem_cause" not in result["semantic_components_evaluated"]


def test_resume_reproduces_uninterrupted_run(demo_cfg,tmp_path):
    cfg=copy.deepcopy(demo_cfg);cfg["training"]["epochs"]=3
    train(cfg,tmp_path/"continuous")
    early=copy.deepcopy(cfg);early["training"]["epochs"]=1
    train(early,tmp_path/"resumed")
    train(cfg,tmp_path/"resumed",tmp_path/"resumed"/"last.pt")
    one=load_checkpoint(tmp_path/"continuous"/"last.pt",torch.device("cpu"))
    two=load_checkpoint(tmp_path/"resumed"/"last.pt",torch.device("cpu"))
    for key in one["model_state"]:
        assert torch.equal(one["model_state"][key],two["model_state"][key]),key


def test_existing_run_not_overwritten(demo_cfg,tmp_path):
    out=tmp_path/"run";out.mkdir();(out/"keep.txt").write_text("keep")
    with pytest.raises(FileExistsError):train(demo_cfg,out)
    assert (out/"keep.txt").read_text()=="keep"


def test_ch4_pressure_only(demo_cfg,tmp_path):
    demo_cfg["model"].update(predict_flow=False,classify=False,sensor_mode="seven_telemetry_features")
    demo_cfg["data"].update(train_fraction=.8,validation_fraction=0.)
    demo_cfg["training"]["epochs"]=1
    train(demo_cfg,tmp_path/"ch4")
    cp=load_checkpoint(tmp_path/"ch4"/"best.pt",torch.device("cpu"))
    assert cp["selected_on"]=="last_epoch_without_validation"
    result=evaluate(tmp_path/"ch4"/"best.pt",tmp_path/"eval_ch4")
    assert "accuracy_percent" not in result["horizons"][0]


def test_full_semantic_training(semantic_cfg,tmp_path):
    train(semantic_cfg,tmp_path/"semantic_run")
    record=json.loads((tmp_path/"semantic_run"/"history.jsonl").read_text().splitlines()[-1])
    assert record["gradients"]["norm_sem"]>0
    result=evaluate(tmp_path/"semantic_run"/"best.pt",tmp_path/"semantic_eval")
    assert "sem_wave" in result["semantic_components_evaluated"]


def test_changed_data_rejected_for_existing_test(demo_cfg,tmp_path):
    train(demo_cfg,tmp_path/"run")
    source=pd.read_csv(demo_cfg["data"]["path"]);source.loc[len(source)-1,"pressure"]+=1
    source.to_csv(demo_cfg["data"]["path"],index=False)
    with pytest.raises(ValueError,match="изменились"):
        evaluate(tmp_path/"run"/"best.pt",tmp_path/"eval")


def test_semantic_future_event_rejected(semantic_cfg,tmp_path):
    path=Path(semantic_cfg["semantics"]["path"])
    with np.load(path,allow_pickle=False) as f:a={k:f[k] for k in f.files}
    a["available_time_ns"][0,0]=a["origin_time_ns"][0]+1
    changed=tmp_path/"future.npz";np.savez(changed,**a)
    with pytest.raises(ValueError,match="позже"):
        SemanticStore(changed,semantic_cfg["semantics"])


def test_cluster_minibatch_path(demo_cfg,tmp_path):
    demo_cfg["training"].update(batch_mode="mini",epochs=1)
    demo_cfg["clustering"].update(enabled=True,clusters=3,pca_components=3)
    result=train(demo_cfg,tmp_path/"clustered")
    assert result["clustering"]["cvr"]>0
    labels=np.load(tmp_path/"clustered"/"train_cluster_labels.npy")
    assert len(labels)==result["data"]["windows"]["train"]


def test_metrics_are_computed_not_injected():
    m=RegressionClassificationMetrics([1],.1,1.,.5,.5)
    m.update(np.array([[10.],[10.]]),np.array([[10.],[12.]]),np.array([[0.],[1.]]),np.array([[0],[1]]),np.array([[.1],[.2]]))
    r=m.result()["horizons"][0]
    assert r["pressure_mae_pa"]==1
    assert r["pressure_within_tolerance_percent"]==50
    assert r["physics_consistency_percent"]==50
    assert r["accuracy_percent"]==50 and r["f1"]==0
    assert r["confusion_matrix"]=={"tn":1,"fp":0,"fn":1,"tp":0}


def test_undefined_class_metrics_are_null():
    m=RegressionClassificationMetrics([1])
    m.update(np.ones((2,1)),np.ones((2,1)),np.zeros((2,1)),np.zeros((2,1)),np.zeros((2,1)))
    assert m.result()["horizons"][0]["f1"] is None


def assessment_args():
    return {"scores":{"stat":.9,"phys":.9,"sem":.9},"quality":{"stat":1.,"phys":1.,"sem":1.},
            "omega":{"stat":1.,"phys":1.,"sem":1.},"thresholds":{"stat":.5,"phys":.5,"sem":.5},"uncertainty":.1}


def test_decision_full_data():
    result=consistency_assessment(**assessment_args())
    assert result["admitted"] and result["psi"]==pytest.approx(.9)
    assert result["information_quality"]==pytest.approx(.81)


def test_missing_semantics_blocks_control_recommendation():
    args=assessment_args();args["scores"]["sem"]=None;args["quality"]["sem"]=0
    result=consistency_assessment(**args)
    assert not result["admitted"] and result["q_source"]==pytest.approx(2/3)
    assert result["information_quality"]==0 and np.isfinite(result["psi"])


def test_zero_physics_hard_gate_survives_smoothing():
    args=assessment_args();args["scores"]["phys"]=0
    args.update(previous_smoothed=.99,smoothing_alpha=.01)
    result=consistency_assessment(**args)
    assert result["smoothed_psi"]>0 and not result["admitted"] and result["information_quality"]==0


def test_statistical_score_requires_confirmed_history():
    with pytest.raises(ValueError):statistical_score([],100.,0.,1.)
    assert statistical_score([0.,0.],100.,0.,1.)==1.


def test_actions_filter_before_ranking():
    actions=[{"id":"unsafe","constraints":{"physical":1.},"scenario_utilities":[100.,100.]},
             {"id":"safe","constraints":{"physical":-1.,"technological":0.,"semantic":0.},"scenario_utilities":[1.,1.]},
             {"id":"missing","constraints":{"physical":None},"scenario_utilities":[2.,2.]}]
    result=rank_actions(actions,[.5,.5],1.)
    assert [a["id"] for a in result]==["safe"]


def test_pinn_derivatives_in_physical_units():
    class Analytic(torch.nn.Module):
        def forward(self,x):
            p=2*x[:,0]+3*x[:,1]
            m=5*x[:,0]+7*x[:,1]
            return torch.stack([p,m,x[:,0]*0],dim=1)
    coords=torch.tensor([[1.,2.],[2.,3.]],requires_grad=True)
    rp,rm,_=gas_pde_residual(Analytic(),coords,2.,2.,0.,1.,False)
    assert torch.allclose(rp,torch.full((2,),3.+4/np.pi*5))
    assert torch.allclose(rm,torch.full((2,),7.+np.pi*2))


def test_pinn_full_train_smoke(tmp_path):
    xx,tt=np.meshgrid(np.linspace(0,10,5),np.arange(6))
    x=xx.ravel();t=tt.ravel();split=np.where(t<2,0,np.where(t<4,1,2))
    data=tmp_path/"pinn_data.npz"
    np.savez(data,x_m=x,t_s=t,pressure_pa=5e6+100*x+250*t,
             mass_flow_kg_s=40.+np.sin(x)+t*.1,accident=(t%2).astype(float),
             split=split,is_initial=t==0,is_boundary=(x==0)|(x==10))
    cfg={"data_path":str(data),"diameter_m":.5,"sound_speed_m_s":300.,"friction_lambda":.01,"baseline_velocity_m_s":1.,
         "adam_steps":2,"lbfgs_steps":1,"collocation_points":16,"lr":.001,
         "width":8,"layers":2,"validation_every":1,"seed":42,"device":"cpu"}
    config=tmp_path/"pinn_config.json";write_json(config,cfg)
    result=train_pinn(config,tmp_path/"pinn_run")
    assert result["test_samples"]==10 and result["optimizer_steps"]==3
    assert result["direct_comparison_to_lstm_physics_percent_valid"] is False
    assert (tmp_path/"pinn_run"/"best_pinn.pt").exists()


@pytest.mark.parametrize('field,value',[('gamma',float('nan')),('previous_smoothed',1.2),('max_uncertainty',2.)])
def test_decision_rejects_nonfinite_or_invalid_params(field,value):
    args=assessment_args();args[field]=value
    with pytest.raises(ValueError):consistency_assessment(**args)


def test_pinn_table_parameter_count():
    from neurolstm.pinn import GasPINN
    model=GasPINN(torch.zeros(2),torch.ones(2),torch.zeros(2),torch.ones(2),64,6)
    assert sum(p.numel() for p in model.parameters())==21187


def test_all_sources_absent_always_blocks():
    args=assessment_args()
    args["quality"]={"stat":0.,"phys":0.,"sem":0.}
    args["required_complete"]=False
    out=consistency_assessment(**args)
    assert not out["admitted"] and "no_source_information" in out["blocking_reasons"]


def test_action_without_required_groups_is_not_accepted():
    actions=[{"id":"unchecked","constraints":{"physical":-1.},"scenario_utilities":[1.]}]
    assert rank_actions(actions,[1.])==[]


def test_action_utility_literal_equation():
    result=action_utility(.4,.2,1.,.2,.3,.1,[1.,2.,.5,1.,1.,1.])
    assert result==pytest.approx(.7)


def test_resume_different_folder_rejected(demo_cfg,tmp_path):
    with pytest.raises(ValueError,match="исходную папку"):
        train(demo_cfg,tmp_path/"other",tmp_path/"old"/"last.pt")


def test_negative_semantic_intensity_rejected(semantic_cfg,tmp_path):
    with np.load(semantic_cfg["semantics"]["path"],allow_pickle=False) as f:
        arrays={k:f[k] for k in f.files}
    arrays["wave_intensity"][0,0,0]=-1.
    path=tmp_path/"bad_intensity.npz"
    np.savez(path,**arrays)
    with pytest.raises(ValueError,match="интенсивность"):
        SemanticStore(path,semantic_cfg["semantics"])
