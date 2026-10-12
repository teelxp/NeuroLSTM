import math
import numpy as np
import pytest
import torch
from neurolstm.physics import gas_inventory,gas_mass_residual,gas_loss,hydro_balance,active_power_residual,apparent_power_violation,rating_curve,muskingum_residual
from neurolstm.gradients import combine_gradients,ramp,learning_rate,flattened_gradient
from neurolstm.semantics import semantic_time,pure_fidelity,graph_coherence,wave_residual,kl_from_embeddings,causal_interval
from neurolstm.clustering import EVKMeans,clustered_order
from neurolstm.config import DEFAULT


def meta():
    x=lambda v:torch.tensor([[v]],dtype=torch.float64)
    return {"pressure_now":x(5e6),"temperature_now":x(293.15),"z_now":x(.9),
            "temperature_future":x(293.15),"z_future":x(.9),"volume":x(1000),
            "molar_mass":x(.018),"mean_mass_in":x(20.),"mean_mass_out":x(20.),"elapsed_seconds":x(60.)}


def test_mass_zero_when_steady():
    p=torch.tensor([[5e6]],dtype=torch.float64,requires_grad=True)
    r=gas_mass_residual(p,meta())
    assert abs(r.item())<1e-12


def test_accumulation_not_false_violation():
    m=meta();m["mean_mass_in"]+=1
    stock=gas_inventory(m["pressure_now"],m["temperature_now"],m["z_now"],m["volume"],m["molar_mass"])
    future=(stock+60)*m["z_future"]*8.31446261815324*m["temperature_future"]/(m["molar_mass"]*m["volume"])
    assert abs(gas_mass_residual(future,m).item())<1e-10


def test_physical_gradient_matches_finite_difference():
    m=meta();p=torch.tensor([[5.02e6]],dtype=torch.float64,requires_grad=True)
    cfg=dict(DEFAULT["physics"])
    loss,_=gas_loss(p,m,cfg,1.)
    grad=torch.autograd.grad(loss,p)[0].item()
    delta=1.
    a=gas_loss(p.detach()+delta,m,cfg,1.)[0].item()
    b=gas_loss(p.detach()-delta,m,cfg,1.)[0].item()
    assert grad!=0
    assert grad==pytest.approx((a-b)/(2*delta),rel=1e-7)


def test_gradient_combination_literal_formula():
    gd=torch.tensor([2.,1.,0.],dtype=torch.float64)
    gp=torch.tensor([-1.,2.,3.],dtype=torch.float64)
    gs=torch.tensor([4.,1.,2.],dtype=torch.float64)
    gr=torch.tensor([1.,1.,1.],dtype=torch.float64)
    eps=1e-8
    p=gp-(gp@gd)/(gd@gd+eps)*gd
    s=gs-(gs@gd)/(gd@gd+eps)*gd-(gs@p)/(p@p+eps)*p
    actual,_=combine_gradients(gd,gp,gs,gr,2.,3.,.4,eps)
    assert torch.allclose(actual,gd+2*p+3*s+.4*gr,atol=1e-10)


def test_epsilon_not_claimed_exact_orthogonality():
    gd=torch.tensor([1.],dtype=torch.float64);gp=-gd;zero=torch.zeros_like(gd)
    out,_=combine_gradients(gd,gp,zero,zero,1.,0.,0.,.1)
    assert out.item()==pytest.approx(1-.1/1.1)


def test_zero_data_gradient_keeps_physics():
    g=torch.tensor([2.,3.]);z=torch.zeros(2)
    out,_=combine_gradients(z,g,z,z,1.,0.,0.)
    assert torch.equal(out,g)


def test_sum_mode_has_no_projection():
    g=torch.tensor([2.,3.])
    out,_=combine_gradients(g,g,g,g,2.,3.,4.,mode="sum")
    assert torch.equal(out,10*g)


@pytest.mark.parametrize("q,fraction",[(0,0.),(10,1-math.exp(-1)),(30,1-math.exp(-3))])
def test_ramp(q,fraction):
    assert ramp(q,2.,10.)==pytest.approx(2*fraction)


def test_clamped_cosine_does_not_restart():
    c=DEFAULT["training"]
    assert learning_rate(0,c)==pytest.approx(c["lr"])
    assert learning_rate(100,c)==pytest.approx(c["eta_min"])
    assert learning_rate(25000,c)==pytest.approx(c["eta_min"])


def test_semantic_time_monotone_and_gradients():
    e=torch.randn(2,4,5,requires_grad=True)
    t=semantic_time(e,.1,5.)
    assert torch.all(t[:,1:]>t[:,:-1])
    assert torch.equal(t[:,0],torch.ones(2))
    t.sum().backward();assert e.grad is not None and torch.isfinite(e.grad).all()


def test_causal_interval_sign():
    pos=torch.tensor([[[0.],[1.]]]);time=torch.tensor([[1.,3.]])
    result=causal_interval(pos,time,1.,torch.ones(1))
    assert result[0,0,1].item()==pytest.approx(-3.)


def test_pure_fidelity_global_phase_invariant():
    states=torch.tensor([[[1.+0j,0j],[1j,0j],[0j,1.+0j]]])
    f=pure_fidelity(states)
    assert f[0,0,1].item()==pytest.approx(1.)
    assert f[0,0,2].item()==pytest.approx(0.)


def test_wave_operator_known_solution():
    t=torch.arange(5,dtype=torch.float64)[:,None]
    x=torch.arange(6,dtype=torch.float64)[None,:]
    field=(t.square()+x.square())[None].to(torch.complex128)
    r=wave_residual(field,torch.zeros(1,3,4,dtype=torch.complex128),torch.ones_like(field.real),1.,1.,1.,0.)
    assert torch.allclose(r,torch.zeros_like(r))


def test_wave_gradient_reaches_field():
    f=torch.randn(1,4,5,dtype=torch.complex128,requires_grad=True)
    r=wave_residual(f,torch.zeros(1,2,3,dtype=torch.complex128),torch.ones(1,4,5),1.,1.,1.,.1)
    r.abs().square().mean().backward()
    assert f.grad.abs().sum()>0


def test_graph_score_empty_and_full():
    sparse=torch.eye(4)[None]
    full=torch.ones(1,4,4)
    assert graph_coherence(sparse,.5,.1).item()==0.
    assert graph_coherence(full,.5,.1).item()>0.999


def test_negative_graph_threshold_rejected():
    with pytest.raises(ValueError):graph_coherence(torch.ones(1,2,2),-.1,.1)


def test_kl_distinct_component_dimension_and_grad():
    e=torch.randn(2,3,8);p=torch.randn(2,3,8,requires_grad=True)
    w=torch.randn(2,4,8);b=torch.zeros(2,4)
    assert kl_from_embeddings(e,e,w,b).item()==pytest.approx(0.)
    loss=kl_from_embeddings(e,p,w,b);loss.backward()
    assert loss.item()>=0 and p.grad.abs().sum()>0


def test_hydro_and_power_residuals():
    x=lambda v:torch.tensor(v,dtype=torch.float64)
    assert hydro_balance(x(100),x(160),x(3),x(2),60.).item()==0
    assert active_power_residual(x(12),x([4,5]),x(3)).item()==0
    assert apparent_power_violation(x(3),x(4),x(4)).item()==1
    assert rating_curve(x(-1),2.,0.,1.5).item()==0
    assert muskingum_residual(x(5),x(5),x(5),x(5),.2,.3,.5).item()==0


def test_evkmeans_expected_separation_and_no_duplicate_sampling():
    rng=np.random.default_rng(2)
    x=np.r_[rng.normal(0,.01,(15,3)),rng.normal(4,.01,(15,3))]
    r=np.r_[np.zeros((15,1)),np.ones((15,1))*4]
    m=EVKMeans(2,2,[.5,0,.5])
    labels=m.fit_predict(x,r)
    assert len(set(labels[:15]))==len(set(labels[15:]))==1 and labels[0]!=labels[-1]
    assert 0<m.training_cvr<=1
    assert sorted(clustered_order(labels,42))==list(range(30))
    assert np.array_equal(labels,m.predict(x,r))


def test_evkmeans_missing_modal_fails():
    with pytest.raises(ValueError):EVKMeans(2,2,[.3,.3,.4]).fit_predict(np.random.randn(8,3),np.random.randn(8,2))


def test_full_5v_criterion_requires_all_components():
    from neurolstm.criteria import architecture_psi
    scores={key:.9 for key in ['Volume','Velocity','Variety','Veracity','Value']}
    assert architecture_psi(scores,.1,1.,.1,1.)['psi']==pytest.approx(.9)
    with pytest.raises(ValueError):architecture_psi({'Velocity':.9},.1,1.,.1,1.)


def test_auxiliary_criterion_not_mislabeled_as_full_5v():
    from neurolstm.criteria import auxiliary_experiment_proxy
    result=auxiliary_experiment_proxy(100.,100.,.8,.8,.8,.2,.8)
    assert result['all_5v_evaluated'] is False
    assert result['c_ent_proxy']==pytest.approx(1.)


def test_5v_criterion_rejects_nan():
    from neurolstm.criteria import architecture_psi
    scores={key:.9 for key in ['Volume','Velocity','Variety','Veracity','Value']}
    with pytest.raises(ValueError):architecture_psi(scores,float('nan'),1.,.1,1.)
