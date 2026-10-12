"""Контролируемые СИНТЕТИЧЕСКИЕ данные исключительно для проверки кода."""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
import pandas as pd
from .config import DEFAULT, validate
from .training import write_json


def make_demo(directory:str|Path,rows:int=480,with_semantics:bool=False)->Path:
    directory=Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Каталог примера не пуст")
    directory.mkdir(parents=True,exist_ok=True)
    if rows<200:
        raise ValueError("Для демонстрационного разбиения требуется не менее 200 строк")
    seed=314159;rng=np.random.default_rng(seed)
    dt=60.;volume=1000.;molar=.018;R=8.31446261815324
    i=np.arange(rows,dtype=float)
    temp=293.15+.1*np.sin(i/13)
    z=np.full(rows,.9)
    net=.2*np.sin(i/10)
    mass_in=40.+2*np.sin(i/20)
    mass_out=mass_in-net
    stock=np.empty(rows);stock[0]=5e6*molar*volume/(.9*R*temp[0])
    stock[1:]=stock[0]+np.cumsum(dt*.5*(net[1:]+net[:-1]))
    pressure=stock*z*R*temp/(molar*volume)
    times=pd.date_range("2024-01-01",periods=rows,freq="60s",tz="UTC")
    # Физические пороги демо заданы заранее, не вычисляются по тестовому ряду.
    accident=(pressure>5.015e6).astype(int)
    frame=pd.DataFrame({"object_id":"synthetic_pipe","timestamp":times.astype(str),
                        "pressure":pressure/1e5,"temperature":temp-273.15,
                        "flow_rate":mass_in/40.,"external_temp":10+np.sin(i/12),
                        "humidity":8+.2*np.cos(i/23),"mass_in":mass_in*3.6,"mass_out":mass_out*3.6,
                        "compressibility":z,"volume_m3":volume,"molar_mass_kg_mol":molar,"accident":accident})
    frame.to_csv(directory/"synthetic_telemetry.csv",index=False)
    cfg=copy.deepcopy(DEFAULT)
    cfg["profile"]="synthetic_smoke_test_NOT_dissertation_experiment"
    cfg["data"].update(path=str((directory/"synthetic_telemetry.csv").resolve()),sample_period_seconds=dt,
                       window=8,horizons=[1,2],provenance="Synthetic inventory-balanced fixture; threshold labels; seed=314159; NOT real telemetry",
                       humidity_description="synthetic absolute humidity g/m3")
    cfg["physics"].update(consistency_tolerance_kg_s=.5,residual_scale_kg_s=1.)
    cfg["model"].update(hidden_size=16,layers=2)
    cfg["training"].update(epochs=3,micro_batch_size=32,tau_phys=1.,tau_sem=1.,checkpoint_every=1,cosine_steps=3,lr=.003,eta_min=.0001)
    if with_semantics:
        e=4;d=8;k=4
        input_emb=np.zeros((rows,e,d),np.float32)
        base=rng.normal(size=(e,d)).astype(np.float32)
        for n in range(rows):
            input_emb[n]=base+(pressure[n]/5e6-1)*np.arange(1,d+1,dtype=np.float32)[None,:]
        reference=.8*input_emb+.1
        cause=np.zeros((rows,e,e),np.float32);cause[:,0,1]=1;cause[:,1,2]=1
        coherent=np.zeros_like(cause);coherent[:,0,1]=1
        separate=np.zeros_like(cause);separate[:,0,3]=1
        ns=times.astype("int64").to_numpy()
        np.savez_compressed(directory/"synthetic_semantics.npz",
             object_id=np.full(rows,"synthetic_pipe"),origin_time_ns=ns,
             available_time_ns=ns[:,None]-np.arange(e-1,-1,-1,dtype=np.int64)[None,:]*1_000_000_000,
             input_embeddings=input_emb,reference_embeddings=reference,
             basis_weight=np.eye(k,d,dtype=np.float32),basis_bias=np.zeros(k,np.float32),
             projection=np.eye(d,2,dtype=np.float32),cause_pairs=cause,coherent_pairs=coherent,separate_pairs=separate,
             wave_source=np.zeros((rows,2,3),np.complex64),wave_index=np.ones((rows,4,5),np.float32),
             wave_intensity=np.ones((rows,4,5),np.float32))
        cfg["semantics"].update(path=str((directory/"synthetic_semantics.npz").resolve()),embedding_size=d,state_components=k,
                                 spatial_weights=[1.,1.])
        cfg["semantics"]["weights"].update(reconstruction=1.0,cause=.05,phase=.1,graph=.1,wave=.05)
    config_path=directory/"demo_config.json"
    write_json(config_path,validate(cfg))
    write_json(directory/"PROVENANCE.json",{"source":"generated_synthetic_fixture","seed":seed,"rows":rows,
                                           "purpose":"functional_tests_only","source_equations":"gas stock balance (4.8)-(4.11)",
                                           "not_evidence_for_dissertation_tables":True})
    return config_path
