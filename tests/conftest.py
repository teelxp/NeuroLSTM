import json
import pytest
import torch
from neurolstm.demo import make_demo
from neurolstm.config import load_config

@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(1)

@pytest.fixture
def demo_cfg(tmp_path):
    path=make_demo(tmp_path/"data",rows=240)
    cfg=load_config(path)
    cfg["training"]["epochs"]=2
    return cfg

@pytest.fixture
def semantic_cfg(tmp_path):
    path=make_demo(tmp_path/"semantic_data",rows=240,with_semantics=True)
    cfg=load_config(path)
    cfg["training"]["epochs"]=2
    return cfg
