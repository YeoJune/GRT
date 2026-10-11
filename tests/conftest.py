import pytest
import torch
from grt.experiments import load_experiment


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def tiny_config(path, model="rmt"):
    return load_experiment(task='remember', model=model, runtime='cpu', output_dir=path,
        overrides={
            'model': {'d_model': 32, 'num_registers': 4,
                      'alu': {'num_layers': 2, 'nhead': 4, 'd_ff': 64},
                      **({'router': {'mlp_hidden': 32}} if model == 'grt' else {})},
            'training': {'curriculum': [
                {'num_pairs': 1, 'key_size': 1, 'max_steps': 2},
                {'num_pairs': 2, 'key_size': 1, 'max_steps': 2}]},
            'data': {'validation_samples': 4, 'test_samples': 4},
            'evaluation': {'every_steps': 1}})
