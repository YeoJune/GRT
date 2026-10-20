"""Cosine LR floor, optimizer groups and exact scheduler replay."""
import copy

import pytest
import torch

from grt.stream_training import cosine_schedule


@pytest.mark.parametrize('ratio', [0., .01, .2, 1.])
@pytest.mark.parametrize('warmup_ratio', [0., .05, .25, .5])
def test_cosine_curve_floor_and_resume(ratio, warmup_ratio):
    def setup():
        a, b = torch.nn.Parameter(torch.zeros(1)), torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.Adam([{'params': [a], 'lr': .0005},
                                      {'params': [b], 'lr': .001}])
        scheduler = cosine_schedule(optimizer, max_steps=40,
            min_lr_ratio=ratio, warmup_ratio=warmup_ratio)
        return optimizer, scheduler

    optimizer, scheduler = setup()
    trace = [scheduler.get_last_lr()]
    for step in range(1, 46):
        optimizer.step()
        scheduler.step()
        trace.append(scheduler.get_last_lr())
        if step == 13:
            saved = copy.deepcopy((optimizer.state_dict(), scheduler.state_dict()))

    warmup_steps = int(40*warmup_ratio)
    assert trace[0] == pytest.approx([0., 0.] if warmup_steps else [.0005, .001])
    assert trace[warmup_steps] == pytest.approx([.0005, .001])
    if warmup_steps:
        assert trace[warmup_steps//2] == pytest.approx([.00025, .0005])
    midpoint = warmup_steps + (40-warmup_steps)//2
    assert trace[midpoint] == pytest.approx([.0005*(1+ratio)/2, .001*(1+ratio)/2])
    assert trace[40] == pytest.approx([.0005*ratio, .001*ratio])
    assert trace[45] == trace[40]
    assert all(trace[i][0] <= trace[i+1][0] for i in range(warmup_steps))
    assert all(trace[i][0] >= trace[i+1][0] for i in range(warmup_steps, 45))

    resumed_optimizer, resumed = setup()
    resumed_optimizer.load_state_dict(saved[0])
    resumed.load_state_dict(saved[1])
    for expected in trace[14:]:
        resumed_optimizer.step()
        resumed.step()
        assert resumed.get_last_lr() == expected


def test_substep_warmup_rounds_down_for_tiny_budget():
    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=.0005)
    scheduler = cosine_schedule(optimizer, max_steps=1, min_lr_ratio=.01, warmup_ratio=.05)
    assert scheduler.get_last_lr() == pytest.approx([.0005])
    optimizer.step()
    scheduler.step()
    assert scheduler.get_last_lr() == pytest.approx([.000005])
