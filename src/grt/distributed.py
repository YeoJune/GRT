"""Single-node torchrun support, without changing native model state_dict keys."""

from contextlib import nullcontext
from datetime import timedelta
import os
from typing import NamedTuple
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from grt.checkpoint import preserve_rng


class TrainingOutput(NamedTuple):
    loss: torch.Tensor


class _LossOnly(torch.nn.Module):
    """Keep the author's dynamic HF ModelOutput outside DDP's pytree boundary."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, **kwargs):
        return TrainingOutput(self.model(**kwargs).loss)


class DistributedContext:
    def __init__(self, device, initialize=False):
        self.device = torch.device(device)
        if initialize and int(os.environ.get("WORLD_SIZE", "1")) > 1:
            local_rank = int(os.environ["LOCAL_RANK"])
            if self.device.type == "cuda":
                torch.cuda.set_device(local_rank)
                self.device = torch.device("cuda", local_rank)
            dist.init_process_group(
                "nccl" if self.device.type == "cuda" else "gloo",
                timeout=timedelta(minutes=20),
            )
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.main = self.rank == 0

    def barrier(self):
        if self.world_size > 1:
            dist.barrier()

    def wrap(self, model):
        if self.world_size == 1:
            return model
        with preserve_rng():
            return DistributedDataParallel(
                _LossOnly(model),
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
                broadcast_buffers=False,
                find_unused_parameters=True,
            )

    def synchronize(self, model, last_microbatch):
        return (
            model.no_sync()
            if self.world_size > 1 and not last_microbatch
            else nullcontext()
        )

    def sum(self, values):
        value = torch.tensor(values, dtype=torch.float64, device=self.device)
        if self.world_size > 1:
            dist.all_reduce(value)
        return value.tolist()

    def max(self, value):
        tensor = torch.tensor(float(value), dtype=torch.float64, device=self.device)
        if self.world_size > 1:
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return tensor.item()

    def decision(self, value):
        tensor = torch.tensor(bool(value) if self.main else False, device=self.device)
        if self.world_size > 1:
            dist.broadcast(tensor, src=0)
        return bool(tensor.item())

    def rng_states(self, state):
        if self.world_size == 1:
            return [state]
        states = [None] * self.world_size
        dist.all_gather_object(states, state)
        return states

    def close(self):
        if self.world_size > 1:
            dist.destroy_process_group()
