"""Common teacher-forcing and generation evaluation through the dataset adapter."""

import torch
from grt.checkpoint import evaluation_context
from grt.data import get_adapter, make_loader, forward_batch


@torch.no_grad()
def evaluate(model, cfg, dataset, device, vary=False):
    ce = correct = count = 0
    adapter = get_adapter(cfg.data.name)
    with evaluation_context(model):
        for batch in make_loader(cfg, dataset, vary=vary):
            output = forward_batch(model, batch, device)
            n = len(batch["input_ids"])
            ce += output.loss.item() * n
            count += n
            generated = model.generate(
                batch["input_ids_generate"].to(device),
                attention_mask=batch["attention_mask_generate"].to(device),
                **adapter.generation_kwargs(cfg),
            )
            target = adapter.generation_target(cfg, batch).to(device)
            if generated.shape[1] >= target.shape[1]:
                correct += (
                    generated[:, -target.shape[1] :].eq(target).all(1).sum().item()
                )
    if not count:
        raise ValueError("Empty evaluation dataset")
    return {"loss": ce / count, "exact_match": correct / count, "samples": count}
