"""Common teacher-forcing and generation evaluation through the dataset adapter."""

import torch
from grt.checkpoint import evaluation_context
from grt.data import get_adapter, make_loader, forward_batch


@torch.no_grad()
def evaluate(model, cfg, dataset, device, vary=False, distributed=None):
    ce = correct = count = 0
    adapter = get_adapter(cfg.data.name)
    with evaluation_context(model):
        world_size = distributed.world_size if distributed else 1
        rank = distributed.rank if distributed else 0
        for batch in make_loader(cfg, dataset, vary=vary, world_size=world_size):
            batch = {k: v[rank::world_size] for k, v in batch.items()}
            if not len(batch["input_ids"]):
                continue
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
    if distributed:
        ce, correct, count = distributed.sum([ce, correct, count])
    if not count:
        raise ValueError("Empty evaluation dataset")
    return {"loss": ce / count, "exact_match": correct / count, "samples": int(count)}


def evaluate_lengths(model, cfg, device, distributed=None):
    """Length tests keep the checkpoint key/value alphabet and segment geometry."""
    import copy
    from types import SimpleNamespace
    from grt.data import make_dataset
    from grt.config import validate_config

    records = []
    for pairs in cfg.evaluation.pair_counts:
        current = copy.deepcopy(cfg)
        current.data.test_samples = cfg.evaluation.generalization_samples
        get_adapter(cfg.data.name).configure_stage(
            current, SimpleNamespace(num_pairs=pairs, key_size=cfg.data.key_size)
        )
        try:
            validate_config(current, require_output=False)
        except ValueError as error:
            records.append(
                {"num_pairs": pairs, "status": "incompatible", "reason": str(error)}
            )
            continue
        result = evaluate(
            model,
            current,
            make_dataset(current, "test", pairs),
            device,
            vary=False,
            distributed=distributed,
        )
        records.append(
            {
                "num_pairs": pairs,
                "key_size": cfg.data.key_size,
                "status": "evaluated",
                **result,
            }
        )
    return records
