"""Stage orchestration for the existing train entry point, with convergence gates."""
import copy
import json
import time
from dataclasses import asdict
from pathlib import Path
from grt.checkpoint import load_checkpoint, restore_model, seed_all
from grt.config import save_config, validate_config
from grt.evaluator import evaluate_lengths
from grt.logger import Logger, metadata, write_json
from grt.models.factory import create_model
from grt.trainer import Trainer
from grt.data import remember_context_capacity


def stage_config(cfg, index):
    stage = cfg.training.curriculum[index]
    result = copy.deepcopy(cfg)
    result.training.curriculum = []
    result.training.max_steps = stage.max_steps
    result.training.scheduler_steps = 2 * stage.max_steps
    result.training.warmup_steps = min(cfg.training.warmup_steps, stage.max_steps // 10)
    result.data.key_size = stage.key_size
    result.model.segment_len = stage.key_size + result.data.value_size + 2
    result.data.train_segments = stage.num_pairs + 1
    result.data.eval_segments = (cfg.data.eval_segments if index == len(cfg.training.curriculum)-1
                                 else [stage.num_pairs + 1])
    result.run.output_dir = str(Path(cfg.run.output_dir) / f"stage_{index+1:02d}_pairs{stage.num_pairs}_key{stage.key_size}")
    if result.data.remember_sampling == "balanced_contexts":
        for split, field in (("train","train_samples"),("validation","validation_samples"),("test","test_samples")):
            capacity = remember_context_capacity(stage.num_pairs, split)[1]
            requested = getattr(result.data, field)
            setattr(result.data, field, min(requested // stage.num_pairs, capacity) * stage.num_pairs)
        result.evaluation.autoregressive_samples = min(result.evaluation.autoregressive_samples,
            result.data.validation_samples) // stage.num_pairs * stage.num_pairs
    return validate_config(result)


def run_curriculum(cfg, device):
    directory = Path(cfg.run.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    saved = directory / "resolved_config.yaml"
    if saved.exists():
        from grt.config import load_config
        previous = load_config(saved, output_dir=directory)
        if asdict(previous) != asdict(cfg):
            raise ValueError("Curriculum settings changed; use a new output directory")
    elif any(directory.iterdir()):
        raise ValueError("Nonempty directory without curriculum configuration")
    save_config(cfg, saved)
    summary_path = directory / "summary.json"
    old_seconds = json.loads(summary_path.read_text()).get("wall_seconds", 0) if summary_path.exists() else 0
    started = time.perf_counter()
    stages = []
    selected = None
    status = "completed"
    for index, stage in enumerate(cfg.training.curriculum):
        stage_cfg = stage_config(cfg, index)
        stage_dir = Path(stage_cfg.run.output_dir)
        candidates = [(path, load_checkpoint(path)) for name in ("last.pt", "best.pt", "converged.pt")
                      if (path := stage_dir / name).exists()]
        latest = max(candidates, key=lambda item: (item[1]['global_step'], bool(item[1].get('converged')))) if candidates else None
        if latest is None and stage_dir.exists() and any(stage_dir.iterdir()):
            raise ValueError(f"Incomplete stage without a checkpoint: {stage_dir}")
        seed_all(cfg.training.model_seed)
        model = create_model(stage_cfg.model).to(device)
        trainer = Trainer(model, stage_cfg)
        if latest is not None:
            _, state = latest
            expected = asdict(stage_cfg)
            if {k:v for k,v in state['config'].items() if k != 'run'} != {k:v for k,v in expected.items() if k != 'run'}:
                raise ValueError(f"Stage settings changed: {stage_dir}")
            trainer.resume(state)
        elif selected is not None:
            # Match the reference curriculum: transfer weights, reset stage optimizer/scheduler.
            model.load_state_dict(load_checkpoint(selected)["model"])
        save_config(stage_cfg, stage_dir / "resolved_config.yaml")
        run_meta = metadata(model, stage_cfg)
        run_meta["reference_commit"] = "24cbb9aed62a5748c4045fd928f7f59899f03b24"
        run_meta["initialized_from"] = str(selected) if selected is not None else None
        meta_path = stage_dir / (f"resume_metadata_step_{trainer.global_step:06d}.json" if latest is not None else "metadata.json")
        write_json(meta_path, run_meta)
        if not (directory / "metadata.json").exists():
            write_json(directory / "metadata.json", run_meta)
        logger = Logger(stage_dir, stage_cfg)
        trainer.logger = logger
        elapsed = old_seconds + time.perf_counter() - started
        remaining = None if cfg.training.max_seconds is None else max(0.0, cfg.training.max_seconds-elapsed)
        print(f"Remember stage {index+1}: {stage.num_pairs} pairs, key={stage.key_size}, "
              f"max_updates={stage.max_steps}, batch={stage_cfg.training.batch_size} "
              f"x accumulation={stage_cfg.training.grad_accum_steps}", flush=True)
        try:
            progress = trainer.train(time_limit_seconds=remaining)
        finally:
            logger.finish()
        selected = stage_dir / ("converged.pt" if progress["converged"] else "best.pt")
        record = {"num_pairs": stage.num_pairs, "key_size": stage.key_size,
                  "checkpoint": str(selected), **progress}
        if stage_cfg.data.remember_sampling == "balanced_contexts":
            record.update(data_sampling="balanced_contexts", train_contexts=stage_cfg.data.train_samples // stage.num_pairs,
                          validation_contexts=stage_cfg.data.validation_samples // stage.num_pairs,
                          test_contexts=stage_cfg.data.test_samples // stage.num_pairs)
        validation_rows = [json.loads(line) for line in (stage_dir / "metrics.jsonl").read_text().splitlines()
                           if '"val/loss"' in line]
        record["last_validation"] = validation_rows[-1] if validation_rows else None
        stages.append(record)
        if not progress["converged"]:
            status = "stage_not_converged"
        wall_seconds = old_seconds + time.perf_counter() - started
        write_json(summary_path, {"task": "remember", "status": status, "stages": stages,
                                  "wall_seconds": wall_seconds, "selected_checkpoint": str(selected)})
        if not progress["converged"]:
            print("Stage did not meet generation exact-match threshold; curriculum stopped.", flush=True)
            break
    best_state = load_checkpoint(selected)
    best_model, evaluated_cfg = restore_model(best_state, device)
    rows = evaluate_lengths(best_model, evaluated_cfg, stages[-1])
    write_json(directory / "evaluation.json", {"checkpoint": str(selected),
               "checkpoint_step": best_state["global_step"], "results": rows})
    print(f"Remember curriculum: {status}; results: {directory}", flush=True)
    return stages
