"""Direct CPU experiments using the real small preset and public interfaces.

Remember uses the native task. The LM stream is synthetic, with the full
WikiText vocabulary size; its scores are interface checks, not WikiText PPL.
"""
import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from grt.checkpoint import seed_all, restore_model, load_checkpoint, evaluation_context
from grt.config import CurriculumStage, validate_config
from grt.experiments import load_experiment
from grt.data import get_adapter, make_dataset, make_loader
from grt.evaluator import evaluate
from grt.logger import write_json
from grt.models.factory import create_model
from grt.trainer import run
from prepare_wikitext import prepare


def synthetic_stream(root):
    source = root/'synthetic_source'
    source.mkdir()
    # Five common entries including EOS, plus 267730 unique training words.
    with (source/'wiki.train.tokens').open('w') as f:
        f.write('alpha beta gamma <unk>\n'*600)
        f.write(' '.join(f'unique{i}' for i in range(267730))+'\n')
    for split in ('valid', 'test'):
        (source/f'wiki.{split}.tokens').write_text('alpha beta gamma <unk>\n'*80)
    destination = root/'synthetic_tokens'
    manifest = prepare(source, destination)
    # Explicitly identify the generated corpus in persisted training metadata.
    manifest['dataset'] = 'synthetic CPU interface check; NOT WikiText-103'
    write_json(destination/'manifest.json', manifest)
    return destination


def remember_diagnostics(model, cfg, dataset):
    adapter = get_adapter(cfg.data.name)
    value_correct = eos_correct = count = 0
    examples = []
    predictions = Counter()
    with evaluation_context(model):
        for batch in make_loader(cfg, dataset, vary=False):
            predicted = model.generate(batch['input_ids_generate'], **adapter.generation_kwargs(cfg))
            target = adapter.generation_target(cfg, batch)
            value_correct += predicted[:, :-1].eq(target[:, :-1]).all(1).sum().item()
            eos_correct += predicted[:, -1].eq(target[:, -1]).sum().item()
            count += len(target)
            predictions.update(tuple(p) for p in predicted.tolist())
            if not examples:
                examples = [dict(target=t, predicted=p) for t,p in zip(target.tolist(),predicted.tolist())]
    return dict(value_accuracy=value_correct/count, eos_accuracy=eos_correct/count, examples=examples,
        prediction_counts=[dict(predicted=list(p),samples=n) for p,n in predictions.most_common()])


def experiment(cfg, corpus):
    validate_config(cfg)
    seed_all(cfg.training.model_seed)
    model = create_model(cfg).cpu()
    adapter = get_adapter(cfg.data.name)
    dataset = make_dataset(cfg, 'test', cfg.data.train_segments-1)
    initial = evaluate(model, cfg, dataset, 'cpu', vary=False)
    parameters = dict(backbone=sum(p.numel() for p in model.backbone.parameters()),
        total=sum(p.numel() for p in model.parameters()))
    del model
    gc.collect()
    started = time.perf_counter()
    summary = run(cfg, torch.device('cpu'))
    seconds = time.perf_counter()-started
    directory = Path(cfg.run.output_dir)
    checkpoint = (directory/'best.pt' if hasattr(adapter, 'train_stream') else
        directory/'stage_01_pairs1_key1/best.pt')
    selected = load_checkpoint(checkpoint)
    selected_step = selected['step']
    restored, restored_cfg = restore_model(selected, 'cpu')
    del selected
    final = evaluate(restored, restored_cfg, dataset, 'cpu', vary=False)
    del restored
    gc.collect()
    last = load_checkpoint(checkpoint.with_name('last.pt'))
    last_step = last['step']
    restored, restored_cfg = restore_model(last, 'cpu')
    del last
    last_test = evaluate(restored, restored_cfg, dataset, 'cpu', vary=False)
    diagnostics = (remember_diagnostics(restored, restored_cfg, dataset)
        if not hasattr(adapter, 'train_stream') else None)
    result = dict(name=directory.name, corpus=corpus, model=cfg.model.name,
        attention_mode=cfg.model.attention_mode, d_model=cfg.model.d_model,
        layers=cfg.model.alu.num_layers, heads=cfg.model.alu.nhead,
        d_ff=cfg.model.alu.d_ff, memory_slots=cfg.model.num_registers,
        conditional_read=cfg.model.register.conditional_read if cfg.model.register else None,
        parameters=parameters, initial_test=initial, selected_checkpoint_test=final,
        selected_checkpoint_step=selected_step, last_checkpoint_step=last_step, last_checkpoint_test=last_test,
        last_diagnostics=diagnostics,
        train_wall_seconds=seconds, summary=summary, checkpoint=str(checkpoint.resolve()))
    write_json(directory/'cpu_experiment.json', result)
    del restored
    gc.collect()
    print(json.dumps({k:v for k,v in result.items() if k not in ('summary',)}, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--remember-steps', type=int, default=60)
    parser.add_argument('--lm-steps', type=int, default=6)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if min(args.remember_steps, args.lm_steps, args.threads) <= 0:
        parser.error('Steps and threads must be positive')
    root = Path(args.output_dir)
    if root.exists():
        parser.error('Use a new output directory to preserve earlier experiments')
    root.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    records = []
    for mode in ('encoder', 'decoder'):
        for name in ('grt', 'rmt'):
            directory = root/f'remember_{mode}_{name}'
            cfg = load_experiment(task='remember', model=name, size='small', output_dir=directory,
                overrides={'model': {'attention_mode': mode},
                           'data': {'options': {'objective': 'parallel' if mode == 'encoder' else 'next_token'}}})
            cfg.training.batch_size = cfg.evaluation.batch_size = 8
            cfg.training.global_batch_size = 8
            cfg.training.max_steps = args.remember_steps
            cfg.training.curriculum = [CurriculumStage(1,1,args.remember_steps)]
            cfg.data.train_samples, cfg.data.validation_samples, cfg.data.test_samples = 256,64,128
            cfg.evaluation.every_steps = max(1,args.remember_steps//3)
            records.append(experiment(cfg,'native Remember, one fact, held-out 128 samples'))
            write_json(root/'results.json', records)
    prepared = synthetic_stream(root)
    for name in ('grt', 'rmt'):
        cfg = load_experiment(task='wikitext103', model=name, size='small', tokenization='word', output_dir=root/f'lm_synthetic_{name}')
        cfg.data.options['path'] = str(prepared.resolve())
        cfg.training.batch_size = cfg.training.grad_accum_steps = cfg.training.global_batch_size = 1
        cfg.training.max_steps = args.lm_steps
        cfg.evaluation.batch_size = 1
        cfg.evaluation.every_steps = max(1,args.lm_steps//3)
        records.append(experiment(cfg,'synthetic word stream, full V267735; NOT WikiText-103'))
        write_json(root/'results.json',records)
    write_json(root/'protocol.json', dict(threads=args.threads, catalog='configs/experiments.yaml', size='small',
        remember_steps=args.remember_steps, lm_steps=args.lm_steps,
        note='Direct run/evaluate/restore interfaces. Small backbone kept intact; CPU batch and update budget reduced. Test is diagnostic, selection uses validation.'))


if __name__ == '__main__':
    main()
