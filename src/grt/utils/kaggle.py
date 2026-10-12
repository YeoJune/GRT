"""Shared WikiText notebook execution for Kaggle and Colab."""
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys


def run_logged(command, *, cwd, log_path):
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(' '.join(map(str, command)), flush=True)
    with log_path.open('a', encoding='utf-8') as log:
        process = subprocess.Popen(list(map(str, command)), cwd=cwd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            bufsize=1, start_new_session=True)
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            if process.wait():
                raise subprocess.CalledProcessError(process.returncode, command)
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                except ProcessLookupError:
                    process.wait()
            process.stdout.close()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def estimate_time(benchmark, manifest, *, steps, eval_every, eval_batch, segment_len):
    if min(steps, eval_every, eval_batch, segment_len) <= 0:
        raise ValueError('Step, batch and segment settings must be positive')
    if benchmark.get('objective') != 'next_token' or benchmark['target_tokens_per_update'] != benchmark['global_batch_size'] * segment_len:
        raise ValueError('Expected a next_token benchmark with matching segment length')
    def segments(split):
        length = manifest['token_counts'][split] // eval_batch - 1
        if length <= 0:
            raise ValueError(f'{split} is too short for the evaluation batch')
        return math.ceil(length / segment_len)
    training = steps * benchmark['update_seconds']
    validation = math.ceil(steps / eval_every) * segments('validation') * benchmark['evaluation_segment_seconds']
    test = segments('test') * benchmark['evaluation_segment_seconds']
    return dict(model=benchmark['model'], score_kind=benchmark['score_kind'],
        steps=steps, target_tokens=steps * benchmark['target_tokens_per_update'],
        training_hours=training / 3600, validation_hours=validation / 3600,
        final_test_hours=test / 3600, total_compute_hours=(training + validation + test) / 3600,
        peak_gib_per_gpu=benchmark['peak_bytes_per_gpu'] / 2**30,
        reserved_gib_per_gpu=benchmark['reserved_bytes_per_gpu'] / 2**30,
        note='Excludes preparation, checkpoint I/O, startup and RTLA probes. Full evaluation repeats on each rank; tail segments approximated as full segments.')


@dataclass
class WikiTextRun:
    repo_dir: Path
    size: str = 'small'
    budget: str = '10k'
    models: tuple = ('rmt',)
    runtime: str = 'kaggle'
    tokenization: str = 'gpt2_bpe'
    steps: int | None = None
    seed: int = 54
    wandb: bool = True
    work_dir: Path = Path('/kaggle/working')
    run_id: str | None = None
    world_size: int = 2
    device: str = 'cuda'
    benchmark_steps: int = 10
    overrides: dict = field(default_factory=dict)

    def __post_init__(self):
        self.repo_dir, self.work_dir = Path(self.repo_dir), Path(self.work_dir)
        if not self.models or len(set(self.models)) != len(self.models) or any(m not in ('rmt', 'grt') for m in self.models):
            raise ValueError('Select rmt/grt once each')
        if self.world_size <= 0 or self.benchmark_steps <= 0:
            raise ValueError('Invalid execution settings')
        name = self.run_id or f'wiki_{self.size}_{self.tokenization}_{self.budget}_seed{self.seed}'
        if not self.run_id and (self.steps is not None or self.overrides):
            signature = json.dumps(dict(steps=self.steps, overrides=self.overrides), sort_keys=True)
            name += '-' + hashlib.sha256(signature.encode()).hexdigest()[:8]
        self.run_dir = self.work_dir / name
        self.prepared_dir = self.work_dir / ('wikitext103_' + self.tokenization)
        # Validate selection before installing, downloading or creating outputs.
        for model in self.models:
            cfg = self.config(model, calibrated=False)
            self.prepared_dir = Path(cfg.data.options['path'])

    def config(self, model, *, calibrated=True):
        from grt.config import merge_dicts
        from grt.utils.batching import apply_batch_override
        from grt.experiments import load_experiment
        environment = {'data': {'options': {'path': str(self.prepared_dir)}, 'data_seed': self.seed},
            'training': {'model_seed': self.seed},
            'wandb': {'enabled': self.wandb, 'run_name': self.run_dir.name + '-' + model}}
        cfg = load_experiment(task='wikitext103', model=model, size=self.size,
            tokenization=self.tokenization, budget=self.budget, runtime=self.runtime, steps=self.steps,
            overrides=merge_dicts(environment, self.overrides), output_dir=self.run_dir / model,
            catalog_path=self.repo_dir / 'configs/experiments.yaml')
        if cfg.model.name != model:
            raise ValueError('Select the model through models, not overrides')
        benchmark = self.run_dir / (model + '_preflight') / 'benchmark.json'
        if calibrated and cfg.training.auto_batch and benchmark.exists():
            report = json.loads(benchmark.read_text())
            expected_batch = cfg.training.batch_size * cfg.training.grad_accum_steps * self.world_size
            if (report['model'] != model or report['world_size'] != self.world_size
                or report['global_batch_size'] != expected_batch
                or report['previous_segments'] != cfg.training.bptt_segments
                or report['target_tokens_per_update'] != expected_batch * cfg.model.segment_len
                or report['batch_tuning']['enabled'] != cfg.training.auto_batch
                or report['batch_tuning']['memory_fraction'] != cfg.training.memory_fraction
                or (self.device == 'cuda' and not report['batch_tuning']['applied'])):
                raise ValueError('Preflight conditions changed; use a new run_id')
            cfg = apply_batch_override(cfg, report['calibrated_override'], self.world_size)
        effective = cfg.training.batch_size * cfg.training.grad_accum_steps * self.world_size
        if cfg.training.global_batch_size is not None and effective != cfg.training.global_batch_size:
            raise ValueError('microbatch × accumulation × world_size must equal global batch')
        stored = self.run_dir / model / 'experiment_config.json'
        if calibrated and stored.exists() and json.loads(stored.read_text()) != asdict(cfg):
            raise ValueError('Existing training config changed; use a new run_id')
        return cfg

    def setup(self):
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ.setdefault('HF_HOME', str(self.work_dir / 'hf_cache'))
        configs = [self.config(model) for model in self.models]
        extras = 'wandb,wikitext' if any(cfg.wandb.enabled for cfg in configs) else 'wikitext'
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-e',
                        str(self.repo_dir) + f'[{extras}]'], check=True)
        import torch
        if self.device == 'cuda':
            if torch.cuda.device_count() < self.world_size:
                raise RuntimeError(f'{self.world_size} CUDA devices required')
            for rank in range(self.world_size):
                print(rank, torch.cuda.get_device_name(rank),
                      torch.cuda.get_device_properties(rank).total_memory / 2**30, 'GiB')
        if any(cfg.wandb.enabled and cfg.wandb.mode == 'online' for cfg in configs):
            import wandb
            if os.environ.get('WANDB_API_KEY'):
                wandb.login()
            elif self.runtime == 'colab':
                from google.colab import userdata
                wandb.login(key=userdata.get('WANDB_API_KEY'))
            else:
                from kaggle_secrets import UserSecretsClient
                wandb.login(key=UserSecretsClient().get_secret('WANDB_API_KEY'))

    def prepare_data(self, *, source=None, prepared_input=None, resume_input=None,
                     dataset_revision='main',
                     tokenizer='openai-community/gpt2', local_files_only=False):
        if resume_input is not None:
            if self.run_dir.exists():
                raise FileExistsError('Run already exists; omit resume_input to reuse it')
            shutil.copytree(resume_input, self.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if prepared_input is not None and not self.prepared_dir.exists():
            shutil.copytree(prepared_input, self.prepared_dir)
        manifest_path = self.prepared_dir / 'manifest.json'
        if not manifest_path.exists():
            if prepared_input is not None:
                raise ValueError('prepared_input has no manifest.json')
            if source is None:
                source = self.work_dir / 'wikitext103_hf_source'
                if not (source / 'download_manifest.json').exists():
                    self._logged([sys.executable, 'scripts/download_wikitext.py', '--output-dir', source,
                                  '--revision', dataset_revision], 'download.txt')
            script = 'prepare_wikitext_bpe.py' if self.tokenization == 'gpt2_bpe' else 'prepare_wikitext.py'
            command = [sys.executable, 'scripts/' + script, '--source', source, '--output-dir', self.prepared_dir]
            if self.tokenization == 'gpt2_bpe':
                command += ['--tokenizer', tokenizer]
                if local_files_only:
                    command += ['--local-files-only']
            self._logged(command, 'prepare.txt')
        manifest = json.loads(manifest_path.read_text())
        if prepared_input is not None and json.loads((Path(prepared_input) / 'manifest.json').read_text()) != manifest:
            raise ValueError('Existing prepared data differs from prepared_input')
        cfg = self.config(self.models[0])
        if manifest['vocab_size'] != cfg.model.vocab_size or manifest.get('tokenization', 'word') != self.tokenization:
            raise ValueError('Prepared vocabulary/tokenization differs from the experiment')
        artifacts = {split + '.pt': digest for split, digest in manifest['sha256'].items()}
        artifacts.update(manifest.get('prepared_artifacts_sha256', {}))
        if 'vocab_sha256' in manifest:
            artifacts['vocab.json'] = manifest['vocab_sha256']
        for relative, digest in artifacts.items():
            if file_hash(self.prepared_dir / relative) != digest:
                raise ValueError(f'Prepared artifact changed: {relative}')
        if source is not None:
            for split, name in [('train', 'train'), ('validation', 'valid'), ('test', 'test')]:
                if file_hash(Path(source) / f'wiki.{name}.tokens') != manifest['source_sha256'][split]:
                    raise ValueError('Prepared source changed; use a new data directory')
        (self.run_dir / 'dataset_manifest.json').write_text(json.dumps(manifest, indent=2))
        print('Prepared:', self.prepared_dir, manifest['token_counts'])
        return manifest

    def _logged(self, command, filename):
        run_logged(command, cwd=self.repo_dir, log_path=self.run_dir / filename)

    def _launch(self, model, output, *, benchmark=False):
        from grt.config import save_config
        self.run_dir.mkdir(parents=True, exist_ok=True)
        path = self.run_dir / (model + '_config.yaml')
        save_config(self.config(model, calibrated=not benchmark), path)
        command = [sys.executable, '-u']
        if self.world_size > 1:
            command += ['-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={self.world_size}']
        command += ['scripts/train.py', '--config', path, '--output-dir', output, '--device', self.device]
        if benchmark:
            command += ['--benchmark-steps', str(self.benchmark_steps)]
        self._logged(command, model + ('_preflight_console.txt' if benchmark else '_train_console.txt'))

    def preflight(self):
        from grt.config import save_config
        reports = {}
        manifest = json.loads((self.prepared_dir / 'manifest.json').read_text())
        for model in self.models:
            probe = self.run_dir / (model + '_preflight')
            if not (self.run_dir / model / 'experiment_config.json').exists():
                self._launch(model, probe, benchmark=True)
            report = json.loads((probe / 'benchmark.json').read_text())
            cfg = self.config(model)
            if self.device == 'cuda' and cfg.training.auto_batch:
                if report['peak_reserved_fraction'] > cfg.training.memory_fraction:
                    raise RuntimeError('Final DDP peak exceeds the memory limit; reduce microbatch and remeasure')
            save_config(cfg, self.run_dir / (model + '_config.yaml'))
            eta = estimate_time(report, manifest, steps=cfg.training.max_steps,
                eval_every=cfg.evaluation.every_steps, eval_batch=cfg.evaluation.batch_size,
                segment_len=cfg.model.segment_len)
            (self.run_dir / (model + '_eta.json')).write_text(json.dumps(eta, indent=2))
            print(json.dumps(eta, indent=2))
            reports[model] = report
        return reports

    def train(self):
        results = {}
        for model in self.models:
            if not (self.run_dir / (model + '_preflight') / 'benchmark.json').exists():
                raise ValueError('Run preflight before training')
            self._launch(model, self.run_dir / model)
            cfg = self.config(model)
            if cfg.wandb.enabled:
                status = json.loads((self.run_dir / model / 'wandb_status.json').read_text())
                print(model, 'W&B:', status)
                if (status['state'] != 'finished' or status.get('failures', 0)
                    or status.get('last_logged_step', 0) < cfg.training.max_steps
                    or status['mode'] != cfg.wandb.mode
                    or (cfg.wandb.mode == 'online' and not status.get('url'))):
                    raise RuntimeError('W&B logging failed; local results remain in the run directory')
            results[model] = json.loads((self.run_dir / model / 'summary.json').read_text())
        return results

    def report(self, *, plot=True):
        results = {}
        if plot:
            import matplotlib.pyplot as plt
            _, ax = plt.subplots()
        for model in self.models:
            root = self.run_dir / model
            for name in ['wandb_status.json', 'summary.json']:
                if (root / name).exists():
                    value = json.loads((root / name).read_text())
                    print(model, name, value)
                    if name == 'summary.json':
                        results[model] = value
            if (root / 'warnings.txt').exists():
                print((root / 'warnings.txt').read_text())
            if plot and (root / 'metrics.jsonl').exists():
                rows = [json.loads(line) for line in (root / 'metrics.jsonl').read_text().splitlines() if line]
                rows = [row for row in rows if 'val/loss' in row]
                ax.plot([r['step'] for r in rows], [r['val/loss'] for r in rows], label=model)
        if plot:
            ax.set(xlabel='optimizer updates', ylabel='validation token NLL')
            ax.grid()
            if ax.lines:
                ax.legend()
            plt.show()
        print('Results:', self.run_dir, '\nData/tokenizer:', self.prepared_dir)
        return results
