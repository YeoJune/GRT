"""Single-checkpoint WikiText posthoc analysis and a portable JSON report."""
from dataclasses import asdict, dataclass
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import time
from uuid import uuid4

import numpy as np
import torch
import transformers

from grt.checkpoint import load_checkpoint, preserve_rng, restore_model
from grt.config import Config, build_config
from grt.logger import metadata
from grt.memory_analysis import analyze_stream, probe_metrics, save_analysis
from grt.register_analysis import (RegisterObserver, analyze_groups, classify_gates,
    group_stability, intervention_forward_calls, intervention_plan)
from grt.utils.kaggle import file_hash
from grt.wikitext import WikiTextAdapter


def portable(value):
    """Encode absent tail observations as null, never as nonstandard JSON NaN."""
    if isinstance(value, np.ndarray):
        return portable(value.tolist())
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def gate_statistics(values, threshold=0.01):
    """Statistics over real example/segment gates, rather than batch averages."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError('Expected finite write gates [segment, example, slot]')
    if values.size == 0 or values.min() < 0 or values.max() > 1:
        raise ValueError('Write gates must lie in [0, 1]')
    histogram, edges = np.histogram(values, bins=np.linspace(0, 1, 21))
    return dict(
        scope='equally weighted measured segments and examples; no training-time history',
        shape=list(values.shape), closed_threshold=threshold, open_threshold=1-threshold,
        mean=float(values.mean()), std=float(values.std()),
        quantiles=dict(zip(['min', 'p10', 'median', 'p90', 'max'],
                           np.quantile(values, [0, .1, .5, .9, 1]).tolist())),
        fraction_near_closed=float((values <= threshold).mean()),
        fraction_near_open=float((values >= 1-threshold).mean()),
        histogram=dict(counts=histogram.tolist(), edges=edges.tolist()),
        per_slot=dict(
            mean=values.mean((0, 1)).tolist(), std=values.std((0, 1)).tolist(),
            fraction_near_closed=(values <= threshold).mean((0, 1)).tolist(),
            fraction_near_open=(values >= 1-threshold).mean((0, 1)).tolist(),
            segment_std_of_batch_mean=values.mean(1).std(0).tolist(),
            mean_within_segment_example_std=values.std(1).mean(0).tolist()))


@dataclass
class WikiTextPosthoc:
    checkpoint: Path
    prepared_dir: Path
    output_dir: Path
    device: str = 'cuda'
    batch_size: int = 8
    segments: int = 8
    warmup_segments: int = 3
    horizon: int = 4
    slots: bool = False
    registers: bool = True
    window_starts: tuple = (0, 16, 32)
    group_repeats: int = 3
    lesion_fractions: tuple = (.1, .25, .5, .75)
    analysis_seed: int = 54

    def __post_init__(self):
        self.checkpoint = Path(self.checkpoint)
        self.prepared_dir = Path(self.prepared_dir)
        self.output_dir = Path(self.output_dir)
        if self.device not in ('cpu', 'cuda'):
            raise ValueError('device must be cpu or cuda')
        for name in ('batch_size', 'segments', 'warmup_segments', 'horizon'):
            if type(getattr(self, name)) is not int:
                raise ValueError(f'{name} must be an integer')
        if self.batch_size < 2 or min(self.segments, self.warmup_segments) <= 0 or self.horizon < 0:
            raise ValueError('Require batch>=2, segments/warmup>0 and horizon>=0')
        if type(self.slots) is not bool:
            raise ValueError('slots must be a boolean')
        if type(self.registers) is not bool:
            raise ValueError('registers must be a boolean')
        if type(self.group_repeats) is not int or self.group_repeats < 1:
            raise ValueError('group_repeats must be a positive integer')
        if type(self.analysis_seed) is not int or self.analysis_seed < 0:
            raise ValueError('analysis_seed must be a nonnegative integer')
        self.window_starts = tuple(self.window_starts)
        self.lesion_fractions = tuple(self.lesion_fractions)
        if (not self.window_starts or self.window_starts[0] != 0 or
            any(type(x) is not int or x < 0 for x in self.window_starts) or
            any(b <= a for a, b in zip(self.window_starts, self.window_starts[1:]))):
            raise ValueError('window_starts must be increasing nonnegative integers starting at 0')
        if (not self.lesion_fractions or any(type(x) not in (int, float) or not 0 < x <= 1
                                            for x in self.lesion_fractions) or
            tuple(sorted(set(self.lesion_fractions))) != self.lesion_fractions):
            raise ValueError('lesion_fractions must be unique increasing fractions in (0, 1]')
        span = self.warmup_segments+self.segments+max(0, self.horizon-1)
        if self.registers and (len(self.window_starts) < 2 or
            any(b-a < span for a, b in zip(self.window_starts, self.window_starts[1:]))):
            raise ValueError('Register analysis needs >=2 windows, disjoint including warmup and horizon')
        self._state = None
        self._report = None

    def inspect(self):
        """Validate inputs and count forwards before allocating a GPU model."""
        print('Checking checkpoint and validation/tokenizer hashes...', flush=True)
        if self._state is None:
            self.checkpoint_sha256 = file_hash(self.checkpoint)
            loaded = load_checkpoint(self.checkpoint)
            self._state = {key: loaded[key] for key in
                           ('format', 'model', 'config', 'dataset_manifest', 'step', 'world_size')}
            self._state['target_tokens_seen'] = loaded.get('target_tokens_seen')
        cfg = build_config(Config, deepcopy(self._state['config']))
        if (cfg.model.name != 'grt' or cfg.data.name != 'wikitext103' or
            cfg.model.attention_mode != 'decoder'):
            raise ValueError('Posthoc notebook requires a current WikiText GRT checkpoint')
        self.training_config = asdict(cfg)
        self.manifest = json.loads((self.prepared_dir/'manifest.json').read_text())
        if self.manifest != self._state['dataset_manifest']:
            raise ValueError('Prepared manifest differs from checkpoint; copy the original artifacts')
        cfg.data.options['path'] = str(self.prepared_dir)
        cfg.rtla.batch_size = self.batch_size
        self.tokens = WikiTextAdapter().dataset(cfg, 'validation')
        self.cfg = cfg
        self.settings = dict(split='validation', precision='fp32', batch_size=self.batch_size,
            segments=self.segments, warmup_segments=self.warmup_segments,
            horizon=self.horizon, slots=self.slots, trace_examples=True,
            segment_len=cfg.model.segment_len, num_registers=cfg.model.num_registers,
            registers=self.registers, window_starts=(self.window_starts if self.registers else (0,)),
            group_repeats=self.group_repeats, lesion_fractions=self.lesion_fractions,
            analysis_seed=self.analysis_seed)
        source_root = Path(__file__).resolve().parents[1]
        source_files = sorted(source_root.rglob('*.py'))
        self.source_hashes = {str(path.relative_to(source_root)): file_hash(path) for path in source_files}
        signature = json.dumps(dict(checkpoint=self.checkpoint_sha256,
            validation=self.manifest['sha256']['validation'], settings=self.settings,
            source=self.source_hashes, device=self.device, python=platform.python_version(),
            gpu=(torch.cuda.get_device_name() if self.device == 'cuda' and torch.cuda.is_available() else None),
            cuda_runtime=torch.version.cuda,
            torch=str(torch.__version__), transformers=transformers.__version__), sort_keys=True)
        key = hashlib.sha256(signature.encode()).hexdigest()[:12]
        self.analysis_dir = self.output_dir/f'm{cfg.model.num_registers}_step{self._state["step"]}_{key}'
        self.report_path = self.analysis_dir/'posthoc_report.json'
        available = (self.tokens.numel()//self.batch_size-2)//cfg.model.segment_len+1
        measured = min(self.segments, available-self.warmup_segments)
        if measured <= 0:
            raise ValueError('Validation probe is too short; reduce warmup/batch')
        span = self.warmup_segments+self.segments+max(0, self.horizon-1)
        if self.registers and available < self.window_starts[-1]+span:
            raise ValueError('Validation is too short for full disjoint windows; reduce starts/segments/horizon')
        stop = min(available, self.warmup_segments+measured+max(0, self.horizon-1))
        calls = stop+2*measured+(measured*cfg.model.num_registers if self.slots else 0)
        calls += sum(max(0, min(self.horizon, stop-self.warmup_segments-i)-1) for i in range(measured))
        # Before calibration the group sizes are unknown: report a safe upper bound.
        if self.registers:
            other_calls = stop+2*measured+sum(max(0, min(self.horizon,
                stop-self.warmup_segments-i)-1) for i in range(measured))
            calls += (len(self.window_starts)-1)*other_calls
            masks = 8+6*self.group_repeats+len(self.lesion_fractions)*self.group_repeats
            calls += len(self.window_starts)*2*masks*measured*max(1, self.horizon)
        info = dict(model=cfg.model.name, M=cfg.model.num_registers,
            router_hidden=cfg.model.router.mlp_hidden, checkpoint_step=self._state['step'],
            training_world_size=self._state['world_size'], analysis_world_size=1,
            checkpoint_sha256=self.checkpoint_sha256, measured_segments=measured,
            forward_calls=calls, forward_count_kind=('upper_bound_until_calibration' if self.registers else 'exact'),
            settings=self.settings, output=str(self.analysis_dir))
        print(json.dumps(info, indent=2, ensure_ascii=False), flush=True)
        return info

    def run(self):
        self.inspect()
        if self.report_path.exists():
            self._report = json.loads(self.report_path.read_text())
            print('Reusing completed analysis:', self.report_path)
            return self._report
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('Select a Colab GPU runtime, or explicitly use device="cpu"')
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        last_print = started

        phase = 'primary probe'
        phase_started = started
        actual_calls = 0

        def progress(done, total):
            nonlocal last_print, phase_started, actual_calls
            actual_calls += 1
            now = time.perf_counter()
            if done == 1:
                phase_started = now
            if done == 1 or done == total or now-last_print >= 10:
                elapsed = now-started
                remaining = (now-phase_started)/done*(total-done)
                print(f'{phase}: forward {done}/{total}; elapsed {elapsed:.1f}s; '
                      f'phase progress {100*done/total:.1f}%, phase ETA ~{remaining:.1f}s', flush=True)
                last_print = now

        matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        cudnn_tf32 = torch.backends.cudnn.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            with preserve_rng(), torch.autocast(self.device, enabled=False):
                model, _ = restore_model(self._state, self.device)
                model.float()
                if self.device == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                windows, observers = [], []
                for start in self.settings['window_starts']:
                    phase = f'window {start} normal/reset/swap'
                    observer = RegisterObserver() if self.registers else None
                    window_summary, window_arrays = analyze_stream(model, self.cfg, self.tokens,
                        segments=self.segments, warmup_segments=self.warmup_segments,
                        horizon=self.horizon, slots=self.slots and start == 0,
                        trace_examples=True, progress=progress, start_segment=start, observer=observer)
                    if start == 0:
                        summary, arrays = window_summary, window_arrays
                    if self.registers:
                        diagnostics = observer.diagnostics(model)
                        windows.append(dict(start_segment=start, role='calibration' if start == 0 else 'evaluation',
                            summary=window_summary,
                            gate_statistics=gate_statistics(diagnostics['write_gates']),
                            diagnostics=diagnostics))
                        observers.append(observer)
                register_analysis = None
                if self.registers:
                    labels = classify_gates(windows[0]['diagnostics']['write_gates'])
                    groups, plans = intervention_plan(labels, self.lesion_fractions,
                                                     self.group_repeats, self.analysis_seed)
                    for window, observer in zip(windows, observers):
                        window['stability'] = group_stability(window['diagnostics']['write_gates'], labels)
                        phase = f'window {window["start_segment"]} group interventions'
                        calls = intervention_forward_calls(plans, self.segments, len(observer.records), self.horizon)
                        print(f'{phase}: {calls} forwards (after mask deduplication)', flush=True)
                        window['interventions'] = analyze_groups(model, observer, self.segments,
                            plans, self.horizon, progress)
                    register_analysis = dict(num_registers=self.cfg.model.num_registers,
                        calibration_start_segment=0, calibration_groups=groups,
                        intervention_plan=plans, windows=windows,
                        protocol=dict(threshold=.01, threshold_sensitivity=[.001, .01, .05],
                            window_state='Reset to learned S0 at each start, then equal warmup',
                            masks='Defined only on calibration; fixed for every window',
                            damage_curve='Nested random subsets per repeat; ceil(fraction*M)',
                            uncertainty='Raw paired example/start/lag deltas; no iid token confidence interval',
                            scope='Gate categories are observations, not established functional roles'))
                del observers
                repo = Path(__file__).resolve().parents[3]
                runtime = metadata(model, self.cfg, repo_dir=repo)
                runtime.update(analysis_world_size=1, precision='fp32', dtype='fp32', tf32=False,
                    analysis_forward_calls=actual_calls,
                    peak_gpu_memory_bytes=(torch.cuda.max_memory_allocated() if self.device == 'cuda' else 0))
                runtime['git_dirty'] = (bool(subprocess.check_output(
                    ['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip())
                    if (repo/'.git').exists() else None)
                if (repo/'source_revision.json').is_file():
                    runtime['source_snapshot'] = json.loads((repo/'source_revision.json').read_text())
                del model
        finally:
            torch.backends.cuda.matmul.allow_tf32 = matmul_tf32
            torch.backends.cudnn.allow_tf32 = cudnn_tf32
        checkpoint = dict(name=self.checkpoint.name, path=str(self.checkpoint),
            sha256=self.checkpoint_sha256, step=self._state['step'],
            format=self._state['format'], training_world_size=self._state['world_size'],
            target_tokens_seen=self._state['target_tokens_seen'])
        metrics = probe_metrics(summary, arrays)
        diagnostics = gate_statistics(arrays['trace_w_gates_per_example'])
        save_analysis(self.analysis_dir, summary, arrays,
            dict(checkpoint=checkpoint, analysis_settings=self.settings), plots=True)
        if register_analysis is not None:
            from grt.plots import plot_register_diagnostics
            plot_register_diagnostics(register_analysis, self.analysis_dir/'registers.png')
        runtime['analysis_wall_seconds'] = time.perf_counter()-started
        self._report = portable(dict(
            report_format='grt.posthoc_report/1', report_id=uuid4().hex,
            created_at=datetime.now(timezone.utc).isoformat(), status='completed',
            checkpoint=checkpoint, training_config=self.training_config,
            dataset_manifest=self.manifest, analysis_settings=self.settings,
            environment=runtime, analysis_source_sha256=self.source_hashes,
            summary=summary, metrics=metrics, gate_statistics=diagnostics, arrays=arrays,
            register_analysis=register_analysis,
            interpretation=dict(
                gate_scope='Per-example gates on the fixed validation probe; not training trajectories',
                slot_scope='Reset one slot to learned initial S0; contributions are not additive',
                retention_scope='Swap once, then propagate the altered branch; lag0=current segment',
                attention_scope='Write pooling attention is not token-to-register read attention',
                test_evaluated=False)))
        temporary = self.report_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self._report, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
        temporary.replace(self.report_path)
        print(f'Completed: normal probe PPL={summary["conditions"]["normal"]["perplexity"]:.4f}; '
              f'wall={runtime["analysis_wall_seconds"]:.1f}s', flush=True)
        return self._report

    def report(self, *, download=False):
        if self._report is None:
            raise RuntimeError('Call run() before report()')
        print('Send this single file:', self.report_path)
        print(f'Report size: {self.report_path.stat().st_size/2**20:.2f} MiB')
        if download:
            from google.colab import files
            files.download(str(self.report_path))
        return self.report_path
