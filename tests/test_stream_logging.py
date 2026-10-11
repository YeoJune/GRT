"""Focused checks for new stream logging; no tokenizer/DDP/regression repetition."""
import hashlib
import json
from pathlib import Path

import pytest
import torch
from grt.config import load_config
from grt.logger import Monitor
from grt import trainer


@pytest.mark.parametrize('name', ['rmt', 'grt'])
def test_tiny_stream_logs_before_validation_with_real_offline_wandb(tmp_path, monkeypatch, name):
    pytest.importorskip('wandb')
    monkeypatch.setenv('WANDB_MODE', 'offline')
    monkeypatch.setenv('WANDB_CACHE_DIR', str(tmp_path / 'wandb_cache'))
    monkeypatch.setenv('WANDB_CONFIG_DIR', str(tmp_path / 'wandb_config'))
    monkeypatch.setenv('XDG_CACHE_HOME', str(tmp_path / 'cache'))
    data = tmp_path / 'tokens'
    data.mkdir()
    hashes, counts = {}, {}
    for split, size in [('train', 900), ('validation', 31), ('test', 31)]:
        ids = torch.arange(size, dtype=torch.long) % 13
        path = data / f'{split}.pt'
        torch.save(ids, path)
        hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        counts[split] = size
    (data / 'manifest.json').write_text(json.dumps(dict(
        dataset='synthetic CPU logging check; NOT WikiText', tokenization='gpt2_bpe',
        vocab_size=50257, sha256=hashes, token_counts=counts)))
    root = tmp_path / name
    cfg = load_config(['configs/base.yaml', 'configs/neox.yaml', 'configs/wikitext103.yaml',
        'configs/wikitext_tiny.yaml', 'configs/wikitext_bpe.yaml', f'configs/neox_{name}.yaml',
        'configs/wikitext_smoke.yaml'], output_dir=root)
    cfg.data.options['path'] = str(data)
    # Actual tiny backbone/vocabulary/L150/K3; shorten only the CPU verification budget.
    cfg.training.max_steps = 2
    cfg.training.batch_size = cfg.training.grad_accum_steps = cfg.training.global_batch_size = 1
    cfg.evaluation.batch_size = 1
    cfg.evaluation.every_steps = 2
    cfg.wandb.enabled = True
    cfg.wandb.mode = 'offline'
    cfg.wandb.every_steps = 1
    calls = []
    class CapturingMonitor(Monitor):
        def log(self, step, payload):
            calls.append((step, dict(payload)))
            super().log(step, payload)
    monkeypatch.setattr(trainer, 'Monitor', CapturingMonitor)
    result = trainer.run(cfg, torch.device('cpu'))
    assert result['step'] == 2
    assert [step for step, _ in calls] == [1, 2]
    assert 'val/loss' not in calls[0][1] and 'val/loss' in calls[1][1]
    rows = [json.loads(line) for line in (root / 'metrics.jsonl').read_text().splitlines()]
    assert [row['step'] for row in rows] == [1, 2]
    assert rows[0]['train/loss'] == calls[0][1]['train/loss']
    status = json.loads((root / 'wandb_status.json').read_text())
    assert status['mode'] == 'offline' and status['state'] == 'finished'
    assert status['failures'] == 0 and status['last_logged_step'] == 2
    assert status['summary_updated']
    assert any(path.stat().st_size > 0 for path in (root / 'wandb').rglob('*.wandb'))
