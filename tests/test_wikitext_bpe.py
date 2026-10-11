"""Offline byte-BPE preparation and the notebook's production stream workflow."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest
import torch
import yaml
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import GPT2TokenizerFast
from grt.config import load_config
from grt.wikitext import WikiTextAdapter
from scripts.prepare_wikitext_bpe import prepare

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def prepared(tmp_path):
    # Offline fixture has all bytes and GPT-2's ID range, not the real GPT-2 merges.
    vocab = {token: index for index, token in enumerate(pre_tokenizers.ByteLevel.alphabet())}
    vocab.update({f'fixture_{i}': i for i in range(256, 50256)})
    vocab['<|endoftext|>'] = 50256
    backend = Tokenizer(models.BPE(vocab, []))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = GPT2TokenizerFast(tokenizer_object=backend, add_prefix_space=False)
    tokenizer_dir = tmp_path / 'fixture_tokenizer'
    tokenizer.save_pretrained(tokenizer_dir)
    source = tmp_path / 'source'
    source.mkdir()
    texts = {'train': ' Alpha beta!\n' * 100,
             'validation': ' Alpha beta!\n\nUnicode 한글!  \n' + 'long ' * 300 + '\n',
             'test': ' Alpha beta!\n' * 4}
    for split, name in [('train', 'train'), ('validation', 'valid'), ('test', 'test')]:
        (source / f'wiki.{name}.tokens').write_text(texts[split], encoding='utf-8')
    destination = tmp_path / 'prepared'
    manifest = prepare(source, destination, str(tokenizer_dir), local_files_only=True)
    return source, destination, manifest, tokenizer, texts


def test_byte_bpe_preserves_text_eos_long_lines_and_hashes(prepared):
    source, destination, manifest, tokenizer, texts = prepared
    assert manifest['vocab_size'] == 50257
    assert manifest['tokenization'] == 'gpt2_bpe'
    for split, text in texts.items():
        ids = torch.load(destination / f'{split}.pt', weights_only=True)
        # A round trip checks whitespace, Unicode, blank lines and no truncation.
        expected = ''.join(line + '<|endoftext|>' for line in text.splitlines())
        assert tokenizer.decode(ids.tolist(), clean_up_tokenization_spaces=False) == expected
        assert (ids == 50256).sum().item() == manifest['line_counts'][split]
        assert ids.numel() == manifest['token_counts'][split]
        assert hashlib.sha256((destination / f'{split}.pt').read_bytes()).hexdigest() == manifest['sha256'][split]
    with pytest.raises(FileExistsError):
        prepare(source, destination, 'not-loaded', local_files_only=True)


def test_adapter_checks_bpe_identity_and_tokenizer_artifacts(prepared):
    _, destination, manifest, _, _ = prepared
    cfg = load_config(['configs/base.yaml', 'configs/neox.yaml', 'configs/wikitext103.yaml',
                       'configs/wikitext_small.yaml', 'configs/wikitext_bpe.yaml', 'configs/neox_rmt.yaml'])
    cfg.data.options['path'] = str(destination)
    adapter = WikiTextAdapter()
    assert adapter.dataset(cfg, 'test').numel() == manifest['token_counts']['test']
    cfg.data.options['tokenization'] = 'word'
    with pytest.raises(ValueError, match='tokenization'):
        adapter.dataset(cfg, 'test')
    cfg.data.options['tokenization'] = 'gpt2_bpe'
    (destination / 'tokenizer/merges.txt').write_text('changed')
    with pytest.raises(ValueError, match='artifact hash'):
        adapter.dataset(cfg, 'test')


def test_kaggle_notebook_cpu_preflight_training_and_resume(prepared, tmp_path, monkeypatch):
    original, destination, _, _, _ = prepared
    # Execute the notebook's actual helpers, config builder, measurement and train cells.
    # CPU, one worker and a tiny stack are explicit test substitutions.
    monkeypatch.setenv('OMP_NUM_THREADS', '1')
    monkeypatch.setenv('MKL_NUM_THREADS', '1')
    monkeypatch.setenv('HF_HOME', str(tmp_path / 'hf_cache'))
    notebook = json.loads((REPO / 'notebooks/kaggle.ipynb').read_text())
    source = lambda index: ''.join(notebook['cells'][index]['source'])
    root = tmp_path / 'run'
    root.mkdir()
    namespace = dict(Path=Path, os=os, sys=sys, subprocess=subprocess, signal=signal,
        json=json, hashlib=hashlib, yaml=yaml, torch=torch,
        REPO_DIR=REPO, RUN_DIR=root, PREPARED_DIR=destination, TOKENIZATION='gpt2_bpe',
        PREPARED_INPUT=destination, SOURCE_DIR=original,
        WANDB=False, MAX_STEPS=2, MODELS=('rmt', 'grt'), TRAIN_MODEL='rmt',
        WORLD_SIZE=1, DEVICE='cpu', BENCHMARK_STEPS=1, MEMORY_FRACTION=.8)
    exec(compile(source(3), 'notebook_helpers', 'exec'), namespace)
    exec(compile(source(4), 'notebook_data', 'exec'), namespace)
    exec(compile(source(5), 'notebook_configs', 'exec'), namespace)
    small = tmp_path / 'tiny.yaml'
    small.write_text(yaml.safe_dump({'model': {'d_model': 32, 'segment_len': 4, 'num_registers': 2,
        'alu': {'num_layers': 1, 'nhead': 4, 'd_ff': 64}},
        'training': {'batch_size': 4, 'grad_accum_steps': 15, 'global_batch_size': 60, 'bptt_segments': 1},
        'evaluation': {'batch_size': 10, 'every_steps': 2}}))
    config_builder = namespace['configs']
    namespace['configs'] = lambda model: config_builder(model) + [str(small)]
    namespace['launcher'] = [sys.executable, '-u', 'scripts/train.py']
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda i: type('Device', (), {'total_memory': 2**40})())
    exec(compile(source(7), 'notebook_preflight', 'exec'), namespace)
    for model in ['rmt', 'grt']:
        namespace['TRAIN_MODEL'] = model
        exec(compile(source(9), 'notebook_train', 'exec'), namespace)
        summary = json.loads((root / model / 'summary.json').read_text())
        assert summary['step'] == 2
        assert summary['perplexity_unit'] == 'bpe_token'
        expected_kind = 'autoregressive' if model == 'rmt' else 'teacher_forced_noncausal'
        assert summary['score_kind'] == expected_kind
        exec(compile(source(9), 'notebook_resume', 'exec'), namespace)
        assert json.loads((root / model / 'summary.json').read_text())['step'] == 2
        assert 'total_compute_hours' in (root / (model + '_eta.txt')).read_text()
