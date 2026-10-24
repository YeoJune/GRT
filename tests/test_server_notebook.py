"""Kaggle transfer notebook: immutable conditions and fresh-Save resume routing."""
import ast
from datetime import datetime, timedelta
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from grt.utils.kaggle import WikiTextRun
from grt.utils import server

REPO = Path(__file__).resolve().parents[1]
CELLS = {c['id']: ''.join(c['source']) for c in json.loads(
    (REPO/'notebooks/kaggle_gate_transfer.ipynb').read_text())['cells'] if c['cell_type'] == 'code'}


def selection(tmp_path, **values):
    tree = ast.parse(CELLS['selection'])
    values['work_dir'] = str(tmp_path)
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
            if key in values:
                node.value = ast.Call(func=ast.Name(id='Path', ctx=ast.Load()),
                    args=[ast.Constant(values[key])], keywords=[]) if key == 'work_dir' else ast.Constant(values[key])
    namespace = {}
    exec(compile(ast.fix_missing_locations(tree), '<selection>', 'exec'), namespace)
    return namespace


class Client:
    def __init__(self, job=None):
        self.job, self.submissions = job, {}

    def check(self):
        return {'kind': 'write'}

    def ensure_schema(self):
        pass

    def ensure_queue(self, name):
        pass

    def request(self, method, route, body=None, *, key=None):
        if route == '/jobs/validate':
            return {}
        if route == '/jobs' and method == 'POST':
            if key in self.submissions:
                assert self.submissions[key] == body
            self.submissions[key] = body
            self.job = {**body, 'id': 'job_test', 'status': 'queued',
                        'resume_artifact_id': None, 'artifacts': [], 'attempts': []}
        if route.startswith('/jobs?'):
            return {'data': [self.job], 'next_cursor': None}
        return self.job


@pytest.mark.parametrize('memory', [8, 75])
@pytest.mark.parametrize('source', ['candidate', 'independent'])
def test_new_job_preserves_transfer_conditions_and_submission_identity(tmp_path, monkeypatch, memory, source):
    namespace = selection(tmp_path, num_registers=memory, write_source=source)
    client = Client()
    def prepare(run, **kwargs):
        run.prepared_dir.mkdir(parents=True, exist_ok=True)
        (run.prepared_dir/'manifest.json').write_text('{"test_manifest": true}')
    monkeypatch.setattr(WikiTextRun, 'setup', lambda *a, **k: None)
    monkeypatch.setattr(WikiTextRun, 'prepare_data', prepare)
    monkeypatch.setattr(WikiTextRun, 'cache_prepared', lambda run: {
        'extensions': {'manifest_sha256': server.digest(run.prepared_dir/'manifest.json')}})
    namespace.update(repo=REPO, WikiTextRun=WikiTextRun, client=client)
    exec(CELLS['submit'], namespace)
    cfg = client.job['spec']['config']
    assert cfg['model']['num_registers'] == memory and cfg['model']['segment_len'] == 75
    assert cfg['model']['router']['write_source'] == source
    assert cfg['model']['router']['write_detach_source'] is False
    assert cfg['model']['register']['dropout_prob'] == 0
    assert cfg['model']['register']['write_logit_coef'] == 1e-4
    assert cfg['model']['register']['write_gate_bias_init'] == 0
    assert cfg['training']['max_steps'] == 40000 and cfg['training']['global_batch_size'] == 256
    assert cfg['training']['warmup_ratio'] == .05 and cfg['training']['min_lr_ratio'] == .01
    assert cfg['training']['max_seconds'] is None  # Cumulative budget would prevent a fresh attempt progressing.
    assert client.job['spec']['world_size'] == 2
    assert client.job['spec']['posthoc_options']['window_starts'] == [0, 16, 32]
    assert client.job['policy']['checkpointable'] and client.job['policy']['require_posthoc']
    assert client.job['queue'].endswith(f'm{memory}-{source}')
    exec(CELLS['submit'], namespace)
    assert len(client.submissions) == 1
    repeated = selection(tmp_path)
    assert repeated['server_deadline'] == namespace['server_deadline']
    assert datetime.fromisoformat(namespace['server_deadline']) == namespace['session_started'] + timedelta(hours=11)


@pytest.mark.parametrize('status,snapshot', [('running', 'art_resume'), ('queued', None), ('queued', 'art_resume')])
def test_new_save_resume_pins_original_code_and_never_submits(tmp_path, monkeypatch, status, snapshot):
    namespace = selection(tmp_path, execution_mode='resume', resume_job_id='job_previous')
    job = {'id': 'job_previous', 'status': status, 'resume_artifact_id': snapshot,
           'queue': 'transfer-m75-independent', 'spec': {
               'code_ref': 'a'*40, 'code_sha256': 'b'*64, 'world_size': 2, 'target_steps': 40000,
               'config': {'model': {'num_registers': 75, 'router': {'write_source': 'independent'}}},
               'dataset_sha256': 'c'*64}}
    client = Client(job)
    commands = []
    monkeypatch.setattr(server, 'ServerClient', lambda *a, **k: client)
    monkeypatch.setattr(server, 'code_identity', lambda repo: ('a'*40, 'b'*64))
    monkeypatch.setattr(namespace['subprocess'], 'check_output', lambda *a, **k: '')
    monkeypatch.setattr(namespace['subprocess'], 'run', lambda command, **kw: commands.append((command, kw)) or SimpleNamespace(returncode=0))
    if status != 'queued' or not snapshot:
        with pytest.raises(RuntimeError, match='이어서 학습'):
            exec(CELLS['checkout'], namespace)
        assert not any('checkout' in cmd for cmd, _ in commands)
        return
    exec(CELLS['checkout'], namespace)
    assert any(cmd[-3:] == ['checkout', '--detach', 'a'*40] for cmd, _ in commands)
    exec(CELLS['submit'], namespace)
    assert not client.submissions
    monkeypatch.setitem(sys.modules, 'kaggle_secrets', SimpleNamespace(
        UserSecretsClient=lambda: SimpleNamespace(get_secret=lambda name: 'test-secret')))
    exec(CELLS['worker'], namespace)
    cmd, kwargs = commands[-1]
    assert '--prepared-from-server' in cmd and '--once' in cmd
    assert cmd[cmd.index('--world-size')+1] == '2'
    assert cmd[cmd.index('--queue')+1] == job['queue']
    assert kwargs['env']['GRT_SERVER_TOKEN'] == 'test-secret'
    assert 'test-secret' not in cmd
    assert 'worker_env' not in namespace
    assert namespace['selected_job']['spec']['config']['model']['num_registers'] == 75
