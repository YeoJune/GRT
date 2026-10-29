"""Unified notebook/controller routing; GPU training and the real API are not mocked as evidence."""
import ast
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import subprocess

import pytest

from grt.utils.kaggle import WikiTextRun
from grt.utils import notebook, server

REPO = Path(__file__).resolve().parents[1]
POSTHOC = dict(batch_size=8, segments=8, warmup_segments=3, horizon=4,
               slots=True, registers=True, window_starts=[0, 16, 32], group_repeats=3)


class Client:
    url = 'https://grt.example'

    def __init__(self, job=None):
        self.job, self.submissions, self.requests = job, {}, []
        self.other_job = False

    def check(self):
        return {'kind': 'write'}

    def ensure_schema(self):
        pass

    def ensure_queue(self, name):
        pass

    def request(self, method, route, body=None, *, key=None):
        self.requests.append((method, route))
        if route == '/jobs/validate':
            return {}
        if route == '/jobs' and method == 'POST':
            if key in self.submissions:
                assert self.submissions[key] == body
            else:
                self.submissions[key] = body
                self.job = {**body, 'id': 'job_test', 'status': 'queued',
                            'resume_artifact_id': None, 'artifacts': [], 'attempts': []}
        if route.startswith('/jobs?'):
            return {'data': [self.job, {'id': 'job_other'}] if self.other_job else [self.job],
                    'next_cursor': None}
        return self.job


@pytest.fixture
def harness(tmp_path, monkeypatch):
    client = Client()
    calls, secrets = [], []
    monkeypatch.setattr(server, 'ServerClient', lambda *a, **k: client)
    monkeypatch.setattr(server, 'code_identity', lambda repo: ('a'*40, 'b'*64))
    monkeypatch.setattr(notebook.ServerNotebookRun, '_checkout', lambda self, *identity: REPO)
    monkeypatch.setattr(notebook.subprocess, 'check_output', lambda *a, **k: '')
    monkeypatch.setattr(WikiTextRun, 'setup', lambda *a, **k: None)
    def prepare(run, **kwargs):
        run.prepared_dir.mkdir(parents=True, exist_ok=True)
        (run.prepared_dir/'manifest.json').write_text('{"test_manifest": true}')
        return {'test_manifest': True}
    monkeypatch.setattr(WikiTextRun, 'prepare_data', prepare)
    monkeypatch.setattr(WikiTextRun, 'cache_prepared', lambda run: {
        'extensions': {'manifest_sha256': server.digest(run.prepared_dir/'manifest.json')}})
    def secret(name, runtime):
        secrets.append((name, runtime))
        return 'secret-'+name
    monkeypatch.setattr(notebook, 'read_secret', secret)
    def worker(command, **kwargs):
        calls.append((command, {**kwargs, 'env': dict(kwargs['env'])}))
        client.job['status'] = 'succeeded'
    monkeypatch.setattr(notebook, 'run_logged', worker)
    return client, calls, secrets


@pytest.mark.parametrize('runtime', ['kaggle', 'colab'])
@pytest.mark.parametrize('memory,source', [(8, 'candidate'), (20, 'separate_head'), (75, 'independent')])
def test_new_job_preserves_conditions_and_is_idempotent(tmp_path, harness, runtime, memory, source):
    client, calls, secrets = harness
    def selection():
        return notebook.notebook_run(REPO, runtime=runtime, execution_mode='new', work_dir=tmp_path,
            models=('grt',), budget='40k', num_registers=memory, write_source=source,
            campaign='test-campaign')
    run = selection()
    run.setup(install=False)
    run.prepare_data()
    run.execute(posthoc_options=POSTHOC, download=False)
    cfg = client.job['spec']['config']
    assert cfg['model']['num_registers'] == memory and cfg['model']['segment_len'] == 75
    assert cfg['model']['router']['write_source'] == source
    assert cfg['model']['router']['write_detach_source'] is False
    assert cfg['model']['register']['dropout_prob'] == 0
    assert cfg['model']['register']['write_logit_coef'] == 1e-4
    assert cfg['model']['register']['write_gate_bias_init'] == 0
    assert cfg['training']['max_steps'] == 40000 and cfg['training']['global_batch_size'] == 256
    assert cfg['training']['warmup_ratio'] == .05 and cfg['training']['min_lr_ratio'] == .01
    assert cfg['training']['max_seconds'] is None
    assert client.job['spec']['world_size'] == (2 if runtime == 'kaggle' else 1)
    assert client.job['spec']['posthoc_options'] == POSTHOC
    assert client.job['policy']['checkpointable'] and client.job['policy']['require_posthoc']
    assert client.job['labels']['campaign'] == 'test-campaign'
    assert len(calls) == 1
    cmd, options = calls[0]
    assert '--prepared-from-server' in cmd and '--once' in cmd
    assert cmd[cmd.index('--world-size')+1] == str(run.world_size)
    assert options['env']['GRT_SERVER_TOKEN'] == 'secret-GRT_SERVER_TOKEN'
    assert not any('secret-GRT' in arg for arg in cmd)
    repeated = selection()
    assert repeated.deadline == run.deadline
    repeated.setup(install=False)
    repeated.prepare_data()
    repeated.execute(posthoc_options=POSTHOC, download=False)
    assert len(client.submissions) == 1 and len(calls) == 1


def old_job(status='queued', snapshot='art_resume', world_size=2):
    return {'id': 'job_previous', 'status': status, 'resume_artifact_id': snapshot,
        'queue': 'original-queue', 'artifacts': [], 'attempts': [], 'spec': {
            'code_ref': 'a'*40, 'code_sha256': 'b'*64, 'world_size': world_size, 'target_steps': 40000,
            'config': {'model': {'num_registers': 75, 'router': {'write_source': 'independent'}},
                       'wandb': {'enabled': False, 'mode': 'online'}},
            'posthoc_options': POSTHOC, 'dataset_sha256': 'c'*64}}


@pytest.mark.parametrize('status,snapshot', [('running', 'art_resume'), ('queued', None), ('queued', 'art_resume')])
def test_resume_uses_original_job_and_never_submits(tmp_path, harness, monkeypatch, status, snapshot):
    client, calls, secrets = harness
    client.job = old_job(status, snapshot)
    # Even invalid new-run choices must not rebuild or replace the original experiment.
    run = notebook.notebook_run(REPO, runtime='kaggle', execution_mode='resume', work_dir=tmp_path,
        resume_job_id='job_previous', budget='invalid', num_registers=-1, write_source='invalid')
    pins = []
    monkeypatch.setattr(run, '_checkout', lambda *identity: pins.append(identity) or REPO)
    if status != 'queued' or not snapshot:
        with pytest.raises(RuntimeError, match='이어서 학습'):
            run.setup(install=False)
        assert not pins and not calls
        return
    run.setup(install=False)
    run.prepare_data(server_cache='ignored-notebook-choice')
    run.execute(posthoc_options={'horizon': 99}, download=False)
    assert pins == [('a'*40, 'b'*64)]
    assert not client.submissions
    assert client.job['spec']['posthoc_options'] == POSTHOC
    assert client.job['spec']['config']['model']['num_registers'] == 75
    assert secrets == [('GRT_SERVER_TOKEN', 'kaggle')]  # W&B was disabled in the original job.
    cmd, kwargs = calls[0]
    assert cmd[cmd.index('--queue')+1] == 'original-queue'
    assert kwargs['env']['GRT_SERVER_TOKEN'] == 'secret-GRT_SERVER_TOKEN'
    assert 'secret-GRT_SERVER_TOKEN' not in json.dumps(json.loads((run.work_dir/'selected_job.json').read_text()))


@pytest.mark.parametrize('problem', ['world_size', 'queue', 'code', 'deadline'])
def test_resume_refuses_incompatible_or_unsafe_worker_start(tmp_path, harness, monkeypatch, problem):
    client, calls, _ = harness
    client.job = old_job(world_size=1 if problem == 'world_size' else 2)
    run = notebook.notebook_run(REPO, runtime='kaggle', execution_mode='resume',
        resume_job_id='job_previous', work_dir=tmp_path)
    if problem == 'world_size':
        with pytest.raises(ValueError, match='world_size'):
            run.setup(install=False)
    else:
        run.setup(install=False)
        if problem == 'queue':
            client.other_job = True
        elif problem == 'code':
            monkeypatch.setattr(server, 'code_identity', lambda repo: ('d'*40, 'b'*64))
        else:
            run.deadline = (datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()
        with pytest.raises(RuntimeError):
            run.execute(download=False)
    assert not calls and not client.submissions


def test_first_cell_deadline_is_not_extended_by_reruns(tmp_path):
    start = datetime.now(timezone.utc)-timedelta(hours=1)
    deadline = notebook.session_deadline(tmp_path, hours=11, started=start.timestamp())
    assert datetime.fromisoformat(deadline) == start+timedelta(hours=11)
    assert notebook.session_deadline(tmp_path, hours=20) == deadline
    (tmp_path/'session_deadline.json').write_text(json.dumps({
        'deadline': (datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()}))
    with pytest.raises(RuntimeError, match='deadline'):
        notebook.session_deadline(tmp_path, hours=20)


def test_original_checkout_is_isolated_and_rejects_modified_worker(tmp_path):
    repo = tmp_path/'repo'
    repo.mkdir()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()
    git('init')
    git('config', 'user.email', 'test@example.invalid')
    git('config', 'user.name', 'Test')
    (repo/'src').mkdir()
    (repo/'src/model.py').write_text('original = True\n')
    git('add', '.')
    git('commit', '-m', 'original')
    original = server.code_identity(repo)
    (repo/'src/model.py').write_text('original = False\n')
    git('commit', '-am', 'new controller')
    current = git('rev-parse', 'HEAD')
    git('remote', 'add', 'origin', str(repo))
    run = notebook.notebook_run(repo, runtime='kaggle', execution_mode='resume',
        resume_job_id='job_previous', work_dir=tmp_path)
    worker = run._checkout(*original)
    assert git('rev-parse', 'HEAD') == current
    assert server.code_identity(worker) == original
    (worker/'src/model.py').write_text('modified = True\n')
    with pytest.raises(RuntimeError, match='hash'):
        run._checkout(*original)


def test_local_execution_uses_existing_training_and_report_without_server(tmp_path, monkeypatch):
    monkeypatch.setattr(server, 'ServerClient', lambda *a, **k: pytest.fail('local must not connect'))
    run = notebook.notebook_run(REPO, runtime='kaggle', models=('grt',), budget='40k',
                                work_dir=tmp_path, wandb=False)
    stages = []
    monkeypatch.setattr(run, 'preflight', lambda: stages.append('preflight'))
    monkeypatch.setattr(run, 'train', lambda: stages.append('train'))
    def report(**kwargs):
        stages.append('report')
        assert kwargs == dict(posthoc=True, download=False, posthoc_options=POSTHOC)
    monkeypatch.setattr(run, 'report', report)
    run.execute(posthoc_options=POSTHOC, download=False)
    assert stages == ['preflight', 'train', 'report']
    assert not (tmp_path/'grt_server').exists()


@pytest.mark.parametrize('runtime', ['colab', 'kaggle'])
def test_notebook_gate_choices_do_not_add_gates_to_baselines(tmp_path, runtime):
    run = notebook.notebook_run(REPO, runtime=runtime, models=('neox', 'rmt', 'grt'),
        budget='40k', work_dir=tmp_path, wandb=False, write_source='separate_head')
    for model in run.models:
        cfg = run.config(model, calibrated=False)
        assert cfg.model.segment_len == 75
        assert cfg.training.batch_size*cfg.training.grad_accum_steps*run.world_size == 256
        if model == 'grt':
            assert cfg.model.router.write_source == 'separate_head'
        else:
            assert cfg.model.router is None and cfg.model.register is None


def test_grt_notebook_keeps_existing_config_and_local_run_identity(tmp_path):
    current = notebook.notebook_run(REPO, runtime='kaggle', models=('grt',), budget='40k',
        seed=54, wandb=False, work_dir=tmp_path)
    from grt.config import merge_dicts
    previous = WikiTextRun(REPO, runtime='kaggle', models=('grt',), budget='40k',
        seed=54, wandb=False, work_dir=tmp_path, num_registers=20, router_hidden=512,
        overrides=merge_dicts(current.overrides, current.grt_overrides))
    assert current.run_dir == previous.run_dir
    assert current.config('grt') == previous.config('grt')


@pytest.mark.parametrize('runtime', ['colab', 'kaggle'])
def test_unified_notebook_selection_runs_in_local_and_resume_modes(tmp_path, harness, monkeypatch, runtime):
    client, calls, _ = harness
    monkeypatch.setattr(notebook.subprocess, 'run', lambda *a, **k: None)
    book = json.loads((REPO/'notebooks'/f'{runtime}.ipynb').read_text())
    code = next(''.join(c['source']) for c in book['cells'] if c['id'] == 'selection')
    for mode in ('local', 'resume'):
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                values = dict(execution_mode=mode, resume_job_id='job_previous' if mode == 'resume' else '')
                if node.targets[0].id in values:
                    node.value = ast.Constant(values[node.targets[0].id])
        client.job = old_job(world_size=2 if runtime == 'kaggle' else 1)
        def factory(*a, **kwargs):
            return notebook.notebook_run(*a, **kwargs, work_dir=tmp_path)
        ns = dict(notebook_run=factory, repo=REPO, session_started=None)
        exec(compile(ast.fix_missing_locations(tree), '<selection>', 'exec'), ns)
        if mode == 'local':
            assert isinstance(ns['job'], WikiTextRun)
        else:
            assert ns['job'].selected_job['id'] == 'job_previous'
    assert not calls and not client.submissions
