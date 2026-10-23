"""CPU notebook -> real Express API -> verified original artifacts."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest
import requests
import torch

from grt.utils.kaggle import WikiTextRun, file_hash
from grt.utils.server import ServerClient, ServerError, SCHEMA_ID, digest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def api_server(tmp_path_factory):
    server_repo = Path(os.environ.get('GRT_TEST_SERVER_REPO', REPO.parent / 'GRT_server'))
    if not (server_repo / 'node_modules/tsx').exists():
        pytest.skip('Set GRT_TEST_SERVER_REPO to an installed GRT_server checkout for live integration')
    root = tmp_path_factory.mktemp('server')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    env = {**os.environ, 'NODE_ENV': 'test', 'HOST': '127.0.0.1', 'PORT': str(port),
        'APP_ORIGIN': url, 'COOKIE_SECURE': 'false', 'DB_CLIENT': 'sqlite',
        'SQLITE_FILENAME': str(root / 'database.sqlite'), 'STORAGE_DRIVER': 'local',
        'STORAGE_DIRECTORY': str(root / 'files'), 'STORAGE_SIGNING_KEY': 'integration-key-' * 4,
        'OWNER_USERNAME': 'owner', 'OWNER_PASSWORD': 'integration-password-123',
        'HEARTBEAT_SECONDS': '1'}
    subprocess.run(['node', '--import', 'tsx', 'server/src/manage.ts', 'setup'],
        cwd=server_repo, env=env, check=True, capture_output=True)
    with (root / 'server.log').open('w') as log:
        process = subprocess.Popen(['node', '--import', 'tsx', 'server/src/index.ts'],
            cwd=server_repo, env=env, stdout=log, stderr=log)
        try:
            for _ in range(100):
                try:
                    if requests.get(url + '/healthz', timeout=1).status_code == 200:
                        break
                except requests.RequestException:
                    pass
                if process.poll() is not None:
                    pytest.fail('API server exited during startup')
                time.sleep(.1)
            else:
                pytest.fail('API server readiness timed out')
            session = requests.Session()
            login = session.post(url + '/v1/sessions', json={'username': 'owner', 'password': env['OWNER_PASSWORD']},
                headers={'Origin': url}, timeout=5)
            login.raise_for_status()
            headers = {'Origin': url, 'X-CSRF-Token': login.json()['csrf_token']}
            tokens = {}
            for kind in ('write', 'read-only'):
                issued = session.post(url + '/v1/tokens', json={'name': 'integration-' + kind, 'kind': kind},
                    headers={**headers, 'Idempotency-Key': 'integration-' + kind}, timeout=5)
                issued.raise_for_status()
                tokens[kind] = issued.json()['token']
            yield SimpleNamespace(url=url, tokens=tokens, directory=root)
        finally:
            process.terminate()
            process.wait(timeout=35)


@pytest.fixture
def client(api_server, monkeypatch):
    monkeypatch.setenv('GRT_SERVER_TOKEN', api_server.tokens['write'])
    value = ServerClient(api_server.url)
    value.check()
    return value


def make_run(tmp_path, client, *, steps=2, model='grt', mode='upload'):
    prepared = tmp_path / 'input'
    prepared.mkdir()
    manifest = dict(vocab_size=128, tokenization='word', token_counts={}, sha256={})
    for split, count in [('train', 80), ('validation', 160), ('test', 20)]:
        torch.save(torch.arange(count, dtype=torch.long) % 128, prepared / f'{split}.pt')
        manifest['token_counts'][split] = count
        manifest['sha256'][split] = file_hash(prepared / f'{split}.pt')
    (prepared / 'manifest.json').write_text(json.dumps(manifest))
    run = WikiTextRun(REPO, runtime='cpu', size='tiny', models=(model,), tokenization='word',
        work_dir=tmp_path / 'working', steps=steps, wandb=False, device='cpu', world_size=1,
        benchmark_steps=1, server_mode=mode, server_url=client.url,
        server_queue='test-' + tmp_path.name,
        server_deadline=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        overrides={'model': {'vocab_size': 128, 'd_model': 32, 'segment_len': 4, 'num_registers': 3,
            'alu': {'num_layers': 1, 'nhead': 4, 'd_ff': 64},
            **({'router': {'mlp_hidden': 32}} if model == 'grt' else {})},
            'training': {'auto_batch': False, 'batch_size': 1, 'grad_accum_steps': 1,
                'global_batch_size': 1, 'bptt_segments': 1},
            'evaluation': {'batch_size': 1, 'every_steps': 2}, 'wandb': {'every_steps': 1},
            'rtla': {'enabled': False}})
    run.prepare_data(prepared_input=prepared)
    return run


def test_off_mode_does_not_import_server_or_http_client():
    command = ("import sys; from grt.utils.kaggle import WikiTextRun; "
        f"WikiTextRun({str(REPO)!r}, wandb=False); "
        "assert 'grt.utils.server' not in sys.modules; assert 'requests' not in sys.modules")
    subprocess.run([sys.executable, '-c', command], check=True)


def test_secret_reader_has_no_fallback(monkeypatch):
    monkeypatch.setenv('GRT_SERVER_TOKEN', 'old-env-value')
    def unavailable(_):
        raise KeyError('secret unavailable')
    monkeypatch.setitem(sys.modules, 'google.colab', SimpleNamespace(userdata=SimpleNamespace(get=unavailable)))
    with pytest.raises(RuntimeError, match='selected secret source'):
        ServerClient('https://example.com', secret_source='colab')
    for url in ('http://example.com', 'https://user:password@example.com', 'https://example.com/v1'):
        with pytest.raises(ValueError):
            ServerClient(url)


def test_read_only_fails_before_gpu_or_subprocess(api_server, monkeypatch, tmp_path):
    monkeypatch.setenv('GRT_SERVER_TOKEN', api_server.tokens['read-only'])
    run = WikiTextRun(REPO, wandb=False, server_mode='upload', server_url=api_server.url, work_dir=tmp_path)
    calls = []
    monkeypatch.setattr('grt.utils.kaggle.subprocess.run', lambda *a, **k: calls.append(a))
    with pytest.raises(ServerError, match='write_token_required'):
        run.setup()
    assert calls == []


def test_real_upload_training_posthoc_and_repeat_are_idempotent(tmp_path, client, monkeypatch):
    monkeypatch.setenv('OMP_NUM_THREADS', '1')
    monkeypatch.setenv('MKL_NUM_THREADS', '1')
    run = make_run(tmp_path, client)
    original_put = requests.put
    transfers = []
    def put(url, **kwargs):
        assert 'Authorization' not in kwargs.get('headers', {})
        transfers.append(url)
        return original_put(url, **kwargs)
    monkeypatch.setattr(requests, 'put', put)
    try:
        run.preflight()
        assert run.train()['grt']['step'] == 2
        session = run._server_sessions['grt']
        job_id = session.job['id']
        wandb_url = 'https://wandb.ai/grt/test/runs/integration'
        (run.run_dir / 'grt/wandb_status.json').write_text(json.dumps({'mode': 'online', 'url': wandb_url}))
        session.heartbeat()
        assert client.request('GET', '/jobs/' + job_id)['summary']['wandb_url'] == wandb_url
        checkpoint_before = digest(run.run_dir / 'grt/last.pt')
        with pytest.raises(subprocess.CalledProcessError):
            run.report(plot=False, posthoc=True, posthoc_options={'window_starts': [0, 1]})
        partial = client.request('GET', '/jobs/' + job_id)
        assert partial['status'] == 'running' and partial['summary']['phase_error'] == 'posthoc_failed'
        assert any(a['name'] == 'experiment_report.json' for a in partial['artifacts'])
        options = dict(batch_size=2, segments=2, warmup_segments=1, horizon=2,
            window_starts=(0, 6, 12), group_repeats=1, lesion_fractions=(.5,))
        run.report(plot=False, posthoc=True, posthoc_options=options)
        detail = client.request('GET', '/jobs/' + job_id)
        assert detail['status'] == 'succeeded'
        assert detail['summary']['wandb_url'] == wandb_url
        assert detail['summary']['partial_result'] is False and detail['summary']['phase_error'] is None
        assert digest(run.run_dir / 'grt/last.pt') == checkpoint_before
        assert len(detail['attempts']) == 1
        artifact = next(a for a in detail['artifacts'] if a['name'] == 'experiment_report.json')
        download = client.request('GET', '/artifacts/' + artifact['id'] + '/download')
        response = requests.get(download['url'], timeout=5)
        assert response.content == run.report_path.read_bytes()
        assert hashlib.sha256(response.content).hexdigest() == artifact['sha256']
        assert {'config', 'provenance', 'checkpoint', 'report', 'result', 'log'} <= {a['kind'] for a in detail['artifacts']}
        state_path = run.run_dir / '.server/grt/state.json'
        state = state_path.read_text()
        assert client._token not in state
        assert state_path.stat().st_mode & 0o777 == 0o600
        assert 'server_url' not in json.loads(run.report_path.read_text())['experiments'][0]['config']
        before = len(detail['artifacts'])
        checkpoint_sha = digest(run.run_dir / 'grt/last.pt')
        run.train()
        run.report(plot=False, posthoc=True, posthoc_options=options)
        assert len(client.request('GET', '/jobs/' + job_id)['artifacts']) == before
        assert digest(run.run_dir / 'grt/last.pt') == checkpoint_sha
        assert transfers
    finally:
        run.close()


def test_queue_rejects_mismatched_code_before_execution(tmp_path, client):
    source = make_run(tmp_path, client, mode='off')
    client.ensure_schema()
    client.ensure_queue(source.server_queue)
    body = source.server_job('grt')
    body['spec']['code_ref'] = '0' * 40
    queued = client.request('POST', '/jobs', body)
    with pytest.raises(ValueError, match='code or world_size'):
        WikiTextRun.claim(REPO, work_dir=tmp_path / 'worker', deadline=source.server_deadline,
            runtime='cpu', device='cpu', world_size=1, server_url=client.url, queue=source.server_queue)
    until = time.monotonic() + 5
    while time.monotonic() < until:
        detail = client.request('GET', '/jobs/' + queued['id'])
        if detail['status'] != 'finalizing':
            break
        time.sleep(.1)
    assert detail['status'] == 'failed'
    assert not list((tmp_path / 'worker').rglob('last.pt'))


def test_wall_time_checkpoint_interval_saves_before_final_validation(tmp_path, client, monkeypatch):
    from grt.distributed import DistributedContext
    from grt.trainer import run as train
    from grt import stream_training
    run = make_run(tmp_path, client, model='rmt', mode='off', steps=3)
    cfg = run.config('rmt')
    cfg.evaluation.every_steps = 99
    ctx = DistributedContext('cpu')
    ctx.checkpoint_seconds = 1e-9
    saved_steps = []
    original_save = stream_training.atomic_save
    def save(value, destination):
        original_save(value, destination)
        if Path(destination).name == 'last.pt':
            saved_steps.append(torch.load(destination, weights_only=False)['step'])
    monkeypatch.setattr(stream_training, 'atomic_save', save)
    result = train(cfg, torch.device('cpu'), ctx)
    assert result['step'] == 3 and saved_steps[:2] == [1, 2]


def test_upload_response_loss_reuses_reservation_without_retraining(tmp_path, client, monkeypatch):
    run = make_run(tmp_path, client, model='rmt')
    try:
        run.preflight()
        run.train()
        session = run._server_sessions['rmt']
        job_id = session.job['id']
        checkpoint_sha = digest(run.run_dir / 'rmt/last.pt')
        original_put = requests.put
        dropped = []
        def lose_response(url, **kwargs):
            response = original_put(url, **kwargs)
            if not dropped:
                dropped.append(url)
                raise requests.ConnectionError('lost response')
            return response
        monkeypatch.setattr(requests, 'put', lose_response)
        with pytest.raises(ServerError, match='upload_unavailable'):
            run.report(plot=False)
        run.report(plot=False)
        detail = client.request('GET', '/jobs/' + job_id)
        assert detail['status'] == 'succeeded'
        reports = [a for a in detail['artifacts'] if a['name'] == 'experiment_report.json']
        assert len(reports) == 1 and reports[0]['status'] == 'verified'
        assert digest(run.run_dir / 'rmt/last.pt') == checkpoint_sha
    finally:
        run.close()


def test_queue_uses_immutable_config_and_returns_none_when_empty(tmp_path, client):
    source = make_run(tmp_path, client, mode='off')
    client.ensure_schema()
    client.ensure_queue(source.server_queue)
    definition = source.server_job('grt')
    queued = client.request('POST', '/jobs', definition)
    run = WikiTextRun.claim(REPO, work_dir=tmp_path / 'worker', deadline=source.server_deadline,
        runtime='cpu', device='cpu', world_size=1, server_url=client.url, queue=source.server_queue)
    assert run.server_mode == 'queue'
    try:
        run.prepare_data(prepared_input=source.prepared_dir)
        run.preflight()
        assert run.train()['grt']['step'] == definition['spec']['target_steps']
        run.report(plot=False)
        detail = client.request('GET', '/jobs/' + queued['id'])
        assert detail['status'] == 'succeeded' and detail['spec'] == definition['spec']
        assert run.config('grt').model.d_model == 32
        assert WikiTextRun.claim(REPO, work_dir=tmp_path / 'worker', deadline=source.server_deadline,
            runtime='cpu', device='cpu', world_size=1, server_url=client.url, queue=source.server_queue) is None
    finally:
        run.close()


def test_worker_cli_runs_one_queued_experiment_without_installing_packages(tmp_path, client):
    source = make_run(tmp_path, client, mode='off', model='rmt')
    client.ensure_schema()
    client.ensure_queue(source.server_queue)
    queued = client.request('POST', '/jobs', source.server_job('rmt'))
    completed = subprocess.run([sys.executable, 'scripts/server_worker.py',
        '--work-dir', str(tmp_path / 'worker'), '--prepared-dir', str(source.prepared_dir),
        '--deadline', source.server_deadline, '--queue', source.server_queue,
        '--runtime', 'cpu', '--device', 'cpu', '--once'], cwd=REPO,
        env={**os.environ, 'GRT_SERVER_URL': client.url, 'GRT_SERVER_TOKEN': client._token,
             'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
        capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert 'pip install' not in completed.stdout
    assert client.request('GET', '/jobs/' + queued['id'])['status'] == 'succeeded'


def test_cancel_saves_checkpoint_and_never_reports_success(tmp_path, client):
    run = make_run(tmp_path, client, steps=100000, model='rmt')
    run.preflight()
    session = run._server_sessions['rmt']
    job_id = session.job['id']
    cancellation = []
    def cancel_when_training():
        until = time.monotonic() + 40
        while time.monotonic() < until:
            detail = client.request('GET', '/jobs/' + job_id)
            attempt = detail['attempts'][0]
            if (attempt.get('progress') or {}).get('current', 0):
                client.request('POST', '/jobs/' + job_id + '/actions', {'action': 'cancel'})
                cancellation.append(True)
                return
            time.sleep(.1)
    thread = threading.Thread(target=cancel_when_training, daemon=True)
    thread.start()
    try:
        with pytest.raises(ServerError, match='cancel_requested'):
            run.train()
        thread.join(timeout=5)
        assert cancellation
        detail = client.request('GET', '/jobs/' + job_id)
        assert detail['status'] == 'cancelled'
        saved = torch.load(run.run_dir / 'rmt/last.pt', weights_only=False)
        assert 0 < saved['step'] < 100000
        assert any(a['kind'] == 'checkpoint' and a['status'] == 'verified' for a in detail['artifacts'])
    finally:
        run.close()


def test_revoked_token_stops_locally_and_new_token_recovers_same_lease(tmp_path, client, monkeypatch):
    issued = client.request('POST', '/tokens', {'name': 'revocation-check', 'kind': 'write'})
    monkeypatch.setenv('GRT_SERVER_TOKEN', issued['token'])
    run = make_run(tmp_path, client, steps=100000, model='rmt')
    run.preflight()
    original = run._server_sessions['rmt']
    job_id = original.job['id']
    lease = original.lease.copy()
    revoked = []
    def revoke_when_training():
        until = time.monotonic() + 40
        while time.monotonic() < until:
            detail = client.request('GET', '/jobs/' + job_id)
            if (detail['attempts'][0].get('progress') or {}).get('current', 0):
                client.request('DELETE', '/tokens/' + issued['id'])
                revoked.append(True)
                return
            time.sleep(.1)
    thread = threading.Thread(target=revoke_when_training, daemon=True)
    thread.start()
    try:
        with pytest.raises(ServerError, match='unauthorized'):
            run.train()
        thread.join(timeout=5)
        assert revoked and (run.run_dir / 'rmt/last.pt').exists()
        assert run.report_path.exists()
        monkeypatch.setenv('GRT_SERVER_TOKEN', client._token)
        restored = run._server_session('rmt')
        assert restored.lease == lease
        detail = client.request('GET', '/jobs/' + job_id)
        assert len(detail['attempts']) == 1 and detail['status'] == 'running'
        client.request('POST', '/jobs/' + job_id + '/actions', {'action': 'cancel'})
        restored.heartbeat()
        restored.finish()
        assert client.request('GET', '/jobs/' + job_id)['status'] == 'cancelled'
    finally:
        run.close()


def test_shared_prepared_cache_reuses_verified_bytes_and_download_is_optional(tmp_path, client, api_server, monkeypatch):
    from grt.utils.server_files import cache_prepared, fetch_prepared
    run = make_run(tmp_path, client, mode='off')
    cached = run.cache_prepared()
    assert cached['kind'] == 'dataset' and cached['status'] == 'verified'
    assert cache_prepared(client, run.prepared_dir)['id'] == cached['id']
    fields = {k: cached[k] for k in ('name', 'kind', 'size_bytes', 'sha256', 'format', 'format_version', 'extensions')}
    assert client.request('POST', '/artifacts/uploads', fields)['id'] == cached['id']
    monkeypatch.setenv('GRT_SERVER_TOKEN', api_server.tokens['read-only'])
    reader = ServerClient(api_server.url)
    sha = digest(run.prepared_dir / 'manifest.json')
    restored = tmp_path / 'restored'
    fetch_prepared(reader, sha, restored)
    for file in run.prepared_dir.rglob('*'):
        if file.is_file():
            assert (restored / file.relative_to(run.prepared_dir)).read_bytes() == file.read_bytes()
    assert fetch_prepared(reader, sha, restored) == restored
    (restored / 'train.pt').write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        fetch_prepared(reader, sha, restored)
    assert len(client.request('GET', '/datasets')['data']) >= 1


def test_prepared_cache_rejects_archive_traversal_before_publishing(tmp_path, client):
    import zipfile
    from grt.utils.server_files import fetch_prepared
    archive = tmp_path / 'unsafe.zip'
    with zipfile.ZipFile(archive, 'w') as bundle:
        bundle.writestr('../escaped.txt', 'untrusted')
    sha = '0' * 64
    item = client.request('POST', '/artifacts/uploads', dict(name='unsafe.zip', kind='dataset',
        format='zip', format_version='1', size_bytes=archive.stat().st_size, sha256=digest(archive),
        extensions={'cache_format': 'grt.prepared/1', 'manifest_sha256': sha}))
    with archive.open('rb') as stream:
        response = requests.put(item['upload']['url'], data=stream, timeout=5)
    assert response.status_code == 204
    client.request('POST', '/artifacts/uploads/' + item['id'] + '/complete', {})
    for _ in range(40):
        if client.request('GET', '/artifacts/uploads/' + item['id'])['status'] == 'verified':
            break
        time.sleep(.1)
    with pytest.raises(ValueError, match='Unsafe prepared archive'):
        fetch_prepared(client, sha, tmp_path / 'destination')
    assert not (tmp_path / 'destination').exists()
    assert not list(tmp_path.rglob('escaped.txt'))
