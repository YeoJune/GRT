"""Optional GRT Server connection; no model imports or notebook-specific fallback."""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import tempfile
import time
from urllib.parse import urlsplit
from uuid import uuid4


SCHEMA_ID = 'grt_wikitext_v1'
SCHEMA = dict(
    id=SCHEMA_ID, name='GRT WikiText notebook v1', runner='generic/1',
    json_schema=dict(type='object', additionalProperties=False,
        required=['target_steps', 'code_ref', 'code_sha256', 'config', 'world_size', 'dataset_sha256', 'posthoc_options'],
        properties=dict(target_steps=dict(type='integer', minimum=1),
            code_ref=dict(type='string', pattern='^[a-f0-9]{40}$'),
            code_sha256=dict(type='string', pattern='^[a-f0-9]{64}$'),
            config=dict(type='object'), world_size=dict(type='integer', minimum=1),
            dataset_sha256=dict(type='string', pattern='^[a-f0-9]{64}$'),
            posthoc_options=dict(type=['object', 'null']))),
    report_schema=dict(type='object', required=['spec_sha256', 'config_sha256', 'completed_steps', 'posthoc_completed'],
        properties=dict(spec_sha256=dict(type='string'), config_sha256=dict(type='string'),
            completed_steps=dict(type='integer', minimum=0), posthoc_completed=dict(type='boolean'))))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_private(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
    try:
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def portable_config(cfg):
    value = asdict(cfg)
    value['run']['output_dir'] = None
    value['data']['options'].pop('path', None)
    value['wandb']['run_name'] = None
    return value


def research_config(value, world_size):
    value = deepcopy(value)
    value.pop('run', None)
    value.pop('wandb', None)
    value['data']['options'].pop('path', None)
    training = value['training']
    value['effective_batch'] = training.pop('batch_size') * training.pop('grad_accum_steps') * world_size
    training.pop('auto_batch')
    training.pop('memory_fraction')
    return value


def code_identity(repo):
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    files = subprocess.check_output(['git', 'ls-files', '-co', '--exclude-standard', '-z', '--',
        'src', 'scripts', 'configs', 'pyproject.toml'], cwd=repo).decode().split('\0')
    value = hashlib.sha256()
    for name in sorted(set(filter(None, files))):
        path = Path(repo) / name
        value.update(name.encode() + b'\0')
        value.update(digest(path).encode() if path.exists() else b'deleted')
    return commit, value.hexdigest()


def job_definition(run, model, *, posthoc_options=None):
    cfg = run.config(model, calibrated=False)
    manifest = run.prepared_dir / 'manifest.json'
    if not manifest.exists():
        raise ValueError('Prepare and verify the dataset before creating a server job')
    commit, source_sha256 = code_identity(run.repo_dir)
    spec = dict(target_steps=cfg.training.max_steps, code_ref=commit, code_sha256=source_sha256,
        config=portable_config(cfg), world_size=run.world_size,
        dataset_sha256=digest(manifest), posthoc_options=posthoc_options)
    return dict(name=run.run_dir.name + '-' + model, queue=run.server_queue,
        runner='generic/1', schema_id=SCHEMA_ID, spec=spec,
        requirements=dict(gpu_count=run.world_size if run.device == 'cuda' else 0,
                          labels=['world_size:' + str(run.world_size)]),
        policy=dict(checkpointable=True, require_posthoc=posthoc_options is not None),
        labels=dict(campaign=run.run_dir.name))


class ServerError(RuntimeError):
    def __init__(self, status, code):
        code = code if isinstance(code, str) and re.fullmatch('[a-z0-9_]{1,80}', code) else 'invalid_response'
        self.status, self.code = status, code
        # Remote messages and HTTP exceptions may contain credentials or signed URLs.
        super().__init__(f'GRT Server: {status} {code}')


def failure_details(error, phase):
    """Record useful locations and API codes without credentials or signed URLs."""
    import traceback
    known_messages = {
        'Code changed after server registration',
        'Dataset changed after server registration',
        'Actual research configuration differs from the server spec',
        'Server request changed; use a new run',
        'Server spec changed; use a new run',
        'Final DDP peak exceeds the memory limit; reduce microbatch and remeasure',
        'Insecure upload destination',
    }
    message = type(error).__name__
    if isinstance(error, ServerError) or str(error) in known_messages:
        message = str(error)
    elif isinstance(error, subprocess.CalledProcessError):
        message = f'Child process exited with status {error.returncode}'
    detail = dict(phase=phase, exception=type(error).__name__, message=message,
        locations=[dict(file=Path(frame.filename).name, line=frame.lineno, function=frame.name)
                   for frame in traceback.extract_tb(error.__traceback__)])
    if isinstance(error, ServerError):
        detail.update(status=error.status, code=error.code)
    return detail


def validate_url(url):
    parsed = urlsplit(url)
    if (parsed.username or parsed.password or parsed.query or parsed.fragment
        or not parsed.hostname or parsed.path not in ('', '/')):
        raise ValueError('Use a server origin without credentials, path or query')
    if parsed.scheme != 'https' and not (parsed.scheme == 'http' and parsed.hostname in ('localhost', '127.0.0.1', '::1')):
        raise ValueError('GRT Server requires HTTPS (HTTP is allowed only for local tests)')
    return url.rstrip('/')


class ServerClient:
    def __init__(self, url=None, *, secret_source='env'):
        self.url = validate_url(url or os.environ.get('GRT_SERVER_URL', ''))
        if secret_source not in ('env', 'colab', 'kaggle'):
            raise ValueError('secret_source must be env, colab or kaggle')
        try:
            if secret_source == 'env':
                token = os.environ.get('GRT_SERVER_TOKEN')
            elif secret_source == 'colab':
                from google.colab import userdata
                token = userdata.get('GRT_SERVER_TOKEN')
            else:
                from kaggle_secrets import UserSecretsClient
                token = UserSecretsClient().get_secret('GRT_SERVER_TOKEN')
        except Exception:
            raise RuntimeError('Cannot read GRT_SERVER_TOKEN from the selected secret source') from None
        if not isinstance(token, str) or not token.strip():
            raise ValueError('Set GRT_SERVER_TOKEN in the selected secret source')
        self._token = token.strip()

    def request(self, method, route, body=None, *, key=None, stop_event=None):
        import requests
        key = key or uuid4().hex
        for attempt in range(3):
            if stop_event is not None and stop_event.is_set():
                raise ServerError(0, 'connection_closed')
            try:
                response = requests.request(method, self.url + '/v1' + route,
                    headers={'Authorization': 'Bearer ' + self._token, 'Idempotency-Key': key},
                    json=body, timeout=(5, 20), allow_redirects=False)
                try:
                    data = None if response.status_code == 204 else response.json()
                except ValueError:
                    raise ServerError(response.status_code, 'invalid_response') from None
                if not response.ok or response.is_redirect:
                    raise ServerError(response.status_code, data.get('code', 'http_error') if isinstance(data, dict) else 'http_error')
                return data
            except requests.RequestException:
                error = ServerError(0, 'network_unavailable')
            except ServerError as error_response:
                error = error_response
            if attempt == 2 or (error.status and error.status < 500 and error.status != 429):
                raise error from None
            if stop_event is None:
                time.sleep(2 ** attempt)
            elif stop_event.wait(2 ** attempt):
                raise ServerError(0, 'connection_closed')

    def check(self):
        me = self.request('GET', '/me')
        if me.get('api_version') != 'v1' or me.get('kind') != 'write':
            raise ServerError(403, 'write_token_required')
        caps = self.request('GET', '/capabilities')
        if 'generic/1' not in caps.get('runners', []):
            raise ServerError(422, 'generic_runner_required')
        return me

    def ensure_schema(self):
        try:
            existing = self.request('GET', '/schemas/' + SCHEMA_ID)
        except ServerError as error:
            if error.status != 404:
                raise
            try:
                return self.request('POST', '/schemas', SCHEMA)
            except ServerError as conflict:
                if conflict.status != 409:
                    raise
            existing = self.request('GET', '/schemas/' + SCHEMA_ID)
        if any(existing.get(key) != value for key, value in SCHEMA.items()):
            raise ServerError(409, 'schema_content_conflict')
        return existing

    def ensure_queue(self, name):
        try:
            self.request('POST', '/queues', {'name': name})
        except ServerError as error:
            if error.code != 'queue_exists':
                raise


class ServerRun:
    """One local lease holder, durable request IDs, and independent heartbeat."""
    def __init__(self, run, model, client, *, assignment=None):
        self.run, self.model, self.client = run, model, client
        self.directory = run.run_dir / '.server' / model
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lock_file = (self.directory / 'lock').open('a')
        try:
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._lock_file.close()
            raise RuntimeError('This run already has an active server connection') from None
        self.path = self.directory / 'state.json'
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {'requests': {}, 'uploads': {}}
        self._mutex = threading.RLock()
        self._done = threading.Event()
        self._thread = None
        self.stage = 'preparing'
        self.error = None
        self.stop_file = self.directory / 'stop'
        self._last_online = time.monotonic()
        self._checkpoint_stamp = None
        self._checkpoint_id = None
        try:
            if self.state.get('server_url', client.url) != client.url:
                raise ValueError('Existing run belongs to a different server')
            self.state['server_url'] = client.url
            if assignment is not None:
                self.state['assignment'] = assignment
            self._save()
        except BaseException:
            self.close()
            raise

    def _save(self):
        write_private(self.path, self.state)

    def mutation(self, name, route, body):
        signature = hashlib.sha256(json.dumps(body, sort_keys=True, allow_nan=False).encode()).hexdigest()
        with self._mutex:
            record = self.state['requests'].get(name)
            if record and record['sha256'] != signature:
                raise ValueError('Server request changed; use a new run')
            if not record:
                record = self.state['requests'][name] = dict(key=uuid4().hex, sha256=signature)
                self._save()
        return self.client.request('POST', route, body, key=record['key'])

    def connect(self):
        self.client.check()
        self.client.ensure_schema()
        self.client.ensure_queue(self.run.server_queue)
        definition = job_definition(self.run, self.model)
        if 'assignment' not in self.state:
            deadline = self.run.server_deadline
            parsed_deadline = datetime.fromisoformat(deadline.replace('Z', '+00:00')) if deadline else None
            if not parsed_deadline or parsed_deadline.tzinfo is None or parsed_deadline <= datetime.now(timezone.utc):
                raise ValueError('Set server_deadline to the session deadline (UTC ISO8601)')
            session_key = self.state.setdefault('session_key', uuid4().hex)
            self._save()
            worker = self.mutation('worker', '/workers', dict(
                name=definition['name'], session_key=session_key, environment=self.run.runtime if self.run.runtime in ('colab', 'kaggle') else 'remote',
                runners=['generic/1'], schema_ids=[SCHEMA_ID],
                gpu_count=self.run.world_size if self.run.device == 'cuda' else 0,
                labels=['world_size:' + str(self.run.world_size)],
                world_size=self.run.world_size, deadline_at=deadline))
            attached = self.mutation('attached', '/jobs', {
                **definition, 'dispatch': 'attached', 'worker_id': worker['id']})
            assignment = attached.get('assignment')
            if not assignment or not assignment.get('lease_token'):
                raise ServerError(409, 'lease_recovery_required')
            self.state['assignment'] = {**assignment, 'worker_id': worker['id']}
            self._save()
        assignment = self.state['assignment']
        self.job = self.client.request('GET', '/jobs/' + assignment['job']['id'])
        if self.job['spec'] != definition['spec']:
            # Queued jobs can additionally require posthoc; their spec is authoritative.
            expected = {**definition['spec'], 'posthoc_options': self.job['spec']['posthoc_options']}
            if self.job['spec'] != expected:
                raise ValueError('Server spec changed; use a new run')
        if self.job['status'] in ('succeeded', 'finalizing'):
            self._done.set()
            return
        if self.job['active_attempt_id'] != assignment['attempt_id'] or self.job['status'] != 'running':
            raise ServerError(409, 'attempt_closed')
        self.stop_file.unlink(missing_ok=True)
        self.heartbeat()
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()

    @property
    def lease(self):
        assignment = self.state['assignment']
        return {key: assignment[key] for key in ('worker_id', 'lease_token', 'generation')}

    def wandb_summary(self):
        status_path = self.run.run_dir / self.model / 'wandb_status.json'
        try:
            status = json.loads(status_path.read_text())
            url = status.get('url') if status.get('mode') == 'online' else None
        except (OSError, ValueError):
            url = None
        return {'wandb_url': url} if url else {}

    def heartbeat(self):
        progress = None
        metrics = self.run.run_dir / self.model / 'metrics.jsonl'
        if metrics.exists():
            try:
                with metrics.open('rb') as stream:
                    stream.seek(max(0, metrics.stat().st_size - 65536))
                    row = json.loads(stream.read().splitlines()[-1])
                progress = dict(current=row['step'], total=self.job['spec']['target_steps'], unit='step')
            except (ValueError, KeyError, IndexError):
                pass
        reply = self.client.request('POST', '/attempts/' + self.state['assignment']['attempt_id'] + '/heartbeat',
            {**self.lease, 'stage': self.stage, **({'progress': progress} if progress else {}),
             'summary': self.wandb_summary(),
             **({'last_verified_checkpoint_id': self._checkpoint_id} if self._checkpoint_id else {})},
            stop_event=self._done)
        self._last_online = time.monotonic()
        if reply['command'] == 'cancel':
            self.stop(ServerError(409, 'cancel_requested'))

    def _heartbeat_loop(self):
        interval = self.state['assignment'].get('heartbeat_seconds', 30)
        while not self._done.wait(interval):
            if datetime.fromisoformat(self.state['assignment']['deadline_at'].replace('Z', '+00:00')) <= datetime.now(timezone.utc):
                self.stop(ServerError(409, 'deadline_reached'))
                return
            try:
                self.heartbeat()
            except ServerError as error:
                if error.status in (401, 403, 409) or time.monotonic() - self._last_online >= 600:
                    self.stop(error)
                    return

    def stop(self, error):
        if not self._done.is_set():
            self.error = error
            self.stop_file.touch(mode=0o600)

    def check(self):
        if self.error:
            raise self.error

    def upload(self, path, kind, *, extensions=None):
        import requests
        path = Path(path)
        sha = digest(path)
        identity = f'{kind}:{path.name}:{sha}'
        with self._mutex:
            previous = self.state['uploads'].get(identity)
        if previous:
            return previous
        body = dict(name=path.name, kind=kind, size_bytes=path.stat().st_size, sha256=sha,
            format=path.suffix.lstrip('.') or 'binary', format_version='1', extensions=extensions or {},
            job_id=self.job['id'], attempt_id=self.state['assignment']['attempt_id'])
        reserved = self.mutation('upload:' + identity, '/artifacts/uploads', body)
        current = self.client.request('GET', '/artifacts/uploads/' + reserved['id'])
        if current['status'] == 'reserved':
            link = self.client.request('POST', '/artifacts/uploads', {'upload_id': reserved['id']})['upload']
            destination = urlsplit(link['url'])
            if destination.scheme != 'https' and not (destination.scheme == 'http' and destination.hostname in ('localhost', '127.0.0.1', '::1')):
                raise ValueError('Insecure upload destination')
            try:
                with path.open('rb') as stream:
                    response = requests.put(link['url'], data=stream, headers=link['headers'], timeout=(5, 120), allow_redirects=False)
            except requests.RequestException:
                raise ServerError(0, 'upload_unavailable') from None
            # A lost PUT response may have stored immutable bytes already. Let the server verify them.
            if response.status_code not in (200, 201, 204, 409):
                raise ServerError(response.status_code, 'upload_failed')
        self.mutation('complete:' + identity, '/artifacts/uploads/' + reserved['id'] + '/complete', {})
        until = time.monotonic() + 120
        while True:
            current = self.client.request('GET', '/artifacts/uploads/' + reserved['id'])
            if current['status'] == 'verified':
                break
            if current['status'] == 'failed' or time.monotonic() >= until:
                raise ServerError(422, 'artifact_not_verified')
            time.sleep(.25)
        with self._mutex:
            self.state['uploads'][identity] = current
            self._save()
        return current

    def checkpoint(self, *, force=False):
        if not force and (self.stage != 'training' or self.error or getattr(self, '_checkpoint_retry_at', 0) > time.monotonic()):
            return
        source = self.run.run_dir / self.model / 'last.pt'
        if not source.exists():
            return
        stamp = source.stat()
        identity = (stamp.st_ino, stamp.st_mtime_ns, stamp.st_size)
        if not force and identity == self._checkpoint_stamp:
            return
        try:
            from grt.utils.server_resume import snapshot_run
            with tempfile.TemporaryDirectory(dir=self.directory, prefix='snapshot-') as temporary:
                snapshot, step = snapshot_run(self, temporary)
                item = self.upload(snapshot, 'checkpoint', extensions={'resume_format': 'grt.resume/1', 'step': step})
            self._checkpoint_id = item['id']
            self._checkpoint_stamp = identity
            self.heartbeat()
        except (ServerError, OSError, ValueError) as error:
            self._checkpoint_retry_at = time.monotonic() + 30
            if isinstance(error, ServerError) and error.status in (401, 403, 409):
                self.stop(error)
            # A later tick retries; local checkpoint always remains available.

    def evidence(self):
        cfg = self.run.config(self.model)
        if digest(self.run.prepared_dir / 'manifest.json') != self.job['spec']['dataset_sha256']:
            raise ValueError('Dataset changed after server registration')
        if code_identity(self.run.repo_dir) != (self.job['spec']['code_ref'], self.job['spec']['code_sha256']):
            raise ValueError('Code changed after server registration')
        if research_config(asdict(cfg), self.run.world_size) != research_config(self.job['spec']['config'], self.run.world_size):
            raise ValueError('Actual research configuration differs from the server spec')
        config_path = self.directory / 'config.json'
        write_private(config_path, {**self.job['spec'], 'actual_config': asdict(cfg)})
        artifact = self.upload(config_path, 'config')
        provenance = self.directory / 'provenance.json'
        write_private(provenance, dict(spec_sha256=self.job['spec_sha256'], config_sha256=artifact['sha256'],
            runner='generic/1', code_ref=self.job['spec']['code_ref'], world_size=self.run.world_size,
            code_sha256=self.job['spec']['code_sha256'],
            dataset_sha256=self.job['spec']['dataset_sha256']))
        return artifact, self.upload(provenance, 'provenance')

    def preserve_partial(self, phase):
        self.evidence()
        for path, kind in [(self.run.report_path, 'result'),
                           (self.run.run_dir / self.model / 'last.pt', 'checkpoint'),
                           (self.run.run_dir / (self.model + '_train_console.txt'), 'log')]:
            if path.exists():
                self.upload(path, kind)
        self.client.request('POST', '/attempts/' + self.state['assignment']['attempt_id'] + '/heartbeat',
            {**self.lease, 'stage': self.stage, 'summary': {'partial_result': True, 'phase_error': phase}})

    def finish(self, *, error=False, posthoc=False):
        if self.job['status'] == 'succeeded':
            return self.job
        if self.job['status'] == 'finalizing':
            return self.wait_finished(allow_failed=error)
        self.stage = 'uploading'
        root = self.run.run_dir / self.model
        summary = json.loads((root / 'summary.json').read_text()) if (root / 'summary.json').exists() else {}
        step = summary.get('step', 0)
        completed = summary.get('status') == 'completed' and step >= self.job['spec']['target_steps']
        outcome = 'failed' if error else 'completed' if completed else 'interrupted'
        if self.error and self.error.code == 'cancel_requested':
            outcome = 'cancelled'
        config, provenance = self.evidence()
        self.checkpoint(force=True)
        artifacts = [config['id'], provenance['id']]
        if self._checkpoint_id:
            artifacts.append(self._checkpoint_id)
        for path, kind in [(self.run.report_path, 'result'), (root / 'experiment_config.json', 'result'),
                           (root / 'metadata.json', 'result'), (root / 'last.pt', 'checkpoint'),
                           (root / 'best.pt', 'checkpoint'), (self.run.run_dir / (self.model + '_train_console.txt'), 'log'),
                           (self.run.run_dir / 'worker_error.json', 'log'),
                           (self.run.run_dir / 'sync_error.json', 'log')]:
            if path.is_file():
                artifacts.append(self.upload(path, kind)['id'])
        analysis = root / 'posthoc_report_path.json'
        analyzed = posthoc and analysis.exists()
        if analyzed:
            path = Path(json.loads(analysis.read_text())['report_path'])
            artifacts.append(self.upload(path, 'result')['id'])
        bundle = self.run.run_dir / 'experiment_results.zip'
        if posthoc and bundle.exists():
            artifacts.append(self.upload(bundle, 'result')['id'])
        report = self.directory / 'report.json'
        write_private(report, dict(spec_sha256=self.job['spec_sha256'], config_sha256=config['sha256'],
            completed_steps=step, posthoc_completed=bool(analyzed), summary=summary))
        artifacts.append(self.upload(report, 'report')['id'])
        manifest = dict(spec_sha256=self.job['spec_sha256'], config_sha256=config['sha256'],
            completed_steps=step, target_steps=self.job['spec']['target_steps'], posthoc_completed=bool(analyzed))
        self.client.request('POST', '/attempts/' + self.state['assignment']['attempt_id'] + '/heartbeat',
            {**self.lease, 'stage': 'uploading', 'summary': {**summary, **self.wandb_summary(), 'partial_result': outcome != 'completed', 'phase_error': None}})
        failure = getattr(self.run, '_worker_failure', None) if error else None
        self.mutation('finish', '/attempts/' + self.state['assignment']['attempt_id'] + '/finish',
            {**self.lease, 'outcome': outcome, 'stage': failure['phase'] if failure else 'uploading',
             'error': f"{failure['phase']}: {failure['message']}" if failure else
                 'Local stage failed; inspect preserved results and logs' if error else None,
             'artifact_ids': artifacts, 'manifest': manifest})
        self._done.set()
        return self.wait_finished(allow_failed=error)

    def wait_finished(self, *, allow_failed=False):
        until = time.monotonic() + 120
        while True:
            self.job = self.client.request('GET', '/jobs/' + self.job['id'])
            if self.job['status'] != 'finalizing':
                if self.job['status'] == 'failed' and not allow_failed:
                    raise ServerError(422, (self.job.get('validation_error') or {}).get('code', 'job_failed'))
                return self.job
            if time.monotonic() >= until:
                raise ServerError(0, 'finalization_pending')
            time.sleep(.25)

    def close(self):
        self._done.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=30)
        if not self._lock_file.closed:
            self._lock_file.close()


def claim_run(repo_dir, *, work_dir, deadline, runtime='remote', device='cuda', world_size=1,
              server_url=None, secret_source='env', queue='research'):
    """Claim one job using a durable worker identity; never fetch or execute remote code."""
    from grt.config import Config, build_config, validate_config
    from grt.utils.kaggle import WikiTextRun
    client = ServerClient(server_url, secret_source=secret_source)
    client.check()
    client.ensure_schema()
    state_dir = Path(work_dir) / '.grt-server' / hashlib.sha256((client.url + ':' + queue).encode()).hexdigest()[:16]
    state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (state_dir / 'lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError('Another worker is already claiming from this local directory') from None
        state_path = state_dir / 'state.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else dict(session_key=uuid4().hex)
        assignment = state.get('assignment')
        if assignment:
            try:
                previous = client.request('GET', '/jobs/' + assignment['job']['id'])
            except ServerError as error:
                if error.status != 404:
                    raise
                previous = {}
            if (previous.get('status') not in ('running', 'finalizing') or
                previous.get('active_attempt_id') != assignment['attempt_id'] or
                previous.get('generation') != assignment['generation']):
                state.pop('assignment')
                state.pop('claim_key', None)
                assignment = None
        worker_body = dict(name='GRT ' + runtime, session_key=state['session_key'],
            environment=runtime if runtime in ('colab', 'kaggle') else 'remote',
            runners=['generic/1'], schema_ids=[SCHEMA_ID], gpu_count=world_size if device == 'cuda' else 0,
            world_size=world_size, deadline_at=deadline, labels=['world_size:' + str(world_size)])
        state.setdefault('claim_key', uuid4().hex)
        write_private(state_path, state)
        worker_key = 'worker-' + hashlib.sha256(json.dumps(worker_body, sort_keys=True).encode()).hexdigest()
        worker = client.request('POST', '/workers', worker_body, key=worker_key)
        if not assignment:
            if worker.get('active_attempt_id'):
                raise ServerError(409, 'lease_recovery_required')
            assignment = client.request('POST', '/workers/' + worker['id'] + '/claim', {'queue': queue}, key=state['claim_key'])
            if assignment is None:
                state.pop('claim_key')
                write_private(state_path, state)
                return None
            if not assignment.get('lease_token'):
                raise ServerError(409, 'lease_recovery_required')
            assignment['worker_id'] = worker['id']
            state['assignment'] = assignment
            write_private(state_path, state)
    spec = assignment['job']['spec']
    try:
        commit, source_sha256 = code_identity(repo_dir)
        if (spec['code_ref'], spec['code_sha256'], spec['world_size']) != (commit, source_sha256, world_size):
            raise ValueError('Queued code or world_size differs from this worker')
        run_id = 'server-' + assignment['job']['id'] + '-g' + str(assignment['generation'])
        raw = deepcopy(spec['config'])
        raw['run']['output_dir'] = str(Path(work_dir) / run_id)
        raw['data']['options']['path'] = str(Path(work_dir) / ('wikitext103_' + raw['data']['options']['tokenization']))
        cfg = validate_config(build_config(Config, raw))
        if cfg.data.name != 'wikitext103' or cfg.training.max_steps != spec['target_steps']:
            raise ValueError('Unsupported or inconsistent queued Config')
        run = WikiTextRun(repo_dir, work_dir=Path(work_dir), runtime=runtime if runtime in ('colab', 'kaggle') else 'cpu',
            models=(cfg.model.name,), tokenization=cfg.data.options['tokenization'],
            run_id=run_id, world_size=world_size, device=device,
            wandb=cfg.wandb.enabled, server_mode='upload', server_url=client.url,
            secret_source=secret_source, server_queue=queue, server_deadline=deadline)
        run.runtime = runtime
        run.size = 'server-defined'
        run._queued_config = spec['config']
        run.prepared_dir = Path(raw['data']['options']['path'])
        run._server_client = client
        run._queued_assignment = assignment
        run.server_mode = 'queue'
        return run
    except Exception:
        client.request('POST', '/attempts/' + assignment['attempt_id'] + '/finish', {
            **{key: assignment[key] for key in ('worker_id', 'lease_token', 'generation')},
            'outcome': 'failed', 'stage': 'preparing',
            'error': 'Worker rejected queued code, world_size or Config before execution',
            'artifact_ids': [], 'manifest': dict(spec_sha256=assignment['job']['spec_sha256'],
                config_sha256='0' * 64, completed_steps=0, target_steps=spec['target_steps'], posthoc_completed=False)},
            key='reject-' + state['claim_key'])
        raise
