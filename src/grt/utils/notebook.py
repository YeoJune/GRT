"""Small notebook entry point for local runs and one durable server job per session.

Training, auto-batching and snapshot restore remain in WikiTextRun/server_worker.
This module only selects the workflow, pins code and starts that existing worker.
"""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import quote

from grt.utils.kaggle import WikiTextRun, run_logged


def notebook_run(repo_dir, *, runtime, execution_mode='local', resume_job_id='',
                 server_url='https://grt.grtserver.cloud', session_hours=11,
                 session_started=None, campaign='wikitext', work_dir=None,
                 segment_len=75, num_registers=20, router_hidden=512,
                 write_source='candidate', write_detach_source=False,
                 write_dropout=0., write_logit_threshold=10., write_logit_coef=1e-4,
                 warmup_ratio=.05, memory_fraction=.95, rtla_enabled=True, **options):
    """Map notebook choices to the existing catalog; resume uses only the server spec."""
    if runtime not in ('kaggle', 'colab'):
        raise ValueError('Notebook runtime must be kaggle or colab')
    if execution_mode not in ('local', 'new', 'resume'):
        raise ValueError('execution_mode must be local, new or resume')
    if execution_mode == 'resume' and not re.fullmatch(r'job_[A-Za-z0-9_-]+', resume_job_id):
        raise ValueError('Set resume_job_id to the server job_... ID')
    if execution_mode != 'resume' and resume_job_id:
        raise ValueError('A resume_job_id requires execution_mode=resume')
    work_dir = Path(work_dir or ('/kaggle/working' if runtime == 'kaggle' else '/content'))
    world_size = 2 if runtime == 'kaggle' else 1
    submission = None
    if execution_mode != 'resume':
        submission = WikiTextRun(repo_dir=repo_dir, runtime=runtime, world_size=world_size,
            work_dir=work_dir, size='small', tokenization='gpt2_bpe',
            num_registers=num_registers, router_hidden=router_hidden,
            server_url=server_url, secret_source=runtime,
            grt_overrides={'model': {
                'router': {'write_source': write_source, 'write_detach_source': write_detach_source},
                'register': {'conditional_read': False, 'write_gate_bias_init': 0.,
                    'dropout_prob': write_dropout, 'write_logit_threshold': write_logit_threshold,
                    'write_logit_coef': write_logit_coef}}},
            overrides={'model': {'segment_len': segment_len},
                'training': {'auto_batch': True, 'memory_fraction': memory_fraction,
                             'warmup_ratio': warmup_ratio},
                'rtla': {'enabled': rtla_enabled}}, **options)
    if execution_mode == 'local':
        return submission
    if submission is not None and len(submission.models) != 1:
        raise ValueError('Server mode runs one model/job per session')
    return ServerNotebookRun(Path(repo_dir), work_dir, runtime, world_size, submission,
        resume_job_id, server_url, session_hours, session_started, campaign)


def session_deadline(directory, *, hours, started=None):
    """Persist the first notebook start; rerunning cells cannot extend this session."""
    from grt.utils.server import write_private
    if isinstance(hours, bool) or not isinstance(hours, (int, float)) or not math.isfinite(hours) or hours <= 0:
        raise ValueError('session_hours must be a positive finite number')
    path = Path(directory) / 'session_deadline.json'
    if not path.exists():
        start = datetime.now(timezone.utc) if started is None else datetime.fromtimestamp(started, timezone.utc)
        write_private(path, {'deadline': (start + timedelta(hours=hours)).isoformat()})
    deadline = json.loads(path.read_text())['deadline']
    if datetime.fromisoformat(deadline) <= datetime.now(timezone.utc):
        raise RuntimeError('Session deadline reached; resume in a new Save/runtime')
    return deadline


def read_secret(name, runtime):
    try:
        if runtime == 'kaggle':
            from kaggle_secrets import UserSecretsClient
            value = UserSecretsClient().get_secret(name)
        else:
            from google.colab import userdata
            value = userdata.get(name)
    except Exception:
        raise RuntimeError(f'Enable the {name} notebook Secret') from None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'Set the {name} notebook Secret')
    return value


class ServerNotebookRun:
    def __init__(self, repo_dir, work_dir, runtime, world_size, submission,
                 resume_job_id, server_url, session_hours, session_started, campaign):
        self.repo_dir, self.work_dir = repo_dir, work_dir / 'grt_server'
        self.runtime, self.world_size = runtime, world_size
        self.submission, self.resume_job_id = submission, resume_job_id
        self.server_url, self.campaign = server_url, campaign
        self.deadline = session_deadline(self.work_dir, hours=session_hours, started=session_started)
        self.client = self.selected_job = self.worker_repo = None

    def _queued(self, job):
        if job['status'] != 'queued' or (self.resume_job_id and not job.get('resume_artifact_id')):
            raise RuntimeError('이전 실행 종료 후 서버에서 이어서 학습을 선택해 checkpoint 재개 대기 상태로 만드세요.')
        return job

    def _checkout(self, commit, source_hash):
        """Keep the notebook controller intact; train in an isolated original checkout."""
        from grt.utils.server import code_identity
        if not re.fullmatch('[a-f0-9]{40}', commit):
            raise ValueError('Server job requires a full commit SHA')
        destination = self.work_dir / 'code' / commit
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            subprocess.run(['git', 'clone', '--shared', '--no-checkout', str(self.repo_dir), str(destination)], check=True)
            origin = subprocess.check_output(['git', '-C', str(self.repo_dir), 'remote', 'get-url', 'origin'], text=True).strip()
            subprocess.run(['git', '-C', str(destination), 'remote', 'set-url', 'origin', origin], check=True)
            subprocess.run(['git', '-C', str(destination), 'fetch', 'origin', commit], check=True)
            subprocess.run(['git', '-C', str(destination), 'checkout', '--detach', commit], check=True)
        if code_identity(destination) != (commit, source_hash):
            raise RuntimeError('Original commit/source hash differs; worker will not start')
        return destination

    def setup(self, *, install=True):
        from grt.utils.server import ServerClient, code_identity
        self.client = ServerClient(self.server_url, secret_source=self.runtime)
        self.client.check()
        if self.resume_job_id:
            self.selected_job = self._queued(self.client.request('GET', '/jobs/' + self.resume_job_id))
            spec = self.selected_job['spec']
            if spec['world_size'] != self.world_size:
                raise ValueError('Resume requires the original world_size/runtime')
            identity = spec['code_ref'], spec['code_sha256']
        else:
            if subprocess.check_output(['git', '-C', str(self.repo_dir), 'status', '--porcelain'], text=True).strip():
                raise RuntimeError('Server submission requires a clean committed checkout')
            identity = code_identity(self.repo_dir)
        self.worker_repo = self._checkout(*identity)
        if self.submission is not None:
            self.submission._server_client = self.client
            self.submission.setup(install=install)
        elif install:
            wandb = self.selected_job['spec']['config']['wandb']['enabled']
            extras = 'wandb,wikitext' if wandb else 'wikitext'
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-e',
                            str(self.worker_repo) + f'[{extras}]'], check=True)
        print('Server:', self.client.url, '\nWorker UTC deadline:', self.deadline,
              '\nPinned commit:', identity[0])

    def prepare_data(self, **options):
        if self.client is None:
            raise RuntimeError('Run setup first')
        if self.submission is None:
            print('Resume: original server dataset and verified snapshot will be restored by the worker.')
            return
        manifest = self.submission.prepare_data(**options)
        cached = self.submission.cache_prepared()
        print('Prepared manifest SHA256:', cached['extensions']['manifest_sha256'])
        self._prepared = True
        return manifest

    def _submit(self, posthoc_options):
        if not getattr(self, '_prepared', False):
            raise RuntimeError('Run prepare_data first; server cache is required')
        body = self.submission.server_job(self.submission.models[0], posthoc_options=posthoc_options)
        # Queue isolation prevents this Save from taking another condition's job.
        signature = hashlib.sha256(json.dumps([self.campaign, body['spec']], sort_keys=True).encode()).hexdigest()
        body['queue'] = 'notebook-' + signature[:24]
        body['labels']['campaign'] = self.campaign
        self.client.ensure_schema()
        self.client.ensure_queue(body['queue'])
        request_id = 'notebook-' + hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        self.client.request('POST', '/jobs/validate', body)
        submitted = self.client.request('POST', '/jobs', body, key=request_id)
        return self.client.request('GET', '/jobs/' + submitted['id'])

    def execute(self, *, posthoc_options=None, download=True):
        from grt.utils.server import code_identity, write_private
        if self.client is None:
            raise RuntimeError('Run setup first')
        if datetime.fromisoformat(self.deadline) <= datetime.now(timezone.utc):
            raise RuntimeError('Session deadline reached; resume in a new Save/runtime')
        if self.selected_job is None:
            self.selected_job = self._submit(posthoc_options or {})
        job = self.client.request('GET', '/jobs/' + self.selected_job['id'])
        self.selected_job = job
        spec = job['spec']
        if (*code_identity(self.worker_repo), self.world_size) != (spec['code_ref'], spec['code_sha256'], spec['world_size']):
            raise RuntimeError('Original code/source hash/world_size differs; worker will not start')
        write_private(self.work_dir / 'selected_job.json', {
            'job_id': job['id'], 'code_ref': spec['code_ref'],
            'dataset_sha256': spec['dataset_sha256'], 'queue': job['queue']})
        print('Job ID:', job['id'], '\nTarget steps:', spec['target_steps'])
        if job['status'] == 'succeeded':
            return self.status(download=download)
        self._queued(job)
        queued = self.client.request('GET', '/jobs?queue=' + quote(job['queue'], safe='') + '&status=queued&limit=2')
        if [item['id'] for item in queued['data']] != [job['id']] or queued.get('next_cursor'):
            raise RuntimeError('Dedicated queue contains other pending jobs; separate the queue before running')
        if datetime.fromisoformat(self.deadline) <= datetime.now(timezone.utc):
            raise RuntimeError('Session deadline reached; resume in a new Save/runtime')
        # Resume deliberately ignores current notebook choices, including posthoc/W&B.
        env = dict(os.environ, GRT_SERVER_URL=self.client.url)
        try:
            env['GRT_SERVER_TOKEN'] = read_secret('GRT_SERVER_TOKEN', self.runtime)
            if spec['config']['wandb']['enabled'] and spec['config']['wandb']['mode'] == 'online':
                env['WANDB_API_KEY'] = read_secret('WANDB_API_KEY', self.runtime)
            run_logged([sys.executable, str(self.worker_repo / 'scripts/server_worker.py'),
                '--work-dir', str(self.work_dir / 'worker'), '--prepared-from-server',
                '--deadline', self.deadline, '--runtime', self.runtime, '--secret-source', 'env',
                '--queue', job['queue'], '--device', 'cuda', '--world-size', str(self.world_size), '--once'],
                cwd=self.worker_repo, log_path=self.work_dir / 'worker_console.txt', env=env)
        finally:
            env.clear()
        return self.status(download=download)

    def status(self, *, download=False):
        detail = self.client.request('GET', '/jobs/' + self.selected_job['id'])
        snapshots = [a for a in detail['artifacts'] if a['status'] == 'verified'
                     and a.get('extensions', {}).get('resume_format') == 'grt.resume/1']
        print('Job:', detail['id'], 'status:', detail['status'], 'attempts:', len(detail['attempts']))
        print('Verified resume steps:', sorted(a['extensions']['step'] for a in snapshots))
        if detail['status'] != 'succeeded' and not snapshots:
            print('No verified resume snapshot yet; check the server storage/logs.')
        if download:
            self._download_results(detail['id'])
        return detail

    def _download_results(self, job_id):
        directories = [p for p in (self.work_dir / 'worker').glob('server-' + job_id + '-g*')
                       if re.fullmatch(r'server-' + re.escape(job_id) + r'-g\d+', p.name)]
        if not directories:
            print('Results and snapshots:', self.client.url)
            return
        latest = max(directories, key=lambda p: int(p.name.rsplit('-g', 1)[1]))
        path = latest / 'experiment_results.zip'
        if not path.exists():
            path = latest / 'experiment_report.json'
        if path.exists():
            if self.runtime == 'colab':
                from google.colab import files
                files.download(str(path))
            else:
                from IPython.display import FileLink, display
                display(FileLink(str(path)))
