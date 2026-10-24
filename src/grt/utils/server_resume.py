"""Snapshot current GRT state and restore it into a newly authorized attempt."""
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

from grt.utils.server import digest, research_config, write_private
from grt.utils.server_files import download_artifact, unpack_archive, pack_archive


def snapshot_run(session, destination):
    from grt.checkpoint import load_checkpoint
    run, model = session.run, session.model
    destination = Path(destination)
    names = [f'{model}/{name}' for name in ('last.pt', 'best.pt', 'experiment_config.json',
        'metrics.jsonl', 'wandb_run.json', 'wandb_status.json')]
    names += [f'{model}_preflight/benchmark.json', f'{model}_preflight/resolved_config.yaml']
    for name in names:
        source = run.run_dir / name
        if source.is_file():
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            # Checkpoints are replaced atomically; a hard link preserves the selected inode.
            if source.suffix == '.pt':
                os.link(source, target)
            elif source.name == 'wandb_status.json':
                try:
                    write_private(target, json.loads(source.read_text()))
                except (OSError, ValueError):
                    pass
            else:
                shutil.copyfile(source, target)
    last = load_checkpoint(destination / model / 'last.pt')
    step = last['step']
    best_path = destination / model / 'best.pt'
    if best_path.exists():
        best = load_checkpoint(best_path)
        if best['step'] > step or best['best_loss'] != last['best_loss']:
            raise ValueError('Checkpoint pair is being updated; retry the snapshot')
        del best
    elif not math.isinf(last['best_loss']):
        raise ValueError('Best checkpoint is required to preserve model selection')
    write_private(destination / 'resume.json', dict(format='grt.resume/1', step=step,
        spec_sha256=session.job['spec_sha256'], dataset_sha256=session.job['spec']['dataset_sha256'],
        world_size=run.world_size))
    del last
    archive = destination / 'resume.zip'
    pack_archive(destination, [str(p.relative_to(destination)) for p in destination.rglob('*') if p.is_file()], archive)
    return archive, step


def restore_run(run):
    from grt.checkpoint import load_checkpoint
    assignment = run._queued_assignment
    if assignment.get('resume_mode') != 'checkpoint':
        return
    spec, model = assignment['job']['spec'], run.models[0]
    root = run.run_dir / model
    marker = run.run_dir / '.server' / model / 'restored.json'
    source_id = assignment['resume_artifact_ids'][0]
    if marker.exists():
        if json.loads(marker.read_text())['artifact_id'] != source_id:
            raise ValueError('Local run belongs to a different resume snapshot')
        return
    if root.exists():
        raise ValueError('Resume requires an empty model directory or its own restore journal')
    client = run._connect_server()
    detail = client.request('GET', '/jobs/' + assignment['job']['id'])
    artifact = next(a for a in detail['artifacts'] if a['id'] == source_id and a['status'] == 'verified')
    with tempfile.TemporaryDirectory(dir=run.run_dir, prefix='.resume-') as temporary:
        temporary = Path(temporary)
        download_artifact(client, artifact, temporary / 'resume.zip')
        unpacked = temporary / 'files'
        unpacked.mkdir()
        names = unpack_archive(temporary / 'resume.zip', unpacked, artifact['size_bytes'])
        allowed = {'resume.json', *(f'{model}/{name}' for name in ('last.pt', 'best.pt',
            'experiment_config.json', 'metrics.jsonl', 'wandb_run.json', 'wandb_status.json')),
            f'{model}_preflight/benchmark.json', f'{model}_preflight/resolved_config.yaml'}
        if names - allowed:
            raise ValueError('Unexpected files in resume snapshot')
        info = json.loads((unpacked / 'resume.json').read_text())
        if (info['format'] != 'grt.resume/1' or info['spec_sha256'] != assignment['job']['spec_sha256'] or
            info['world_size'] != run.world_size or info['dataset_sha256'] != digest(run.prepared_dir / 'manifest.json')):
            raise ValueError('Resume spec, data or world_size differs')
        saved = load_checkpoint(unpacked / model / 'last.pt')
        manifest = json.loads((run.prepared_dir / 'manifest.json').read_text())
        config = json.loads((unpacked / model / 'experiment_config.json').read_text())
        if (saved['step'] != info['step'] or not 0 <= saved['step'] <= spec['target_steps'] or
            saved['world_size'] != run.world_size or saved['dataset_manifest'] != manifest or
            research_config(config, run.world_size) != research_config(spec['config'], run.world_size) or
            research_config(saved['config'], run.world_size) != research_config(config, run.world_size)):
            raise ValueError('Checkpoint does not preserve the experiment conditions')
        best_path = unpacked / model / 'best.pt'
        if best_path.exists():
            best = load_checkpoint(best_path)
            if (best['step'] > saved['step'] or best['best_loss'] != saved['best_loss'] or
                best['world_size'] != run.world_size or best['dataset_manifest'] != manifest or
                research_config(best['config'], run.world_size) != research_config(config, run.world_size)):
                raise ValueError('Best checkpoint differs from resume history')
            del best
        elif not math.isinf(saved['best_loss']):
            raise ValueError('Resume snapshot has lost its best checkpoint')
        del saved
        config['run']['output_dir'] = str(root)
        config['data']['options']['path'] = str(run.prepared_dir)
        config['wandb']['run_name'] = run.run_dir.name + '-' + model
        write_private(unpacked / model / 'experiment_config.json', config)
        metrics = unpacked / model / 'metrics.jsonl'
        if metrics.exists():
            rows = []
            for line in metrics.read_text().splitlines():
                try:
                    row = json.loads(line)
                    if row['step'] <= info['step']:
                        rows.append(row)
                except (ValueError, KeyError):
                    continue  # The snapshot may end in a partially written log line.
            metrics.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        status = unpacked / model / 'wandb_status.json'
        if status.exists():
            previous = json.loads(status.read_text())
            previous['last_logged_step'] = min(previous.get('last_logged_step', 0), info['step'])
            write_private(status, previous)
        for name in (model, model + '_preflight'):
            if (run.run_dir / name).exists() or not (unpacked / name).is_dir():
                raise ValueError('Resume requires model and preflight state in a clean directory')
        for name in (model, model + '_preflight'):
            (unpacked / name).rename(run.run_dir / name)
        write_private(marker, dict(artifact_id=source_id, step=info['step']))
    run._server_session(model).heartbeat()
