"""Execute queued WikiText jobs through the same notebook helper, then exit when empty."""
import argparse
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from grt.utils.kaggle import WikiTextRun
from grt.utils.server import ServerClient, failure_details, write_private


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, required=True)
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument('--prepared-dir', type=Path)
    data.add_argument('--prepared-from-server', action='store_true')
    parser.add_argument('--deadline', required=True, help='UTC ISO8601 session deadline; never extended automatically')
    parser.add_argument('--queue', default='research')
    parser.add_argument('--runtime', choices=('remote', 'colab', 'kaggle', 'cpu'), default='remote')
    parser.add_argument('--secret-source', choices=('env', 'colab', 'kaggle'), default='env')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--world-size', type=int, default=1)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    if args.world_size < 1:
        parser.error('--world-size must be positive')
    repo = Path(__file__).resolve().parents[1]
    while True:
        run = WikiTextRun.claim(repo, work_dir=args.work_dir, deadline=args.deadline,
            runtime=args.runtime, device=args.device, world_size=args.world_size,
            secret_source=args.secret_source, queue=args.queue)
        if run is None:
            print('No compatible queued job; worker stopped.')
            return 0
        phase = 'preparing'
        try:
            run.setup(install=False)  # Prepare dependencies once before processing jobs.
            if args.prepared_from_server:
                run.prepare_data(server_cache=run._queued_assignment['job']['spec']['dataset_sha256'])
            else:
                run.prepare_data(prepared_input=args.prepared_dir)
            phase = 'preflight'
            run.preflight()
            phase = 'training'
            run.train()
            options = run._queued_assignment['job']['spec']['posthoc_options']
            phase = 'posthoc' if options is not None else 'uploading'
            run.report(plot=False, posthoc=options is not None, posthoc_options=options)
        except BaseException as error:
            detail = failure_details(error, phase)
            run._worker_failure = detail
            print('Worker error:', json.dumps(detail), file=sys.stderr, flush=True)
            try:
                write_private(run.run_dir / 'worker_error.json', detail)
            except OSError:
                print('Could not save worker_error.json; the original error is printed above.', file=sys.stderr)
            connected = bool(run._server_sessions)
            synced = run._preserve_partial(phase=phase + '_failed')
            client = ServerClient(secret_source=args.secret_source)
            assignment = run._queued_assignment
            current = client.request('GET', '/jobs/' + assignment['job']['id'])
            if (not connected or not synced) and current['status'] == 'running':
                # Failed attempts may finish without uploaded evidence. Preserve the
                # actual stage/error instead of misclassifying every failure as setup.
                try:
                    progress = (current.get('summary') or {}).get('step', 0)
                    client.request('POST', '/attempts/' + assignment['attempt_id'] + '/finish', {
                        **{key: assignment[key] for key in ('worker_id', 'lease_token', 'generation')},
                        'outcome': 'failed', 'stage': phase,
                        'error': f"{phase}: {detail['message']}",
                        'artifact_ids': [], 'manifest': dict(spec_sha256=assignment['job']['spec_sha256'],
                            config_sha256='0' * 64, completed_steps=progress,
                            target_steps=assignment['job']['spec']['target_steps'], posthoc_completed=False)})
                except Exception as sync_error:
                    print('Failure reporting error:', json.dumps(failure_details(sync_error, 'synchronizing')),
                          file=sys.stderr, flush=True)
            print('Job stopped; inspect local logs and the server job detail.', file=sys.stderr)
            return 1  # Never claim another job after a failed or unsynchronized attempt.
        finally:
            run.close()
        if args.once:
            return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        # API helpers sanitize remote failures; avoid dumping credentials via exception repr/tracebacks.
        print('Worker error:', json.dumps(failure_details(error, 'claiming')), file=sys.stderr, flush=True)
        sys.exit(1)
