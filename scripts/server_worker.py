"""Execute queued WikiText jobs through the same notebook helper, then exit when empty."""
import argparse
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from grt.utils.kaggle import WikiTextRun
from grt.utils.server import ServerClient, failure_details, write_private, finish_failure


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
            print('Worker error:', json.dumps(detail), file=sys.stderr, flush=True)
            try:
                write_private(run.run_dir / 'worker_error.json', detail)
            except OSError:
                print('Could not save worker_error.json; the original error is printed above.', file=sys.stderr)
            run.close()  # A dead training process must not keep its lease alive during cleanup.
            try:
                finish_failure(ServerClient(secret_source=args.secret_source), run._queued_assignment, detail)
            except Exception as sync_error:
                print('Failure reporting error:', json.dumps(failure_details(sync_error, 'synchronizing')),
                      file=sys.stderr, flush=True)
            try:
                run.export_report()
            except Exception as local_error:
                print('Local report error:', json.dumps(failure_details(local_error, 'reporting')), file=sys.stderr)
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
