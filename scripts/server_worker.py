"""Execute queued WikiText jobs through the same notebook helper, then exit when empty."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from grt.utils.kaggle import WikiTextRun


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--prepared-dir', type=Path, required=True)
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
    failed = False
    while True:
        run = WikiTextRun.claim(repo, work_dir=args.work_dir, deadline=args.deadline,
            runtime=args.runtime, device=args.device, world_size=args.world_size,
            secret_source=args.secret_source, queue=args.queue)
        if run is None:
            print('No compatible queued job; worker stopped.')
            return int(failed)
        try:
            run.setup(install=False)  # Prepare dependencies once before processing jobs.
            run.prepare_data(prepared_input=args.prepared_dir)
            run.preflight()
            run.train()
            options = run._queued_assignment['job']['spec']['posthoc_options']
            run.report(plot=False, posthoc=options is not None, posthoc_options=options)
        except Exception:
            failed = True
            run._preserve_partial()
            from grt.utils.server import ServerClient
            client = ServerClient(secret_source=args.secret_source)
            detail = client.request('GET', '/jobs/' + run._queued_assignment['job']['id'])
            print('Job stopped; inspect local logs and the server job detail.', file=sys.stderr)
            if detail['status'] in ('running', 'finalizing'):
                return 1  # Resolve synchronization/lease ownership before any further claim.
        finally:
            run.close()
        if args.once:
            return int(failed)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        # API helpers sanitize remote failures; avoid dumping credentials via exception repr/tracebacks.
        print(type(error).__name__ + ': worker stopped before a new job could start', file=sys.stderr)
        sys.exit(1)
