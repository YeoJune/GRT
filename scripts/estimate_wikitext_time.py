"""Convert a WikiText preflight and prepared manifest into a compute-only ETA."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from grt.utils.kaggle import estimate_time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=200000)
    parser.add_argument('--eval-every', type=int, default=4000)
    parser.add_argument('--eval-batch', type=int, default=10)
    parser.add_argument('--segment-len', type=int, default=150)
    args = parser.parse_args()
    try:
        report = estimate_time(json.loads(args.benchmark.read_text()),
            json.loads(args.manifest.read_text()), steps=args.steps,
            eval_every=args.eval_every, eval_batch=args.eval_batch, segment_len=args.segment_len)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
