"""Convert a WikiText preflight and prepared manifest into a compute-only ETA."""
import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=200000)
    parser.add_argument('--eval-every', type=int, default=4000)
    parser.add_argument('--eval-batch', type=int, default=10)
    parser.add_argument('--segment-len', type=int, default=150)
    args = parser.parse_args()
    if min(args.steps, args.eval_every, args.eval_batch, args.segment_len) <= 0:
        parser.error('Step, batch and segment settings must be positive')
    benchmark = json.loads(args.benchmark.read_text())
    counts = json.loads(args.manifest.read_text())['token_counts']
    if benchmark.get('objective') != 'next_token':
        parser.error('Expected a WikiText next_token benchmark')
    if benchmark['target_tokens_per_update'] != benchmark['global_batch_size'] * args.segment_len:
        parser.error('Segment length differs from the benchmark')
    def segments(split):
        length = counts[split] // args.eval_batch - 1
        if length <= 0:
            parser.error(f'{split} is too short for the evaluation batch')
        return math.ceil(length / args.segment_len)
    valid_segments, test_segments = segments('validation'), segments('test')
    training = args.steps * benchmark['update_seconds']
    validation = math.ceil(args.steps / args.eval_every) * valid_segments * benchmark['evaluation_segment_seconds']
    test = test_segments * benchmark['evaluation_segment_seconds']
    print(json.dumps(dict(
        model=benchmark['model'], score_kind=benchmark['score_kind'],
        steps=args.steps, target_tokens=args.steps * benchmark['target_tokens_per_update'],
        training_hours=training / 3600, validation_hours=validation / 3600,
        final_test_hours=test / 3600, total_compute_hours=(training + validation + test) / 3600,
        peak_gib_per_gpu=benchmark['peak_bytes_per_gpu'] / 2**30,
        reserved_gib_per_gpu=benchmark['reserved_bytes_per_gpu'] / 2**30,
        note='Excludes preparation, checkpoint I/O and startup. Evaluation repeats on each rank; no world-size division. Verify eval settings against resolved_config.yaml. Tail segments are approximated as full segments.'
    ), indent=2))


if __name__ == '__main__':
    main()
