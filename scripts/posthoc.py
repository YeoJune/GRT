"""Run the notebook's full posthoc protocol in a clean Python process."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from grt.utils.posthoc import WikiTextPosthoc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--prepared-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--selected-step', type=int, required=True)
    parser.add_argument('--options-json', default='{}')
    parser.add_argument('--report-index', type=Path, required=True)
    args = parser.parse_args()
    job = WikiTextPosthoc(checkpoint=args.checkpoint, prepared_dir=args.prepared_dir,
        output_dir=args.output_dir, device=args.device, **json.loads(args.options_json))
    info = job.inspect()
    if info['checkpoint_step'] != args.selected_step:
        raise ValueError('best.pt step differs from selected training result')
    job.run()
    path = job.report().resolve()
    temporary = args.report_index.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(report_path=str(path)))+'\n')
    temporary.replace(args.report_index)


if __name__ == '__main__':
    main()
