"""Download Hugging Face WikiText-103 v1 and export its text rows."""
import argparse
import hashlib
import json
from pathlib import Path


def download(destination, revision='main'):
    from datasets import load_dataset

    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f'{destination} is not empty; reuse it or choose a new directory')
    dataset = load_dataset('Salesforce/wikitext', 'wikitext-103-v1', revision=revision)
    destination.mkdir(parents=True, exist_ok=True)
    splits = {}
    for split, name in [('train', 'train'), ('validation', 'valid'), ('test', 'test')]:
        path = destination / f'wiki.{name}.tokens'
        digest = hashlib.sha256()
        with path.open('wb') as output:
            for row in dataset[split]:
                text = row['text']
                # An empty HF row is a blank source line. Preserve existing newlines.
                payload = (text if text.endswith('\n') else text + '\n').encode('utf-8')
                output.write(payload)
                digest.update(payload)
        splits[split] = dict(rows=len(dataset[split]), fingerprint=dataset[split]._fingerprint,
                             file=path.name, sha256=digest.hexdigest())
    manifest = dict(dataset='Salesforce/wikitext', configuration='wikitext-103-v1',
        revision=revision, export='UTF-8 text rows; append LF only when absent; empty rows become blank lines',
        splits=splits)
    (destination / 'download_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--revision', default='main')
    args = parser.parse_args()
    print(json.dumps(download(args.output_dir, args.revision), indent=2))
