"""Prepare original tokenized WikiText-103 files (not wikitext-103-raw)."""
import argparse
from array import array
from collections import Counter
import hashlib
import json
from pathlib import Path
import torch


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare(source, destination, expected_vocab=267735):
    source, destination = Path(source), Path(destination)
    paths = {split: source / f"wiki.{name}.tokens" for split, name in
        [('train', 'train'), ('validation', 'valid'), ('test', 'test')]}
    counter = Counter()
    with paths['train'].open(encoding='utf-8') as stream:
        for line in stream:
            counter.update(line.split() + ['<eos>'])
    # Original Transformer-XL vocabulary: EOS first, frequency-ordered training words.
    words = ['<eos>'] + [w for w, _ in counter.most_common() if w != '<eos>']
    if '<unk>' not in words:
        words.append('<unk>')
    if expected_vocab and len(words) != expected_vocab:
        raise ValueError(f"Expected {expected_vocab} vocabulary entries, got {len(words)}; use tokenized v1 files")
    vocab = {w: i for i, w in enumerate(words)}
    destination.mkdir(parents=True, exist_ok=True)
    hashes, raw_hashes, counts = {}, {}, {}
    for split, path in paths.items():
        ids = array('q')
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                ids.extend(vocab.get(w, vocab['<unk>']) for w in line.split() + ['<eos>'])
        target = destination / f'{split}.pt'
        torch.save(torch.frombuffer(ids, dtype=torch.long), target)
        hashes[split] = sha256(target)
        raw_hashes[split] = sha256(path)
        counts[split] = len(ids)
    vocab_path = destination / 'vocab.json'
    vocab_path.write_text(json.dumps(words, ensure_ascii=False), encoding='utf-8')
    manifest = dict(tokenizer='word, case-sensitive, line EOS, train-only vocabulary',
        dataset='wikitext-103-v1 tokenized', vocab_size=len(words), sha256=hashes,
        source_sha256=raw_hashes, token_counts=counts, vocab_sha256=sha256(vocab_path))
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, help='Directory containing wiki.train/valid/test.tokens')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--expected-vocab', type=int, default=267735)
    args = p.parse_args()
    print(prepare(args.source, args.output_dir, args.expected_vocab))
