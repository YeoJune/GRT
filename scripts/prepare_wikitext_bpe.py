"""Prepare tokenized WikiText-103 with the existing GPT-2 byte-level BPE."""
import argparse
from array import array
from itertools import islice
import hashlib
import json
from pathlib import Path
import torch
from transformers import GPT2TokenizerFast


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare(source, destination, tokenizer_name='openai-community/gpt2',
            local_files_only=False, revision='main'):
    source, destination = Path(source), Path(destination)
    paths = {split: source / f'wiki.{name}.tokens' for split, name in
             [('train', 'train'), ('validation', 'valid'), ('test', 'test')]}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f'{destination} is not empty; reuse the existing manifest or choose a new directory')
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name, revision=revision,
        local_files_only=local_files_only, add_prefix_space=False)
    if len(tokenizer) != 50257 or tokenizer.vocab_size != 50257 or tokenizer.eos_token_id != 50256:
        raise ValueError('Expected the unmodified GPT-2 tokenizer: vocabulary 50257, EOS 50256')
    tokenizer.backend_tokenizer.no_truncation()
    tokenizer.backend_tokenizer.no_padding()
    destination.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = destination / 'tokenizer'
    tokenizer.save_pretrained(tokenizer_dir)
    artifacts = {str(path.relative_to(destination)): sha256(path)
                 for path in sorted(tokenizer_dir.iterdir()) if path.is_file()}
    hashes, source_hashes, counts, lines = {}, {}, {}, {}
    for split, path in paths.items():
        ids = array('q')
        line_count = 0
        with path.open(encoding='utf-8', newline='') as stream:
            while batch := list(islice(stream, 256)):
                texts = [line.rstrip('\r\n') for line in batch]
                # No model max-length truncation, BOS, padding or newline token.
                encoded = tokenizer.backend_tokenizer.encode_batch(texts, add_special_tokens=False)
                for item in encoded:
                    ids.extend(item.ids)
                    ids.append(tokenizer.eos_token_id)
                line_count += len(batch)
        if len(ids) < 2:
            raise ValueError(f'{split} requires at least two tokens')
        target = destination / f'{split}.pt'
        torch.save(torch.frombuffer(ids, dtype=torch.long), target)
        hashes[split], source_hashes[split] = sha256(target), sha256(path)
        counts[split], lines[split] = len(ids), line_count
    manifest = dict(dataset='wikitext-103-v1 tokenized, GPT-2 BPE variant',
        tokenization='gpt2_bpe', tokenizer='GPT-2 byte-level BPE; CR/LF removed; line EOS; no prefix space, BOS or truncation',
        tokenizer_source=str(tokenizer_name), tokenizer_revision=revision,
        tokenizer_commit=tokenizer.init_kwargs.get('_commit_hash'),
        tokenizer_sha256=artifacts['tokenizer/tokenizer.json'], prepared_artifacts_sha256=artifacts,
        vocab_size=len(tokenizer), eos_token_id=tokenizer.eos_token_id,
        sha256=hashes, source_sha256=source_hashes, token_counts=counts, line_counts=lines)
    if (source / 'download_manifest.json').exists():
        manifest['source_dataset'] = json.loads((source / 'download_manifest.json').read_text())
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='Directory containing wiki.train/valid/test.tokens')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--tokenizer', default='openai-community/gpt2', help='Hugging Face ID or saved GPT-2 tokenizer directory')
    parser.add_argument('--revision', default='main', help='Remote tokenizer revision; saved files are hashed')
    parser.add_argument('--local-files-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output_dir, args.tokenizer,
        args.local_files_only, args.revision), indent=2))
