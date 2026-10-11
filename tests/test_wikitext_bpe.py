"""Offline byte-BPE preparation and prepared data identity."""
import hashlib

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import GPT2TokenizerFast
from grt.experiments import load_experiment
from grt.wikitext import WikiTextAdapter
from scripts.prepare_wikitext_bpe import prepare

@pytest.fixture
def prepared(tmp_path):
    # Offline fixture has all bytes and GPT-2's ID range, not the real GPT-2 merges.
    vocab = {token: index for index, token in enumerate(pre_tokenizers.ByteLevel.alphabet())}
    vocab.update({f'fixture_{i}': i for i in range(256, 50256)})
    vocab['<|endoftext|>'] = 50256
    backend = Tokenizer(models.BPE(vocab, []))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = GPT2TokenizerFast(tokenizer_object=backend, add_prefix_space=False)
    tokenizer_dir = tmp_path / 'fixture_tokenizer'
    tokenizer.save_pretrained(tokenizer_dir)
    source = tmp_path / 'source'
    source.mkdir()
    texts = {'train': ' Alpha beta!\n' * 100,
             'validation': ' Alpha beta!\n\nUnicode 한글!  \n' + 'long ' * 300 + '\n',
             'test': ' Alpha beta!\n' * 4}
    for split, name in [('train', 'train'), ('validation', 'valid'), ('test', 'test')]:
        (source / f'wiki.{name}.tokens').write_text(texts[split], encoding='utf-8')
    destination = tmp_path / 'prepared'
    manifest = prepare(source, destination, str(tokenizer_dir), local_files_only=True)
    return source, destination, manifest, tokenizer, texts


def test_byte_bpe_preserves_text_eos_long_lines_and_hashes(prepared):
    source, destination, manifest, tokenizer, texts = prepared
    assert manifest['vocab_size'] == 50257
    assert manifest['tokenization'] == 'gpt2_bpe'
    for split, text in texts.items():
        ids = torch.load(destination / f'{split}.pt', weights_only=True)
        # A round trip checks whitespace, Unicode, blank lines and no truncation.
        expected = ''.join(line + '<|endoftext|>' for line in text.splitlines())
        assert tokenizer.decode(ids.tolist(), clean_up_tokenization_spaces=False) == expected
        assert (ids == 50256).sum().item() == manifest['line_counts'][split]
        assert ids.numel() == manifest['token_counts'][split]
        assert hashlib.sha256((destination / f'{split}.pt').read_bytes()).hexdigest() == manifest['sha256'][split]
    with pytest.raises(FileExistsError):
        prepare(source, destination, 'not-loaded', local_files_only=True)


def test_adapter_checks_bpe_identity_and_tokenizer_artifacts(prepared):
    _, destination, manifest, _, _ = prepared
    cfg = load_experiment(task='wikitext103', model='rmt', size='small', tokenization='gpt2_bpe')
    cfg.data.options['path'] = str(destination)
    adapter = WikiTextAdapter()
    assert adapter.dataset(cfg, 'test').numel() == manifest['token_counts']['test']
    cfg.data.options['tokenization'] = 'word'
    with pytest.raises(ValueError, match='tokenization'):
        adapter.dataset(cfg, 'test')
    cfg.data.options['tokenization'] = 'gpt2_bpe'
    (destination / 'tokenizer/merges.txt').write_text('changed')
    with pytest.raises(ValueError, match='artifact hash'):
        adapter.dataset(cfg, 'test')
