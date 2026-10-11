"""Prepared word/BPE WikiText files and ordered token streams."""
from pathlib import Path
import hashlib
import json
import math
import torch


class WikiTextAdapter:
    def validate(self, cfg):
        if cfg.model.attention_mode != "decoder":
            raise ValueError("WikiText next-token training requires the new decoder; encoder objective is unspecified")
        if cfg.data.options.get('objective', 'next_token') != 'next_token':
            raise ValueError('WikiText supports only the explicit next-token objective')
        if cfg.training.curriculum:
            raise ValueError("WikiText uses max_steps, not the Remember pair curriculum")
        if not cfg.data.options.get("path"):
            raise ValueError("WikiText requires data.options.path to prepared token streams")
        if cfg.rtla.enabled:
            raise ValueError("Automatic RTLA dataset sampling supports Remember; WikiText segment traces need an explicit batch")
        if cfg.training.global_batch_size is not None and cfg.training.global_batch_size <= 0:
            raise ValueError("Invalid global batch")

    def dataset(self, cfg, split, pairs=None):
        path = Path(cfg.data.options["path"])
        manifest = json.loads((path / 'manifest.json').read_text())
        if manifest['vocab_size'] != cfg.model.vocab_size:
            raise ValueError("Prepared vocabulary does not match model.vocab_size")
        expected = cfg.data.options.get('tokenization')
        if expected and manifest.get('tokenization', 'word') != expected:
            raise ValueError('Prepared tokenization does not match data.options.tokenization')
        for relative, expected_hash in manifest.get('prepared_artifacts_sha256', {}).items():
            if hashlib.sha256((path / relative).read_bytes()).hexdigest() != expected_hash:
                raise ValueError('Prepared tokenizer artifact hash mismatch')
        filename = path / f"{split}.pt"
        digest = hashlib.sha256()
        with filename.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != manifest['sha256'][split]:
            raise ValueError("Token stream hash mismatch")
        tokens = torch.load(filename, weights_only=True, mmap=True)
        if tokens.ndim != 1 or tokens.dtype != torch.long or tokens.numel() < 2 or tokens.min() < 0 or tokens.max() >= cfg.model.vocab_size:
            raise ValueError("Invalid prepared token stream")
        return tokens

    def train_stream(self, cfg, device, monitor, ctx):
        from grt.stream_training import train_stream
        return train_stream(self, cfg, device, monitor, ctx)

    def benchmark(self, cfg, ctx, steps):
        from grt.stream_training import benchmark_stream
        return benchmark_stream(self, cfg, ctx, steps)

    @staticmethod
    def streams(tokens, batch_size):
        length = tokens.numel() // batch_size
        if length < 2:
            raise ValueError("At least two tokens per stream are required")
        return tokens[:batch_size * length].view(batch_size, length)

    @staticmethod
    def score_kind(cfg):
        if cfg.model.name == "rmt" or (cfg.model.register_read_source == "fixed_state"
                and not cfg.model.register.conditional_read):
            return "autoregressive"
        return "teacher_forced_noncausal"

    def evaluate(self, model, cfg, tokens, device):
        from grt.checkpoint import evaluation_context
        stream = self.streams(tokens, cfg.evaluation.batch_size)
        state = None
        nll = count = 0
        with evaluation_context(model):
            for offset in range(0, stream.shape[1]-1, cfg.model.segment_len):
                n = min(cfg.model.segment_len, stream.shape[1]-1-offset)
                ids = stream[:, offset:offset+n].to(device)
                targets = stream[:, offset+1:offset+n+1].to(device)
                out = model(ids, target_ids=targets, memory_state=state)
                state = out.memory_state
                nll += out.loss.item() * targets.numel()
                count += targets.numel()
        mean = nll / count
        return dict(loss=mean, perplexity=math.exp(mean), nll_sum=nll,
            target_tokens=count, discarded_tail_tokens=tokens.numel()-stream.numel(),
            tokenization=cfg.data.options.get('tokenization', 'word'),
            perplexity_unit='bpe_token' if cfg.data.options.get('tokenization') == 'gpt2_bpe' else 'word_token',
            score_kind=self.score_kind(cfg))
