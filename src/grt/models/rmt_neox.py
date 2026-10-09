"""RMT memory wrapping from RodkinIvan/associative-recurrent-memory-transformer.

Reference: 24cbb9, modeling_rmt/language_modeling.py (Apache-2.0).
The GPT-NeoX backbone is supplied by Transformers; loss remains external.
No associative block, cross-segment KV cache, or detached recurrent state.
"""
import torch
from torch import nn
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
from grt.config import validate_model
from .interface import ModelOutput, validate_inputs


class NeoXRMTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        validate_model(cfg)
        self.cfg = cfg
        self.internal_segment_length = cfg.segment_len + 2 * cfg.num_registers
        backbone_cfg = GPTNeoXConfig(
            vocab_size=cfg.vocab_size, hidden_size=cfg.d_model,
            num_hidden_layers=cfg.alu.num_layers, num_attention_heads=cfg.alu.nhead,
            intermediate_size=cfg.alu.d_ff, hidden_act="gelu",
            max_position_embeddings=2048, rotary_pct=0.25, rotary_emb_base=10000,
            attention_dropout=cfg.alu.dropout, hidden_dropout=cfg.alu.dropout,
            initializer_range=0.02, layer_norm_eps=1e-5,
            use_parallel_residual=True, tie_word_embeddings=False,
            bos_token_id=101, eos_token_id=102, use_cache=False,
        )
        # Eager attention preserves the reference arithmetic for CPU parity checks.
        backbone_cfg._attn_implementation = "eager"
        self.backbone = GPTNeoXForCausalLM(backbone_cfg)
        self.mem0 = nn.Parameter(torch.randn(cfg.num_registers, cfg.d_model)
                                 * self.backbone.get_input_embeddings().weight.detach().std())

    def initial_memory(self, batch_size):
        return self.mem0.repeat(batch_size, 1, 1)

    def forward_segment(self, ids, memory):
        m = self.cfg.num_registers
        embeddings = self.backbone.get_input_embeddings()(ids)
        sequence = torch.cat([memory, embeddings, memory], dim=1)
        output = self.backbone(inputs_embeds=sequence, use_cache=False,
                               output_hidden_states=True, return_dict=True)
        return output.logits[:, m:-m], output.hidden_states[-1][:, -m:]

    def forward(self, input_ids, attention_mask=None):
        validate_inputs(input_ids, attention_mask, self.cfg.segment_len, self.cfg.vocab_size)
        memory = self.initial_memory(input_ids.shape[0])
        logits = []
        for ids in input_ids.split(self.cfg.segment_len, dim=1):
            output, memory = self.forward_segment(ids, memory)
            logits.append(output)
        return ModelOutput(torch.cat(logits, dim=1))

    def generate_answer(self, prompt, max_new_tokens):
        """Process complete fact segments, then generate from the query prefix."""
        n = self.cfg.segment_len
        boundary = prompt.shape[1] // n * n
        if boundary == prompt.shape[1]:
            raise ValueError("Generation requires an incomplete final query segment")
        memory = self.initial_memory(prompt.shape[0])
        for ids in prompt[:, :boundary].split(n, dim=1):
            _, memory = self.forward_segment(ids, memory)
        prefix = prompt[:, boundary:]
        answer = []
        for _ in range(max_new_tokens):
            logits, _ = self.forward_segment(prefix, memory)
            token = logits[:, -1].argmax(dim=-1)
            answer.append(token)
            prefix = torch.cat([prefix, token[:, None]], dim=1)
        return torch.stack(answer, dim=1)
