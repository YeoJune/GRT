"""Construct the unmodified author's RMT without adding state_dict prefixes."""

from transformers import GPTNeoXConfig
from grt.vendor.armt.modeling_gpt_neox import GPTNeoXForCausalLM
from grt.vendor.armt.language_modeling import MemoryCell, RecurrentWrapper

REFERENCE_COMMIT = "24cbb9aed62a5748c4045fd928f7f59899f03b24"


def make_rmt(cfg):
    m = cfg.model
    config = GPTNeoXConfig(
        vocab_size=m.vocab_size,
        hidden_size=m.d_model,
        num_hidden_layers=m.alu.num_layers,
        num_attention_heads=m.alu.nhead,
        intermediate_size=m.alu.d_ff,
        max_position_embeddings=2048,
        bos_token_id=101,
        eos_token_id=102,
        hidden_act="gelu",
        rotary_pct=0.25,
        rotary_emb_base=10000,
        attention_dropout=m.alu.dropout,
        hidden_dropout=m.alu.dropout,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        use_cache=True,
        tie_word_embeddings=False,
        use_parallel_residual=True,
    )
    return RecurrentWrapper(
        MemoryCell(GPTNeoXForCausalLM(config), m.num_registers, wrap_pos=False),
        segment_size=m.segment_len,
        max_n_segments=cfg.data.train_segments,
        k2=cfg.data.train_segments,
        segment_alignment="left",
    )
