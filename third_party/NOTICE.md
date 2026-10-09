# LM-RMT attribution

`src/grt/models/rmt_relative.py` adapts the relative-position attention, Post-LN
and recurrent memory computations from `booydar/LM-RMT`,
commit `e2895080868afc390386220f2b68a56c4a218126`,
`pytorch/mem_transformer.py`. LM-RMT is distributed under Apache License 2.0;
the upstream license is retained in `LM-RMT-LICENSE`.

The adapted module removes Transformer-XL caching, adaptive softmax and internal
loss computation; uses this project's configuration and output interface; and
adds segment and greedy Copy generation methods. It is not an unmodified
vendoring of the upstream repository.

# ARMT repository RMT attribution

`src/grt/models/rmt_neox.py` adapts the RMT memory wrapping from
`RodkinIvan/associative-recurrent-memory-transformer`, commit
`24cbb9aed62a5748c4045fd928f7f59899f03b24`,
`modeling_rmt/language_modeling.py` (Apache-2.0; `ARMT-LICENSE`).
GPT-NeoX is provided by Hugging Face Transformers 4.44.2 (Apache-2.0).
The reference repository's backbone was copied from Transformers 4.31.0.
The golden test fixture is generated from that pinned author backbone and wrapper;
its regeneration script removes only an unused optional adapter import.

The D128/L4/H4/FF128/V128 backbone configuration follows the repository's later
`base_models/gptconfigs/create_config.py` at commit `5297db7`; the original 2024
generated JSON is absent. Appendix E specifies approximately 500k parameters,
four layers and hidden size 128, but does not specify every backbone option.
This is a short Remember validation using the paper task and RMT recurrence,
not a claim to reproduce all Figure 2 training conditions or results.
