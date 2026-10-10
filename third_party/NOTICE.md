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
GPT-NeoX is provided by Hugging Face Transformers 4.45.2 (Apache-2.0).
Its tokenizers 0.20.x dependency provides Python 3.13 wheels used by Colab.
The reference repository's backbone was copied from Transformers 4.31.0.
The golden test fixture is generated from that pinned author backbone and wrapper;
its regeneration script removes only an unused optional adapter import.
The committed fixture was generated using Transformers 4.44.2; its logits,
memory states, gradients and generated tokens are also checked against 4.45.2.

The D128/L4/H4/FF128/V128 backbone configuration follows the repository's later
`base_models/gptconfigs/create_config.py` at commit `5297db7`; the original 2024
generated JSON is absent. Appendix E specifies approximately 500k parameters,
four layers and hidden size 128, but does not specify every backbone option.
This is a short Remember validation using the paper task and RMT recurrence,
not a claim to reproduce all Figure 2 training conditions or results.

## Native author reference path

`src/grt/reference/` vendors `base_models/modeling_gpt_neox.py`,
`modeling_rmt/language_modeling.py`, and `adapters.py` from author commit
`24cbb9aed62a5748c4045fd928f7f59899f03b24`. The only edit to these three
files is making the adapter import package-relative. `SOURCE.json` records
the original and vendored SHA256 hashes; the license is included in the package.
`data.py` extracts the original `generate_pairs`, `ARDataset`, and `collate_fn`
bodies, with the collator's globals captured by a factory. This path supports
Remember, not Rewrite.

`runner.py` is local orchestration: it calls the native wrapper with raw labels
and the author's labels_mask, and backpropagates the wrapper's internal loss.
It retains original AdamW/linear scheduling, random-length collator and
best-exact-match stage transfer. Checkpoint/resume, CPU/one-device execution,
deterministic split generation, and additional fixed-length evaluation are local.
It does not depend on the common logits-only model adapter or shifted labels.
The historical generated backbone JSON is still unavailable; the small-model
configuration follows the later public generator noted above.
