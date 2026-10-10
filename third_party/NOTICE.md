# ARMT RMT attribution

`src/grt/vendor/armt/` contains code from
[RodkinIvan/associative-recurrent-memory-transformer](https://github.com/RodkinIvan/associative-recurrent-memory-transformer),
commit `24cbb9aed62a5748c4045fd928f7f59899f03b24`:

- `base_models/modeling_gpt_neox.py`
- `modeling_rmt/language_modeling.py`
- `adapters.py`
- Remember dataset and collator functions from `run_ar.py`

The backbone's adapter import is package-relative. The wrapper and adapters are
unchanged; the data functions are extracted into a collator factory. `SOURCE.json`
records hashes and extraction details. Apache-2.0 is retained in the vendor package
and `ARMT-LICENSE`. The author's backbone derives from Transformers 4.31.0; the
installed compatibility dependency is Transformers 4.45.2.

The historical generated backbone JSON is absent from the pinned 2024 checkout.
D128/L4/H4/FF128/V128 options follow the later public generator
`base_models/gptconfigs/create_config.py` at commit `5297db7`. Appendix E specifies
four layers, hidden size128 and approximately500k parameters, without every option.
This project does not claim exact recovery of the missing configuration or the
paper's complete training schedule.

The common trainer, deterministic split generation, resume, fixed-length validation,
T4 batch override, GRT and W&B integration are local code. The RMT factory returns
the native MemoryCell/RecurrentWrapper and uses its internal loss. GRT implements
that same external contract with its own router, ALU and gated registers.

`tests/fixtures/neox_rmt_reference.pt` is generated from the pinned author backbone
and wrapper by the included fixture script. Its generation used Transformers4.44.2;
logits, memory states, loss, gradients and generated tokens are checked with4.45.2.
The fixture and short CPU checks establish module consistency, not GPU convergence
or reproduction of all paper results.
