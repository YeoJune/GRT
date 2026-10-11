# ARMT code attribution

`src/grt/vendor/armt/` contains code from
[RodkinIvan/associative-recurrent-memory-transformer](https://github.com/RodkinIvan/associative-recurrent-memory-transformer),
commit `24cbb9aed62a5748c4045fd928f7f59899f03b24`:

- `base_models/modeling_gpt_neox.py`: current shared NeoX block/attention base.
- `adapters.py`: dependency of the vendored block source.
- Remember dataset and collator functions from `run_finetuning_associative_retrieval.py`.

The backbone's adapter import is package-relative. Data functions are extracted
into a collator factory. `SOURCE.json` records provenance, hashes and extraction
details. Apache-2.0 is retained in the vendor package and `ARMT-LICENSE`.
The author's backbone derives from Transformers 4.31.0; this project uses
Transformers 4.45.2.

The current encoder/decoder, memory masks, RMT/GRT state updates, trainer,
checkpoint, tokenizer preparation and W&B integration are local code. The old
MemoryCell/RecurrentWrapper execution path and its fixture are removed.
Historical observations are documented in `docs/archive/`; they do not define
current execution or establish reproduction of all paper results.
