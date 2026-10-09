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
