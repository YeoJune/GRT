# WikiText-103 승인된 구현 설정

2026-10-11. 사용자가Remember호환항목외앞선제안을승인했다.
문서최신화후구현한다. 기존author baseline/checkpoint는보존한다.

| 항목 | base | small |
|---|---:|---:|
| layers |16|6|
| hidden |410|512|
| heads |10|8|
| head dim |41|64|
| FF |2100|2048|

공통: GPT-NeoX,parallel residual,RoPE pct.25/base10000,tied full-softmax head.
base는2022RMT WikiText 규모만맞춘NeoX변형이며원문모델의정확한재현이아니다.
small은흔한FF=4D구성이며유일한표준을뜻하지않는다. 30M예산/router총규모제한없음.
WikiText word tokenizer vocabulary267735,원문데이터처리유지.
segment150,memory10,이전3segment BPTT,global batch60,
Adam lr2.5e-4,cosine,warmup0,normclip.25,dropout.1,max200000updates.
attention dropout0은원문기준이며backbone세부설정에기록한다.
GPU/시간은미제공이며microbatch/accumulation은실측으로조정한다.
기본FP32경로와작은CPU검증을준비하고장시간GPU는사용자가실행한다.
인코더/디코더구현모두지원. read=false,intermediate 기본/fixed_state 옵션,
prefix재계산제거,전체stack종료후write-back1회.
intermediate decoder NLL은엄격한AR PPL과구분한다.
WikiText인코더MLM/분류목표는후속지정. Remember는새인코더지원가능성을별도검토했다.
