# GRT/RMT/NeoX · 현재 구현 명세

갱신: 2026-10-20 · 실행 계약만 유지한다. 이전 설계·검증 이력은 [아카이브](archive/README.md), 측정 결과는 [실험 보고서](README.md)에 둔다.

## 1. 범위와 소스

공통 GPT-NeoX backbone으로 GRT·RMT·메모리 없는 NeoX를 비교한다. Remember 인코더/디코더와 WikiText-103 디코더를 지원한다. 현재 수치 형식은 FP32, checkpoint는 `memory_experiment/4`다. 모델 경로는 `src/grt/models/`, 학습 경로는 `trainer.py`와 `stream_training.py`, 설정은 `config.py`·`configs/experiments.yaml`이다.

| 크기 | D / layers / heads / FF |
|---|---|
| base | 410 / 16 / 10 / 2100 |
| small | 256 / 6 / 8 / 1024 |
| tiny | 128 / 3 / 4 / 512 |

Token embedding과 LM head를 공유한다. GELU·LayerNorm eps1e-5·선형/embedding 초기화 std.02를 사용한다. GPT-2 BPE vocabulary는50,257, word vocabulary는267,735다. Backbone은 공통이고 router·memory 파라미터는 별도다. Vendor 출처·라이선스·hash는 [SOURCE.json](../src/grt/vendor/armt/SOURCE.json)에 있다.

## 2. Segment 시작 state와 write-back

GRT는 시작 state S를 모든 layer의 token 읽기 source로 사용한다. 각 layer의 LayerNorm/KV projection은 다르지만 입력 state는 segment 안에서 고정된다. 고정은 detach를 뜻하지 않는다. Candidate 계산용 register hidden은 token/state를 읽되, 그 중간 표현을 token의 K/V에 노출하지 않는다.

```text
X = embedding(segment)
u = AttentionPool(q_pool, X)
v = RegisterAttention(u, S, S)
w = sigmoid(MLP_write(concat(u, v)))
S_read = read_gate(S) * S if conditional_read else S
[Y; C] = NeoXStack([X; S_read], token_read_source=S_read)
logits = tied_lm_head(Y)
S_next = (1-w) * S + w * C
```

Gate는 슬롯별 scalar[B,M,1], state/candidate는[B,M,D]다. Router는 원래 token embedding을 요약하며 contextual backbone hidden을 입력하지 않는다. Read gate는 시작 S만 참조하고 기본 off다. 전체 stack 뒤 write-back을 한 번 수행하며 새 state는 다음 segment에서 사용한다. GRT의 S0는 학습 가능한0 초기화다.

위 식은 기본 `router.write_source=independent`다. 후속 [정보 경로 실험](experiments/20261019_l75_gate_paths.md)의2번 `separate_head`는 기존 최종 출력 정규화 직전 register hidden에서 D→D gate linear로 분기하고,3번 `candidate`는 기존 최종 norm 뒤 C를 그대로 사용한다. Candidate 출력 경로·명칭은 유지하고2번에만 head를 추가한다. 세 경로 모두 `V=RegisterAttention(F,S,S)`를 유지하고 concat(F,V)의2D를 write MLP에 넣는다. State on/off 옵션은 없다. 공유 경로의 write는 stack 뒤 계산하지만 token은 시작 S만 읽는다. `write_detach_source`는 공유 backbone 입력 연결만 차단하며2번에서는 head 앞에 적용해 head·state attention·MLP는 계속 학습한다. 기본 false이며 독립 경로에서는 허용하지 않는다. Write-back·A·write dropout은 공통이다.

| Query → Key | 인코더 | 디코더 |
|---|---|---|
| Tokens → Tokens | 전체 | causal |
| Tokens → Registers | 전체, 시작 S_read | 전체, 시작 S_read |
| Registers → Tokens/Registers | 전체 | 전체 |

GRT token 위치는0..n−1, register 위치는segment_len..segment_len+M−1로 짧은 prefix에서도 고정한다. RMT는[read;tokens;write] 순서이며 read memory 뒤에서 token 위치가 시작된다. NeoX는 register가 없고 cross-segment memory를 사용하지 않는다. RMT는 read/write block으로 메모리를 갱신한다.

## 3. 현재 gate 안정화 장치

일반 sigmoid write에 **logit tail regularization**과 선택적인 **write dropout**을 지원한다. 현재 노트북은 small·L75/M20·bias0·read off·threshold10/coef1e-4·dropout0을 선택한다. M8/20/75의 dropout0/.1 탐색은 완료했으며, 후속 실험은 모든 M에서 dropout0으로 통일한다. Dropout의 효과는 더 긴 학습에서 재검증한다. 판단 범위는 [안정화 보고서](experiments/20261018_l75_gate_stabilization.md#logit-tail-expansion)를 따른다.

- `write_logit_threshold`, `write_logit_coef`: `relu(abs(raw_logit)-threshold)^2`의 슬롯 평균을 유효 predictor token 수로 가중한다. Config의 coef 기본값0은 off다.
- 과거 BPTT replay(target=-100)의 직접 패널티는0이며 state를 통한 gradient는 유지한다. M/L 추가 배율이나 run별 표준편차 보정은 없다.
- `dropout_prob`: 학습 중 write를 Bernoulli mask로0으로 만들고 rescale하지 않는다. 패널티는 mask 이전 raw logits에서 계산한다.
- 실제 backward는 LM+coef*R, logging은 LM/raw R/weighted R/total을 분리한다. Evaluation·PPL·best 선택에는 규제를 넣지 않는다.
- Training 관측은raw write와abs logit>20 비율만 유지한다. 안정화 전용 logit 평균/최대·threshold 초과율·sigmoid slope·hidden norm 계산 경로는 제거했으며 상세 관측은 사후 분석에서 수행한다.
- `write_gate_bias_init`는 공통 초기화 기록이다. 현재 노트북은0으로 고정하며 과거 수치 실험은 결과에서만 비교한다.

Refine head·Uniform 초기화 전용 구현은 제거했다. Gate 전용 warmup·write mean loss·router normalization·softmax write는 현재 구현하지 않는다. 이들의 제안·종료 이력을 실행 선택지로 되살리지 않는다.

## 4. 과제별 loss 계약

Remember fact는[key,SEP100,value,EOS102]다. 인코더 query에는[value,EOS] 대신 MASK103을 넣고 label의 value+EOS slot과 직접 정렬한 CE를 사용한다. 정답은 embedding 입력에 들어가지 않는다. 디코더는 기존 predictor-mask convention의 next-token shift CE와 autoregressive 생성을 사용한다. 두 모델에 같은 입력·target을 준다.

WikiText adapter는 명시적인 `target_ids`를 다음 token에 정렬한다. 모델은 유효 target 위치의 `F.cross_entropy`를 계산하며 shift를 다시 하지 않는다. Word/BPE는 준비 manifest와 vocabulary·tokenizer artifact hash로 구분한다. BPE-token PPL과 word PPL은 직접 비교하지 않는다.

## 5. Stream·BPTT·평가

각 split을 global batch 수의 연속 stream으로 나누고 나머지 소수 token을 버린다. EOS에서도 state를 유지하며 split 시작/epoch 순환에서 reset한다. 평가 batch가 partition과 문맥을 바꾸므로 비교 중 고정한다. 평가 targets는 `B*(floor(N/B)-1)`이다.

매 update는 새 segment의 target만 loss로 삼는다. 이전 K segments를 같은 forward에서 재계산하고 그 앞 boundary state는 detach된 저장 state다. 현재 GRT/RMT는K3, NeoX는K0이다. Update 사이 graph는 저장하지 않는다. 마지막 짧은 segment를 포함한 실제 target 수를 기록한다. Nominal token 예산은global batch×L×updates이고 L75/20k/global256은384M이다.

WikiText는 Adam(.9,.999), decay0, global norm clip.25, cosine·warmup0·LR5e-4다. Remember는 AdamW·value clipping·linear schedule/curriculum을 사용한다. 두 optimizer 프로토콜을 섞지 않는다. 모든 평가·분석은 dropout off, RNG/model mode를 보존한다. WikiText best는 전체 validation mean NLL로 선택한다.

## 6. 배치·재개·분석

`training.auto_batch`는 VRAM 목표 안에서 microbatch와 accumulation을 조절하며 global batch와 평가 batch는 유지한다. 현재 노트북 목표는.95, 카탈로그 기본은.8이다. 같은 world size·데이터·최종 Config로만 재개한다. Checkpoint에는 model/optimizer/scheduler/RNG, rank별 history·boundary states·offset·target 수·manifest·best loss가 들어간다.

학습 중 RTLA는 고정 validation probe의 정상/reset/swap과 실제 update를 측정한다. W&B에는 Quality·Optimization·Memory·Runtime의 핵심 지표만 전송하며, 노트북과 JSON의 `overview`도 같은 best/final·단위 계약을 사용한다. 사후에는 슬롯·그룹·retention·gate/logit·rank/CKA 분석을 수행한다. Read off의 계산값을 읽기량으로 보고하지 않으며 candidate와 실제 update를 구분한다. 자세한 파일 전달·실행은 [workflow](20261011_experiment_workflow.md), 지표 해석은 [공통 사전](experiments/20261018_l75_gate_stabilization.md#2-공통-비교-기준)을 따른다.
