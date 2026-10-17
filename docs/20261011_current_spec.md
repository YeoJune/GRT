# GRT/RMT/NeoX 최신 실험·구현 명세

갱신일: 2026-10-17. 사용자 합의를 통합한 유일한 최신 설계 기준이다.
상태: 문서 정리 후 새 인코더·디코더, Remember 병렬 slot, WikiText 학습·평가 경로 구현. 검증 결과는 9절에 기록한다.
이전 문서는 archive에 보존하며 현재 설정을 선택하는 근거로 사용하지 않는다.
현재 실행 구조는 공통 NeoX 하나다. 과거 모델/설정/checkpoint 호환 경로는 제공하지 않는다.

2026-10-12 추가: WikiText의 메모리 없는 `neox` baseline을 공통 backbone/cell 인터페이스로 지원한다. register/router parameter 없이 M0·BPTT0으로 각 segment만 causal하게 읽는다. 현재 NeoX baseline은 WikiText decoder 전용이며 Remember는 RMT/GRT를 사용한다. 이전 small 용량 비교는 [L150 결과](experiments/20261012_small_register_sweep.md), 현재 dropout/bias 비교는 [별도 L75 계획](experiments/20261015_l75_write_gates.md)을 따른다.

## 1. 범위와 순서

기존 Remember 실험을 지원하면서 WikiText-103 지원을 추가한다.
문서 최신화 후 코드 구현, 작은 CPU 검증 순서로 작업한다.
인코더와 디코더는 간결한 공통 backbone/데이터/학습 경계를 사용한다.
과거 설명과 검증 기록은 문서로만 archive에 남긴다. 과거 config/모델 wrapper/fixture와 checkpoint 변환 코드는 유지하지 않는다.
커밋은 명시 승인 후, push는 지시 후 수행한다.
장시간 GPU 학습은 사용자가 수행하며 에이전트는 작은 CPU 기능 검증을 수행한다.

## 2. 공통 backbone와 크기

GPT-NeoX 계열 Transformer, parallel residual, RoPE, tied embedding/출력 head.
동일 실험에서 RMT/GRT backbone의 구조와 차원을 일치시킨다.
30M 예산과 router 크기 제한은 없다. 전체 parameter 수 차이는 기록한다.

| 항목 | base | small | tiny |
|---|---:|---:|---:|
| layers | 16 | 6 | 3 |
| d_model | 410 | 256 | 128 |
| heads | 10 | 8 | 4 |
| head dimension | 41 | 32 | 32 |
| FFN dimension | 2100 | 1024 | 512 |

RoPE rotary_pct=0.25, base=10000을 사용한다.
base는 RMT 2022 WikiText의 차원을 맞춘 NeoX 변형이며 정확한 원문 모델 재현이 아니다.
small은 후속 사용자 지시에 따라 D256으로 변경했고 FFN=4D(1024)를 유지한다. tied full softmax를 두 모델에 공통 적용한다.
크기·예산 선택과 노트북 운영은 [실험 workflow](20261011_experiment_workflow.md)를 따른다. tiny와 200-update 조건은 유지한다.
WikiText word-level vocabulary는 267735, 별도 GPT-2 BPE 옵션은 50257이다. Remember는 기존 V128/token 규칙을 기반으로 한다.
공통 NeoX 경로는 기존 vendor의 GELU, LayerNorm eps=1e-5, 선형/embedding 초기화 std=0.02를 계승한다.
후보와 token hidden은 모두 final norm 뒤에서 추출한다. 새 비교 프로토콜을 추가한 것으로 해석하지 않는다.
위치는 각 segment에서 reset한다. RMT는 [read;tokens;write] 순서로 0부터 매겨 본문이 M부터 시작한다.
GRT는 token 위치 0..n-1, register 위치 segment_len..segment_len+M-1을 사용한다. 짧은 segment에서도
register 위치를 고정해야 가변 길이 prefix가 token-to-memory RoPE를 바꾸지 않는다.
전체 길이 segment에서는 [tokens;registers]의 연속 위치와 동일하다.
RoPE scaling은 없고 초기 cache 길이는 2048이며 더 긴 길이에서는 vendor cache를 확장한다.
같은 model seed로 생성하면 두 모델의 공통 backbone 초기 weights도 동일하다.

## 3. GRT 공통 계산

20260608의 입력 기반 router/ALU/register write-back을 기준으로 한다.

```text
X = embedding(segment_ids)
u, alpha = AttentionPool(q_pool, X)
v = RegisterCrossAttention(u, S, S)
u_read = AttentionPool(q_pool, S)
v_read = RegisterCrossAttention(u_read, S, S)
R = sigmoid(MLP_R(concat(u_read, v_read)))
W = sigmoid(MLP_W(concat(u, v)))
S_read = R*S if conditional_read else S
[Y; C] = TransformerStack([X; S_read], mask, position_ids,
                          token_read_state=S_read)
logits = tied_lm_head(Y)
S_next = (1-W)*S + W*C
```

2026-10-17 추가: `model.register.refine_gate`(기본false)를 켜면 write MLP hidden h를 공유하는 중립 초기화 보정 head를 추가한다. p는 기존 write sigmoid, q는 새 head의 sigmoid이며 유효 W=p+(2q−1)p(1−p)로 바뀐다. 새 head weight·bias는0이고 초기화 RNG를 소모하지 않아 시작 시 q=.5, W=p다. Refine 옵션 자체는 Uniform Gate Initialization이나 추가 loss를 켜지 않는다. Read gate와 독립적이며 candidate·고정 state 읽기·종료 후 write-back은 동일하다. Write dropout은 on에서 유효 W를0으로 mask한다. [실험 조건과 전달 방법](experiments/20261017_l75_refine_gate.md)을 따른다.

2026-10-18 추가: `model.register.uniform_gate_init`(기본false)는 write 출력층 bias만 `logit(U[1/M,1−1/M])`로 초기화한다(M≥3). On이면 `write_gate_bias_init` 대신 사용하며 공통 가중치·이후 CPU RNG 상태를 보존한다. Refine과 독립적이며 현재 [Uniform 단독 비교](experiments/20261018_l75_uniform_gate.md)는 Refine off다. 추가 loss·파라미터·forward 변경 없이 초기 bias만 바꾼다.

R/W는 register별 scalar [B,M,1], S/C는 [B,M,D].
read gate는 on/off 가능하고 기본 conditional_read=false. write gate는 항상 적용한다.
read gate는 segment 시작 S만으로 계산하며 현재 token X를 참조하지 않는다. write gate는 기존 입력 기반 pooling을 유지하고 alpha는 이 write pooling의 token attention weights다.
위치별 prefix 재계산은 새 구조에서 제거한다.
세그먼트의 전체 Transformer stack 완료 후 write-back을 한 번 수행한다.
개별 layer마다 저장 state를 write-back하지 않으며 S_next는 다음 segment에서 사용한다.
저장 state S와 layer별 register hidden state H_R는 구분한다.

## 4. 인코더/디코더의 표준 register 읽기

GRT는 두 버전 모두 segment 시작 S_read를 모든 layer의 token 읽기 source로 사용한다. 각 layer의 LayerNorm과 K/V projection은 서로 다르지만 그 입력 state는 segment 안에서 고정된다. fixed_state는 선택 옵션이 아니라 유일한 표준 구조이며 register_read_source 설정 필드는 제거한다.
candidate 계산용 register hidden은 모든 token을 읽고 layer마다 계산된다. 이 중간 표현은 token의 K/V로 제공하지 않는다. 전체 stack 완료 후 최종 candidate로 write-back을 한 번 수행하며 새 state는 다음 segment에서만 사용한다.
읽기 state를 고정하는 것은 detach가 아니다. 이전 segment와의 state gradient 및 BPTT를 유지한다. read gate는 model.register.conditional_read로 제어하며 기본 off다.

| Query → Key | 인코더 | 디코더 |
|---|---|---|
| Tokens → Tokens | 양방향 전체 | causal j<=i |
| Tokens → Registers | 전체, segment 시작 S_read | 전체, segment 시작 S_read |
| Registers → Tokens | 전체 | 전체 |
| Registers → Registers | 전체 | 전체 |

디코더는 read gate on/off 모두 위치 i의 각 layer 표현이 segment 시작 state와 i까지의 token에만 의존한다. gate도 S만으로 계산하므로 병렬 입력의 미래 token에 영향을 받지 않는다. 따라서 dropout off, 동일한 RoPE 위치/segment 경계에서 전체 segment 병렬 계산과 prefix 단위 직렬 추론의 token 출력이 일치한다. 추론에서도 prefix의 임시 write-back 결과를 다음 token에 넣지 않고 segment 완료 시에만 저장한다. 완료 시 전체 token/state가 같으므로 candidate, write gate, 다음 저장 state도 일치한다.

이전 중간 register 읽기 방식은 token-token mask가 causal이어도 병렬 teacher forcing에서 미래 token -> register hidden -> 앞쪽 token의 층간 경로로 leakage가 발생할 수 있었다. 종료 후 write-back의 타이밍 문제가 아니라 token에 노출되는 K/V source의 문제다. 이 방식은 설명 기록으로만 남기며 실행 코드·설정·호환 분기를 유지하지 않는다.
NeoX 기본 전체 triangular mask를 그대로 쓰면 뒤쪽 register를 token이 읽지 못한다.
인코더 양방향 및 디코더 block mask 지원을 명시적으로 구현한다.

## 5. Remember 새 인코더: 병렬 정답 slot 확정

기존 fact 데이터, key 선택, split, curriculum을 재사용한다.
fact는 [key, SEP100, value, EOS102]이며 각 fact를 한 번 인코딩 후 write-back한다.

```text
query input: [query_key, GEN101, MASK_VALUE ... , MASK_EOS]
target:                              value ... , EOS102
```

정답 value/EOS는 label에만 존재하며 embedding 입력에는 넣지 않는다.
query를 한 번 인코딩하고 masked slot의 tied head에서 value+EOS를 병렬 예측한다.
loss는 slot과 target을 직접 정렬한 CE이며 기존 next-token shift loss와 구분한다.
value_size>1도 고정 길이 masked slots로 표현한다.
평가 target은 value+EOS exact match이며 autoregressive 생성과 다른 prediction protocol을 기록한다.
query write-back도 stack 종료 후 한 번이며 출력 state를 현재 답 예측에 되먹이지 않는다.
MASK는 기존 데이터에서 사용하지 않는 V128의 103을 모든 answer slot에 공유한다.
새 병렬 collator는 data seed로 초기화한 독립 Torch generator를 사용하고 해당 상태도 checkpoint에 저장한다.
기존 autoregressive collator의 전역 RNG 동작은 보존한다.
RMT encoder 비교에도 동일 query 입력/labels/loss를 적용한다.
Remember 디코더도 현재 공통 NeoX를 사용하며 next-token shift loss와 autoregressive 생성으로 평가한다.

## 6. WikiText-103 승인 조건

원문 word tokenizer와 데이터 처리, vocabulary 267735.
full softmax는 원문 adaptive softmax와 다르며 메모리 비용을 실제 측정한다.
실행 방법은 [실험 workflow](20261011_experiment_workflow.md)를 따른다.
GPT-2 BPE는 `openai-community/gpt2`의 50257 vocabulary를 사용한다. 각 원본 줄에서 CR/LF만 제거하고 공백·대소문자를 보존한다. 빈 줄을 포함해 줄마다 EOS50256을 붙이며 BOS/padding/truncation/prefix-space 추가는 하지 않는다. tokenizer만 다운로드하고 모델 weights는 초기화한다. tokenizer와 prepared artifact의 hash를 manifest에 기록한다.
단어 기준 원문 조건과 구분하며 BPE-token PPL을 기록한다.

| 항목 | 설정 |
|---|---|
| segment length | 150 |
| memory slots | 10 |
| BPTT | 이전 3 segments |
| global batch | 256 |
| optimizer / LR | Adam / 5e-4 |
| scheduler | cosine, warmup 0 |
| gradient clipping | global norm 0.25 |
| dropout / attention dropout | 0.1 / 0 |
| max updates | 100000 |
| validation / logging | 1000 / 10 updates (smoke: 20 / 10) |

위 표는 기본 설정이다. Kaggle 노트북에서 segment_len·num_registers·bptt_segments를 RMT/GRT 공통 override로 지정한다. BPTT는 재계산할 이전 segment 수이며 전체 입력의 segment 수를 제한하지 않는다. WikiText는 문장별 truncation 없이 연속 stream을 끝까지 처리하고 마지막 짧은 segment도 사용한다. 길이를 바꿔도 BPTT segment 수·updates·배치·LR은 자동 변경하지 않는다.

Kaggle T4 × 2는 microbatch32/accumulation4로 시작하며 auto_batch=true, memory_fraction=0.8로 global batch256을 유지해 조절한다. rank당 실질 배치128의 약수는 모두 2의 거듭제곱이므로 후보는128→64→32→16→8→4→2→1이다. 모든 크기와 두 모델에 같은 학습 설정을 사용한다. 공통 auto_batch 기본은 false이며 Kaggle/Colab runtime에서 true로 켠다. Colab GPU1개는 micro32/accumulation8로 같은 global256을 유지한다.
LR은 이전 배치60의2.5e-4에서 제곱근 비례값 약5.16e-4를 반올림해5e-4로 정했다. 최적 LR을 실측한 결과는 아니며 자동 microbatch 보정은 LR을 바꾸지 않는다. warmup0/cosine/Adam/clip/dropout/평가 batch10은 유지한다.
기존 FP32 경로를 기준으로 CPU 기능 검사를 준비한다.
데이터 순서/target token 예산을 공유하고 모델 RNG와 데이터 RNG를 구분한다.
연속 state의 입출력, stream ID/reset, BPTT detach 경계가 필요하다.
새 학습은 vendor manage_gradients에 의존하지 않는다. 자세한 replay/detach 경계는 8절을 따른다.
segment boundary target을 빠뜨리지 않고 loss shift를 중복 적용하지 않는다.
NLL 합/유효 target 수로 mean NLL을 계산한 후 exp로 PPL을 계산한다.
현재 GRT decoder 기본과 RMT decoder는 모두 autoregressive NLL/PPL을 기록한다. 이전 중간 register 읽기로 얻은 GRT PPL은 leakage 가능성이 있으므로 현재 모델 또는 AR baseline과 같은 조건의 결과로 해석하지 않는다.
WikiText 인코더 MLM/분류 목적은 후속 지정이며 자동으로 next-token 목적을 채택하지 않는다.

## 7. 구현 경계와 검증

공통 backbone/모델 factory에 인코더·디코더와 고정 state 읽기를 구현한다. 읽기 source 선택 분기는 두지 않는다.
데이터 adapter에서 task별 입력/label 계약을 구성하고 trainer의 token-ID 기반 분기는 피한다.
trainer/evaluator에는 objective/state/metric capability를 명시적으로 확장한다.
현재 모델에서 사용하는 vendor 코드의 출처·hash·라이선스를 유지한다.

검증: 공통 backbone 일치, tied save/load, read off 영향, stack/write-back 각1회,
prefix 호출 제거, masked Remember label 변경시 입력/logits 불변,
fact->query gradient, parallel value+EOS EM,
고정 state 읽기의 미래 token 불변, prefix/전체 segment 출력과 종료 state 일치,
RoPE positions, state reset/BPTT 경계, boundary CE, token-weighted PPL, 재개.
GPU 수렴/VRAM/throughput은 작은 CPU 검사와 구분하여 보고한다.

## 8. 실제 인터페이스와 학습 상태

공통 모델은 `src/grt/models/neox.py`이며 factory는 항상 현재 `RecurrentNeoX`를 생성한다. `implementation=neox`는 실행 구조의 식별 값이다.
인코더·디코더는 같은 stack 구현에 attention mask만 바꾼다. factory의 과거 구현 분기는 없다.

| 모델 입력/출력 | 계약 |
|---|---|
| input_ids | int64 [B,L], unpadded, segment_len 단위로 순서대로 처리 |
| memory_state | 선택 입력 [B,M,D]; 없으면 learnable initial memory로 시작 |
| labels / labels_mask | Remember raw label: parallel slot-aligned CE 또는 디코더 next-token shift |
| target_ids | WikiText adapter가 정렬한 next-token target [B,L]; -100은 replay loss 제외 |
| 출력 | loss, logits(일반 forward), memory_state(마지막 write-back) |

`labels`와 `target_ids`는 함께 사용할 수 없다. WikiText target_ids 경로는 유효 위치만
full-vocabulary head를 계산하고 logits를 반환하지 않아 replay 위치의 큰 logits 할당을 피한다.
Remember 병렬 generate는 answer slot argmax를 한 번 읽는다. 디코더 generate는 KV cache 없는
참조 구현으로 각 생성 단계에서 prompt를 재실행한다. 이는 **학습 prefix 재계산**과 별개다.

WikiText 준비는 원래 tokenized `wiki.train/valid/test.tokens` 파일에서 수행한다.
train 단어 빈도 순 vocabulary, EOS 우선, line별 EOS, 대소문자 유지, unknown은 train `<unk>`로 매핑한다.
train vocabulary 267735가 아니면 기본 준비 명령이 실패한다. raw/BPE 데이터로 자동 대체하지 않는다.
manifest에 split token 수, 원본/준비 파일/vocabulary SHA256을 저장한다.
Kaggle 기본 입력은 Hugging Face `Salesforce/wikitext`의 `wikitext-103-v1`이며
`scripts/download_wikitext.py`로 원본 text rows를 내보낸 뒤 같은 준비 경로를 사용한다.
로컬 원본 파일도 계속 지원한다. BPE tokenizer는 `openai-community/gpt2`에서 내려받는다.

각 split을 B개의 연속 stream으로 나누고 끝의 B 미만 token만 버린다. 각 stream 첫 token은 입력만으로
쓰이므로 평가 target 수는 `B*(floor(N/B)-1)`이다. batch partition 경계는 연결하지 않는다.
문서 EOS에서는 state를 유지하며 split 시작/epoch 순환에서만 reset한다. 문서 간 context 조건을 기록한다.
평가 batch 크기를 변경하면 partition과 context도 달라지므로 두 모델은 같은 평가 batch를 사용한다.

학습 update마다 현재 150 token의 다음 token만 loss로 삼는다. 앞선 최대 3 segment를 연결해
같은 forward 안에서 재계산하고, 그 앞 boundary state는 detach된 저장 state로 공급한다.
따라서 초기에는 1→2→3→4 segments, steady state에서는 최대 600 input token을 처리한다.
이전 segment들에는 target=-100을 주지만 state를 통해 현재 loss gradient가 전달된다.
optimizer update 사이에는 graph를 보관하지 않는다. K=0이면 직전 detached state만 읽는다.
이 경우 현재 write gate를 다음 segment의 loss로 학습할 경로도 끊기므로 GRT 학습의 기본 K=3을 유지한다.
마지막 짧은 segment도 boundary next-token target을 포함하고 실제 target 수를 token budget에 누적한다.
update당 target 예산 상한은38400(256×150)이다. 10k의10000 updates는384M, 100k의100000 updates는3.84B이며 짧은 tail/reset을 반영한 실제 수를 기록한다. budget 이름은 update 수를 뜻한다.

현재 checkpoint 형식은 `memory_experiment/4`이며 과거 형식 변환/읽기는 제공하지 않는다. 고정 state 표준화 이전 config/checkpoint로 재개하지 않고 새 run을 사용한다. checkpoint는 model/Adam/scheduler/RNG뿐 아니라 rank별 history, detached boundary states,
stream offset, 실제 target token 수, manifest와 best loss를 저장한다. 동일 설정/동일 world_size로만 재개한다.
validation은 dropout off로 순차 state를 유지하고 training RNG를 보존한다. best는 validation mean NLL로 선택한다.
DDP 학습은 stream을 rank별로 나누며 평가 전체 stream은 각 rank에서 동일하게 읽는다. 평가를 중복 합산하지 않는다.

`--benchmark-steps`는 WikiText에서도 실제 replay/full softmax/Adam/gradient norm clip을 측정한다.
`training.auto_batch=false`는 config 배치를 그대로 사용한다. true는 GPU별 peak reserved VRAM을 측정해 목표 `training.memory_fraction`(기본0.8) 안에서 microbatch를 늘리거나 줄이고 accumulation으로 global batch를 유지한다. rank당 실질 배치를 넘지 않으며 평가 batch는 고정한다. 출력 calibrated_override를
utils에서 적용해 모델당 최종 Config YAML 하나를 저장하고 본 학습에 사용한다. 환경·logging·보정용 YAML을 별도로 늘리지 않는다.
평가 batch는 자동으로 변경하지 않으며 평가에서도 메모리 여유 조건을 만족해야 한다.
utils의 예상 시간은 학습·주기별 validation·마지막 test를 포함하며 checkpoint I/O·데이터 준비·시작 시간은 제외한다.

WikiText는 Adam betas=(0.9,0.999), eps=1e-8, weight decay=0이다. Remember는 기존
AdamW/value clip/linear schedule/curriculum을 재사용한다. 두 과제의 optimizer 프로토콜을 혼동하지 않는다.

고유 Parameter 수(optimizer에서 tied weight 중복 없음, buffer 제외):

| 크기 | 공통 backbone | RMT 전체 | GRT 전체 |
|---|---:|---:|---:|
| WikiText base | 148175210 | 148179310 | 150378764 |
| WikiText small | 73279232 | 73281792 | 74343956 |
| WikiText small GPT-2 BPE | 17604864 | 17607424 | 18669588 |
| WikiText base GPT-2 BPE | 59009230 | 59013330 | 61212784 |
| WikiText tiny GPT-2 BPE | 7027968 | 7029248 | 7434900 |

small의 D256×267735 tied vocabulary는 68540160 parameters이며 전체 backbone은 약 73.28M이다.
30M 예산은 적용하지 않는다. word vocabulary의 full logits 한 segment만으로도 microbatch32에서
약4.79 GiB(fp32)이며 gradient/activation/Adam/CE 중간 버퍼는 추가다. 단일 GPU의 논리 batch256을 한 번에 계산하면 logits만 약38.30 GiB이므로 실제 GPU에서는 사전 측정과 accumulation이 필요하다. GPT-2 BPE의 microbatch32 logits는 약0.90 GiB이다. T4 적합성과 속도는 실제 사전 측정으로 확인한다.

새 GRT Remember에도 RTLA trace를 지원하며 query를 포함한 모든 segment의 stack 뒤 write-back을 기록한다.
GRT WikiText RTLA는 학습 중 고정 validation probe와 사후 checkpoint 개입 분석을 지원한다. 학습 중에는 gate/state/실제 write-back 변화량과 정상·초기화·문맥 교환의 token NLL을 기록한다. 사후에는 같은 분석에 slot별 초기화와 한 번 교환한 state의 이후 segment 영향을 추가한다. 입력·target·segment 경계는 동일하며 dropout off, gradient off, RNG와 model mode 복원으로 학습 경로에 영향을 주지 않는다. 자세한 설정·해석은 실험 workflow의 RTLA 절을 따른다.
read gate off의 계산값은 읽기량으로 해석하지 않으며 write pooling attention과 token-to-register attention은 구분한다. candidate distance와 gate 적용 후 실제 state update norm을 별도 기록한다. WikiText 인코더 MLM objective는 지원 범위 밖이다.
Kaggle/Colab 노트북은 WikiText base/small/tiny, GPT-2 BPE/word-level과 smoke/10k/20k/100k 예산을 선택한다. 2026-10-15 학습 노트북 선택 기본은 별도 L75 계획의 GRT/small/20k/M8/bias0/write dropout0/read off다. 데이터·학습·분석은 같은 WikiTextRun을 사용한다. 로컬 Remember 실행은 현재 template 노트북을 사용한다.

## 9. 현재 검증과 해석 범위

20261012 RTLA 개입 분석 추가는 관련 CPU 검사5개와 사후 CLI 실행으로 확인했다. 메모리 사용/무시 예제, token 가중·짧은 tail·개입 후 회복, 실제 갱신량 및 분석 on/off의 실제 GRT 가중치/RNG 동일성을 검사했다. W&B payload는 모의 클라이언트로 확인했으며 온라인 업로드·GPU 비용·실제 WikiText 메모리 효과는 아직 실측하지 않았다.

고정 state 표준화 후 관련 CPU 검사13개가 합계3.13초에 통과했다. read gate on/off 모두 미래 token 변경 불변성과 모든 prefix의 출력을 확인했고, 비영 초기 state·복수 segment·짧은 tail에서 layer별 읽기 source/토큰 attention 출력, 종료 candidate/write gate/저장 state를 병렬 실행과 대조했다. prefix 비교는 FP32의 연산 shape 차이를 고려해 rtol=1e-5/atol=1e-7을 사용했다. Remember masked slot/gradient/trace, replay detach/gradient 및 현재 형식의 Remember·WikiText 짧은 실행과 재개도 확인했다. GPU 수렴·VRAM·속도는 새 구조로 아직 실측하지 않았다.

현재 코드 기준 관련 CPU 검사 54개가 9.16초에 통과했다(설정 선택·단일 파일 입력·형식 거부·짧은 실행/재개와 vendor 출처 검사). 전체 회귀나 GPU 수렴 검사 결과로 해석하지 않는다. 현재 카탈로그와 공통 NeoX 인코더·디코더 계약을 검사한다. 모델/과제 검사는 현재 preset에서 작은 CPU override를 적용하고, archive의 코드나 설정에 의존하지 않는다. config 단일 파일 입력과 현재 checkpoint 재개를 지원하며 옛 경로/형식은 거부한다.

사용자가 제공한 tiny/smoke GPU 실행은 두 모델 모두 200 updates 완료했고 온라인 W&B도 failures=0, last_logged_step=200으로 종료했다. small BPE/T4 × 2 preflight는 두 모델 약0.48초/update였다. 이는 과거 관측이며 이번 코드 정리 후 GPU 실행을 다시 수행한 결과가 아니다. tiny/100k 수렴과 시간은 아직 확인하지 않았다.

과거 중간 register 읽기 GRT 결과는 leakage 가능성이 있어 현재 autoregressive GRT/RMT 결과와 동일 조건으로 해석하지 않는다. 정확한 RMT 2022 Transformer-XL/adaptive-softmax 재현도 아니다.

## 10. 출처

원 설계: [20260608 spec](archive/20260608_spec.md).
이전 구현 기준: [20261010 design](archive/20261010_design.md).
원 논문: https://papers.neurips.cc/paper_files/paper/2022/file/47e288629a6996a17ce50b90a056a0e1-Paper-Conference.pdf
현재 운영 코드: src/grt/models, data.py, trainer.py, evaluator.py.
