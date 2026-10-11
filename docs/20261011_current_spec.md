# GRT/RMT 최신 실험·구현 명세

갱신일: 2026-10-11. 사용자 합의를 통합한 유일한 최신 설계 기준이다.
상태: 문서 정리 후 새 인코더·디코더, Remember 병렬 slot, WikiText 학습·평가 경로 구현. 검증 결과는 9절에 기록한다.
이전 문서는 archive에 보존하며 현재 설정을 선택하는 근거로 사용하지 않는다.
현재 실행 구조는 공통 NeoX 하나다. 과거 모델/설정/checkpoint 호환 경로는 제공하지 않는다.

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
g = concat(u, v)
R = sigmoid(MLP_R(g)); W = sigmoid(MLP_W(g))
S_read = R*S if conditional_read else S
[Y; C] = TransformerStack([X; S_read], mask, position_ids)
logits = tied_lm_head(Y)
S_next = (1-W)*S + W*C
```

R/W는 register별 scalar [B,M,1], S/C는 [B,M,D].
read gate는 on/off 가능하고 기본 conditional_read=false. write gate는 항상 적용한다.
위치별 prefix 재계산은 새 구조에서 제거한다.
세그먼트의 전체 Transformer stack 완료 후 write-back을 한 번 수행한다.
개별 layer마다 저장 state를 write-back하지 않으며 S_next는 다음 segment에서 사용한다.
저장 state S와 layer별 register hidden state H_R는 구분한다.

## 4. 인코더/디코더와 register 읽기 옵션

register_read_source=intermediate가 두 버전의 기본값이다.
각 layer의 token은 이전 layer의 register 중간 표현을 읽는다.
fixed_state 옵션은 token이 모든 layer에서 segment 시작 S_read를 읽는다.
후보 계산 경로는 유지하되 candidate 중간 표현을 token 읽기 source로 제공하지 않는다.
fixed_state는 detach가 아니며 state gradient/BPTT와 독립이다.
read gate와 register_read_source는 별도 옵션이다.
`--set model.register_read_source=fixed_state` 또는 동일한 Python override로 fixed_state를 선택한다. read gate는 `model.register.conditional_read`로 제어한다.

| Query → Key | 인코더 | 디코더 |
|---|---|---|
| Tokens → Tokens | 양방향 전체 | causal j<=i |
| Tokens → Registers | 전체, 선택한 read source | 전체, 선택한 read source |
| Registers → Tokens | 전체 | 전체 |
| Registers → Registers | 전체 | 전체 |

디코더 기본 intermediate는 token-token causal이다. 다층 register를 경유한
미래 token -> register 중간 표현 -> 과거 token 경로가 가능하므로 엄격한 AR decoder와 구분한다.
read=false는 router read gate 경로를 제거하지만 이 층간 경로는 제거하지 않는다.
fixed_state + read=false는 해당 우회 경로를 막으며 CPU 미래변경 불변성 검사를 수행한다.
read=true의 full-segment gate는 별도 미래 정보 경로를 만들 수 있어 평가 조건에 기록한다.
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
| max updates | 200000 |
| validation / logging | 1000 / 10 updates (smoke: 20 / 10) |

Kaggle T4 × 2는 microbatch32/accumulation4로 시작하며 auto_batch=true, memory_fraction=0.8로 global batch256을 유지해 조절한다. rank당 실질 배치128의 약수는 모두 2의 거듭제곱이므로 후보는128→64→32→16→8→4→2→1이다. 모든 크기와 두 모델에 같은 학습 설정을 사용한다. 공통 auto_batch 기본은 false이며 Kaggle runtime에서만 true로 켠다.
LR은 이전 배치60의2.5e-4에서 제곱근 비례값 약5.16e-4를 반올림해5e-4로 정했다. 최적 LR을 실측한 결과는 아니며 자동 microbatch 보정은 LR을 바꾸지 않는다. warmup0/cosine/Adam/clip/dropout/평가 batch10은 유지한다.
기존 FP32 경로를 기준으로 CPU 기능 검사를 준비한다.
데이터 순서/target token 예산을 공유하고 모델 RNG와 데이터 RNG를 구분한다.
연속 state의 입출력, stream ID/reset, BPTT detach 경계가 필요하다.
새 학습은 vendor manage_gradients에 의존하지 않는다. 자세한 replay/detach 경계는 8절을 따른다.
segment boundary target을 빠뜨리지 않고 loss shift를 중복 적용하지 않는다.
NLL 합/유효 target 수로 mean NLL을 계산한 후 exp로 PPL을 계산한다.
intermediate decoder의 NLL/PPL은 엄격한 AR baseline과 동등하게 해석하지 않는다.
WikiText 인코더 MLM/분류 목적은 후속 지정이며 자동으로 next-token 목적을 채택하지 않는다.

## 7. 구현 경계와 검증

공통 backbone/모델 factory에 인코더·디코더와 read source 정책을 둔다.
데이터 adapter에서 task별 입력/label 계약을 구성하고 trainer의 token-ID 기반 분기는 피한다.
trainer/evaluator에는 objective/state/metric capability를 명시적으로 확장한다.
현재 모델에서 사용하는 vendor 코드의 출처·hash·라이선스를 유지한다.

검증: 공통 backbone 일치, tied save/load, read off 영향, stack/write-back 각1회,
prefix 호출 제거, masked Remember label 변경시 입력/logits 불변,
fact->query gradient, parallel value+EOS EM,
fixed_state/read=false 미래 token 불변, intermediate의 경로 기록,
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
update당 target 예산 상한은38400(256×150)이다. 20k의20000 updates는768M, full의200000 updates는7.68B로 이전 배치60 대비 약4.27배이며 짧은 tail/reset을 반영한 실제 수를 기록한다. budget 이름은 계속 update 수를 뜻한다. 이전20k의180M token과 비교하려면 steps=4688, 이전full의1.8B와 비교하려면 steps=46875를 명시하고 최종 실제 token 수를 확인한다.

현재 checkpoint 형식은 `memory_experiment/3`이며 과거 형식 변환/읽기는 제공하지 않는다. checkpoint는 model/Adam/scheduler/RNG뿐 아니라 rank별 history, detached boundary states,
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
WikiText 자동 RTLA 샘플링과 인코더 MLM objective는 지원 범위 밖이며 해당 설정을 명시적으로 거부한다.
Kaggle 노트북은 WikiText base/small/tiny, GPT-2 BPE/word-level과 smoke/20k/full 예산을 선택한다. 기본은 tiny/smoke/RMT·GRT 두 모델이다. 로컬 Remember 실행은 현재 template 노트북을 사용한다.

## 9. 현재 검증과 해석 범위

현재 코드 기준 관련 CPU 검사 54개가 9.16초에 통과했다(설정 선택·단일 파일 입력·형식 거부·짧은 실행/재개와 vendor 출처 검사). 전체 회귀나 GPU 수렴 검사 결과로 해석하지 않는다. 현재 카탈로그와 공통 NeoX 인코더·디코더 계약을 검사한다. 모델/과제 검사는 현재 preset에서 작은 CPU override를 적용하고, archive의 코드나 설정에 의존하지 않는다. config 단일 파일 입력과 현재 checkpoint 재개를 지원하며 옛 경로/형식은 거부한다.

사용자가 제공한 tiny/smoke GPU 실행은 두 모델 모두 200 updates 완료했고 온라인 W&B도 failures=0, last_logged_step=200으로 종료했다. small BPE/T4 × 2 preflight는 두 모델 약0.48초/update였다. 이는 과거 관측이며 이번 코드 정리 후 GPU 실행을 다시 수행한 결과가 아니다. tiny/full 수렴과 시간은 아직 확인하지 않았다.

GRT intermediate의 teacher_forced_noncausal PPL을 autoregressive RMT PPL과 동일 조건으로 해석하지 않는다. 정확한 RMT 2022 Transformer-XL/adaptive-softmax 재현도 아니다.

## 10. 출처

원 설계: [20260608 spec](archive/20260608_spec.md).
이전 구현 기준: [20261010 design](archive/20261010_design.md).
원 논문: https://papers.neurips.cc/paper_files/paper/2022/file/47e288629a6996a17ce50b90a056a0e1-Paper-Conference.pdf
현재 운영 코드: src/grt/models, data.py, trainer.py, evaluator.py.
