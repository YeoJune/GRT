# GRT/RMT 최신 실험·구현 명세

갱신일: 2026-10-11. 사용자 합의를 통합한 유일한 최신 설계 기준이다.
상태: 문서 정리 후 새 인코더·디코더, Remember 병렬 slot, WikiText 학습·평가 경로 구현. 검증 결과는 9절에 기록한다.
이전 문서는 archive에 보존하며 현재 설정을 선택하는 근거로 사용하지 않는다.
`implementation=neox`가 새 구조이며 `legacy`는 기존 Remember/저자 reference 호환 경로다.

## 1. 범위와 순서

기존 Remember 실험을 지원하면서 WikiText-103 지원을 추가한다.
문서 최신화 후 코드 구현, 작은 CPU 검증 순서로 작업한다.
인코더와 디코더는 간결한 공통 backbone/데이터/학습 경계를 사용한다.
기존 저자 baseline, fixture, checkpoint와 실행 결과는 보존한다.
새 모델은 기존 checkpoint와 자동 호환된다고 가정하지 않는다.
커밋은 명시 승인 후, push는 지시 후 수행한다.
장시간 GPU 학습은 사용자가 수행하며 에이전트는 작은 CPU 기능 검증을 수행한다.

## 2. 공통 backbone와 크기

GPT-NeoX 계열 Transformer, parallel residual, RoPE, tied embedding/출력 head.
동일 실험에서 RMT/GRT backbone의 구조와 차원을 일치시킨다.
30M 예산과 router 크기 제한은 없다. 전체 parameter 수 차이는 기록한다.

| 항목 | base | small |
|---|---:|---:|
| layers | 16 | 6 |
| d_model | 410 | 256 |
| heads | 10 | 8 |
| head dimension | 41 | 32 |
| FFN dimension | 2100 | 1024 |

RoPE rotary_pct=0.25, base=10000을 사용한다.
base는 RMT 2022 WikiText의 차원을 맞춘 NeoX 변형이며 정확한 원문 모델 재현이 아니다.
small은 후속 사용자 지시에 따라 D256으로 변경했고 FFN=4D(1024)를 유지한다. tied full softmax를 두 모델에 공통 적용한다.
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
`configs/fixed_state.yaml`로 fixed_state를 선택한다. read gate는 `model.register.conditional_read`로 제어한다.

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
원본 autoregressive Remember baseline은 별도 참조 경로로 보존한다.

## 6. WikiText-103 승인 조건

원문 word tokenizer와 데이터 처리, vocabulary 267735.
full softmax는 원문 adaptive softmax와 다르며 메모리 비용을 실제 측정한다.
추가 GPT-2 BPE 실험은 [BPE 계약과 Kaggle 안내](20261011_wikitext_bpe.md)를 따른다.
단어 기준 원문 조건과 구분하며 BPE-token PPL을 기록한다.

| 항목 | 설정 |
|---|---|
| segment length | 150 |
| memory slots | 10 |
| BPTT | 이전 3 segments |
| global batch | 60 |
| optimizer / LR | Adam / 2.5e-4 |
| scheduler | cosine, warmup 0 |
| gradient clipping | global norm 0.25 |
| dropout / attention dropout | 0.1 / 0 |
| max updates | 200000 |

GPU/시간 예산은 미제공이며 microbatch/accumulation은 실측으로 조절한다.
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
기존 Remember author reference를 몰래 바꾸거나 vendor 원본을 무기록 변경하지 않는다.

검증: 원본 baseline 회귀, tied save/load, read off 영향, stack/write-back 각1회,
prefix 호출 제거, masked Remember label 변경시 입력/logits 불변,
fact->query gradient, parallel value+EOS EM,
fixed_state/read=false 미래 token 불변, intermediate의 경로 기록,
RoPE positions, state reset/BPTT 경계, boundary CE, token-weighted PPL, 재개.
GPU 수렴/VRAM/throughput은 작은 CPU 검사와 구분하여 보고한다.

## 8. 실제 인터페이스와 학습 상태

공통 모델은 `src/grt/models/neox.py`, factory 선택은 `model.implementation=neox`다.
인코더·디코더는 같은 stack 구현에 attention mask만 바꾼다. legacy/vendor 원본은 수정하지 않는다.

| 모델 입력/출력 | 계약 |
|---|---|
| input_ids | int64 [B,L], unpadded, segment_len 단위로 순서대로 처리 |
| memory_state | 선택 입력 [B,M,D]; 없으면 learnable initial memory로 시작 |
| labels / labels_mask | Remember raw label: parallel slot-aligned CE 또는 legacy next-token shift |
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
200000 updates의 target 예산 상한은 1.8B(60×150×200000)이며 짧은 tail/reset을 반영한 실제 수를 기록한다.

checkpoint는 model/Adam/scheduler/RNG뿐 아니라 rank별 history, detached boundary states,
stream offset, 실제 target token 수, manifest와 best loss를 저장한다. 동일 설정/동일 world_size로만 재개한다.
validation은 dropout off로 순차 state를 유지하고 training RNG를 보존한다. best는 validation mean NLL로 선택한다.
DDP 학습은 stream을 rank별로 나누며 평가 전체 stream은 각 rank에서 동일하게 읽는다. 평가를 중복 합산하지 않는다.

`--benchmark-steps`는 WikiText에서도 실제 replay/full softmax/Adam/gradient norm clip을 측정한다.
microbatch를 줄일 때 accumulation을 늘려 global batch를 유지한다. 출력 calibrated_override를
새 YAML로 저장해 본 학습 명령의 마지막에 추가한다. 학습 설정을 자동으로 덮어쓰지 않는다.
평가 batch는 자동으로 변경하지 않으며 평가에서도 메모리 여유 조건을 만족해야 한다.
측정 예상 시간은 checkpoint I/O/데이터 준비/마지막 test 시간을 제외한다.

WikiText는 Adam betas=(0.9,0.999), eps=1e-8, weight decay=0이다. Remember는 기존
AdamW/value clip/linear schedule/curriculum을 재사용한다. 두 과제의 optimizer 프로토콜을 혼동하지 않는다.

고유 Parameter 수(optimizer에서 tied weight 중복 없음, buffer 제외):

| 크기 | 공통 backbone | RMT 전체 | GRT 전체 |
|---|---:|---:|---:|
| WikiText base | 148175210 | 148179310 | 150378764 |
| WikiText small | 73279232 | 73281792 | 74343956 |
| WikiText small GPT-2 BPE | 17604864 | 17607424 | 18669588 |

small의 D256×267735 tied vocabulary는 68540160 parameters이며 전체 backbone은 약 73.28M이다.
30M 예산은 적용하지 않는다. full logits 한 segment만으로도 base global batch60에서
약 8.98 GiB(fp32)이며 gradient/activation/Adam은 추가다. preset microbatch60은 논리 배치 기준이고
실제 GPU에서는 사전 측정과 accumulation override가 필요하다.

새 GRT Remember에도 RTLA trace를 지원하며 query를 포함한 모든 segment의 stack 뒤 write-back을 기록한다.
WikiText 자동 RTLA 샘플링과 인코더 MLM objective는 지원 범위 밖이며 해당 설정을 명시적으로 거부한다.
최신 Kaggle 노트북은 small WikiText의 GPT-2 BPE/word-level 실행용이다. 기존 Remember 노트북은 archive에 보존한다.

## 9. 검증 결과와 실행 전 남은 항목

공통 구조 구현 시 `python -m pytest -q`: **51 passed**, 93.96초(CPU). 새 경로 검사 18개를 포함한다.
GPT-2 BPE와 WikiText Kaggle 노트북 추가 후 전체 CPU 회귀는 **54 passed**, 108.49초다.
후속 Hugging Face 기본 다운로드 경로 변경에서는 사용자 지시에 따라 추가 학습·회귀 검사를 실행하지 않았다.
추가 검사 범위와 실제 tokenizer/GPU 미검증 조건은 [BPE 안내](20261011_wikitext_bpe.md)의 검증 절을 따른다.

- vendor 원본 hash와 저자 RMT loss/gradient/generate fixture 회귀.
- 두 모델 backbone 초기 weight 일치, tied Parameter 공유 및 save/load 후 유지.
- Remember 정답 제외 입력, value_size=1/2, slot-aligned CE, fact→query gradient,
  독립 sampling RNG, segment별 stack/write-back 횟수, read off의 출력 불변.
- fixed_state/read=false와 RMT의 미래 token 변경 불변 및 nonzero incoming memory에서
  가변 길이 prefix 불변. intermediate의 register 우회 경로 영향 확인.
- replay 이전 detached state에는 gradient 없음, replay의 이전 3 segments에는 gradient 전달.
- encoder Remember 학습과 정확한 재개, WikiText 두 모델의 dropout 포함 정확한 재개.
- WikiText boundary target 수/NLL 정렬/PPL, CPU 2-rank 학습·state checkpoint·preflight,
  microbatch×accumulation×world_size 예산 유지, 인코더 LM 목적의 명시적 거부.

별도 CLI로 새 GRT 인코더의 4 update 학습, checkpoint 평가와 NPZ/JSON/PNG trace 생성을 확인했다.
기능 확인용 결과는 `runs/neox_encoder_check/`, `runs/neox_encoder_trace/`에 있다.
4 update의 exact match는 0으로, 이 검사는 학습 경로 확인이며 수렴 결과가 아니다.
`git diff --check`와 Python compile 검사도 통과했다.

GPU 수렴, 실제 WikiText-103 전체 데이터 학습, GPU VRAM/throughput은 아직 검증하지 않았다.
기본 intermediate의 teacher-forced_noncausal PPL을 엄격한 autoregressive RMT PPL과
동일 조건으로 해석해서는 안 된다. 정확한 RMT 2022 Transformer-XL/adaptive-softmax 재현도 아니다.

후속 small D256 변경과 실제 인터페이스 호출 결과: [CPU 실험 보고서](20261011_small256_cpu.md).
Remember 4조건과 full-vocabulary 합성 LM 2조건을 실행했다. 본 WikiText 데이터 결과와 구별한다.

## 10. 출처

원 설계: [20260608 spec](archive/20260608_spec.md).
이전 구현 기준: [20261010 design](archive/20261010_design.md).
원 논문: https://papers.neurips.cc/paper_files/paper/2022/file/47e288629a6996a17ce50b90a056a0e1-Paper-Conference.pdf
현재 운영 코드: src/grt/models, data.py, trainer.py, evaluator.py.
