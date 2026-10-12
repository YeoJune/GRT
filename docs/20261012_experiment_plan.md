# GRT 메모리 활용성 우선 실험 계획

작성일: 2026-10-12 · 검토 코드: `a3f13dc` · 상태: 사용자 지시를 반영한 단계별 계획

[기존 하이퍼파라미터 탐색 보고서](20261012_hyperparameter_search_report.md)를 바탕으로 후속 지시를 반영했다. **small·segment150·10k baseline → segment를 고정한 GRT의 메모리 활용성 탐색 → 활용 후보의 성능 개선 → baseline과 공통 조건 탐색 → size 확장** 순서로 진행한다. 기존 보고서의 품질 우선 순서와 초기 L/M grid는 이번 순서로 대체하며, 지표 정의와 비교·재개 주의사항은 활용한다.

이번 변경은 문서뿐이다. 코드·카탈로그·노트북을 변경하거나 GPU 학습·Vast.ai 대여를 실행하지 않았다. 후보값·trial 수·품질 허용 폭은 제안이며 최적값이나 실측 결과가 아니다. 구현 기준은 [통합 명세](20261011_current_spec.md), 실행 계약은 [실험 workflow](20261011_experiment_workflow.md)를 따른다.

## 1. 확정 방향과 계획상 가정

| 구분 | 내용 |
|---|---|
| 사용자 지정 | small, segment=150, 10k baseline 선행; GRT 고유 설정 → 공통 조건 → size 확장 |
| 사용자 지정 | 메모리 활용성 우선, 활용성이 좋은 후보의 성능 개선; 극단적인 용량과 WB 게이트·어텐션 변형 포함 |
| 사용자 지정 | Kaggle 주30시간, Colab 하루4~5시간; 유료 RTX4080급 Vast.ai 계획은 별도 관리 |
| 현재 문맥을 따른 가정 | 주 과제는 WikiText-103-v1, GPT-2 BPE; ours는 현재 fixed-state GRT decoder |
| baseline의 우선 해석 | 실행 가능한 RMT를 1차 baseline으로 사용. 메모리 자체의 효용을 주장할 때 동일 NeoX의 메모리 없는 decoder도 필요 |
| 예산의 의미 | `10k`는 10,000 optimizer updates. 10,000 token/샘플이라는 뜻이 아님 |

출발점은 D256·6 layers·8 backbone heads·FF1024, L150·M10·이전 BPTT K3, global batch B256, Adam LR5e-4, cosine/warmup0, clip0.25, residual dropout0.1, attention dropout0, FP32다. GRT는 router hidden512, pooling heads8, write bias=-2, write gate dropout0, read gate off다. tokenizer·prepared 데이터 hash와 evaluation batch10·stream partition을 공통으로 유지한다.

이 조건의 supervision 상한은 `256 × 150 × 10,000 = 384M target tokens`다. 이전 gradient 범위는450 tokens, 현재까지 재계산하는 범위는 최대600 tokens다. 실제 학습량은 tail을 반영한 `target_tokens_seen`을 기록하고 replay token은 계산 비용으로 따로 센다. 기존 PPL 약34는 metadata 확인 전까지 참고 관측으로만 둔다.

## 2. 전체 진행 순서

| 단계 | 할 일 | 종료 시 남길 결과 |
|---|---|---|
| A. 기준점 | RMT small/L150/M10/K3/10k 먼저 학습 → 동일 조건 GRT 기준 run | checkpoint·validation·GRT RTLA·실측 시간/VRAM |
| B. 활용성 탐색 | L150 고정, GRT의 M·write bias·router 용량·WB heads·read gate 순차 탐색 | 과거 문맥 기여가 재현되는 설정2~3개, 실패/비효율 영역 |
| C. 성능 개선 | B의 후보에서 LR·dropout·K와 필요한 WB 변형을 좁혀 탐색 | 활용성을 유지하면서 NLL을 낮춘 후보1~2개 |
| D. 공통 조건 | 후보를 고정하고 L·BPTT 등 변경; RMT와 대응 탐색 | 동일 target 예산의 L별 baseline/GRT 비교 |
| E. 확인·확장 | seed 반복 → small에서 base로 이동, 필요한 길이/용량 확인 | 품질·활용성·비용의 재현성과 확장 결과 |

전체 조합을 곱하지 않는다. 한 축씩 방향을 찾은 뒤 **M×write bias, M×router 용량, gate×pooling 방식**의 중요한 결합만 확인한다. 단일 축의 결과는 후속 탐색 출발점이지 최종 최적값이 아니다.

## 3. A단계: baseline을 먼저 고정

1. 데이터·commit·seed54·resolved Config를 고정하고 해당 조건에서 preflight를 한다.
2. RMT를 small/L150/M10/K3/10k로 먼저 완료한다. validation 곡선, best/last checkpoint, 처리 속도와 VRAM을 보존한다.
3. 같은 조건의 GRT 기준 run을 완료한다. RTLA를 켜 normal/reset/swap을 validation마다 기록한다.
4. 같은 checkpoint 선택 규칙으로 비교한다. best를 쓰면 해당 best에서 validation과 RTLA를 다시 함께 측정한다.

첫 두 run이 불안정할 때만 LR2.5e-4/5e-4/1e-3의 제한된 점검을 추가한다. 먼저 기준점을 얻고, 이후 GRT 튜닝 결과를 비교할 때 RMT에도 대응 LR·M·K 탐색 예산을 배정한다. 공통 학습 조건 비교와 모델별 tuning 후 비교를 함께 보고한다.

**현재 RMT/GRT만 지원하고 메모리 없는 NeoX는 미구현이다.** `num_registers=0`은 Config가 거부하고 state 초기화 개입은 메모리 없이 학습한 baseline을 대신하지 못한다. RMT 대비 연구는 현재 비교군으로 시작할 수 있지만, 메모리 없는 모델보다 유용하다는 주장은 local NeoX 구현·학습 후 확인한다. 필요하면 학습부터 state를 매 segment 갱신하지 않는 static-register control을 추가한다.

탐색 전에 test 격리를 준비한다. 현재 `train_stream`은 정상 종료와 시간 제한 종료 모두 `best.pt`의 test를 평가하고 summary 최상위 `loss/perplexity`도 test 값이다. test 평가를 선택적으로 생략하는 경로가 필요하며 HPO 순위에는 명시적인 validation 값만 사용한다. 이번 보고서에서 해당 코드를 고치지는 않았다.

## 4. B단계: 활용성을 먼저 찾는 GRT 탐색

### 4.1 활용성 판단

주 지표는 `Δ_swap = NLL(같은 시점의 다른 stream state) − NLL(정상 state)`다. `Δ_reset = NLL(학습된 S0로 초기화) − NLL(정상 state)`를 함께 본다. gate·state norm·실제 update norm·slot cosine은 진단용이며 활용 점수 자체로 최대화하지 않는다. write pooling attention은 token이 memory를 읽는 attention과 다르다.

순서는 **동작/gradient 확인 → 과거 문맥 기여 확인 → 기여가 있는 후보의 품질 개선**이다.

| 관측 | 처리 |
|---|---|
| swap/reset 효과가 여러 window에서 재현되고 정상 품질도 양호 | C단계 주 후보 |
| memory 효과가 강하지만 현재 품질이 낮음 | 개선 가능한 기전 연구 후보1개까지 별도 보존 |
| gate/norm만 크고 delta는 미미하거나 reset만 큼 | 더 긴 warmup·다른 window로 재확인 후 우선순위 하향 |

Δ_swap=0.01 nats/token은 개입 PPL 약1% 악화에 해당하는 효과 크기 예시다. 확정 성공선은 pilot의 window/seed 변동을 보고 정한다. 주 후보의 임시 품질 범위는 RMT reference PPL 대비 +5% 이내를 제안한다. 초기 기전 탐색용 제안이며 최종 후보에는 동등하거나 더 좋은 품질을 목표로 한다. 큰 delta와 나쁜 정상 품질만으로 유용한 기억을 주장하지 않는다.

후속에는 정상 GRT가 local NeoX보다 좋고 memory 교란으로 그 이득이 줄어드는지 확인한다. slot 개별 효과가 작아도 중복·분산 저장일 수 있어 큰 M에서는 묶음 개입도 검토한다.

### 4.2 현재 Config로 가능한 탐색

L150·small·B256·10k·K3을 유지한다. 기준과 같은 run은 재사용한다.

| 순서/축 | 후보 제안 | 확인할 질문 |
|---|---|---|
| 1. memory M | 10 → 40 → 160 → 512 | token 길이보다 큰 memory까지 활용이 늘어나는가? |
| 2. write bias | -4 / -2 / 0; 포화 진단 시 +2 | 보존/갱신 성향이 활용을 늦추거나 정보를 지우는가? |
| 3. router hidden | 128 / 512 / 1024 | 큰 memory에서 router가 병목인가? |
| 4. WB attention heads | 1 / 4 / 8 / 16 | 동일 D256의 head 분할이 pooling·state 선택에 도움이 되는가? |
| 5. read gate | off / on | 현재 gate가 logits·다음 state에 실제 영향을 주는가? |
| 6. 필요시 gate dropout | 0 / 0.05 / 0.1 | 과도한 갱신/과적합이 있는 후보에 도움이 되는가? |

필드는 각각 `model.num_registers`, `model.register.write_gate_bias_init`, `model.router.mlp_hidden`, `model.router.pool_nhead`, `model.register.conditional_read`, `model.register.dropout_prob`다. bias의 sigmoid는 대략2%/12%/50%/88%지만 실제 gate는 입력과 weights를 포함해 실측한다.

용량 효과가 포화되면5/20/80/320 등으로 경계 주변을 좁힌다. M512가 preflight에서 너무 느리거나 맞지 않으면256/320에서 상한을 찾고512를 보류한다. 최초에는 M만 바꾸고 hidden512를 고정한 뒤 큰 M에서 gate 용량 부족이 의심될 때 hidden을 넓힌다.

WB heads는 backbone heads8과 별개다. 현재 `pool_nhead` 하나가 token pooling과 state cross-attention 모두를 바꾼다. heads를 늘려도 총 attention 차원 D256이나 projection parameter 규모가 커지지 않으므로 attention 차원 확장으로 해석하지 않는다.

기준 이후 **대략10~14개 추가 조건**을 1차 상한으로 제안한다. 용량3개, 선택 용량의 bias2~3개, hidden2개, heads2~3개, read on1개 정도를 결과에 따라 배분한다. 전부를 한 번에 예약하거나 각 축을 전조합하지 않는다. 활용 후보2~3개와 강한 기전의 진단 후보 최대1개를 남긴다.

### 4.3 극단적인 memory의 목적과 비용

M은 저장 slot 수, VRAM은 실행 장치 메모리다. small/global B256/FP32의 **state 한 벌만**의 합산 크기는 `B×M×256×4 bytes`다.

| M | M/L | GRT 위치 L+M | RMT 위치 L+2M | global state 한 벌 |
|---:|---:|---:|---:|---:|
| 10 | 0.067 | 160 | 170 | 2.5 MiB |
| 40 | 0.267 | 190 | 230 | 10 MiB |
| 160 | 1.067 | 310 | 470 | 40 MiB |
| 512 | 3.413 | 662 | 1174 | 128 MiB |

이는 GPU peak 예측값이 아니다. rank별 stream 분배, history/boundary 복제, replay activation, attention, optimizer, softmax와 평가 메모리가 따로 든다. M512 GRT의 입력 길이 제곱은 M10 대비 약17배지만 전체 시간/FLOPs가17배라는 뜻은 아니다. RMT는 read/write 블록이 각각 있어 같은 M의 비용을 따로 측정한다.

극단 용량에서는 활용 발생 여부, router/credit 병목, slot 중복과 포화를 조사한다. 매 M마다 peak allocated/reserved VRAM·target tokens/s·wall time을 preflight와 본 run에서 기록한다. OOM은 나쁜 품질 대신 실행 불가로 기록한다. auto_batch는 global256을 유지하며 microbatch/accumulation만 바꾸고 micro1 또는 평가 batch10도 맞지 않으면 그 장치에서는 보류한다.

M512의 전체 slot 개입은 수백 번의 추가 forward가 필요하므로 normal/reset/swap과 일부 slot부터 확인한다. 묶음 개입·slot sampling·read attention 계측은 현재 자동 제공되지 않는 후속 분석 작업이다.

### 4.4 WB 구조 변형: 구현 후 별도 탐색

현재 WB는 single learned query의 token attention pooling → state cross-attention → register별 sigmoid scalar gate이며 최종 candidate와 state를 `S_next=(1−W)S+WC`로 섞는다. 다음은 단순 override로 실행할 수 없는 후보다.

| 변형 축 | 후속 후보 | 목적·검증 |
|---|---|---|
| attention 총 차원 | router 전용 projection D_attn=64/128/256/512 | backbone D256과 head 수를 고정한 차원·계산 비교 |
| pooling 방식 | mean pooling control / single-query / multi-query / slot별 query | 전역 요약 하나가 slot 다양성을 막는지 확인 |
| 두 attention의 head | token pool과 state cross-attention의 head를 분리 | 현재 공유 설정의 두 효과를 분리 |
| write gate 해상도/방식 | slot-scalar / slot×channel sigmoid / 제한된 write budget의 slot 경쟁 gate | 선택적 갱신·보존, 실제 update·gradient·안정성 확인 |
| read gate 위치 | 현재 state-LN 이전 / 정규화된 memory 경로나 읽기 기여에 적용 | scalar gate의 LN 상쇄를 분리 |
| write gate 입력 | 현재 raw embedding pool / 완료된 segment hidden pool | 내용 기반 갱신 효과; write는 다음 segment에만 반영 |

현재 `R*S`는 layer별 LayerNorm을 거쳐 양의 scalar 크기가 대부분 상쇄될 수 있다. gate 값으로 읽기 강도를 해석하지 않고 동일 incoming state의 logits와 next state 차이를 먼저 진단한다.

Config 탐색에서 드러난 병목에 따라1~2개 축만 구현하고 기존 구조와 paired 비교한다. 동시에 M·LR·L을 바꾸지 않으며 parameter 수·추가 비용을 기록한다. **token은 segment 시작 state만 읽고, read gate도 시작 state만 사용하며, write-back은 stack 뒤 한 번**이라는 AR 계약을 유지한다. candidate 중간 hidden을 token K/V로 되돌리는 방식은 후보에 넣지 않는다.

구조 변경 뒤 미래 token 변경 불변성, 병렬/prefix 출력·종료 state 동등성, 다음 segment loss→write gradient를 작은 CPU 검사로 검증하고 새 run을 사용한다.

## 5. C단계: 활용성을 유지하며 성능 개선

B의 활용 후보2~3개에서 LR2.5e-4/5e-4/1e-3를 중심으로 좁힌다. 이후 dropout0/0.1/0.2, K1/3/6 중 필요한 축을 탐색한다. K 변경은 과거 credit 범위와 replay 비용도 바꾼다. clip0.25는 우선 유지하고 clipping 빈도 계측이 있을 때0.5/1.0을 검토한다. 작은 초기 write gate가 memory 효과를 늦출 수 있어 초반 delta만으로 제거하지 않는다.

**K0은 일반 GRT 후보에서 제외한다.** stream state는 다음 update 전에 detach하고 현재 loss는 현재 write candidate를 읽지 않는다. K0이면 다음 segment loss가 이전 write-back으로 돌아가지 않는다. read on에서 공유 router 일부가 학습되더라도 write credit 문제는 남는다. K0은 별도 negative control로만 다룬다.

선택 checkpoint에서 Δ_swap/Δ_reset도 재측정한다. 품질만 좋아지고 활용이 사라진 후보는 별도 기록한다. 최종1~2개와 RMT에는 같은 seed 목록으로 최소3 seeds를 권장하며 첫 screen은 seed54로 시작한다. 품질·재현되는 문맥 기여·wall time/VRAM으로 후보를 좁힌다.

10k 끝에도 학습 곡선이 빠르게 개선되면 baseline과 후보를 함께 더 긴 예산으로 새로 학습할지 결정한다. cosine을 소진한10k checkpoint의 max_steps만 늘리는 것은 exact-resume가 아니고 schedule도 달라진다. 처음부터 긴 최종 schedule을 고정한 promotion을 원하면 별도 execution stop 인터페이스가 필요하다.

## 6. D단계: segment와 baseline 공동 탐색

GRT M/WB 후보1~2개를 고정하고 L150/75/50을 비교한다. L25는 더 강한 병목이 필요한 경우, L170~172는 원 논문의 word150에 평균 문맥량을 맞추려는 별도 참고 조건이다. RMT도 같은 L에서 기준 M10과 대응 용량/최적화 후보를 평가한다. 같은 slot 수는 총 비용·구조 동등성이 아니므로 비용을 병기한다.

주 비교는 B256·target 예산384M 고정이다. 모든 L에서10k만 실행하면 짧은L의 학습량이 줄어든다.

| L | target/update 상한 | 384M의 updates | K3 과거 범위 | 과거450 tokens의 K | RTLA warmup/측정 segments |
|---:|---:|---:|---:|---:|---:|
| 150 | 38,400 | 10,000 | 450 | 3 | 3 / 8 |
| 75 | 19,200 | 20,000 | 225 | 6 | 6 / 16 |
| 50 | 12,800 | 30,000 | 150 | 9 | 9 / 24 |
| 25 | 6,400 | 60,000 | 75 | 18 | 18 / 48 |

tail을 포함한 실제 target count를 확인한다. steps와 validation 간격을 명시하여 cosine·평가를 같은 token 진행률에 맞춘다. L75의 2,000, L50의 3,000 updates마다 평가하면 L150의 1,000과 같은 token 간격이다. K3 고정의 저비용 screen은 L과 gradient token 범위가 함께 바뀐 비교로 기록하고 최종 후보에서 K3/6/9의 450-token 통제 비교를 한다. L25/K18은 필수로 두지 않는다.

RTLA는 동일 prepared stream·batch8·절대 offset·대상 token을 사용한다. 위 표는 450-token warmup과 1,200-token 측정을 맞추는 예시다. 큰 M에는 더 긴 warmup도 확인한다. horizon도 segment 수 외에 token 거리로 비교하며, 경계 횟수/위치 차이는 구조 효과로 병기한다.

L별 좁은 LR 재탐색, M 고정과 M/L 비율 고정의 소수 비교, write bias 재확인을 baseline/GRT에 배정한다. 짧은 L에서는 token당 write 횟수가 늘어 같은 gate 값이 같은 기억 수명을 뜻하지 않는다. B 변경은 stream partition까지 바꾸므로 이후 별도 실험으로 둔다.

## 7. E단계: size 확장과 최종 확인

small 후보를 고정하고 base(D410·16 layers·10 heads·FF2100)로 옮긴다. tiny는 기능/속도 점검이며 small의 주 순위를 대신하지 않는다. size×L×M×WB 전체 grid는 실행하지 않는다.

1. base RMT/GRT를 같은 L·target 예산으로 비교한다.
2. router hidden의 절대값 고정/D 비례, M의 절대값/M/L 방침을 명시하고 이전이 나쁘면 소수 재탐색한다.
3. base의 WB heads는 D410을 나누는 값이어야 한다. small의 4/8/16은 그대로 못 쓰고 1/2/5/10 등을 검토한다.
4. 같은 token 예산의 품질 비교와 같은 시간 예산의 비용 비교를 구분하고 memory 효과·VRAM·tokens/s 변화를 확인한다.

base는 width/depth가 동시에 바뀌므로 순수 width scaling이라고 해석하지 않는다. 최종 후보와 baseline의 seed 반복 뒤 test를 평가하고 test를 보고 재선택하지 않는다. local NeoX와 긴 raw-context baseline은 주장에 필요한 범위로 추가한다.

## 8. 무료 리소스 운영 계획

사용자가 제시한 시간을 계획 상한으로 사용하며 실제 GPU 부여·session 상한·quota 단위는 실행 화면에서 확인한다. GPU-hours와 경과 시간을 구분한다. Kaggle T4×2의 한 DDP trial은 2 GPU를 점유하므로 같은 방식의 2 trial 병렬 실행은 불가하다.

| 환경 | 우선 배치 | 시간 배분 제안 |
|---|---|---|
| Kaggle 주30h | baseline, small10k 주 비교, 안정적인 긴 run | 주 학습20h, validation/RTLA/준비·저장4h, 중단/실패 여유6h |
| Colab 하루4~5h | 새 조건 preflight, WB 짧은 진단, 독립 small run, checkpoint 사후 RTLA | 한 session에 완결되는 계측 우선; 학습은 실측 후 시간 내 가능성 판단 |
| 공통 | prepared 데이터·로그·checkpoint 외부 저장 | Config/hash/world_size를 확인하고 같은 조건으로 재개 |

Colab을 매일 쓸 수 있으면 주28~35h 상당이지만 매일 GPU 보장으로 보지 않는다. Kaggle30h가 GPU 시간인지 notebook 경과 시간인지도 확인한다. 경과 시간이면 T4×2의 30h는 60 GPU-hours다. 위 배분은 quota 확인 전의 경과 시간 예시다.

각 조건에서 `h_run=(updates×초/update + validation + RTLA + 저장·준비)/3600`을 추산하고 전체 실측으로 보정한다. `가능 trial 수≈주 학습 시간/h_run`으로 주간 조건 수를 정한다. 가령 6h/run이면 주 학습20h에 약3본, 12h/run이면 약1본으로 baseline2본부터 여러 주가 걸릴 수 있다. 이는 예시이며 현재 ETA가 아니다.

첫 무료 시간은 preflight와 RMT baseline, 다음은 GRT 기준 run에 쓴다. 이후 주별로 용량 → bias/router → 활용 후보 성능 개선을 배정하고 이전 결과를 본 뒤 다음 군을 실행한다. 추가10~14조건을 한 주에 끝내는 계획으로 잡지 않는다. Colab에서는 진단/분석을 병행한다.

### 세션 중단·재개

- Kaggle world_size2 run을 Colab world_size1로 exact-resume할 수 없다. 독립 trial 또는 checkpoint 평가·분석으로 분담한다.
- 같은 global256이라도 보정된 microbatch/accumulation은 다를 수 있다. GPU가 바뀌어 원래 batch가 맞지 않으면 새 run에서 측정한다.
- 현재 `max_seconds`는 run 누적 시간 상한이며 Config에 저장된다. 상한에 도달한 checkpoint는 같은 상한으로 재개해도 바로 멈추고, 값을 바꾸면 Config 일치 검사에 걸린다. 매일4h씩 재개하는 session별 상한 기능이 아니다.
- 실행 시간/step 구간의 graceful stop을 immutable Config와 분리하고 저장·resume·종료 status를 검증하는 작업이 필요하다. 그전에는 시간 내 완결 run/분석을 우선하며 강제 kill의 완전 복원을 약속하지 않는다.
- checkpoint는 현재 validation마다와 종료 시에 저장한다. 갑작스러운 종료는 저장 간격만큼 학습을 잃을 수 있다. prepared 데이터, last/best, Config, manifest, metrics를 session 밖에 보존한다. W&B만으로는 학습을 재개할 수 없다.

## 9. 유료 Vast.ai RTX4080급 별도 계획

**무료 시간에서 후보와 비용 병목을 파악한 뒤 필요한 run만 유료로 옮기며 현재는 대여하지 않는다.** 일반 RTX4080은 VRAM16GB여서 T4보다 큰 용량이 아니다. 속도 상승은 기대하되 M512/base의 OOM 해결은 보장하지 않는다. 용량 부족이면 더 큰 VRAM 장치와의 비용 비교를 따로 검토한다.

| 투입 구분 | 사용 내용 | 선행 조건 |
|---|---|---|
| P0. 계측 | 선택된 small/base/큰 M의 preflight, 저장/복원 확인 | 단가·실제 VRAM·volume·비용 상한 확인 |
| P1. 우선 사용 | 무료 session에서 완결하기 어려운 유망 run, 후보와 baseline의 seed 반복 | 무료 환경에서 활용 효과 확인, 총시간 실측 견적 |
| P2. 확장 | base 대조 비교, 극대 M, WB 추가 구조 | 16GB 적합 확인, 가설·비교 대상·중지 조건 명시 |

small GRT 후보1본+대응 RMT1본 또는 seed 보완을1단위로 시작하고 결과를 본 뒤 다음 단위를 정한다. 한쪽만 길게 학습해서 품질 우위를 주장하지 않는다. 대여 전에 데이터/tokenizer·commit·의존 환경·trial 표를 준비한다.

가격은 미확인이므로 금액은 단정하지 않는다. `총비용=GPU 시간 단가×청구 시간+storage+전송 등 추가 비용`, `청구 시간≈계측+Σrun 시간+저장+여유20%`로 견적한다. 실제 offer의 idle 과금·정지/삭제·storage 조건을 확인하고 run 상한과 주/월 상한을 분리한다. 결과를 외부에 저장한 후 과금 대상을 정지/해제하는 절차도 준비한다.

현재 Vast 전용 runtime은 없다. single GPU는 기존 Colab runtime의 batch/schedule을 활용하되 world_size1, 작업 path, Colab Secret에 의존하지 않는 W&B 인증을 명시하고 새 run을 쓴다. Kaggle2GPU checkpoint의 학습 재개는 계획하지 않는다. 필요시 환경 설정만 다루는 runtime을 추가한다. FP32 비교를 유지하고 mixed precision은 별도 속도/수치 검증으로 둔다.

## 10. 추가로 사용자가 결정할 사항

지정된 순서와 small/L150/10k는 확정이다. 아래는 실험 시작/다음 단계에 앞서 정할 항목이다.

| 시점/결정 | 권장 초기안 | 영향 |
|---|---|---|
| 시작 전: baseline 범위 | RMT 선행, local NeoX는 메모리 효용 주장까지 추가 | 구현량·주장 범위 |
| 시작 전: 데이터/tokenizer | WT103-v1 전량, 현재 GPT-2 BPE artifact 고정 | PPL/token 예산 비교; subset은 별도 새 실험 |
| 시작 전: 무료 quota 단위/지속성 | Kaggle30h 계수와 Colab 가용일 확인 | 실행 수·달력 계획 |
| 활용 screen 전: 품질 허용 폭 | 주 후보 RMT PPL+5% 이내, 강한 기전 후보1개 별도 보존 | 활용 우선의 범위 |
| pilot 후: 최소 효과와 반복 | window 변동에서 Δ_swap 기준 결정, 최종3 seeds | 미세 차이의 성공 판정 |
| WB 구현 전: 변형 범위 | M/bias/hidden/heads 먼저, 구조 추가1~2축 | pooling/gate/projection 우선순위 |
| 유료 투입 전: Vast 예산 | 1회 상한·주/월 상한·통화, 대조 비교1조부터 | 대여 시간·더 큰 VRAM 대안 |
| C/D: 튜닝 공정성 | 공통 조건/모델별 튜닝 비교 병기, trial 수와 GPU-hours 모두 기록 | baseline 재탐색 예산 |
| 확장 전: 최종 목표 | 동품질의 활용 확보 → 품질 개선·비용 평가 | PPL 우위/기억 기전/효율 중 주장 초점 |

100k까지 갈지, base를 필수로 할지, 최종 test 반복 수는 small 결과와 비용 실측 뒤 결정한다. 큰 memory를 탐색에 포함하지만 최대 M을 최종 모델로 삼는 것은 전제하지 않는다.

## 11. 놓치기 쉬운 점과 준비 작업

| 항목 | 대응 |
|---|---|
| test 선택 오염 | 종료 test 생략 필요, summary 최상위 PPL을 HPO 목표로 쓰지 않음 |
| best PPL+last RTLA 혼합 | 동일 checkpoint에서 재평가 |
| 짧은 L의10k 학습량 감소 | D단계 target 수 고정과 steps/평가 주기 명시 |
| K0 write credit | 주 탐색 K≥1, K0는 negative control |
| read gate LN 상쇄 | logits/next state/개입 NLL로 진단 |
| 큰 M 중복·분석 비용 | 필요시 sampling/묶음 개입 추가 |
| probe 과적합 | 고정 monitor와 추가 held-out validation windows 분리, 마지막만 test |
| 불확실성 과소평가 | 현재 arrays는 batch 평균; window/block 반복 집계 추가 검토 |
| 문서 경계 오인 | EOS는 원문 줄마다, reset은 split/epoch; 기사 경계 metadata 필요 |
| 10k→긴 run 연장 | Config/schedule 변경은 새 run, promotion은 execution stop 설계 |
| 환경 간 resume | world_size/calibration 차이 확인, 독립 trial/분석 분담 |
| base WB head | D410 divisibility에 맞춰 변경 |

최소 준비는 test 격리, checkpoint별 validation/RTLA 재평가, 외부 저장과 필요시 session별 graceful stop이다. local NeoX·추가 WB·RMT 개입·multiwindow 집계는 목적에 따라 추가한다. 현재 RTLA는 GRT 전용이므로 RMT memory delta도 비교하려면 확장이 필요하며 미계측을0으로 보고하지 않는다.

초기 HPO 관리는 Config override, run 기록표, W&B로 충분하다. 자동 runner/Optuna는 실패·중단·resume·checkpoint 선택 계약을 갖춘 뒤 도입하고 관리 도구를 중첩하지 않는다.

## 12. 초기 실행 방법과 공유할 결과

[Kaggle](../notebooks/kaggle.ipynb)과 [Colab](../notebooks/colab.ipynb)의 선택 cell에서 small/10k/segment150/register10/BPTT3을 지정한다. 처음은 `models=('rmt',)`, 다음 독립 run은 `models=('grt',)`와 RTLA on이다. 다음은 현재 API의 설정 예시이며 test 격리나 session 중단 보완을 구현한 예시는 아니다.

```python
from grt.utils.kaggle import WikiTextRun

job = WikiTextRun(
    repo_dir=repo, size='small', budget='10k', models=('rmt',),
    runtime='kaggle', world_size=2,
    overrides={
        'model': {'segment_len': 150, 'num_registers': 10},
        'training': {'bptt_segments': 3},
    },
)
job.setup()
job.prepare_data()
job.preflight()  # ETA/VRAM을 확인하고 시간 내 가능한 run을 학습한다.
```

Colab은 노트북의 `runtime='colab', world_size=1, work_dir=Path('/content')`를 쓴다. GRT 선택/RTLA·M/WB 변경은 새 run이며, preflight 이후 기존 `job.train()`, 결과 확인은 `job.report()`를 쓴다.

기록할 최소 열은 trial ID·commit·data/tokenizer hash·model/size/L/M/K·WB/gate·seed·target 수·선택 checkpoint/step·validation NLL/PPL·동일 checkpoint의 Δ_swap/Δ_reset·정상 probe NLL·GPU/world_size·precision·확정 microbatch/accumulation·wall time/GPU-hours·peak VRAM·유료 비용·종료 status다. OOM/중단/발산은 구별한다.

근거는 기존 탐색 보고서, 최신 명세, `configs/experiments.yaml`, `src/grt/config.py`, `src/grt/models/router.py`, `src/grt/models/neox.py`, `src/grt/stream_training.py`, `src/grt/utils/kaggle.py`와 사후 분석 CLI의 확인이다. 새로운 시간·가격·성능 실측은 하지 않았다.
