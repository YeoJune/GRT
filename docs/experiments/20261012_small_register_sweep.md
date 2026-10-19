# small 모델의 레지스터 용량 비교 실험

실험 시작: 2026-10-12 · 갱신: 2026-10-17 · 상태: NeoX/RMT baseline과 GRT M10/M40/M160의 10k 학습·다중 window/그룹 사후 분석 완료, 선택 M 미정

[공통 실행 안내](../20261011_experiment_workflow.md) · [Colab 사후 분석 방법](../20261011_experiment_workflow.md#posthoc-colab) · [원본 JSON 보관 규칙](../../results/README.md)

## 1. 현재 결론과 결정할 점

**전체 validation 품질은 M160, 학습 비용은 M10이 유리하다. 첫 probe에서 M40의 이점은 다른 두 window에서 크게 줄어들어 probe 하나로 M을 선택하지 않는다.** 새 분석에서 M10/M40의 write 포화가 문맥을 바꿔도 유지됐고, M160은 중심화한 state·update의 표현 rank와 그룹 간 비가산성이 더 크게 나타났다. 세 유형은 재현되는 write 동작 분류지만 닫힌 슬롯의 reset과 swap 효과가 크게 달라, 비율만으로 역할·실효 용량을 확정하지 않는다.

| 사용자 결정 | 판단 / 근거 |
|---|---|
| 품질·메모리 사용 방식·비용을 고려한 선택 M | |
| M512 또는 M256/320의 추가 용량 확인 여부 | |
| 정규화/초기화 제어·긴 warmup·추가 seed 중 다음 검증 | |
| 선택 M 이후 MLP128/256/512/1024 탐색 우선순위 | |

2026-10-17 추가 확인: 세 GRT run의 bias는 모두−2였다. 같은−2를 사용한 L75/M20과의 [교차 비교](20261018_l75_gate_stabilization.md)는 안정화 보고서의 L150 참고 절에 모았다. 같은M/L에서도 gate 분할이 달라 bias나 비율 하나로 분화를 보장하지 않는다.

L150·L75의 공통 gate 지표·그림·출처는 [GPU 실측 종합 비교](20261018_l75_gate_stabilization.md#all-gate-gpu-measurements)를 따른다. 현재18개 GRT source pair를 같은 기준으로 집계하며 CPU 모델 출력·bias 교체 계산은 제외한다. 이전13개 상세 snapshot은 그 문서의 이력 링크에 보존한다.

이 문서는 이 실험 주제의 계획·결과·해석·판단을 계속 갱신하는 단일 기록이다. MLP·segment·모델 크기 탐색을 시작하면 각각 새 실험 문서에 조건과 결과를 모은다.

## 2. 질문과 실험 조건

질문은 고정된 small backbone에서 레지스터 수 M이 segment 사이의 정보 전달 용량과 압축 압력, 실제 예측 기여에 어떤 차이를 만드는가이다. M10/40/160은 segment150 대비 강한 압축·중간 용량·비슷한 크기의 state를 비교한다. register 하나를 token 하나와 대응시키거나 M≈segment를 최적점으로 가정하지 않는다. state는 여러 segment의 정보를 누적하며 K는 gradient 범위이지 기억 길이의 상한이 아니다.

| 항목 | 고정 조건 |
|---|---|
| 데이터 / tokenizer | WikiText-103-v1 / 동일 GPT-2 BPE artifact, vocabulary50,257 |
| Backbone | small D256, 6 layers, 8 heads, FF1024, tied embedding, 공통 NeoX |
| segment / 학습량 | 150 BPE tokens / 10,000 updates / 실제383,917,824 targets |
| 학습 | global256, Adam LR5e-4·beta2=.999·decay0, cosine·warmup0, clip.25 |
| 정규화 / 수치 | residual dropout.1, attention dropout0, FP32 |
| 과거 BPTT | RMT/GRT K3, 현재까지 최대4 segments 재계산; NeoX K0 |
| GRT | MLP hidden512, router attention D256/head8, read gate off, write bias−2, gate dropout0 |
| 평가 / seed | validation batch10, 1k마다 평가, model/data seed54, test 미평가 |
| GPU 배치 | T4×2, auto_batch on/예약 VRAM 목표95%, microbatch×accumulation×GPU수=global256 |

95%는 예약 VRAM의 목표 상한이며 정확히 채운다는 뜻은 아니다. 준비된 데이터·tokenizer hash, 학습 target 수, 평가 범위는 다섯 run에서 같다. validation은248,420 BPE targets를 사용하고 모든 best checkpoint는 step10,000이다.

비교 조건의 차이는 다음과 같다. M0/K0 NeoX와 K3 메모리 모델은 계산량이 다르며 parameter 수도 다르다. M10은 micro64×accum2, M40/M160은 micro32×accum4여서 dropout RNG·수치 차이가 남는다. M160 학습은 Python3.12.13/PyTorch2.10.0+cu128, 나머지는3.13.15/2.11.0+cu128이며 Transformers4.45.2/CUDA12.8은 같다. 실행 코드 commit 간 차이는 결과 문서·원본 JSON이고 학습 코드는 동일함을 확인했다. 용량만 완전히 통제한 비교나 여러 seed 평균으로 해석하지 않는다.

## 3. 학습 품질·비용과 학습 곡선

### 3.1 완료 결과

| ID | 모델 | M / K | parameters | best val NLL | best val PPL |
|---|---|---|---:|---:|---:|
| B0 | neox | 0 / 0 | 17,604,864 | 3.636507 | 37.9590 |
| B1 | rmt | 10 / 3 | 17,607,424 | 3.551781 | 34.8754 |
| R0 | grt | 10 / 3 | 18,669,588 | 3.555054 | 34.9897 |
| R1 | grt | 40 / 3 | 18,708,048 | 3.548059 | 34.7458 |
| R2 | grt | 160 / 3 | 18,861,888 | 3.530795 | 34.1511 |

RMT는 NeoX보다 PPL3.0836 낮다. GRT M10은 RMT보다0.1143 높고 M40/M160은 각각0.1296/0.7243 낮다. M160은 M40보다0.5947 낮으나 단일 seed의 결과다. 전체 품질과 아래 고정 probe의 품질 순위는 다르므로 구분한다.

| ID | micro × accum × GPU | wall h | GPU-hours | 학습 allocated GiB/GPU | preflight reserved GiB/GPU |
|---|---|---:|---:|---:|---:|
| B0 | 64 × 2 × 2 | 2.610 | 5.220 | 6.899 | 8.686 |
| B1 | 64 × 2 × 2 | 5.366 | 10.731 | 11.049 | 13.064 |
| R0 | 64 × 2 × 2 | 5.812 | 11.623 | 11.269 | 13.156 |
| R1 | 32 × 4 × 2 | 7.009 | 14.018 | 6.538 | 7.436 |
| R2 | 32 × 4 × 2 | 13.150 | 26.300 | 9.713 | 10.752 |

시간은 학습·평가·checkpoint 저장과 GRT의 RTLA를 포함하며 준비/preflight는 제외한다. GPU-hours는 시간×GPU수이며 Kaggle quota 차감량을 뜻하지 않는다. M40의 VRAM이 M10보다 작은 것은 microbatch 차이가 있어 큰 M 자체의 메모리가 더 적다는 뜻이 아니다. RMT 시간에는 GRT 전용 RTLA가 없으므로 GRT와의 차이를 순수 모델 연산 비용으로 보지 않는다. M160의 학습 시간은 M40의 약1.88배다.

### 3.2 validation 학습 곡선

| update | NeoX | RMT | GRT M10 | GRT M40 | GRT M160 |
|---:|---:|---:|---:|---:|---:|
| 1,000 | 90.4616 | 91.0054 | 123.4827 | 208.9937 | 737.4491 |
| 2,000 | 62.2051 | 60.1944 | 87.7662 | 432.5057 | 964.1900 |
| 3,000 | 50.7699 | 49.0132 | 79.3278 | 727.0352 | 1244.3458 |
| 4,000 | 46.1166 | 42.9223 | 43.1153 | 43.8405 | 45.5080 |
| 5,000 | 43.1198 | 39.9501 | 40.1375 | 40.5349 | 43.9151 |
| 6,000 | 40.6935 | 37.6279 | 37.7818 | 37.8219 | 41.9157 |
| 7,000 | 39.3211 | 36.2604 | 36.4037 | 36.3005 | 36.0280 |
| 8,000 | 38.6221 | 35.5079 | 35.6394 | 35.4525 | 35.4295 |
| 9,000 | 38.0851 | 35.0009 | 35.1090 | 34.9164 | 34.9406 |
| 10,000 | 37.9590 | 34.8754 | 34.9897 | 34.7458 | 34.1511 |

M40/M160은 초기1k→3k validation이 악화됐다가4k에 크게 회복한다. 세 GRT run은 초기 reset 효과가 음수였고3k→4k 평가 사이에 양수로 바뀐다. 짧은 pilot의 gate나 PPL만으로 후보를 너무 일찍 제외하면 이 전환을 놓칠 수 있다.

| 모델 / 구간 | write mean 전→후 | relative update 전→후 | reset ΔNLL 전→후 |
|---|---:|---:|---:|
| R0 · 3k→4k | 0.6999→1.0000 | 0.0769→0.0860 | -0.4302→0.2206 |
| R1 · 3k→4k | 0.3211→0.8250 | 0.0180→0.0766 | -1.8893→0.2527 |
| R2 · 3k→4k | 0.0671→0.3809 | 0.0077→0.0475 | -2.0430→0.2395 |
| R2 · 6k→7k | 0.3811→0.3270 | 0.0610→0.0751 | 0.2707→0.2893 |

write 변화와 품질 전환의 선후관계는1k 간격 로그로 구분하지 못한다. M160은6k→7k에 평균 write가 줄어도 품질·실제 relative update가 개선됐다. 따라서 큰 평균 write 하나를 학습 성공의 기준으로 삼지 않는다.

마지막1k PPL 감소는 NeoX0.1261, RMT0.1255, GRT M10/M40/M160이0.1193/0.1706/0.7895다. 마지막 LR0인10k cosine 결과이므로 고정 예산의 비교로 사용하며 완전 수렴을 확정하지 않는다. 더 긴 확인은 새 schedule의 새 run으로 진행한다.

## 4. 동일 checkpoint의 메모리 활용 분석

최신 확장 사후 보고서3개는 모두 이전과 같은 best step10k checkpoint다. validation·batch8/warmup3/측정8/segment150/horizon4/slots=true, window 시작0/16/32, 무작위 반복3·seed54·요청 손상 비율10/25/50/75%를 사용했다. 각 window는 원래 stream partition을 유지하고 지정 위치에서 학습된 S0로 reset한 뒤 같은 warmup을 거친다. 첫 window로 closed/open/other 그룹을 정하고 그 슬롯 묶음을 다른 두 window에서 평가한다. window별 정상 target은9,600개이며 아래 후속 평균은 calibration을 제외한 window16/32의 동일 target 수 평균이다. 합친 PPL은 NLL을 합쳐 exp로 계산하며 PPL 자체를 산술평균하지 않는다.

Colab Tesla T4 단일 GPU·FP32·TF32 off, Python3.13.15/PyTorch2.11.0+cu130/Transformers4.45.2/CUDA13.0, 분석 코드 `89dd4c2`·전체 Python source hash가 세 report에서 동일하고 저장소와 일치한다. 학습 Config는 각 학습 report와 경로까지 정확히 일치한다. checkpoint SHA256은 이전 사후 report와 같고, **첫 window의 정상/reset/swap·slot·retention·gate 통계도 이전 결과와 정확히 일치**했다. 따라서 기존 원본3개를 이를 포함하는 최신 확장본3개로 교체했다. 학습 report 자체에는 checkpoint hash가 없어 최초 학습 파일의 bytes까지 소급 증명하는 것은 아니다.

### 첫 window: 기존 결과의 재현

다음4.1~4.3과 처음 표는 window0의 관측이다. 다른 window까지 포함한 갱신 결론은4.4 이후에 구분한다.

`Δ_reset=NLL(S0 초기화)−NLL(정상 state)`, `Δ_swap=NLL(다른 stream state)−NLL(정상 state)`다. 양수는 정상 과거 state가 그 probe에서 도움이 됐다는 근거다. 큰 delta는 잘못된 state에 대한 민감성도 나타낼 수 있어 단독으로 최대화하지 않는다. baseline의 GRT 전용 RTLA 미측정값은0으로 대체하지 않는다.

| 지표 | M10 | M40 | M160 |
|---|---:|---:|---:|
| 정상 probe PPL | 32.9314 | 32.4541 | 34.0898 |
| reset ΔNLL | 0.294231 | 0.345704 | 0.323459 |
| swap ΔNLL | 0.422916 | 0.524916 | 0.294080 |
| write mean | 1.000000 | 0.825000 | 0.325191 |
| slot cosine | 0.999348 | 0.759750 | 0.529918 |
| relative update | 0.078745 | 0.083919 | 0.080673 |

probe PPL과 전체 validation PPL은 서로 다른 target 범위다. 이번 평가 환경의 통일은 이미 학습된 가중치의 학습 조건 차이까지 통제하지 않는다. relative update는 실제 갱신 norm/(이전 state norm+candidate norm)이며, candidate 변화와 gate 적용 후 실제 update를 구분한다. read gate off의 값을 읽기량으로 쓰지 않고 write pooling attention을 token→register 읽기 attention으로 해석하지 않는다.

### 4.1 gate 평균의 실체: 포화와 변화

각 슬롯에서8 segments×8 examples=64개 gate 원값을 분석했다. near-closed는<=0.01, near-open은>=0.99다. 관측 비율과 전체64개 관측에 걸친 슬롯 분류를 구분한다.

| 지표 | M10 | M40 | M160 |
|---|---:|---:|---:|
| near-closed 관측 비율 (%) | 0.00 | 17.50 | 60.13 |
| 중간값 관측 비율 (%) | 0.00 | 0.00 | 14.26 |
| near-open 관측 비율 (%) | 100.00 | 82.50 | 25.62 |
| 모든 관측에서 near-closed인 슬롯 | 0 | 7 | 77 |
| 모든 관측에서 near-open인 슬롯 | 10 | 33 | 26 |
| 위 두 고정 포화 분류에 속하지 않는 슬롯 | 0 | 0 | 57 |
| probe 내 gate 범위(max−min)>=0.1인 슬롯 | 0 | 0 | 45 |
| closed/open 양쪽 threshold를 넘나든 슬롯 | 0 | 0 | 17 |

![gate와 retention 비교](../figures/20261015_posthoc_comparison.png)

#### M10: 전면 write 포화

640개 gate 관측이 모두 정확히1이다. 이번 probe에서는 입력·segment에 따른 gate 조절이 보이지 않는다. state는 매번 candidate로 교체되지만 candidate가 이전 state를 읽으므로 장기 기억이 매번 지워진다는 뜻은 아니다. cosine≈0.99935는 state 방향의 유사성을 보여주나, 위치·norm·projection 효과가 있어 동일 기능이나 낮은 유효 rank를 직접 증명하지 않는다. 양의 reset/swap과 아래 retention은 유용한 과거 정보 전달이 존재함을 보여준다.

#### M40: 평균0.825는 사실상 고정된 슬롯 분할

33개 슬롯은64개 관측 내내 near-open,7개는 내내 near-closed다. 닫힌 slot index는 **1,2,10,24,25,31,37 (0-based)**다. 평균0.825가 모든 슬롯의 부드러운0.825 혼합에서 나온다는 해석은 배제할 수 있다. segment·example에 따른 gate 변화는 수치적으로 극히 작다(평균 표준편차 약10⁻²⁹). 이번 probe에서는 동적인 write 선택보다 거의 고정된 write 경로 분할로 동작한다.

다만 write가 닫힌 슬롯도 token이 읽을 수 있다. warmup 이전의 갱신이나 학습된 S0의 역할을 이 gate 통계로 판정할 수 없으므로7개를 불필요·미사용 슬롯으로 부르지 않는다. 또한 최종 checkpoint의 포화가 학습 초기부터 지속됐다고 추정하지 않는다.

#### M160: 포화된 부분과 변하는 부분이 공존

전체 gate 관측의60.13%는 near-closed,25.62%는 near-open,14.26%는 중간이다. 전체64개 관측에서 고정 near-closed인 슬롯은77개, 고정 near-open은26개, 나머지는57개다. 이57개 중45개는 관측 범위가0.1 이상이며17개는 closed와 open 양쪽 threshold를 넘나든다. 따라서 평균write0.325만으로 거의 갱신하지 않는 모델로 해석하기 어렵다.

평균 slot별 segment 표준편차(batch 평균의 변화)는0.02759, 동일 segment 내 example 표준편차의 평균은0.04003이다. M10/M40보다 입력·state·segment에 따라 gate가 달라지는 현상이 분명하다. 어떤 요인이 그 변화를 일으켰는지는 이 관측만으로 분리할 수 없다. 고정 분류 밖57개가 모두 유용하거나 모두 강하게 변한다는 뜻도 아니다.

![실제 example별 gate](../figures/20261015_posthoc_gate_heatmaps.png)

### 4.2 한 번의 state 교환 이후 영향과 회복

각 측정 segment 시작점에서 state를 한 번 교환하고, 이후에는 교환된 branch를 계속 진행한다. lag0은 현재 segment, lag1/2/3은 다음1/2/3 segments다. 각 lag는9,600 targets이며, 서로 겹치는 정상 target·개입 시작점을 사용하므로 lag들을 독립 표본으로 세지 않는다.

| lag | M10 ΔNLL | M40 ΔNLL | M160 ΔNLL |
|---:|---:|---:|---:|
| 0 | 0.422916 | 0.524916 | 0.294080 |
| 1 | 0.092503 | 0.121971 | 0.051419 |
| 2 | 0.027552 | 0.040327 | 0.019505 |
| 3 | 0.009960 | 0.017808 | 0.011249 |

| lag0 대비 잔여 효과 | M10 | M40 | M160 |
|---|---:|---:|---:|
| lag1 (%) | 21.87 | 23.24 | 17.48 |
| lag2 (%) | 6.51 | 7.68 | 6.63 |
| lag3 (%) | 2.36 | 3.39 | 3.83 |

세 모델 모두 교환 피해가 다음 segment에서 크게 줄고 이후에도 감소한다. M40의 절대 ΔNLL은 모든 lag에서 가장 크며 lag3에도0.017808이다. 다만 초기 피해로 정규화한 lag3 잔여 비율은 M160 3.83%, M40 3.39%, M10 2.36%로 M160이 조금 더 크다. **절대 민감도와 감쇠 비율은 서로 다른 지표이므로 M40의 기억이 무조건 가장 오래 간다고 결론내리지 않는다.**

lag3의 PPL 악화율 `exp(ΔNLL)−1`은 M10 1.00%, M40 1.80%, M160 1.13%다. 각 모델의8개 개입 시작점에 대한 batch 평균 delta는 모든 lag에서 양수였다. 이는 해당 probe에서 방향이 일관된다는 관측이며, 겹치는 token/문맥을 가진8개 시작점으로 독립 seed 검증이나 통계적 유의성을 대신하지 않는다.

이 결과는 잘못된 기억의 영향이 남아 있다가 회복되는 양상이다. 정보가 정확히 몇 segment 저장되는지, gate가 직접 복사한 state와 candidate를 거친 경로 중 어느 것이 영향을 유지하는지, 일반적인 장기 기억 능력이 얼마인지는 아직 구분하지 못한다.

### 4.3 슬롯별 초기화 기여: 현재 segment의 조건부 영향

슬롯 j의 delta는 `NLL(S의 j만 S0로 초기화)−NLL(정상 S)`다. 양수면 그 초기화가 예측을 악화시키고 음수면 이 probe 평균에서 개선시킨다. 다른 슬롯을 그대로 둔 상태의 단일 개입 효과이며 저장 정보량·읽기 횟수·독립적인 기여 지분은 아니다.

| 지표 | M10 | M40 | M160 |
|---|---:|---:|---:|
| 양의 평균 delta 슬롯 (>1e−6) | 10 | 40 | 140 |
| 음의 평균 delta 슬롯 (<−1e−6) | 0 | 0 | 19 |
| 작은 평균 delta 슬롯 (절대값<=1e−6) | 0 | 0 | 1 |
| 최소 단일 슬롯 ΔNLL | 0.00178523 | 0.00008886 | -0.00029821 |
| 최대 단일 슬롯 ΔNLL | 0.00308472 | 0.00095180 | 0.00205707 |
| 슬롯별 평균 ΔNLL | 0.00232307 | 0.00048377 | 0.00034899 |
| 8개 개입 시작점의 batch 평균이 모두 양수인 슬롯 | 8 | 2 | 2 |

1e−6은 작은 관측값을 별도로 표시하기 위한 기준이며 유의수준이나 검증된 수치 오차 범위가 아니다. M160의 작은 값1개는 실제로 약−3.11e−7이다. 여러 시작점에서 부호가 바뀌는 슬롯이 많으므로 평균 부호만으로 안정적인 역할을 확정하지 않는다.

![슬롯별 초기화와 gate](../figures/20261015_posthoc_slot_contributions.png)

#### 슬롯 기여 상위5개 · index는0-based

| M | 슬롯 | ΔNLL | mean write | probe 내 gate 분류 |
|---:|---:|---:|---:|---|
| 10 | 9 | 0.00308472 | 1.000000 | 고정 near-open |
| 10 | 8 | 0.00294604 | 1.000000 | 고정 near-open |
| 10 | 7 | 0.00271820 | 1.000000 | 고정 near-open |
| 10 | 6 | 0.00261607 | 1.000000 | 고정 near-open |
| 10 | 5 | 0.00232447 | 1.000000 | 고정 near-open |
| 40 | 0 | 0.00095180 | 1.000000 | 고정 near-open |
| 40 | 39 | 0.00081048 | 1.000000 | 고정 near-open |
| 40 | 3 | 0.00080621 | 1.000000 | 고정 near-open |
| 40 | 10 | 0.00073328 | 0.000000 | 고정 near-closed |
| 40 | 38 | 0.00072606 | 1.000000 | 고정 near-open |
| 160 | 156 | 0.00205707 | 0.999931 | 고정 near-open |
| 160 | 30 | 0.00195264 | 0.853732 | 기타/가변 |
| 160 | 157 | 0.00179062 | 0.000285 | 고정 near-closed |
| 160 | 159 | 0.00169288 | 0.999755 | 고정 near-open |
| 160 | 43 | 0.00166439 | 0.885462 | 기타/가변 |

#### gate가 닫힌 슬롯도 예측에 영향을 준다

| M / gate 그룹 | 슬롯 수 | 양 / 음 / 작은 delta 개수 | 그룹 내 평균 ΔNLL |
|---|---:|---:|---:|
| M10 / 고정 near-open | 10 | 10 / 0 / 0 | 0.00232307 |
| M40 / 고정 near-closed | 7 | 7 / 0 / 0 | 0.00031645 |
| M40 / 고정 near-open | 33 | 33 / 0 / 0 | 0.00051927 |
| M160 / 고정 near-closed | 77 | 66 / 11 / 0 | 0.00026898 |
| M160 / 고정 near-open | 26 | 26 / 0 / 0 | 0.00070713 |
| M160 / 기타/가변 | 57 | 48 / 8 / 1 | 0.00029371 |

**M40은40개 슬롯 모두 평균 delta가 양수이며 닫힌7개도 전부 포함된다.** 특히 닫힌 slot10은0.00073328로 전체4위다. 따라서 열린33개만을 실효 메모리 용량으로 세는 해석은 맞지 않는다. 닫힘은 측정8 segments에서 관측한 write 동작이다. warmup 때의 쓰기 여부는 이 trace에 없고, 이미 형성된 state는 write가 작아도 읽힐 수 있다.

**M160의 고정 near-closed77개 중66개는 양의 평균 효과**이며 slot157/158은 각각0.00179062/0.00147925로 상위권이다. 고정 near-open26개는 전부 양수지만, gate 평균이 크다고 반드시 기여 순위가 높은 것은 아니다. 낮은 평균 write를 사용하지 않는 메모리로 해석하지 않는다.

M10은10개 모두 양수이고 개별 제거 피해가 다른 두 모델의 평균보다 크다. 높은 cosine에도 모든 슬롯이 정확히 같은 역할이거나 무의미하다고 볼 수 없다. M160은 일부 슬롯에 영향이 집중되면서 다수의 작은 효과가 함께 나타난다. 양의 단일 슬롯 delta 합에서 상위 약25% 슬롯(M10은3개로 올림)의 비중은 M10 상위3/10개37.66%, M40 상위10/40개38.24%, M160 상위40/160개71.65%다. 이 비중은 단일 개입값의 분포 요약이며 전체 예측 이득의 지분이 아니다.

#### 합산·음의 delta의 해석 한계

| 지표 | M10 | M40 | M160 |
|---|---:|---:|---:|
| 단일 슬롯 ΔNLL의 단순 합 | 0.023231 | 0.019351 | 0.055839 |
| 전체 state reset ΔNLL | 0.294231 | 0.345704 | 0.323459 |

단일 슬롯 효과의 합은 전체 reset과 크게 다르다. 서로 겹치는 표현, 슬롯 간 상호작용, attention 재분배, joint reset의 비선형 효과가 있으므로 합을 전체 기억량이나 설명된 성능 비율로 쓰지 않는다. 어느 원인이 지배적인지는 그룹 개입 등으로 구분해야 한다.

M160의 가장 음수인 slot40은−0.00029821로, 이 probe에서 초기화하면 NLL이 소폭 낮아졌다. 이것만으로 해당 슬롯을 제거하면 일반화 성능이 좋아진다고 주장하지 않는다. 단일 window의 작은 효과이며 다른 문맥·seed·그룹 개입에서 확인해야 한다. 최대 단일 슬롯 제거의 probe PPL 악화율도 M10 약0.309%, M40 약0.095%, M160 약0.206%로 작다.


### 4.4 다른 문맥에서도 유형과 품질 차이가 유지되는가

| M | window | closed / open / other | calibration 유형 유지율 | 정상 PPL | reset ΔNLL | swap ΔNLL |
|---:|---:|---|---:|---:|---:|---:|
| 10 | 0 | 0 / 10 / 0 | 100.00% | 32.9314 | 0.294231 | 0.422916 |
| 10 | 16 | 0 / 10 / 0 | 100.00% | 33.9761 | 0.296579 | 0.406403 |
| 10 | 32 | 0 / 10 / 0 | 100.00% | 32.5404 | 0.274736 | 0.362157 |
| 40 | 0 | 7 / 33 / 0 | 100.00% | 32.4541 | 0.345704 | 0.524916 |
| 40 | 16 | 7 / 33 / 0 | 100.00% | 33.9671 | 0.350879 | 0.486469 |
| 40 | 32 | 7 / 33 / 0 | 100.00% | 32.4971 | 0.333732 | 0.441217 |
| 160 | 0 | 77 / 26 / 57 | 100.00% | 34.0898 | 0.323459 | 0.294080 |
| 160 | 16 | 76 / 21 / 63 | 96.25% | 35.8486 | 0.323245 | 0.301744 |
| 160 | 32 | 76 / 25 / 59 | 98.75% | 33.6877 | 0.322281 | 0.277891 |

M10은 전 슬롯 open, M40은 같은7개 closed/33개 open이 모든 window·threshold에서 유지된다. M160은 window16에서 closed1개(slot96)·open5개(slot21/100/103/128/135)가 other로 바뀌고, window32에서는 slot96/100만 other가 된다. 직접 closed↔open 전환은 없다. 세 유형의 큰 틀은 재현되지만 고정된 기능 partition이라고 확정하지 않는다.

**M160 분류는 threshold에 민감하다.** window0에서 threshold .001/.01/.05의 closed/open/other는40/1/119,77/26/57,79/32/49다. window16은31/0/129,76/21/63,76/31/53, window32는39/1/120,76/25/59,76/32/52다. .01 기준 비율은 해당 관측 정의와 함께 보고하며 자연적인 세 군집이나 최적 비율이 발견됐다고 주장하지 않는다.

M40은 첫 window에서 M10보다 PPL0.4773 낮지만 평가 window16/32 차이는 각각0.0090/0.0433으로 작아진다. 두 평가 window의 합친 PPL은 M10 33.2505, M40 33.2240, M160 34.7513다. 전체 validation의 M160 우위와 probe의 다른 순위는 공존한다. 짧은 warmup·국소 문맥 선택이 원인인지는 아직 분리하지 못하며, 전체 평가를 대체하지 않는다.

### 4.5 포화는 bias 변화보다 context·hidden 크기와 함께 나타난다

다음은 평가 window16/32의 범위 또는 평균이다. sigmoid 전 logit은 z, slope는 w(1−w)다.

| M | logit 전체 범위 | abs(z)>20 비율 | token / memory context norm 평균 | MLP hidden norm 평균 | sigmoid slope 평균 |
|---:|---:|---:|---:|---:|---:|
| 10 | 39.84 ~ 100.79 | 100.00~100.00% | 1.70 / 121.53 | 365.38 | 0.000e+00 |
| 40 | -116.67 ~ 199.53 | 100.00~100.00% | 4.49 / 156.21 | 551.06 | 4.103e-29 |
| 160 | -43.87 ~ 59.43 | 2.40~5.19% | 2.74 / 16.07 | 50.97 | 1.662e-02 |

M10은 측정한 모든 gate가 정확히1이며 이 sigmoid 경로의 국소 미분은0이다. M40도 open33개는 정확히1이고 나머지는 극히 작은 값이다. 반면 M160의 중간 gate는 작지만 비영 slope를 남긴다. 이는 평가 probe의 국소 성질이며 학습 전체의 gradient 소실이나 회복 불가능성을 증명하지 않는다.

세 모델의 최종 bias 범위는 약−2.02~−1.95로 초기−2 근처이고 출력 weight 행 norm 평균은0.679/0.702/0.700이다. 따라서 bias가 크게 이동해서만 포화됐다는 설명은 지지되지 않는다. M10/M40은 큰 memory context와 hidden activation, 큰 양/음 logit이 함께 관측된다. **router context 정규화·출력층 초기 크기를 제어하는 비교의 우선순위가 높아졌다.** 다만 서로 다른 학습 가중치의 norm 차이는 원인 증명이 아니며, 입력 내용·projection·MLP 방향성 효과를 분리해야 한다.

### 4.6 슬롯 표현의 차이: 공통 성분을 빼도 M10은 낮은 rank

평가 window16/32에서 각64개 segment/example state의 슬롯×D256 행렬을 SVD한 통계를 평균했다. 에너지 entropy rank는 p_i=σ_i²/Σσ², exp(−Σp_i log p_i)다. context-centered는 각 슬롯의64개 관측 평균을 빼고, update는 실제 S_next−S다.

| M | raw rank | state−S0 rank | context-centered rank | update rank | 슬롯 간 centered linear CKA 중앙값 범위 |
|---:|---:|---:|---:|---:|---:|
| 10 | 1.005 | 1.005 | 1.054 | 1.195 | 0.9991~0.9993 |
| 40 | 1.864 | 1.864 | 1.833 | 2.403 | 0.8078~0.8205 |
| 160 | 3.371 | 3.370 | 9.571 | 11.278 | 0.4401~0.4612 |

M10은 정적인 공통 성분을 뺀 뒤에도 rank≈1.05이고 CKA≈.999로, 관측한 문맥별 변화도 매우 비슷하다. M40은 중간, M160은 중심화한 state≈9.57/update≈11.28로 더 다양한 변화가 나타난다. 다만 M마다 가능한 rank 상한이 다르고 rank는 저장 bit 수·정확한 메모리 bandwidth가 아니다. CKA는 같은64개 관측·D256의 finite-sample 추정치이며 관측 간 상관·작은 표본 편향이 있다. 높은 유사도만으로 기능 중복을 확정하지 않고, 낮은 유사도만으로 성능 향상을 설명하지 않는다.

### 4.7 그룹 개입: 닫힌 슬롯과 문맥별 정보 전달을 구분

그룹은 calibration window0의 슬롯 index로 고정한다. 아래는 평가 window16/32의 현재 segment ΔNLL 평균이다. 괄호는 같은 개수의 슬롯을 뽑은 무작위 mask3개의 평균이며 신뢰구간이 아니다. empty 그룹은 미사용/0기여 추정과 구분해 생략한다.

| M / 그룹 | 슬롯 수 | reset (동일 크기 무작위) | swap (동일 크기 무작위) |
|---|---:|---:|---:|
| M10 / open | 10 | 0.285657 (전체 슬롯) | 0.384280 (전체 슬롯) |
| M40 / closed | 7 | 0.004055 (0.006211) | 0.000450 (0.010053) |
| M40 / open | 33 | 0.276792 (0.122428) | 0.439823 (0.312719) |
| M160 / closed | 77 | 0.063031 (0.090221) | 0.000931 (0.070793) |
| M160 / open | 26 | 0.042071 (0.025539) | 0.041587 (0.013874) |
| M160 / other | 57 | 0.095451 (0.065180) | 0.115033 (0.053405) |

M40의 open33개는 같은 크기의 무작위 그룹보다 reset/swap 피해가 크다. closed7개는 reset 피해는 작게 양수지만 swap 피해는 훨씬 작다. M160도 closed77개의 reset0.063031 대비 swap0.000931로 차이가 크며, open26개와 other57개는 같은 크기의 무작위 그룹보다 swap 피해가 크다. **gate 유형과 문맥별 개입 반응의 차이가 연결된다는 근거**가 생겼다. 다만 closed 그룹에 공통 state 성분·warmup에 형성된 상태·학습된 prefix 같은 역할이 있는지는 별도 대조가 필요하다. 낮은 swap을 미사용으로 해석하지 않는다.

비가산성도 크다. M160 open+other를 함께 swap하면0.284787로, 따로 swap한 효과의 합0.156620보다0.128168 크다. reset도0.251671 대 개별 합0.137522다. 전체 swap0.289818에서 closed를 제외한 두 그룹의 공동 효과가 크지만 이를 성능 기여 지분으로 환산하지 않는다. 중복·상호작용·attention 재분배·joint corruption의 비선형성을 이 값 하나로 구분하지 못한다.

복원 효과는 `NLL(전체 변경)−NLL(해당 그룹만 정상으로 보존)`이다. M160 swap에서 open만 정상으로 두면0.171213, other만 정상으로 두면0.246173의 회복이 관측되며 해당 그룹만 손상했을 때0.041587/0.115033과 다르다. 어떤 배경 state에서 개입했는지가 중요하다. M40의 closed7개는 단독 reset 피해가 lag0의0.004055에서 lag3의0.009568로 커지지만, 다른 슬롯이 모두 reset된 배경에서 closed만 정상으로 보존하는 복원 효과는 lag1/2에−0.044941/−0.010475로 음수다. M160 open 그룹의 단독 reset도 lag3에는−0.005032다. 이런 조건부 부호 변화는 그룹 효과를 고정적인 유익/유해 순위로 바꾸기 어렵게 하며, 작은 음수를 근거로 슬롯 제거를 권하지 않는다.

### 4.8 손상 비율과 이후 회복

평가 window16/32·무작위 반복3의 현재 segment ΔNLL 평균이다. 동일 repeat에서는 작은 mask가 큰 mask에 포함된다. 비율은 요청값이고 실제 개수는 ceil(fraction×M)이다.

| 요청 비율 | M10 reset / swap | M40 reset / swap | M160 reset / swap |
|---:|---:|---:|---:|
| 10% | 0.001937 / 0.005249 | 0.003153 / 0.002419 | 0.014961 / 0.007286 |
| 25% | 0.011728 / 0.033567 | 0.010633 / 0.017436 | 0.043416 / 0.029522 |
| 50% | 0.029428 / 0.088960 | 0.035795 / 0.099947 | 0.100648 / 0.093145 |
| 75% | 0.090241 / 0.234801 | 0.094893 / 0.230789 | 0.164691 / 0.159198 |

M10의 실제 개수는1/3/5/8개(10/30/50/80%), M40은4/10/20/30개, M160은16/40/80/120개다. 같은 비율은 같은 개수가 아니므로 M160의 reset 피해가 큰 것을 슬롯당 기여나 분화의 단독 증거로 쓰지 않는다. 곡선은 대체로 비선형이며 세 random mask의 결과로 안전한 pruning 비율을 정하지 않는다.

| 전체 state 개입 / M | lag0 | lag1 | lag2 | lag3 |
|---|---:|---:|---:|---:|
| reset / M10 | 0.285657 | 0.115679 | 0.018739 | 0.004973 |
| reset / M40 | 0.342306 | 0.166248 | 0.054247 | 0.030120 |
| reset / M160 | 0.322763 | 0.233616 | 0.131512 | 0.101067 |
| swap / M10 | 0.384280 | 0.078817 | 0.020891 | 0.006073 |
| swap / M40 | 0.463843 | 0.094583 | 0.027645 | 0.009612 |
| swap / M160 | 0.289818 | 0.048314 | 0.019244 | 0.011476 |

M160은 전체 reset의 영향이 lag3에도0.101067로 남아 M40 0.030120/M10 0.004973보다 크다. 반면 swap lag3은0.011476으로 훨씬 작다. **기억을 초기 상태로 지웠을 때의 재구성과 정상적인 다른 state를 받았을 때의 회복은 다른 현상**이다. 첫 window에서 모든 lag의 절대 swap 효과가 가장 컸던 M40의 순위도 평가 window 평균의 lag3에는 유지되지 않는다. 기억 지속 시간을 하나의 순위로 요약하지 않는다.

![다중 window·표현·그룹 개입 비교](../figures/20261015_register_diagnostics.png)

## 5. 해석과 한계

| 가설 | 이번 관측이 주는 근거 | 남아 있는 구분 |
|---|---|---|
| M10은 bandwidth가 제한된다 | 모든 window에서 포화·중심화 rank≈1.05·CKA≈.999; 더 큰 M에서 전체 품질이 개선됨 | M10도 유용한 state를 사용하며, 명목 M이 실제 정보량을 직접 재지는 않음 |
| M40이 압축과 전달량의 적절한 균형이다 | M10보다 낮은 PPL, 강한 probe 개입 효과, M160보다 짧은 학습 시간 | gate는 고정 포화 분할이고 다른 window에서 M10 대비 PPL 이점이 작아짐; 전체 품질은 M160 우위, 최적 비율 미확정 |
| M160은 압축 압력이 약해진다 | 큰 명목 용량·낮은 cosine에 더해 중심화 rank와 그룹 비가산성이 큼 | 포화된 부분·가변적인 부분이 공존하고 정상 품질도 개선됨; 과거 여러 segments를 저장하므로 무압축이라고 할 수 없음 |
| write 변화가 학습 전환을 결정한다 | 기존3k→4k 로그에서 gate·품질·reset 효과가 함께 바뀜 | 이번은 최종 checkpoint의 probe만 관측; 원인/결과의 선후관계는 초기 checkpoint 또는 제어 학습이 필요 |

새로 가장 분명해진 것은 **M10/M40의 포화는 다른 문맥에서도 유지되고, M160은 더 다양한 state 변화와 강한 그룹 비가산성을 보인다**는 점이다. 세 유형의 비율은 유용한 관측값이지만 threshold에 의존하며 동작 분류·문맥 정보·공통 state 역할을 분리해야 한다. 높은 swap 효과·낮은 cosine·높거나 낮은 gate 중 하나를 건강함의 단독 기준으로 쓰지 않는다. 다음 M 선택은 전체 품질·문맥 개입·gate 동작·비용을 함께 보고 사용자가 판단할 수 있도록 남긴다.


## 6. 후속 실험·운영 계획

| 단계 | 비교 / 변경할 값 | 현재 상태 |
|---|---|---|
| Baseline | B0 NeoX, B1 RMT M10 | 완료 |
| 레지스터 | R0/R1/R2 M10/40/160, MLP512 고정 | 학습·다중 window/그룹 사후 분석 완료 |
| 추가 용량 | R3 M512, T4에 맞지 않으면 M256/320 검토 | 비용·목적 검토 후 결정, 미실행 |
| 레지스터 선택 | 전체 품질·개입 효과·gate/slot·시간/VRAM | 사용자 판단 전 |
| MLP | H0/H1/H2/H3 hidden128/256/512/1024, 선택 M 고정 | 선택 후 별도 실험; hidden512는 해당 R 결과 재사용 |

현재 단계에서는 attention D256/head8을 고정한다. MLP 탐색에서도 attention 차원/head는 바꾸지 않는다. 노트북의 수동 선택값·실행·파일 내보내기는 공통 workflow를 따른다.

### 추가 분석과 후속 주제

- 다중 window·logit·중심화한 state/update의 rank/CKA·그룹 초기화/교환·동일 크기 무작위 대조·손상 곡선은 최신 report로 완료했다. 다음 제어 후보는 router context 정규화와 초기 logit 크기/슬롯별 bias 분포이며 아직 새 학습 조건을 확정하지 않는다. 긴 warmup, 여러 swap permutation, 과거 입력 구간 교란, 초기 checkpoint/gradient가 원인과 기능을 구분할 후속 분석이다.
- 초기 write 전환의 원인을 보려면 당시 checkpoint/trace와 필요시 write bias −4/−2/0 등의 제어 학습을 사용한다. raw gate·norm·cosine이나 최종 checkpoint 분석만으로 원인을 확정하지 않는다.
- segment150/75/50 등은 별도 실험으로 NeoX/RMT baseline도 함께 비교한다. 저비용 탐색 후보는 L과 M을 함께 줄여 M/L을 유지하고 처음부터 학습하는 조건이며, M 고정의 segment 축소와 구분한다. global B, segment L, target 예산 T에서 updates≈T/(B×L)이므로 update 수만 같게 하지 않는다. K3 고정과 과거450-token BPTT(K3/6/9)를 구분하고 replay 비용도 기록한다.
- segment를 바꾸는 RTLA는 같은 token 범위·stream partition을 맞춰야 한다. 예를 들어450-token warmup/1200-token 측정은 L150/75/50에서 warmup3/6/9, 측정8/16/24다. 현재 segment 수만 고정하면 비교 token과 과거 범위가 달라진다.
- read gate on/off, write dropout, clip/LR/beta2/optimizer/warmup은 해당 질문을 정한 뒤 탐색한다. state의 scalar read gate는 LayerNorm에서 크기 효과가 상쇄될 수 있어 logits와 다음 state 경로를 먼저 확인한다. K0은 현재 GRT write router의 다음 segment credit 경로가 끊기는 별도 control이다. AdamW·stream warmup·clip 빈도 metric은 실제 사용 전에 지원 범위를 확인한다.
- 더 큰 모델은 small에서 선택한 설정을 출발점으로 삼되 baseline과 용량·batch·속도를 다시 측정한다. 작은 품질 차이의 최종 주장은 추가 seed와 필요시 더 긴 새 schedule로 확인하고, test는 선택 후 평가한다.
- raw-v1·WT2·WT103 문서 subset은 별도 데이터 계열로 다룬다. BPE PPL을 word PPL과 직접 비교하지 않는다. WT2/WT103 평가 기사는 겹치며, 현재 v1의 `<unk>` 치환은 BPE로 복구되지 않는다. 줄 EOS를 기사 경계로 가정하지 않는다.
- 첫 탐색은 수동 grid와 기존 W&B를 유지한다. 필요할 때 Optuna/W&B Sweeps 등을 검토하되 NLL·RTLA를 임의 가중합으로 합치지 않고 품질 허용 후보의 Pareto·재현성을 본다. token milestone 중단/promotion과 자동 runner는 별도 구현이 필요하다.

이 항목들은 후속 선택지이며 실행 완료 결과가 아니다. 초기 HPO 제안의 미확정 grid·예산을 현재 확정 조건으로 사용하지 않는다.

### 자원 배치

Kaggle 주30h는 baseline·본 학습, Colab 하루4~5h는 사전 측정·사후 분석 또는 시간 내 독립 run에 사용한다. 실제 quota 정책과 세션 길이를 확인하고 checkpoint·prepared 데이터를 세션 밖에 보존한다. 사후 분석은 단일 GPU에서 가능하지만 학습 재개는 같은 Config·world_size·데이터 조건을 지켜야 한다. `max_seconds`는 누적 상한이므로 일별 세션 제한으로 쓰지 않는다.

Vast.ai의 RTX4080 등 유료 자원은 무료 quota 부족 시 별도 예산을 정하고 필요한 후보만 실행하는 선택지다. T4 실측을 다른 GPU에 단순 비례 환산하지 않고 새 환경에서 batch·ETA를 측정한다. preflight ETA는 RTLA·준비·저장 비용을 제외하므로 본 학습 wall time과 구분한다.

### 참고 근거

[RMT 2022](https://papers.neurips.cc/paper_files/paper/2022/file/47e288629a6996a17ce50b90a056a0e1-Paper-Conference.pdf)의 word-level/Transformer-XL 결과는 현재 NeoX·GPT-2 BPE·full-softmax PPL의 직접 목표값이 아니다. 데이터 계열은 [WikiText 설명](https://huggingface.co/datasets/Salesforce/wikitext/blob/main/README.md)과 [WikiText 원 논문](https://arxiv.org/pdf/1609.07843)을 참고한다. 자동화 도입 시 [Optuna 다목적](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/002_multi_objective.html)·[Trial API](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html)의 pruning 제약과 [W&B sweep schema](https://github.com/wandb/sweeps/blob/master/src/sweeps/config/schema.json)의 단일 metric 계약을 확인한다.

## 7. 원본과 재현성

전체 설정·hash·학습/분석 배열은 다음 JSON을 기준으로 확인한다. 학습5개와 최신 slots/registers=true 사후3개를 보관하며 다른 환경 결과를 CSV로 누적하지 않는다. 새 결과가 오면 이 문서의 해당 표·해석을 갱신한다. 미실행·미측정은0으로 채우지 않는다.

| 학습 ID | 원본 | 학습 코드 | W&B |
|---|---|---|---|
| B0 | [JSON](../../results/0f4be0ee5d5c4a77b8ce50d591932c69.json) | [e238ac3](https://github.com/YeoJune/GRT/commit/e238ac356bbfcbf5766fc20466105617a7944354) | [run](https://wandb.ai/joyyoj1-korea-university/grt-research/runs/633dd52b) |
| B1 | [JSON](../../results/24b2e511de2e419eb0ac7f9b52769d89.json) | [e238ac3](https://github.com/YeoJune/GRT/commit/e238ac356bbfcbf5766fc20466105617a7944354) | [run](https://wandb.ai/joyyoj1-korea-university/grt-research/runs/92d08d99) |
| R0 | [JSON](../../results/22ab31a56cb546f993197e5ce937e5de.json) | [e238ac3](https://github.com/YeoJune/GRT/commit/e238ac356bbfcbf5766fc20466105617a7944354) | [run](https://wandb.ai/joyyoj1-korea-university/grt-research/runs/e284b89f) |
| R1 | [JSON](../../results/22cfd16b44964befbe42912c4aba28c5.json) | [c36920d](https://github.com/YeoJune/GRT/commit/c36920d0e4423ecf714b6d34e5785fe3a3beae72) | [run](https://wandb.ai/joyyoj1-korea-university/grt-research/runs/710b2330) |
| R2 | [JSON](../../results/0e0ead4744594b59a7d9a771049c79a1.json) | [555d837](https://github.com/YeoJune/GRT/commit/555d8371e852bab78fcfada5ed909ea34425f817) | [run](https://wandb.ai/joyyoj1-korea-university/grt-research/runs/19167540) |

| M | checkpoint SHA256 앞12자 | 사후 원본 | 분석 시간 s / peak allocated GiB |
|---:|---|---|---:|
| 10 | `cc0daecfb590` | [JSON](../../results/posthoc/bb780916fdef402189e4c73ea0c45b2d.json) | 79.550 / 0.553 |
| 40 | `520315a746d3` | [JSON](../../results/posthoc/602b5d3014244be6a6ae7f7bd3f55fcc.json) | 150.874 / 0.554 |
| 160 | `cbac55d5f46f` | [JSON](../../results/posthoc/85aa236c51ae451a8df185b90cdd4f92.json) | 345.607 / 0.562 |

데이터 식별값 앞12자는 train `2e43bd7c329c`, validation `a2bc335cb8a1`, tokenizer `1fe93b615295`다. JSON은 checkpoint 가중치 본체를 포함하지 않으며 gate bias·weight 행 norm 같은 진단값은 포함한다. 사후 시간은 복원·3개 window·슬롯/그룹 개입·표현 분석·그림 저장을 포함하며 순수 forward 속도가 아니다. 실제 forward는 M10/40/160 각각2,738/4,514/8,546회이며 시간 합은약9.60분이다. SVD/CKA는 CPU 계산도 포함하므로 시간 전체를 GPU kernel 시간으로 보지 않는다.

보고서 정리 과정에서 실제 GPU 학습을 추가 실행하지 않았다. 사용자 제공 완료 report로 T4×2 학습과 Colab T4 사후 분석을 확인했다. 구현의 CPU 검증 범위·실행 방법은 workflow에서 관리하며 단일 seed·고정 probe의 관측과 기능 검증을 구분한다.
