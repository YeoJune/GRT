# 연구 요약

현재 실행 선택과 수신 GPU 결과의 공통 지표를 관리한다. 모델 계약은 [명세](specification.md), 실행 방법은 [workflow](workflow.md)를 따른다. 실험별 질문·조건·결과·판단은 [실험 목록](experiments/README.md), 학습·사후 JSON 원본은 [results](../results/README.md)에 보존한다.

## 현재 선택

| 항목 | 값과 적용 범위 |
|---|---|
| 기본 실행 | GRT small·L75/M20·seed54·FP32·global batch256·40k updates |
| Write 경로 | `candidate`, `write_detach_source=False`; `independent`·`separate_head`도 지원 |
| Gate | bias0·read off·logit tail threshold10/coef1e-4(A)·write dropout0 |
| 최적화 | Adam LR5e-4·clip.25·이전3segments BPTT·warmup5% 뒤 cosine·최소 LR 비율1% |
| 평가 | 전체 validation NLL로 best 선택; best와 final-budget 결과를 별도 표시 |

M20 40k 비교의 best PPL은 candidate28.1781 / independent28.2550 / separate_head28.4120이다. 작은 품질 차이와 reset/swap 진단, 파라미터·연산량을 함께 고려해 candidate를 기본값으로 사용한다. 두 대안의 구현은 유지한다. M8 40k는 candidate/independent가 거의 동등하고 separate_head의 best/final은 두 경로보다 높다. M75 candidate는 independent보다 best 약4.06%·final 약4.56% 낮아 선택을 지지한다. M75 separate_head까지9개40k 비교·사후 분석을 완료했다. Separate_head는 모든 M에서 best/final PPL이 가장 높았고 candidate 기본값을 유지한다. 여러 seed·완전 수렴 후 우열·선택적 detach 효과는 미검증이다.

패널티 A와 dropout0을 모든 M의 후속 비교에 공통 적용한다. M8에서 A+dropout.1의20k 이득은 관측됐지만 M20·M75의 추가 효과는 혼합이었다. 긴 예산에서 같은 조건의 dropout0/.1 재검증이 필요하다. 예산 내 PPL·분화 지표와 수렴 후 우열을 구분한다.

비교 원문: [안정화 탐색](experiments/20261018_l75_gate_stabilization.md) · [M20 write 경로](experiments/20261019_l75_gate_paths.md) · [L150 용량](experiments/20261012_small_register_sweep.md).

## 공통 지표와 해석

전체 validation과 짧은 사후 probe를 구분하고, 학습 통계·선택 checkpoint probe·사후 window 관측을 같은 평균으로 합치지 않는다. 표의 연속 사후 지표는 평가 window16/32의 동일 가중 평균이며 유형은 calibration window0에서 분류한다. C/O/T는 모든 관측에서 gate가 각각 ≤.01 / ≥.99 / 나머지인 슬롯 수다. 유형은 문맥과 threshold에 민감하다.

| 지표 | 정의 | 해석 범위 |
|---|---|---|
| Best / final PPL·NLL | 전체 validation의 best step / 최종 budget | 사후 probe PPL과 다르며 수렴 후 순위를 뜻하지 않음 |
| Write 평균 | 실제 유효 gate w 평균 | 정보량·기여율과 다름; raw logit과 구분 |
| 슬롯 내 gate std | 문맥별 std를 슬롯 평균 | 전체 슬롯을 섞은 std와 다름; 가변성이 내용 분화를 증명하지 않음 |
| 절대 logit 중앙값 / >20 | raw write logit scale·극단 비율 | 저장된 gate 정의와 함께 해석 |
| MLP hidden L2 norm | GELU 뒤·출력 Linear 직전 norm | state·weight·gradient norm과 다름 |
| 문맥 중심화 state rank | 슬롯별 문맥 평균을 제거한 state의 energy effective rank | M에 따라 상한이 달라 절댓값을 M 간 효율 순위로 쓰지 않음 |
| 실제 update rank | S_next−S의 같은 rank | candidate 변화와 다르며 품질 점수가 아님 |
| 평균 slot CKA | off-diagonal 슬롯쌍 linear CKA 평균 | 낮은 값만으로 기능 분화가 입증되지 않음 |
| Reset / swap ΔNLL | 정상 대비 같은 probe의 전체 state 개입 NLL 증가 | 의존성 근거이며 기억 품질·기여율의 절대 점수가 아님 |
| Swap lag3 | write 재개 뒤3segments의 잔여 영향 | 기억 수명과 다르며 복구 속도도 포함 |

공통6패널은 validation PPL, validation probe write, 사후 절대 logit ECDF, 슬롯 내 gate std, 중심화 state/update rank, reset/swap ΔNLL(lag0)이다. 학습 곡선의 가로축은 실제 target tokens다. ECDF는 동수의 평가 window를 합치므로 window별 중앙값 평균과 다를 수 있다.

## 실험별 기록과 갱신

실험 문서는 완료 여부와 관계없이 `docs/experiments/`에 주제별 하나씩 유지한다. 새 조건·수신 결과·판단은 해당 문서에 반영하고 원본 JSON은 `results/`에 보존한다. 설계·운영 계획과 일회성 검증 기록만 `docs/archive/`에 둔다.

표·그림·구조화 집계를 갱신하려면 다음을 실행한다. 과거 모델을 실행하지 않고 저장된 GPU JSON만 집계한다.

```bash
python scripts/report_gate_stabilization.py
```

단일 seed·유한 예산·짧은 사후 window와 라이브러리 차이의 한계를 유지한다. 같은 seed/global batch가 같은 RNG 경로를 보장하지 않으며, 지표 하나로 기능 분화나 수렴을 확정하지 않는다.
