# 연구 요약

현재 실행 선택과 수신 GPU 결과의 공통 지표를 관리한다. 모델 계약은 [명세](specification.md), 실행 방법은 [workflow](workflow.md)를 따른다. 완료된 실험의 질문·경과·판단 원문은 [아카이브](archive/README.md), 학습·사후 JSON 원본은 [results](../results/README.md)에 보존한다.

## 현재 선택

| 항목 | 값과 적용 범위 |
|---|---|
| 기본 실행 | GRT small·L75/M20·seed54·FP32·global batch256·40k updates |
| Write 경로 | `candidate`, `write_detach_source=False`; `independent`·`separate_head`도 지원 |
| Gate | bias0·read off·logit tail threshold10/coef1e-4(A)·write dropout0 |
| 최적화 | Adam LR5e-4·clip.25·이전3segments BPTT·warmup5% 뒤 cosine·최소 LR 비율1% |
| 평가 | 전체 validation NLL로 best 선택; best와 final-budget 결과를 별도 표시 |

M20 40k 비교의 best PPL은 candidate28.1781 / independent28.2550 / separate_head28.4120이다. 작은 품질 차이와 reset/swap 진단, 파라미터·연산량을 함께 고려해 candidate를 기본값으로 사용한다. 두 대안의 구현은 유지한다. 다른 M·여러 seed의 전달성과 선택적 detach 효과는 미검증이다.

패널티 A와 dropout0을 모든 M의 후속 비교에 공통 적용한다. M8에서 A+dropout.1의20k 이득은 관측됐지만 M20·M75의 추가 효과는 혼합이었다. 긴 예산에서 같은 조건의 dropout0/.1 재검증이 필요하다. 예산 내 PPL·분화 지표와 수렴 후 우열을 구분한다.

비교 원문: [안정화 탐색](archive/20261018_l75_gate_stabilization.md) · [M20 write 경로](archive/20261019_l75_gate_paths.md) · [L150 용량](archive/20261012_small_register_sweep.md).

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

## 수신 결과의 현재 집계

아래는 저장된 GPU JSON의 산술 집계다. 모델을 실행해 과거 결과를 재구성하지 않는다. 안정화 탐색은 주로20k·warmup0, gate 경로 비교는40k·warmup5%·최소 LR1%이며 L150의 별도 조건도 포함한다. 표·출처의 seed·스케줄·라이브러리·예산을 함께 비교한다. 원본 JSON은 수정하지 않는다.

### M20 write 경로

<!-- gate-report:gate_paths:start -->
| 조건 | best step / PPL | 최종 PPL / NLL | wall h / peak GiB | 학습 / 사후 원본 |
|---|---|---|---|---|
| M20 Independent | 38,000 / 28.2550 | 28.2876 / 3.342424 | 12.530 / 11.797 | [학습](../results/c4b4660b71164924a0bef45863d42a92.json) / [사후](../results/posthoc/d5afa997eeb84bd99d34f15640d37b57.json) |
| M20 Separate head | 40,000 / 28.4120 | 28.4120 / 3.346810 | 13.670 / 11.778 | [학습](../results/d598c992d4734bd9aa327cbff78b96c5.json) / [사후](../results/posthoc/7b46ce088d5a428da7819b3ae15c1303.json) |
| M20 Candidate | 39,000 / 28.1781 | 28.2030 / 3.339429 | 12.488 / 11.766 | [학습](../results/73881fe675194b53b499eab9e754a6d2.json) / [사후](../results/posthoc/8f43885f4f3e4ab890c84186e6bc4594.json) |

| 공통 지표 | M20 Independent | M20 Separate head | M20 Candidate |
|---|---:|---:|---:|
| C/O/T (window0) | 0/1/19 | 0/0/20 | 0/1/19 |
| Write 평균 | 0.442457 | 0.409248 | 0.404264 |
| 슬롯 내 gate std | 0.196136 | 0.328546 | 0.249819 |
| 절대 logit 중앙값 | 1.958 | 2.097 | 2.702 |
| 절대 logit >20 (%) | 0.00 | 0.00 | 0.00 |
| MLP hidden L2 norm | 26.068 | 63.144 | 17.645 |
| 문맥 중심화 state rank | 5.842 | 5.937 | 6.259 |
| 실제 update rank | 5.017 | 3.696 | 4.607 |
| 평균 slot CKA | 0.526 | 0.607 | 0.565 |
| 전체 reset ΔNLL, lag0 | +0.557522 | +0.540895 | +0.594935 |
| 전체 swap ΔNLL, lag0 | +0.779388 | +0.692431 | +0.780243 |
| 전체 swap ΔNLL, lag3 | +0.090288 | +0.091688 | +0.139034 |

![Gate paths: L75/M20, 40k updates — 공통 6패널](figures/gate_stabilization/gate_paths.png)
<!-- gate-report:gate_paths:end -->

### 패널티 수치

<!-- gate-report:penalty:start -->
| 조건 | best step / PPL | 최종 PPL / NLL | wall h / peak GiB | 학습 / 사후 원본 |
|---|---|---|---|---|
| M20 Off | 19,000 / 33.7436 | 33.7718 / 3.519627 | 6.695 / 11.721 | [학습](../results/86fb2481ac1e45f29da0164863b28d66.json) / [사후](../results/posthoc/5435927747a34aa895d5804c2b5772a6.json) |
| M20 A | 20,000 / 32.3376 | 32.3376 / 3.476231 | 7.028 / 11.757 | [학습](../results/b0420e8b63f64d0094ed84a8d273f14b.json) / [사후](../results/posthoc/0661b28486074874ba3688183326aa8c.json) |
| M20 B | 20,000 / 33.0368 | 33.0368 / 3.497622 | 6.146 / 11.757 | [학습](../results/31a7d55d4d64400e81578e055d88b2e1.json) / [사후](../results/posthoc/dd6827efb9ad4505a3b973cc77b152b3.json) |
| M20 C | 20,000 / 32.9088 | 32.9088 / 3.493740 | 6.961 / 11.757 | [학습](../results/93d019b472434814b3e057ea5732f616.json) / [사후](../results/posthoc/3a96051e1cd946b2944a95f8468eeb4d.json) |

| 공통 지표 | M20 Off | M20 A | M20 B | M20 C |
|---|---:|---:|---:|---:|
| C/O/T (window0) | 14/4/2 | 0/1/19 | 0/0/20 | 0/1/19 |
| Write 평균 | 0.229279 | 0.471292 | 0.407538 | 0.486842 |
| 슬롯 내 gate std | 0.001158 | 0.183397 | 0.194820 | 0.208015 |
| 절대 logit 중앙값 | 18.078 | 1.595 | 2.190 | 2.345 |
| 절대 logit >20 (%) | 35.16 | 0.00 | 0.00 | 0.00 |
| MLP hidden L2 norm | 108.064 | 28.508 | 33.932 | 33.427 |
| 문맥 중심화 state rank | 3.033 | 3.245 | 2.567 | 2.626 |
| 실제 update rank | 2.410 | 2.813 | 2.008 | 2.164 |
| 평균 slot CKA | 0.902 | 0.667 | 0.786 | 0.846 |
| 전체 reset ΔNLL, lag0 | +0.354744 | +0.502921 | +0.525509 | +0.423115 |
| 전체 swap ΔNLL, lag0 | +0.748015 | +0.722187 | +0.649062 | +0.683123 |
| 전체 swap ΔNLL, lag3 | +0.059669 | +0.082597 | +0.102224 | +0.098842 |

![Penalty settings: L75/M20 — 공통 6패널](figures/gate_stabilization/penalty.png)
<!-- gate-report:penalty:end -->

### 패널티의 M별 비교

<!-- gate-report:transfer:start -->
| 조건 | best step / PPL | 최종 PPL / NLL | wall h / peak GiB | 학습 / 사후 원본 |
|---|---|---|---|---|
| M8 Off | 20,000 / 34.1048 | 34.1048 / 3.529438 | 5.511 / 10.849 | [학습](../results/ccf041592f3e4ebe8f287e69195205a5.json) / [사후](../results/posthoc/10b47cdf69034d8d91a842f60f226ede.json) |
| M8 A | 19,000 / 33.5383 | 33.6620 / 3.516369 | 6.328 / 10.881 | [학습](../results/1006dcd4d956442097dba6780d07ab69.json) / [사후](../results/posthoc/88d85d847bfb4762902538ea4f303bb0.json) |
| M20 Off | 19,000 / 33.7436 | 33.7718 / 3.519627 | 6.695 / 11.721 | [학습](../results/86fb2481ac1e45f29da0164863b28d66.json) / [사후](../results/posthoc/5435927747a34aa895d5804c2b5772a6.json) |
| M20 A | 20,000 / 32.3376 | 32.3376 / 3.476231 | 7.028 / 11.757 | [학습](../results/b0420e8b63f64d0094ed84a8d273f14b.json) / [사후](../results/posthoc/0661b28486074874ba3688183326aa8c.json) |
| M75 Off | 20,000 / 33.7629 | 33.7629 / 3.519364 | 10.093 / 8.326 | [학습](../results/b4b4414578524de0a6e9de50851b91b2.json) / [사후](../results/posthoc/f5a1825cc5d84f6da400eb249ec409ac.json) |
| M75 A | 20,000 / 32.0361 | 32.0361 / 3.466865 | 10.994 / 8.354 | [학습](../results/70bea6e32c584668ad3078de871cd66c.json) / [사후](../results/posthoc/e9d2ee22c7344315be1a64b908873405.json) |

| 공통 지표 | M8 Off | M8 A | M20 Off | M20 A | M75 Off | M75 A |
|---|---:|---:|---:|---:|---:|---:|
| C/O/T (window0) | 0/8/0 | 0/0/8 | 14/4/2 | 0/1/19 | 72/3/0 | 0/3/72 |
| Write 평균 | 1.000000 | 0.529519 | 0.229279 | 0.471292 | 0.040000 | 0.372126 |
| 슬롯 내 gate std | 0.000000 | 0.169983 | 0.001158 | 0.183397 | 0.000000 | 0.234121 |
| 절대 logit 중앙값 | 59.242 | 2.217 | 18.078 | 1.595 | 88.294 | 5.637 |
| 절대 logit >20 (%) | 100.00 | 0.00 | 35.16 | 0.00 | 100.00 | 0.00 |
| MLP hidden L2 norm | 272.113 | 32.866 | 108.064 | 28.508 | 484.428 | 58.487 |
| 문맥 중심화 state rank | 1.216 | 2.225 | 3.033 | 3.245 | 2.065 | 10.027 |
| 실제 update rank | 1.582 | 1.859 | 2.410 | 2.813 | 1.804 | 8.700 |
| 평균 slot CKA | 0.993 | 0.781 | 0.902 | 0.667 | 0.976 | 0.676 |
| 전체 reset ΔNLL, lag0 | +0.454872 | +0.432938 | +0.354744 | +0.502921 | +0.438900 | +0.504464 |
| 전체 swap ΔNLL, lag0 | +0.655296 | +0.689234 | +0.748015 | +0.722187 | +0.680441 | +0.607893 |
| 전체 swap ΔNLL, lag3 | +0.028658 | +0.077033 | +0.059669 | +0.082597 | +0.049721 | +0.077487 |

![Penalty A transfer: L75/M8, M20 and M75 — 공통 6패널](figures/gate_stabilization/transfer.png)
<!-- gate-report:transfer:end -->

### 패널티와 dropout 결합

<!-- gate-report:combined:start -->
| 조건 | best step / PPL | 최종 PPL / NLL | wall h / peak GiB | 학습 / 사후 원본 |
|---|---|---|---|---|
| M8 A | 19,000 / 33.5383 | 33.6620 / 3.516369 | 6.328 / 10.881 | [학습](../results/1006dcd4d956442097dba6780d07ab69.json) / [사후](../results/posthoc/88d85d847bfb4762902538ea4f303bb0.json) |
| M8 A+Dropout | 19,000 / 33.0576 | 33.0829 / 3.499015 | 5.719 / 10.881 | [학습](../results/d3b3358f21f949bd84e58b7b1ed0aee5.json) / [사후](../results/posthoc/247fde592b8e4698a9f4700e40c247cb.json) |
| M20 A | 20,000 / 32.3376 | 32.3376 / 3.476231 | 7.028 / 11.757 | [학습](../results/b0420e8b63f64d0094ed84a8d273f14b.json) / [사후](../results/posthoc/0661b28486074874ba3688183326aa8c.json) |
| M20 A+Dropout | 19,000 / 33.1184 | 33.1606 / 3.501361 | 6.544 / 11.757 | [학습](../results/8c2eec49e4cc43989a9847ccd7b942c9.json) / [사후](../results/posthoc/bc0e8167eafd473b918dca1d6685f173.json) |
| M75 A | 20,000 / 32.0361 | 32.0361 / 3.466865 | 10.994 / 8.354 | [학습](../results/70bea6e32c584668ad3078de871cd66c.json) / [사후](../results/posthoc/e9d2ee22c7344315be1a64b908873405.json) |
| M75 A+Dropout | 19,000 / 32.4633 | 32.4831 / 3.480719 | 11.133 / 8.354 | [학습](../results/ba51ccddc849423a9e7aa5bc26729b9c.json) / [사후](../results/posthoc/e3f7126def0a4e37b0b6512c74cd312f.json) |

| 공통 지표 | M8 A | M8 A+Dropout | M20 A | M20 A+Dropout | M75 A | M75 A+Dropout |
|---|---:|---:|---:|---:|---:|---:|
| C/O/T (window0) | 0/0/8 | 0/3/5 | 0/1/19 | 0/0/20 | 0/3/72 | 0/0/75 |
| Write 평균 | 0.529519 | 0.633081 | 0.471292 | 0.622220 | 0.372126 | 0.420362 |
| 슬롯 내 gate std | 0.169983 | 0.145452 | 0.183397 | 0.153758 | 0.234121 | 0.238975 |
| 절대 logit 중앙값 | 2.217 | 2.492 | 1.595 | 3.917 | 5.637 | 4.029 |
| 절대 logit >20 (%) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.06 |
| MLP hidden L2 norm | 32.866 | 28.835 | 28.508 | 41.954 | 58.487 | 43.767 |
| 문맥 중심화 state rank | 2.225 | 2.298 | 3.245 | 2.889 | 10.027 | 13.433 |
| 실제 update rank | 1.859 | 1.746 | 2.813 | 3.026 | 8.700 | 10.952 |
| 평균 slot CKA | 0.781 | 0.798 | 0.667 | 0.802 | 0.676 | 0.531 |
| 전체 reset ΔNLL, lag0 | +0.432938 | +0.479568 | +0.502921 | +0.409239 | +0.504464 | +0.470103 |
| 전체 swap ΔNLL, lag0 | +0.689234 | +0.751375 | +0.722187 | +0.728671 | +0.607893 | +0.630010 |
| 전체 swap ΔNLL, lag3 | +0.077033 | +0.088983 | +0.082597 | +0.088281 | +0.077487 | +0.063782 |

![Write dropout with penalty A: L75 — 공통 6패널](figures/gate_stabilization/combined.png)
<!-- gate-report:combined:end -->

### Dropout의 최종 예산 효과

<!-- gate-report:effects:start -->
| M | Off: dropout0 → .1 최종 PPL | A: dropout0 → .1 최종 PPL | dropout ΔNLL (Off) | dropout ΔNLL (A) | 차이의 차이 |
|---|---|---|---:|---:|---:|
| 8 | 34.1048 → 34.2791 | 33.6620 → 33.0829 | +0.005099 | -0.017354 | -0.022453 |
| 20 | 33.7718 → 34.2129 | 32.3376 → 33.1606 | +0.012977 | +0.025130 | +0.012153 |
| 75 | 33.7629 → 32.8984 | 32.0361 → 32.4831 | -0.025941 | +0.013855 | +0.039796 |
<!-- gate-report:effects:end -->

### Dropout 단독 비교

<!-- gate-report:dropout:start -->
| 조건 | best step / PPL | 최종 PPL / NLL | wall h / peak GiB | 학습 / 사후 원본 |
|---|---|---|---|---|
| M8 Off | 20,000 / 34.1048 | 34.1048 / 3.529438 | 5.511 / 10.849 | [학습](../results/ccf041592f3e4ebe8f287e69195205a5.json) / [사후](../results/posthoc/10b47cdf69034d8d91a842f60f226ede.json) |
| M8 Dropout | 20,000 / 34.2791 | 34.2791 / 3.534537 | 6.054 / 10.849 | [학습](../results/746ab4ab41e84ec0bdd0f3f0ed6bffdb.json) / [사후](../results/posthoc/17f1bc032be54c249e645fa01275bac2.json) |
| M20 Off | 19,000 / 33.7436 | 33.7718 / 3.519627 | 6.695 / 11.721 | [학습](../results/86fb2481ac1e45f29da0164863b28d66.json) / [사후](../results/posthoc/5435927747a34aa895d5804c2b5772a6.json) |
| M20 Dropout | 20,000 / 34.2129 | 34.2129 / 3.532604 | 6.442 / 11.721 | [학습](../results/51bafa3f611d48c2808a7a55d732c91d.json) / [사후](../results/posthoc/1ae377fe64e243749040cf61991c37f0.json) |
| M75 Off | 20,000 / 33.7629 | 33.7629 / 3.519364 | 10.093 / 8.326 | [학습](../results/b4b4414578524de0a6e9de50851b91b2.json) / [사후](../results/posthoc/f5a1825cc5d84f6da400eb249ec409ac.json) |
| M75 Dropout | 20,000 / 32.8984 | 32.8984 / 3.493423 | 10.041 / 8.326 | [학습](../results/06fe396a3e6b4e258b779481409de1ba.json) / [사후](../results/posthoc/df797486d1cd4606acc91fc478535e33.json) |

| 공통 지표 | M8 Off | M8 Dropout | M20 Off | M20 Dropout | M75 Off | M75 Dropout |
|---|---:|---:|---:|---:|---:|---:|
| C/O/T (window0) | 0/8/0 | 6/2/0 | 14/4/2 | 18/2/0 | 72/3/0 | 68/2/5 |
| Write 평균 | 1.000000 | 0.250000 | 0.229279 | 0.100000 | 0.040000 | 0.087262 |
| 슬롯 내 gate std | 0.000000 | 0.000000 | 0.001158 | 0.000000 | 0.000000 | 0.000590 |
| 절대 logit 중앙값 | 59.242 | 153.255 | 18.078 | 276.328 | 88.294 | 10.633 |
| 절대 logit >20 (%) | 100.00 | 100.00 | 35.16 | 100.00 | 100.00 | 4.34 |
| MLP hidden L2 norm | 272.113 | 396.977 | 108.064 | 778.406 | 484.428 | 37.535 |
| 문맥 중심화 state rank | 1.216 | 1.790 | 3.033 | 1.897 | 2.065 | 2.214 |
| 실제 update rank | 1.582 | 1.049 | 2.410 | 1.051 | 1.804 | 2.435 |
| 평균 slot CKA | 0.993 | 0.931 | 0.902 | 0.965 | 0.976 | 0.915 |
| 전체 reset ΔNLL, lag0 | +0.454872 | +0.362176 | +0.354744 | +0.305347 | +0.438900 | +0.300231 |
| 전체 swap ΔNLL, lag0 | +0.655296 | +0.619064 | +0.748015 | +0.593159 | +0.680441 | +0.637313 |
| 전체 swap ΔNLL, lag3 | +0.028658 | +0.023372 | +0.059669 | +0.022567 | +0.049721 | +0.039209 |

![Write dropout: L75, penalty off — 공통 6패널](figures/gate_stabilization/dropout.png)
<!-- gate-report:dropout:end -->

### Bias 초기값 비교

<!-- gate-report:bias:start -->
| 조건 | best step / PPL | 최종 PPL / NLL | wall h / peak GiB | 학습 / 사후 원본 |
|---|---|---|---|---|
| M20 Off | 19,000 / 33.7436 | 33.7718 / 3.519627 | 6.695 / 11.721 | [학습](../results/86fb2481ac1e45f29da0164863b28d66.json) / [사후](../results/posthoc/5435927747a34aa895d5804c2b5772a6.json) |
| M20 Bias-2 | 19,000 / 32.6100 | 32.6552 / 3.486005 | 6.423 / 11.721 | [학습](../results/3001e7328e87456395275745ada7b946.json) / [사후](../results/posthoc/3cb39cb5ceaf40e4b2ac4790af009b6d.json) |
| M20 Bias+2 | 19,000 / 33.0599 | 33.0679 / 3.498563 | 6.698 / 11.721 | [학습](../results/0bb9e51192d54a398d771bab611e9fb1.json) / [사후](../results/posthoc/b9d1e1ea89bd4499989c6968dfc63957.json) |

| 공통 지표 | M20 Off | M20 Bias-2 | M20 Bias+2 |
|---|---:|---:|---:|
| C/O/T (window0) | 14/4/2 | 0/0/20 | 13/7/0 |
| Write 평균 | 0.229279 | 0.394144 | 0.350000 |
| 슬롯 내 gate std | 0.001158 | 0.248303 | 0.000000 |
| 절대 logit 중앙값 | 18.078 | 3.495 | 94.247 |
| 절대 logit >20 (%) | 35.16 | 0.39 | 95.00 |
| MLP hidden L2 norm | 108.064 | 24.485 | 286.954 |
| 문맥 중심화 state rank | 3.033 | 3.995 | 3.079 |
| 실제 update rank | 2.410 | 3.817 | 2.236 |
| 평균 slot CKA | 0.902 | 0.617 | 0.815 |
| 전체 reset ΔNLL, lag0 | +0.354744 | +0.447834 | +0.391926 |
| 전체 swap ΔNLL, lag0 | +0.748015 | +0.693603 | +0.707364 |
| 전체 swap ΔNLL, lag3 | +0.059669 | +0.059706 | +0.065762 |

![Bias initialization: L75/M20 — 공통 6패널](figures/gate_stabilization/bias.png)
<!-- gate-report:bias:end -->

### 완료된 초기화 비교

<!-- gate-report:initialization:start -->
| 조건 | best step / PPL | 최종 PPL / NLL | wall h / peak GiB | 학습 / 사후 원본 |
|---|---|---|---|---|
| M20 Off | 19,000 / 33.7436 | 33.7718 / 3.519627 | 6.695 / 11.721 | [학습](../results/86fb2481ac1e45f29da0164863b28d66.json) / [사후](../results/posthoc/5435927747a34aa895d5804c2b5772a6.json) |
| M20 Refine | 20,000 / 33.8667 | 33.8667 / 3.522433 | 6.954 / 11.721 | [학습](../results/89571cd181a4465b8e818a2407be8924.json) / [사후](../results/posthoc/450dd855d0634475af09aefef2239125.json) |
| M20 Uniform | 20,000 / 33.9524 | 33.9524 / 3.524960 | 6.605 / 11.721 | [학습](../results/51a6bf7d37db4e5b82ce96709c15885f.json) / [사후](../results/posthoc/dafd436622a6482092a621db3bbd0b3c.json) |

| 공통 지표 | M20 Off | M20 Refine | M20 Uniform |
|---|---:|---:|---:|
| C/O/T (window0) | 14/4/2 | 17/2/1 | 17/3/0 |
| Write 평균 | 0.229279 | 0.149114 | 0.150000 |
| 슬롯 내 gate std | 0.001158 | 0.000527 | 0.000000 |
| 절대 logit 중앙값 | 18.078 | 55.444 | 73.258 |
| 절대 logit >20 (%) | 35.16 | 94.69 | 100.00 |
| MLP hidden L2 norm | 108.064 | 298.757 | 217.123 |
| 문맥 중심화 state rank | 3.033 | 2.379 | 2.066 |
| 실제 update rank | 2.410 | 1.532 | 1.206 |
| 평균 slot CKA | 0.902 | 0.945 | 0.959 |
| 전체 reset ΔNLL, lag0 | +0.354744 | +0.401240 | +0.387380 |
| 전체 swap ΔNLL, lag0 | +0.748015 | +0.683010 | +0.642072 |
| 전체 swap ΔNLL, lag3 | +0.059669 | +0.035829 | +0.033352 |

![Retired gate variants: L75/M20 — 공통 6패널](figures/gate_stabilization/initialization.png)
<!-- gate-report:initialization:end -->

### L150 참고 비교

<!-- gate-report:reference:start -->
| 조건 | best step / PPL | 최종 PPL / NLL | wall h / peak GiB | 학습 / 사후 원본 |
|---|---|---|---|---|
| M10 L150 | 10,000 / 34.9897 | 34.9897 / 3.555054 | 5.812 / 11.269 | [학습](../results/22ab31a56cb546f993197e5ce937e5de.json) / [사후](../results/posthoc/bb780916fdef402189e4c73ea0c45b2d.json) |
| M40 L150 | 10,000 / 34.7458 | 34.7458 / 3.548059 | 7.009 / 6.538 | [학습](../results/22cfd16b44964befbe42912c4aba28c5.json) / [사후](../results/posthoc/602b5d3014244be6a6ae7f7bd3f55fcc.json) |
| M160 L150 | 10,000 / 34.1511 | 34.1511 / 3.530795 | 13.150 / 9.713 | [학습](../results/0e0ead4744594b59a7d9a771049c79a1.json) / [사후](../results/posthoc/85aa236c51ae451a8df185b90cdd4f92.json) |
| M20 Bias-2 | 19,000 / 32.6100 | 32.6552 / 3.486005 | 6.423 / 11.721 | [학습](../results/3001e7328e87456395275745ada7b946.json) / [사후](../results/posthoc/3cb39cb5ceaf40e4b2ac4790af009b6d.json) |

| 공통 지표 | M10 L150 | M40 L150 | M160 L150 | M20 Bias-2 |
|---|---:|---:|---:|---:|
| C/O/T (window0) | 0/10/0 | 7/33/0 | 77/26/57 | 0/0/20 |
| Write 평균 | 1.000000 | 0.825000 | 0.322063 | 0.394144 |
| 슬롯 내 gate std | 0.000000 | 0.000000 | 0.066238 | 0.248303 |
| 절대 logit 중앙값 | 52.779 | 123.633 | 8.755 | 3.495 |
| 절대 logit >20 (%) | 100.00 | 100.00 | 3.79 | 0.39 |
| MLP hidden L2 norm | 365.376 | 551.057 | 50.974 | 24.485 |
| 문맥 중심화 state rank | 1.054 | 1.833 | 9.571 | 3.995 |
| 실제 update rank | 1.195 | 2.403 | 11.278 | 3.817 |
| 평균 slot CKA | 0.999 | 0.801 | 0.500 | 0.617 |
| 전체 reset ΔNLL, lag0 | +0.285657 | +0.342306 | +0.322763 | +0.447834 |
| 전체 swap ΔNLL, lag0 | +0.384280 | +0.463843 | +0.289818 | +0.693603 |
| 전체 swap ΔNLL, lag3 | +0.006073 | +0.009612 | +0.011476 | +0.059706 |

![Reference: L150 vs L75, bias -2 — 공통 6패널](figures/gate_stabilization/reference.png)
<!-- gate-report:reference:end -->

## 출처와 갱신

현재 집계는 학습·사후24개 source pair를 사용한다. [구조화 집계](tables/gate_stabilization_metrics.json)에서 조건·환경·checkpoint hash와 원본 ID를 확인한다. 역사 spreadsheet와 그림은 `tables/history/`, `figures/history/`에 보존하며 최신 집계를 대체하지 않는다.

<!-- gate-report:sources:start -->
| 조건 (L/M) | best step | train / post commit | train / post PyTorch | micro×accum×GPU | checkpoint SHA256 |
|---|---:|---|---|---|---|
| Off (75/8) | 20,000 | `90d407c` / `fbb9a53` | 2.11.0+cu128 / 2.11.0+cu130 | 128×1×2 | `5aa9d458db8495bb70197fd910576b640db270f16eeb62857845f2263f64daf6` |
| Dropout (75/8) | 20,000 | `90d407c` / `fbb9a53` | 2.11.0+cu128 / 2.11.0+cu130 | 128×1×2 | `be6f249c6745d8e34c41b0ad69b2754fb31ef4395d6d678397fd6b06305eed27` |
| A (75/8) | 19,000 | `2876bcf` / `2876bcf` | 2.11.0+cu128 / 2.11.0+cu128 | 128×1×2 | `deb4e0ca4a352af4aaa23ea1a16986ecee770a8958add78824c518a42eb885bb` |
| A+Dropout (75/8) | 19,000 | `403e9de` / `403e9de` | 2.11.0+cu128 / 2.11.0+cu128 | 128×1×2 | `0b4e95747bd5b91cd6aba7d9c2a6c3cb2c3e13a25bd9bacb2d4412e422a4f4f9` |
| Off (75/20) | 19,000 | `90d407c` / `fbb9a53` | 2.11.0+cu128 / 2.11.0+cu130 | 128×1×2 | `919e91014cda899df36e4346a952627beb258c00624a8340e8065dd3eda1281a` |
| Dropout (75/20) | 20,000 | `90d407c` / `fbb9a53` | 2.11.0+cu128 / 2.11.0+cu130 | 128×1×2 | `d56a848c64267afb71a09ae951a12cd8397a5512f6986b332c399cdd35740a59` |
| A (75/20) | 20,000 | `2876bcf` / `2876bcf` | 2.10.0+cu128 / 2.10.0+cu128 | 128×1×2 | `f7f45bf28d90f1e6e61ae6310d3e2d2ee460795699bc1343229ba7d638f8bf71` |
| A+Dropout (75/20) | 19,000 | `d7d2995` / `d7d2995` | 2.11.0+cu128 / 2.11.0+cu128 | 128×1×2 | `566e064c5966a82cc7e2191facb33a999d08e7bc680801e598f5383833fc9666` |
| B (75/20) | 20,000 | `2876bcf` / `2876bcf` | 2.11.0+cu128 / 2.11.0+cu128 | 128×1×2 | `dbdddf174ebe41f8baa4ddd270a3ac0503b01fd6f068aaf91b6a5dbb8087ed45` |
| C (75/20) | 20,000 | `2876bcf` / `2876bcf` | 2.11.0+cu128 / 2.11.0+cu128 | 128×1×2 | `322a7ce5dbe2c1ea5671aecfeacab3371e72705cde00155b4bddb37b1f034b13` |
| Bias-2 (75/20) | 19,000 | `56510e6` / `1d495c8` | 2.11.0+cu128 / 2.11.0+cu130 | 128×1×2 | `9e5c3815d522be15e6068fbc14553966f3ebd0dc84876f2c941ca696ea6658dc` |
| Bias+2 (75/20) | 19,000 | `56510e6` / `1d495c8` | 2.11.0+cu128 / 2.11.0+cu130 | 128×1×2 | `ffc7bfd5f8b614125a4bba0bbddf2e292df7d6aa9e872cfdb7d84056b75660c3` |
| Refine (75/20) | 20,000 | `454cd58` / `454cd58` | 2.10.0+cu128 / 2.10.0+cu128 | 128×1×2 | `86bbb718a2f313ce3d1a01c05988da39ce303c3e03cc1fb0bd37750a51894dff` |
| Uniform (75/20) | 20,000 | `1a493c1` / `1a493c1` | 2.10.0+cu128 / 2.10.0+cu128 | 128×1×2 | `12a6ff8bc60a01f8b9a3581a6cc499dd5a3975a9dbd6d654c5c3ef1ef159f5ad` |
| Off (75/75) | 20,000 | `90d407c` / `fbb9a53` | 2.11.0+cu128 / 2.11.0+cu130 | 64×2×2 | `1de8e1540d5cbd216acdf2e20d25e0b93e245ea7302f0c0b80c63ee594d71ed0` |
| Dropout (75/75) | 20,000 | `90d407c` / `fbb9a53` | 2.11.0+cu128 / 2.11.0+cu130 | 64×2×2 | `9f53f8caa51467c29beee91f098dae758dcec9635c5336d267ac26cbcbb877e3` |
| A (75/75) | 20,000 | `2876bcf` / `2876bcf` | 2.11.0+cu128 / 2.11.0+cu128 | 64×2×2 | `e9ea651a272e5a52e6a09857f666af37d586184662dae9a65a810458ac8028ec` |
| A+Dropout (75/75) | 19,000 | `403e9de` / `403e9de` | 2.11.0+cu128 / 2.11.0+cu128 | 64×2×2 | `88fd59ece752eaa801b7dc971b9a03d470c3d57a13f66296777be82d8204f471` |
| L150 (150/10) | 10,000 | `e238ac3` / `89dd4c2` | 2.11.0+cu128 / 2.11.0+cu130 | 64×2×2 | `cc0daecfb5904df79c9fa460092d3c5ed7abb54be720528bdbed29319519c129` |
| L150 (150/40) | 10,000 | `c36920d` / `89dd4c2` | 2.11.0+cu128 / 2.11.0+cu130 | 32×4×2 | `520315a746d332cc6c979035e5f29b86ec22e02c9a3cd78086d19429bace4f0f` |
| L150 (150/160) | 10,000 | `555d837` / `89dd4c2` | 2.10.0+cu128 / 2.11.0+cu130 | 32×4×2 | `cbac55d5f46f5f6480cfea76b6da47e8182e950551a4b3e7b04055691d85231c` |
| Independent (75/20) | 38,000 | `fb9ba36` / `fb9ba36` | 2.11.0+cu128 / 2.11.0+cu128 | 128×1×2 | `f78cb5eda1a7d7e80030ded88f1edc3a358404d9975d6431dfa928852c681089` |
| Separate head (75/20) | 40,000 | `fb9ba36` / `fb9ba36` | 2.11.0+cu128 / 2.11.0+cu128 | 128×1×2 | `daa53fded26478ac62c4650b91f15d1ce61fb8db44b1fd077129d20e0d9d8074` |
| Candidate (75/20) | 39,000 | `fb9ba36` / `fb9ba36` | 2.11.0+cu128 / 2.11.0+cu128 | 128×1×2 | `552928bb2b51d48da0e9bb26744bfcb53022de92a6b683e72e98ec8f8278e0c8` |
<!-- gate-report:sources:end -->

새 결과는 원본 bytes를 `results/`에 보존하고 `scripts/report_gate_stabilization.py`의 source/group 정의에 등록한다. 다음 명령은 이 문서의 표시 블록, 공통 그림과 JSON 집계를 갱신한다. 최신 해석은 표시 블록 밖에서 관리하고 완료된 연구 경과는 Markdown 아카이브로 보존한다.

```bash
python scripts/report_gate_stabilization.py
```

단일 seed·유한 예산·짧은 사후 window와 라이브러리 차이의 한계를 유지한다. 같은 seed/global batch가 같은 RNG 경로를 보장하지 않으며, gate 포화·rank·개입 지표 하나로 기능 분화나 수렴을 확정하지 않는다.
