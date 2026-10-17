# L75/M20: Uniform Gate Initialization 비교

작성: 2026-10-18 · 상태: 구현·실행 설정 준비, GPU 학습 미실행

[실행 workflow](../20261011_experiment_workflow.md) · [Refine 결과](20261017_l75_refine_gate.md) · [기존 dropout/bias 결과](20261015_l75_write_gates.md) · [원논문 §3.2·식12·부록A Listing 2](https://arxiv.org/abs/1910.09890)

## 1. 목적과 비교

Refine 단독은 seed54에서 품질 개선·초기 고착 완화 효과가 없었다. 추가 Refine seed는 진행하지 않고, **Refine off에서 write bias 초기화만 Uniform으로 바꾼 run 하나**를 먼저 실행한다. 질문은 서로 다른 초기 보존·갱신 성향을 주는 것만으로 초기 고착과 후반 문맥별 갱신·품질이 개선되는가이다. 분화 비율이나 write 평균의 목표값은 지정하지 않는다.

| 항목 | 고정 조건 |
|---|---|
| 모델 / 데이터 | GRT small D256·6 layers·head8·FF1024, 기존 WikiText-103 GPT-2 BPE artifact |
| L / M / router | 75 / 20 / attention D256·head8·MLP hidden512 |
| Gate / state | Refine off·read off·write dropout0·학습 가능한 S0의0 초기화 |
| 학습 | seed54·새 초기화20k·FP32·global256·BPTT 이전3segments |
| Optimizer | Adam LR5e−4·betas(.9,.999)·decay0·clip.25·cosine20k·warmup0 |
| 평가 / 사후 | 매1k 전체 validation·기본 RTLA, best 선택·test 미평가. 사후 batch8·warmup3·측정8·horizon4·windows0/16/32·random3/seed54 |
| 변경 변수 | 고정 bias0 → 슬롯별 Uniform Gate Initialization. 추가 loss·파라미터 없음 |

기존 bias0/seed54는 20k 학습을 마친 A20-0을 재사용한다. 과거 Refine on은 참고 결과이며 이번 대조군은 아니다. 처음에는 Uniform run 하나만 실행하고, 결과를 본 뒤 추가 대조군·seed 투자 여부를 정한다. NeoX/RMT는 새로 학습하지 않는다.

## 2. 초기화의 정확한 의미

원논문의 forget gate는 bias를 `logit(U[ε,1−ε])`로 초기화한다. Listing 2의 `hidden_size`는 독립 gate 수이며, GRT는 D256이나 MLP hidden512가 아닌 **슬롯별 scalar gate 수 M20**에 대응시킨다.

```text
ε = 1 / M                         # M≥3; M20에서 .05
u_i ~ Uniform(ε, 1−ε)            # 슬롯마다 한 번 독립 추출
b_i = log(u_i / (1−u_i))         # M20에서 약 −2.944~+2.944
w_i(x,S) = sigmoid(W_i h(x,S) + b_i)
S_next_i = (1−w_i)*S_i + w_i*C_i
```

GRT는 write gate를 쓰고 원논문은 forget gate를 쓰지만, `f=1−w`의 균등분포도 같은 대칭 분포다. 이번에는 Refine head 없이 UGI만 분리해 적용하므로 논문의 전체 UR-LSTM 재현은 아니다. M을 바꾸면 ε도 자동 변경된다. M1/2는 ε=1/M의 범위가 퇴화하므로 명시적으로 거부한다.

`sigmoid(b_i)`의 모집단 분포가 균등하다는 뜻이다. M20의 실제 표본 평균이 정확히 .5이거나 모든 구간에 슬롯이 같은 수로 배정된다는 뜻이 아니다. 기존 출력층 weight를 유지하므로 **실제 입력에 대한 w까지 균등한 것은 아니다.** 초기 bias만 설정하고 이후에는 task CE로 자유롭게 학습한다. Bias·gate를 계속 clip하거나 평균·직교성·분화 비율을 강제하지 않는다. 학습 중 포화나 초기 조건 민감성을 해소한다는 보장은 없으며 이것이 이번 검증 대상이다.

균등 표본은 모델 초기화 시점의 CPU RNG를 복제한 범위에서 추출해 바깥 RNG를 소비하지 않는다. 같은 환경·seed에서 초기 write bias를 제외한 모든 공통 가중치와 이후 학습 RNG 상태를 보존한다. 이것은 같은 소프트웨어·장치·배치 조건에서의 통제이며 환경을 넘는 수치 trajectory 동일성을 보장하지 않는다. Resume·사후 checkpoint 복원에서는 학습된 bias를 그대로 읽는다.

## 3. 기존 노트북 실행

[Kaggle](../../notebooks/kaggle.ipynb) / [Colab](../../notebooks/colab.ipynb)의 기존 선택 셀을 사용한다. 별도 노트북·실험별 YAML은 없다. 이번 기본값은 다음과 같다.

```python
models = ('grt',)
segment_len = 75
num_registers = 20
budget = '20k'
router_hidden = 512
write_gate_bias = 0.0
write_dropout = 0.0
refine_gate = False
uniform_gate_init = True
seed = 54
```

`uniform_gate_init`는 `overrides.model.register.uniform_gate_init`로 전달한다. Typed Config·카탈로그 기본은false이며, on에서는 `write_gate_bias_init` 대신 균등 bias를 사용한다. 따라서 이번 Config에 남는 `write_gate_bias_init=0.0`을 실제 모든 슬롯의 초기 bias0으로 해석하지 않는다. CLI는 `--set model.register.uniform_gate_init=true --set model.register.refine_gate=false`로 선택한다. 설정 hash가 달라 기존 run과 분리되며 기존 checkpoint에서 이어 학습하지 않는다. 대조군 재실행은 `uniform_gate_init=False`, `refine_gate=False`, `write_gate_bias=0.0`이다.

setup → 기존 prepared 데이터 복원 → preflight → train → 마지막 `job.report(posthoc=True, download=True, ...)` 순서다. 마지막 셀에서 기본 RTLA 외에 슬롯·분화·개입·retention 사후 분석까지 수행한다. **experiment_results.zip 하나**를 보내면 학습·사후 JSON을 보관하고 아래 표와 해석을 채운다.

기존 A20-0은 Kaggle T4×2·micro128×accum1·Python3.13.15/PyTorch2.11.0+cu128이다. 가급적 같은 환경·실제 배치를 쓰고 preflight에서 확인한다. 환경이 다르면 첫 run은 탐색 비교로 기록하고, 유망할 때 같은 환경의 bias0 대조군을 먼저 추가한다. 동일 global256만으로 dropout RNG·연산 경로까지 같아지는 것은 아니다. 기존 plain run의 학습 wall time은약6.7시간(T4×2, 약13.4 GPU-hours)이었고 사후 분석은 별도다. 실제 ETA는 preflight로 갱신한다.

## 4. 결과를 판단할 기준

주 기준은 같은20k에서 best·최종 validation NLL/PPL이다. 15k~20k 곡선을 함께 확인한다. 동작은 학습 중 write 평균이 일찍 고정되는지, 사후 슬롯별 gate 변동·logit 포화·sigmoid slope·실제 update rank·CKA·그룹 swap/reset·retention으로 평가한다. 평균 write의 이동이나 other 슬롯 수 증가만으로 성공으로 판단하지 않는다. 다른 초기 평균 때문에 전역 write 평균이 달라지는 것과 실제 문맥별 갱신을 구분한다.

Uniform의 초기 gate 분포는 첫1k 이전에 사라질 수도 있다. 기존 학습 RTLA는1k 간격이므로 초반 변화 순서를 확정할 수 없다. 초기 bias 차이는 구현·seed로 재현 가능하지만 중간 checkpoint가 없으면 그 시점의 슬롯별 trajectory를 복원하지 않는다.

**후속 판단:** 품질을 해치지 않으면서 고착·문맥 활용이 개선되면 동일 환경 대조군과 추가 paired seed를 검토한다. 초기 분포만 달라지고 다시 고착하면 UGI의 지속 효과가 없었던 조건으로 기록한다. 추가 loss나 Refine 결합은 다음 별도 비교이며 이번 run에 섞지 않는다.

| 조건 / seed | 상태 | best step / NLL / PPL | 최종20k PPL | wall h / peak GiB | 학습 / 사후 원본 |
|---|---|---|---|---|---|
| 고정 bias0 / 54 | 기존 A20-0 | 19,000 / 3.518790 / 33.7436 | 33.7718 | 6.695 / 11.721 | [학습](../../results/86fb2481ac1e45f29da0164863b28d66.json) / [사후](../../results/posthoc/5435927747a34aa895d5804c2b5772a6.json) |
| Uniform / 54 | 미실행 | | | | |

| 사후 지표: window16/32 평균 | 고정 bias0 / 54 | Uniform / 54 |
|---|---:|---|
| 유효 write 평균 / 슬롯별 gate std 평균 | .229279 / .001158 | |
| abs logit>20 비율 / sigmoid slope 평균 | 35.156% / .012296 | |
| 실제 update effective rank / 평균 slot CKA | 2.4098 / .9016 | |
| 전체 reset / swap ΔNLL, lag0 | .354744 / .748015 | |
| 전체 reset / swap ΔNLL, lag3 | .049962 / .059669 | |
| calibration closed / open / other | 14 / 4 / 2 | |

환경·배치 차이: 미기입. 사용자 판단: 미기입. 실패·중단·미측정을0이나 완료로 기록하지 않는다.

## 5. 검증 범위

관련 단일 프로세스 CPU 검사110개가 통과했다(38.01초). 분산 CLI 검사3개는 이번 검증에서 제외했다. Uniform on/off의 공통 가중치·RNG 보존, 균등 표본의 inverse-logit 규칙·seed 재현성·범위, task gradient·head 갱신, checkpoint 복원·정확한 학습 재개, 실제 두 노트북 선택 셀의 설정·global256 보존·별도 run, 확장 사후 JSON·캐시 연결을 확인했다. 노트북 코드 문법·문서 링크와 기존/수신 결과 원본 bytes 보존도 확인했다. GPU 학습·수렴·실제 고착 완화는 사용자 실험 후 판단한다.
