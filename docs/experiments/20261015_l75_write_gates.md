# L75 small: write dropout과 bias 비교

작성: 2026-10-15 · 상태: 실행 준비 · 완료 run: 0 · 학습은 사용자가 Kaggle/Colab에서 수행

[공통 실행 안내](../20261011_experiment_workflow.md) · [이전 L150 용량 실험](20261012_small_register_sweep.md) · [원본 보관](../../results/README.md)

## 1. 목적과 고정 조건

L과 M을 함께 줄인 별도 실험으로, write dropout이 품질·분화에 미치는 영향과 초기 write bias에 따른 차이를 확인한다. 현재 단계에서 NeoX/RMT를 새로 학습하지 않는다. 이전 L150 결과는 참고 자료이며 L75 baseline이나 동일 조건 대조군으로 사용하지 않는다.

| 항목 | 조건 |
|---|---|
| 모델 / 데이터 | GRT small D256·6 layers·FF1024 / 기존 WikiText-103 GPT-2 BPE prepared artifact 재사용 |
| L / M | L75 고정, M8 / 20 / 75 |
| M/L | 0.1067 / 0.2667 / 1.0000; 이전10/150·40/150·160/150와 유사한 범위, 정확한 축소 복제는 아님 |
| Router | attention D256/head8, MLP hidden512 고정; head 수는 M의 배수 조건이 아님 |
| Read gate / S0 | read gate 항상 off; 학습 가능한 S0를 0으로 초기화 |
| 기본 write | bias 초기값0, write dropout0; 아래 두 단계에서만 변경 |
| 학습 예산 | 새 초기화에서20,000 optimizer updates; nominal targets=256×75×20,000=384M, 실제 tail 반영 수는 report 기준 |
| BPTT | 이전3 segments 유지: 과거225 tokens, 현재 포함 최대300 tokens 재계산 |
| Optimizer / schedule | Adam LR5e−4, betas(.9,.999), decay0, cosine20k·warmup0, global norm clip.25 |
| Dropout / precision | backbone residual.1 / backbone·router attention0 / FP32 |
| 배치 / seed | global256, model/data seed54; auto_batch·예약 VRAM 목표95%, micro×accum×GPU수 유지 |
| 평가 | validation batch10·매1k updates·best validation checkpoint, test 미평가; 학습 중 RTLA on |

20k는 update 수다. L150/10k와 nominal target 예산은 같지만 optimizer update 횟수, 과거 token 기준 BPTT, write 빈도·메모리 용량이 달라 동일 학습 조건은 아니다. 이번 두 단계 안에서는 L/M별 고정 조건과20k schedule을 유지한다.

## 2. 실험 순서: 총12개 고유 run

**A. 각 M에서 bias0으로 dropout0과0.1을 비교한다(6회).** best validation NLL/PPL을 우선 보고 학습 곡선·시간·분화 지표를 함께 확인한다. 품질 차이를 확신하기 어려우면 dropout0을 우선하는 잠정 선택을 기록한다. 한 seed에서 작은 차이를 확정하지 않는다.

**B. 각 M에서 선택한 dropout을 고정하고 bias−2/0/+2를 비교한다(추가6회).** bias0은 A의 해당 run/checkpoint/report를 그대로 재사용한다. bias만 바꾼 두 조건은 매번 새로 초기화해20k 학습한다. dropout 선택은 M별로 기록한다. 서로 다른 dropout을 고른 M 사이의 비교는 선택된 조건의 비교이며 순수 M 효과로 해석하지 않는다.

Bias는 모든 슬롯의 write MLP 마지막 bias 초깃값이며 학습 가능하다. bias−2/0/+2의 sigmoid 기준값은 약.119/.500/.881이지만 실제 gate는 hidden 출력에 따라 달라진다. write dropout.1은 학습 중 example·slot별 확률.1로 write logit을−∞로 만들어 기존 state를 보존하며, state를 지우거나 rescaling하지 않는다. 평가·사후 분석에서는 dropout을 끈다. 새 정규화·분화 보조 손실은 이번 비교에 넣지 않는다.

| M | A의 선택 dropout | 근거 / 비교한 report | B 실행 여부 |
|---:|---|---|---|
| 8 | | | |
| 20 | | | |
| 75 | | | |

## 3. 학습 결과 기록

각 행은 별도 run이다. B의 bias0 결과는 A에서 선택한 행을 참조해 중복 학습하지 않는다. 실제 micro/accum/GPU·환경·target 예산·코드 hash는 원본 report에 보관하고 차이가 있으면 판단에 적는다.

| ID | M | bias init | write dropout | 상태 | best step / val PPL | wall h / peak GiB | 학습 JSON / 사후 JSON |
|---|---:|---:|---|---|---|---|---|
| A8-0 | 8 | 0 | 0 | 미실행 | | | |
| A8-1 | 8 | 0 | .1 | 미실행 | | | |
| A20-0 | 20 | 0 | 0 | 미실행 | | | |
| A20-1 | 20 | 0 | .1 | 미실행 | | | |
| A75-0 | 75 | 0 | 0 | 미실행 | | | |
| A75-1 | 75 | 0 | .1 | 미실행 | | | |
| B8− | 8 | −2 | A8 선택값 | 미실행 | | | |
| B8+ | 8 | +2 | A8 선택값 | 미실행 | | | |
| B20− | 20 | −2 | A20 선택값 | 미실행 | | | |
| B20+ | 20 | +2 | A20 선택값 | 미실행 | | | |
| B75− | 75 | −2 | A75 선택값 | 미실행 | | | |
| B75+ | 75 | +2 | A75 선택값 | 미실행 | | | |

## 4. 분화 분석과 판단란

기존 Colab 사후 분석 노트북에 각 best checkpoint를 넣는다. batch8·warmup3·측정8·horizon4·window0/16/32·random 반복3/seed54·slots/registers=true를 공통 사용한다. L/M은 checkpoint에서 읽는다. L75에서 warmup225/측정600 tokens이며, 이전 L150 probe와 절대 token 범위는 다르다. 이번 run끼리는 같은 조건으로 비교한다.

| 비교 | 전체 val 품질 / 학습 안정성 | gate 유형·logit 포화 / window 재현 | 중심화 state·update rank / 그룹 reset·swap | 판단 |
|---|---|---|---|---|
| M8 dropout0 vs .1 | | | | |
| M20 dropout0 vs .1 | | | | |
| M75 dropout0 vs .1 | | | | |
| M8 bias−2/0/+2 | | | | |
| M20 bias−2/0/+2 | | | | |
| M75 bias−2/0/+2 | | | | |

세 유형의 균등 분배나 높은 rank 자체를 성공 기준으로 삼지 않는다. threshold 민감도·동일 개수 무작위 그룹·단일 효과의 비가산성·reset과swap 차이를 함께 본다. 학습 중 RTLA로 전환 시점을 보고, 최종 checkpoint 사후 관측으로 학습 원인을 확정하지 않는다. 추가 seed·긴 학습·다음 구조 실험은 결과를 받은 후 결정한다.

## 5. 기존 노트북에서 실행

새 노트북이나 run별 YAML을 만들지 않는다. 기존 [Kaggle](../../notebooks/kaggle.ipynb) / [Colab 학습](../../notebooks/colab.ipynb)의 선택 셀을 사용한다.

```python
models = ('grt',)
segment_len = 75
num_registers = 8        # 8 / 20 / 75
budget = '20k'
router_hidden = 512
write_gate_bias = 0.0    # A: 0, B: -2 / 0 / +2
write_dropout = 0.0      # A: 0 / .1, B: 해당 M에서 선택한 값
```

Read gate off는 model override에 명시한다. 값을 바꾸면 기존 공통 helper의 설정 hash가 달라져 별도 run이 된다. 같은 값·환경은 기존 last.pt를 재개하므로, 동일 조건을 새 폴더에서 재실행하려면 명시적 run_id를 사용하고, 독립 seed 검증은 seed를 바꾼다. GPU 수가 다른 환경으로 학습을 재개하지 않는다. 초기값·dropout·M·L을 바꾼 기존 checkpoint에서 이어 학습하지 않는다.

순서는 setup → 기존 prepared 데이터 복원 → preflight → train → report다. preflight의 실제 micro/accum·target tokens/update·예상 시간을 확인한다. 새 L/M의 시간을 이전 T4 실측에서 단순 환산하지 않는다. 예약 VRAM 목표95%는 정확한 점유율 보장이 아니다. 장시간 run/checkpoint·데이터는 세션 밖에 보존한다.

학습 후 `job.report(download=True)`의 **experiment_report.json**, 사후 분석 후 **posthoc_report.json**을 보내면 이 문서의 해당 ID에 원본 링크·표·판단을 갱신한다. 여러 환경 결과를 CSV로 누적하지 않는다. 미실행·미측정은 빈칸/상태로 남긴다.

## 6. 준비 검증

관련 CPU 검사55개가 통과했다. 기존 카탈로그에20k 예산을 추가하고, 실제 Kaggle/Colab 선택 셀에서 M3개×bias3개×dropout2개의 설정 전달·고유 run hash·read off·실질 배치256·384M nominal targets를 확인했다. 기존 gate/write 구현과 공통 utils를 사용하며 모델 구조·checkpoint 형식·기존 결과는 바꾸지 않았다. 새20k GPU 학습·수렴·속도는 아직 실측하지 않았다.
