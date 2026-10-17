# L75/M20: 중립 초기화 Refine gate 비교

작성·갱신: 2026-10-17 · 상태: 구현·CPU 검증 완료, GPU Refine on 미실행

[실행 workflow](../20261011_experiment_workflow.md) · [기존 dropout/bias 결과](20261015_l75_write_gates.md) · [원논문](https://arxiv.org/abs/1910.09890)

## 1. 비교 조건

질문은 초기 write 분포를 유지한 채 gate 보정 경로를 추가하면, 초반 고착·문맥별 갱신·품질이 달라지는가이다. 첫 seed54의 off 결과는 기존 A20-0을 재사용하고 on만 새로 학습한다. 유망하면 추가 두 seed에서 on/off 각각 학습해 조건별3개 seed로 확인한다. Uniform Gate Initialization, 추가 regularization loss는 이번에 넣지 않는다.

| 항목 | 조건 |
|---|---|
| 모델 / 데이터 | GRT small D256·6 layers·head8·FF1024, 동일 WikiText-103 GPT-2 BPE artifact |
| L / M / router | 75 / 20 / attention D256·head8·MLP hidden512 |
| Gate | base write bias0, write dropout0, read gate off, 학습 가능한 S0의0 초기화 |
| 학습 | seed54, 새 초기화20k, FP32·global256·BPTT 이전3segments |
| Optimizer | Adam LR5e−4·betas(.9,.999)·decay0·clip.25·cosine20k·warmup0 |
| 평가 | 매1k 전체 validation·기본 RTLA, best validation CE 선택, test 미평가 |
| 첫 비교 환경 | 기존 off는 Kaggle T4×2·micro128×accum1. on도 같은 환경을 우선하며 preflight에서 실제 배치를 확인 |

## 2. 이번 구현의 범위

`model.register.refine_gate`의 카탈로그/typed Config 기본은false다. on은 기존 write hidden h에서 기본 logit a와 추가 logit b를 만든다.

```text
p = sigmoid(a)                         # 기존 write MLP 출력
q = sigmoid(refine_head(h))            # read gate와 별개의 보정 gate
w = p + (2*q - 1)*p*(1-p)             # 최종 유효 write
S_next = (1-w)*S + w*C
```

추가 head의 weight·bias를 모두0으로 초기화해 q=.5, w=p에서 시작한다. 이 초기화는 RNG를 소모하지 않아 같은 seed의 공통 파라미터와 초기 학습 RNG도 보존한다. M20/hidden512에서10,260 parameters가 추가된다. 기존 pooling·MLP hidden을 공유하는 GRT용 적용이며, 논문의 전체 UR-LSTM 재현은 아니다. off는 기존 sigmoid·dropout 경로와 state-dict parameter 구성을 유지한다.

Refine on에서 write dropout을 별도로 켜면 최종 유효 w를0으로 mask해 상태를 보존한다. 이번 비교에서는0으로 고정한다. Gate는 segment 전체 입력과 시작 state로 결정하지만 write-back은 stack 뒤 한 번이며 다음 segment에만 영향을 준다. 기존 token causal 읽기 구조를 유지한다.

## 3. 실행과 전달

기존 [Kaggle](../../notebooks/kaggle.ipynb) / [Colab](../../notebooks/colab.ipynb) 선택 셀을 사용한다. 새 노트북·개별 YAML은 만들지 않는다. 현재 노트북 기본은 다음 on 조건이다.

```python
models = ('grt',)
segment_len = 75
num_registers = 20
budget = '20k'
router_hidden = 512
write_gate_bias = 0.0
write_dropout = 0.0
refine_gate = True
seed = 54
```

`refine_gate`는 기존 `overrides['model']['register']`로, seed는 `WikiTextRun(seed=seed)`로 전달한다. CLI에서는 `--set model.register.refine_gate=true`를 사용한다. 설정 hash가 달라 on/off는 별도 run이다. 기존 off checkpoint에서 on 학습을 재개하지 않는다. off 학습은 이번 첫 비교에서 새로 실행할 필요가 없다.

setup → prepared 데이터 복원 → preflight → train → 기존 마지막 `job.report(posthoc=True, download=True, ...)`를 실행한다. **experiment_results.zip 하나**를 전달하면 학습·사후 JSON을 원본으로 보관하고 아래 결과란에 반영한다.

## 4. 기록할 지표와 판단

주 기준은 같은20k 예산에서 best validation NLL/PPL이며 최종20k도 함께 기록한다. 초기 고착·문맥별 갱신·메모리 기여의 개선은 보조 근거다. 유효 gate의 극단화 자체는 실패 기준이 아니며, 단일 seed 결과를 재현성 주장으로 쓰지 않는다.

- 기존 `grt/write_mean`과 슬롯 유형·그룹 개입은 **유효 w** 기준이다.
- validation마다 `grt/refine/base_write_mean`, `refine_mean`, 두 `*_abs_logit_mean`, 두 `*_logit_derivative_mean`을 학습 JSON·W&B에 추가한다. 국소 logit 미분이며 parameter gradient 측정은 아니다.
- 확장 사후 JSON의 `diagnostics.gate_logits`는 기본 p의 logit/기본 sigmoid slope다. on에는 scope를 명시하고 `diagnostics.refine_gate`에 p·q·q logit 원값과 두 국소 미분을 따로 기록한다. `diagnostics.write_gates`는 최종 w다.
- 같은 사후 protocol: batch8·warmup3·측정8·horizon4·window0/16/32·slots/registers true·random3/seed54.

| Seed | Refine | 상태 | best step / val NLL / PPL | 최종20k PPL | wall h / peak GiB | 학습 / 사후 원본 |
|---:|---|---|---|---:|---|---|
| 54 | off | 기존 A20-0 재사용 | 19,000 / 3.518790 / 33.7436 | 33.7718 | 6.695 / 11.721 | [학습](../../results/86fb2481ac1e45f29da0164863b28d66.json) / [사후](../../results/posthoc/5435927747a34aa895d5804c2b5772a6.json) |
| 54 | on | 미실행 | | | | |
| 55 | off | 1차 판단 후 | | | | |
| 55 | on | 1차 판단 후 | | | | |
| 56 | off | 1차 판단 후 | | | | |
| 56 | on | 1차 판단 후 | | | | |

사용자 판단: seed54에서 후속 seed로 진행할지, 유지할 품질·동작 기준과 근거를 여기에 기록한다.

## 5. 검증 범위

관련 CPU 검사126개가 통과했고 optional W&B package가 필요한2개는 미설치로 생략됐다. on/off 중립 초기화·RNG·초기 forward 일치, recurrent task gradient와 head 갱신, 유효 gate dropout·범위, decoder causality, 정확한 학습 재개, 2-rank CPU 분산 학습·preflight, checkpoint 복원 후 확장 사후 JSON까지 확인했다. 별도로 off router를 변경 전 HEAD 구현과 대조해 parameter·RNG·forward·gradient가 write dropout을 포함해 정확히 일치함을 확인했다. GPU 속도·20k 학습·품질·분화 개선은 사용자 실행 전까지 미검증이다.
