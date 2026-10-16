# L75 small: write dropout과 bias 비교

작성: 2026-10-15 · 갱신: 2026-10-16 · 상태: dropout A단계6회 완료, bias B단계 미실행 · 확장 사후 분석 미수신

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

| M | A의 잠정 선택 dropout | 근거 / best PPL (p0 → p.1) | B 실행 여부 |
|---:|---:|---|---|
| 8 | 0 | 34.1048 → 34.2791; 차이가 작아 p0 우선 | 미실행 |
| 20 | 0 | 33.7436 → 34.2129; 한 seed에서 더 낮은 validation NLL | 미실행 |
| 75 | 0.1 | 33.7629 → 32.8984; 한 seed에서 더 낮은 validation NLL | 미실행 |

선택은 B단계를 진행하기 위한 잠정 기준이다. 사용자 최종 판단 전이며, seed54 한 번의 결과로 유의성을 주장하지 않는다. 선택한 bias0 run을 재사용하고 새 bias−2/+2만 학습하면 된다.

## 3. 학습 결과 기록

각 행은 별도 run이다. B의 bias0 결과는 A에서 선택한 행을 참조해 중복 학습하지 않는다. 실제 micro/accum/GPU·환경·target 예산·코드 hash는 원본 report에 보관하고 차이가 있으면 판단에 적는다.

| ID | M | bias init | write dropout | 상태 | best step / PPL(best·20k) | wall h / peak GiB | 학습 JSON / 확장 사후 JSON |
|---|---:|---:|---|---|---|---|---|
| A8-0 | 8 | 0 | 0 | 완료 | 20,000 / 34.1048 · 34.1048 | 5.511 / 10.849 | [JSON](../../results/ccf041592f3e4ebe8f287e69195205a5.json) / 미수신 |
| A8-1 | 8 | 0 | 0.1 | 완료 | 20,000 / 34.2791 · 34.2791 | 6.054 / 10.849 | [JSON](../../results/746ab4ab41e84ec0bdd0f3f0ed6bffdb.json) / 미수신 |
| A20-0 | 20 | 0 | 0 | 완료 | 19,000 / 33.7436 · 33.7718 | 6.695 / 11.721 | [JSON](../../results/86fb2481ac1e45f29da0164863b28d66.json) / 미수신 |
| A20-1 | 20 | 0 | 0.1 | 완료 | 20,000 / 34.2129 · 34.2129 | 6.442 / 11.721 | [JSON](../../results/51bafa3f611d48c2808a7a55d732c91d.json) / 미수신 |
| A75-0 | 75 | 0 | 0 | 완료 | 20,000 / 33.7629 · 33.7629 | 10.093 / 8.326 | [JSON](../../results/b4b4414578524de0a6e9de50851b91b2.json) / 미수신 |
| A75-1 | 75 | 0 | 0.1 | 완료 | 20,000 / 32.8984 · 32.8984 | 10.041 / 8.326 | [JSON](../../results/06fe396a3e6b4e258b779481409de1ba.json) / 미수신 |
| B8− | 8 | -2 | 0 (잠정) | 미실행 | | | |
| B8+ | 8 | +2 | 0 (잠정) | 미실행 | | | |
| B20− | 20 | -2 | 0 (잠정) | 미실행 | | | |
| B20+ | 20 | +2 | 0 (잠정) | 미실행 | | | |
| B75− | 75 | -2 | 0.1 (잠정) | 미실행 | | | |
| B75+ | 75 | +2 | 0.1 (잠정) | 미실행 | | | |

### 3.1 조건 검증과 품질·비용

6개 모두 Kaggle Tesla T4×2·FP32·global256·동일 model/data seed54·코드 `90d407c`에서20k를 완료했다. Python3.13.15/PyTorch2.11.0+cu128/Transformers4.45.2/CUDA12.8과 prepared manifest가 일치한다. 실제 학습 target은 각383,975,424개, validation은248,420개이며 test는 평가하지 않았다. 같은 M의 두 Config는 write dropout·출력/run 이름 이외에 동일하다. M8/20은 micro128×accum1×GPU2, M75는 micro64×accum2×GPU2다. M 사이의 RNG·수치 차이는 남지만 dropout 쌍 안의 배치는 같다.

| M | p.1−p0 best ΔNLL | p.1−p0 best ΔPPL | PPL 상대 변화 | 잠정 선택 |
|---:|---:|---:|---:|---:|
| 8 | +0.005099 | +0.1744 | +0.51% | 0 |
| 20 | +0.013814 | +0.4694 | +1.39% | 0 |
| 75 | -0.025941 | -0.8646 | -2.56% | 0.1 |

M8은 작은 차이로 p0을 유지하고, M20은 p0, M75는 p.1을 우선한다. 가장 낮은 best PPL은 M75/p.1의32.8984다. **M20/p0의 best는19k(33.7436), 최종20k는33.7718**이므로 selected RTLA도19k 기준으로 정리한다. 나머지는 best=20k다. 최종20k끼리 비교해도 dropout 선택 방향은 같다.

학습 wall time은6개 합계44.836h(T4×2, 단순 GPU-hours89.672)다. 평가·기본 RTLA·저장을 포함하며 preflight·준비는 제외한다. GPU-hours는 Kaggle quota 차감량을 뜻하지 않는다. p.1의 시간 차이는 M별 방향이 달라 dropout의 고유 속도 효과로 단정하지 않는다. 경고·metrics parse 오류·W&B 실패는 없고 모든 run이20k까지 기록됐다.

### 3.2 학습 전환과 후반 경향

| step | M8 p0 | M8 p.1 | M20 p0 | M20 p.1 | M75 p0 | M75 p.1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 111.5782 | 119.7384 | 146.7764 | 155.7687 | 339.4793 | 310.9276 |
| 6,000 | 55.3052 | 90.3683 | 141.2861 | 132.7306 | 766.3398 | 393.8043 |
| 7,000 | 43.1608 | 55.9326 | 45.1793 | 56.1340 | 44.1882 | 53.3275 |
| 12,000 | 36.5148 | 55.1371 | 38.4342 | 49.4047 | 38.3717 | 64.3067 |
| 13,000 | 35.9216 | 36.9852 | 35.8741 | 36.6402 | 40.2011 | 35.9088 |
| 18,000 | 34.2506 | 34.8845 | 34.0446 | 34.4091 | 34.8954 | 33.8390 |
| 19,000 | 34.1368 | 34.4075 | 33.7436 | 34.2452 | 33.7674 | 32.9583 |
| 20,000 | 34.1048 | 34.2791 | 33.7718 | 34.2129 | 33.7629 | 32.8984 |

여러 run이6k까지 악화/정체했다가7k에 크게 회복한다. p.1의 세 run은12k→13k에도 큰 품질 전환이 나타나므로 짧은 pilot로 결과를 판단하면 이번 최종 순위를 놓친다. 마지막1k 개선은 M8 p0/.1이0.0320/0.1283, M20 p0는0.0282 악화·p.1은0.0323 개선, M75 p0/.1은0.0045/0.0600 개선이다. cosine20k 종료의 작은 변화는 고정 예산 후반 안정화 근거이며 완전 수렴 증명은 아니다.

![L75 dropout 비교: 품질·학습 전환·기본 RTLA](../figures/20261016_l75_dropout.png)

## 4. 분화 분석과 판단란

새 학습은 마지막 report 셀에서 각 best checkpoint를 자동 분석한다. 이미 완료된 run은 기존 Colab 사후 분석 노트북에 best checkpoint를 넣어 같은 helper로 분석할 수 있다. batch8·warmup3·측정8·horizon4·window0/16/32·random 반복3/seed54·slots/registers=true를 공통 사용한다. L/M은 checkpoint에서 읽는다. L75에서 warmup225/측정600 tokens이며, 이전 L150 probe와 절대 token 범위는 다르다. 이번 run끼리는 같은 조건으로 비교한다.

| 비교 | 전체 val 품질 / 학습 안정성 | 현재 기본 RTLA에서 보이는 차이 | 판단 / 확장 분석 |
|---|---|---|---|
| M8 dropout0 vs .1 | p0가0.1744 PPL 낮음; p.1의 후반 회복이 늦음 | p.1에서 write·cosine 감소, 품질 개선으로 이어지지는 않음 | p0 잠정; raw gate·그룹 분석 미수신 |
| M20 dropout0 vs .1 | p0가0.4694 PPL 낮음 | p.1에서 write 감소·cosine 증가 | p0 잠정; raw gate·그룹 분석 미수신 |
| M75 dropout0 vs .1 | p.1이0.8646 PPL 낮음 | p.1에서 write·relative update 증가, cosine 감소 | p.1 잠정; raw gate·그룹 분석 미수신 |
| M8 bias−2/0/+2 | 미실행 | | |
| M20 bias−2/0/+2 | 미실행 | | |
| M75 bias−2/0/+2 | 미실행 | | |

### 4.1 받은 report에 실제 포함된 기본 RTLA

best checkpoint의 고정 window0·batch8/warmup3/측정8·L75, 정상 target4,800개의 결과다. eval mode이므로 p.1 학습 run에서도 write dropout은 꺼져 있다. write 평균 차이를 평가 때10%를 무작위로 끈 결과나0.9 rescaling으로 해석하지 않는다.

| ID | best step | probe PPL | write mean | relative update | slot cosine | reset ΔNLL | swap ΔNLL |
|---|---:|---:|---:|---:|---:|---:|---:|
| A8-0 | 20,000 | 36.1425 | 1.000000 | 0.142461 | 0.991730 | 0.478807 | 0.779855 |
| A8-1 | 20,000 | 35.8145 | 0.250000 | 0.037800 | 0.800519 | 0.368968 | 0.750425 |
| A20-0 | 19,000 | 35.8678 | 0.229072 | 0.037654 | 0.693341 | 0.384911 | 0.852954 |
| A20-1 | 20,000 | 35.8817 | 0.100000 | 0.017578 | 0.825199 | 0.330863 | 0.696968 |
| A75-0 | 20,000 | 36.5850 | 0.040000 | 0.006818 | 0.911851 | 0.454511 | 0.767104 |
| A75-1 | 20,000 | 34.9636 | 0.087290 | 0.019172 | 0.633101 | 0.327613 | 0.740750 |

모든 reset/swap 평균이 양수로 해당 probe에서 과거 state가 예측에 도움을 준다. 다만 큰 delta는 잘못된 state에 대한 민감성도 나타내므로 단독으로 최대화하지 않는다. M8/p.1은 probe PPL이 더 낮아도 전체 validation은 p0가 더 낮다. 선택에는 전체 validation을 우선한다.

M8/p0는 write 평균1·cosine≈.992로 강한 포화/유사화가 의심되고, p.1에서는 write.25·cosine≈.801로 바뀌었다. M20에서는 반대로 p.1의 cosine이.825로 높아져 dropout이 항상 분화를 늘린다고 할 수 없다. M75/p.1은 p0보다 cosine이.912→.633, state norm이15.799→2.035로 낮고, write.040→.087·실제 update norm.362→.773은 커졌다. **평균 gate와 norm만으로 세 유형의 비율·동적 분화를 확정하지 않는다.** 이번 report에는 raw gate/logit·슬롯/그룹 개입·rank/CKA가 없다.

write 평균만으로 학습 전환을 설명할 수 없다. M8/p.1은12k→13k에 write.25를 유지하면서 PPL55.1371→36.9852로, M75/p.1도 write≈.0933인 채64.3067→35.9088로 바뀐다. 후자는 cosine.868→.635·reset ΔNLL−.212→+.282 변화도 동반한다. candidate state·backbone·읽기 등의 변화를 구분하려면 확장 분석이 필요하다.

### 4.2 Colab 자동 실행 범위 확인

| 경로 | 자동 실행 내용 | 이번 수신 확인 |
|---|---|---|
| `colab.ipynb` / `kaggle.ipynb`의 `job.train()` | RTLA on이면 validation마다 및 종료 후 best 복원 시 정상/reset/swap·gate/state/update 평균 | 6run 모두 step별 RTLA20건과 selected_rtla 포함 |
| 학습 노트북 마지막 `job.report(posthoc=True, ...)` | best.pt의 확장 분석 후 experiment_results.zip에 학습·사후 JSON 동봉 | 2026-10-16 연결; 기존 수신6파일의 확장 분석은 여전히 미수신 |
| 별도 `colab_analysis.ipynb`의 `job.run()` | 지정 checkpoint의 slot·retention·3window·logit·rank/CKA·group 개입을 일괄 실행 | 이번6파일에 미포함, posthoc_report.json 미수신 |

학습 중 기본 RTLA는 `WikiTextRun` → `train_stream` → `adapter.log_analysis` → `analyze_stream(horizon=0, slots=False)`를 유지한다. 기존 수신6개 report도 horizon0·slot/retention 배열이 비어 있다. **2026-10-16부터 학습 노트북 마지막 셀에서 `job.report(posthoc=True, ...)`가 별도 `WikiTextPosthoc`을 호출해 전체 사후 분석을 실행한다.** 학습 JSON을 먼저 보존하고 두 결과를 ZIP 하나로 묶는다. 이번 코드 연결이 기존6개 checkpoint의 확장 분석 완료를 의미하지는 않는다. 근거는 [학습 루프](../../src/grt/stream_training.py), [공통 notebook helper](../../src/grt/utils/kaggle.py), [기본 분석](../../src/grt/memory_analysis.py)이다.

분화 비교를 위해6조건의 best.pt를 보관하고, 기존 [Colab 분석 노트북](../../notebooks/colab_analysis.ipynb)에서 동일 옵션으로 각 checkpoint를 실행한다. 출력 posthoc_report.json을 보내면 이 문서에 합친다. M20/p0의 best는19k로 다른20k best와 step이 다름을 기록하고, 엄밀한 동일 step 비교가 필요하면 모든 조건의 last.pt/20k를 별도로 분석한다.

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

학습 노트북 마지막 `job.report(posthoc=True, download=True, posthoc_options=...)`의 **experiment_results.zip** 하나를 보내면 안의 학습·사후 JSON을 이 문서의 해당 ID에 반영한다. 마지막 셀에서 batch8·warmup3·측정8·horizon4·slots/registers true·window0/16/32·group_repeats3을 직접 바꿀 수 있다. 기존 학습 JSON과 별도 사후 JSON도 계속 받을 수 있다. 여러 환경 결과를 CSV로 누적하지 않는다. 미실행·미측정은 빈칸/상태로 남긴다.

## 6. 준비 검증

관련 CPU 검사55개가 통과했다. 기존 카탈로그에20k 예산을 추가하고, 실제 Kaggle/Colab 선택 셀에서 M3개×bias3개×dropout2개의 설정 전달·고유 run hash·read off·실질 배치256·384M nominal targets를 확인했다. 기존 gate/write 구현과 공통 utils를 사용하며 모델 구조·checkpoint 형식·기존 결과는 바꾸지 않았다. 2026-10-16 사용자 제공6개 report로20k GPU 학습·시간·기본 RTLA 완료를 확인했다. 확장 사후 분석은 아직 미수신이다.

## 7. 원본과 재현성

| ID | report ID / 원본 | W&B |
|---|---|---|
| A8-0 | [ccf041592f3e4ebe8f287e69195205a5](../../results/ccf041592f3e4ebe8f287e69195205a5.json) | [run](https://wandb.ai/joyyoj1-korea-university/grt-research/runs/748b71bf) |
| A8-1 | [746ab4ab41e84ec0bdd0f3f0ed6bffdb](../../results/746ab4ab41e84ec0bdd0f3f0ed6bffdb.json) | [run](https://wandb.ai/joyyoj1-korea-university/grt-research/runs/edf5664c) |
| A20-0 | [86fb2481ac1e45f29da0164863b28d66](../../results/86fb2481ac1e45f29da0164863b28d66.json) | [run](https://wandb.ai/joyyoj1-korea-university/grt-research/runs/54d2c635) |
| A20-1 | [51bafa3f611d48c2808a7a55d732c91d](../../results/51bafa3f611d48c2808a7a55d732c91d.json) | [run](https://wandb.ai/joyyoj1-korea-university/grt-research/runs/48443a79) |
| A75-0 | [b4b4414578524de0a6e9de50851b91b2](../../results/b4b4414578524de0a6e9de50851b91b2.json) | [run](https://wandb.ai/joyyoj1-korea-university/grt-research/runs/feb72c65) |
| A75-1 | [06fe396a3e6b4e258b779481409de1ba](../../results/06fe396a3e6b4e258b779481409de1ba.json) | [run](https://wandb.ai/joyyoj1-korea-university/grt-research/runs/761c6ca9) |

원본 JSON bytes를 그대로 보관했다. best·최종20k·selected RTLA를 구분해 검증했고, 학습 curve20개 평가점·기본 RTLA20건·환경·Config를 대조했다. 결과 정리 시 학습을 추가 실행하지 않았다. 이후 2026-10-16 사용자 요청으로 학습 노트북 마지막 셀의 확장 분석과 ZIP 전달을 연결했다. 기존 수신 원본은 수정하지 않았다.
