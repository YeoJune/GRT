# L75 small: write dropout과 bias 비교

작성: 2026-10-15 · 갱신: 2026-10-17 · 상태: dropout A단계6개 및 M20 bias−2/+2 학습·확장 사후 완료; B단계 M8/M75 미실행

[공통 실행 안내](../20261011_experiment_workflow.md) · [이전 L150 용량 실험](20261012_small_register_sweep.md) · [원본 보관](../../results/README.md) · [Write 구조·regularization 설계 검토](../20261016_write_regularization.md)

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

**B. 사용자 최신 결정에 따라 모든 M에서 dropout0을 고정하고 bias−2/0/+2를 비교한다(추가6회).** bias0은 A8-0/A20-0/A75-0의 run/checkpoint/report를 그대로 재사용한다. bias만 바꾼 두 조건은 매번 새로 초기화해20k 학습한다. A의 품질 기준 dropout 선택은 M별 결과로 남기되 이번 B에는 적용하지 않는다. B는 세 M 모두 dropout0으로 통제한다.

Bias는 모든 슬롯의 write MLP 마지막 bias 초깃값이며 학습 가능하다. bias−2/0/+2의 sigmoid 기준값은 약.119/.500/.881이지만 실제 gate는 hidden 출력에 따라 달라진다. write dropout.1은 학습 중 example·slot별 확률.1로 write logit을−∞로 만들어 기존 state를 보존하며, state를 지우거나 rescaling하지 않는다. 평가·사후 분석에서는 dropout을 끈다. 새 정규화·분화 보조 손실은 이번 비교에 넣지 않는다.

| M | A의 품질 기준 잠정 dropout | B의 고정 dropout | 근거 / best PPL (p0 → p.1) | B 실행 여부 |
|---:|---:|---:|---|---|
| 8 | 0 | 0 | 34.1048 → 34.2791; 차이가 작아 p0 우선 | 미실행 |
| 20 | 0 | 0 | 33.7436 → 34.2129; 한 seed에서 더 낮은 validation NLL | −2/0/+2 완료 |
| 75 | 0.1 | 0 | 33.7629 → 32.8984; 한 seed에서 더 낮은 validation NLL | 미실행 |

A의 선택은 seed54 한 번의 품질 기준이며 유의성을 주장하지 않는다. B에서는 dropout0의 bias0 결과를 재사용하고 새 bias−2/+2만 학습한다.

## 3. 학습 결과 기록

각 행은 별도 run이다. B의 bias0 결과는 A의 dropout0 행을 참조해 중복 학습하지 않는다. 실제 micro/accum/GPU·환경·target 예산·코드 hash는 원본 report에 보관하고 차이가 있으면 판단에 적는다.

| ID | M | bias init | write dropout | 상태 | best step / PPL(best·20k) | wall h / peak GiB | 학습 JSON / 확장 사후 JSON |
|---|---:|---:|---|---|---|---|---|
| A8-0 | 8 | 0 | 0 | 완료 | 20,000 / 34.1048 · 34.1048 | 5.511 / 10.849 | [학습](../../results/ccf041592f3e4ebe8f287e69195205a5.json) / [사후](../../results/posthoc/10b47cdf69034d8d91a842f60f226ede.json) |
| A8-1 | 8 | 0 | 0.1 | 완료 | 20,000 / 34.2791 · 34.2791 | 6.054 / 10.849 | [학습](../../results/746ab4ab41e84ec0bdd0f3f0ed6bffdb.json) / [사후](../../results/posthoc/17f1bc032be54c249e645fa01275bac2.json) |
| A20-0 | 20 | 0 | 0 | 완료 | 19,000 / 33.7436 · 33.7718 | 6.695 / 11.721 | [학습](../../results/86fb2481ac1e45f29da0164863b28d66.json) / [사후](../../results/posthoc/5435927747a34aa895d5804c2b5772a6.json) |
| A20-1 | 20 | 0 | 0.1 | 완료 | 20,000 / 34.2129 · 34.2129 | 6.442 / 11.721 | [학습](../../results/51bafa3f611d48c2808a7a55d732c91d.json) / [사후](../../results/posthoc/1ae377fe64e243749040cf61991c37f0.json) |
| A75-0 | 75 | 0 | 0 | 완료 | 20,000 / 33.7629 · 33.7629 | 10.093 / 8.326 | [학습](../../results/b4b4414578524de0a6e9de50851b91b2.json) / [사후](../../results/posthoc/f5a1825cc5d84f6da400eb249ec409ac.json) |
| A75-1 | 75 | 0 | 0.1 | 완료 | 20,000 / 32.8984 · 32.8984 | 10.041 / 8.326 | [학습](../../results/06fe396a3e6b4e258b779481409de1ba.json) / [사후](../../results/posthoc/df797486d1cd4606acc91fc478535e33.json) |
| B8− | 8 | -2 | 0 | 미실행 | | | |
| B8+ | 8 | +2 | 0 | 미실행 | | | |
| B20− | 20 | -2 | 0 | 완료 | 19,000 / 32.6100 · 32.6552 | 6.423 / 11.721 | [학습](../../results/3001e7328e87456395275745ada7b946.json) / [사후](../../results/posthoc/3cb39cb5ceaf40e4b2ac4790af009b6d.json) |
| B20+ | 20 | +2 | 0 | 완료 | 19,000 / 33.0599 · 33.0679 | 6.698 / 11.721 | [학습](../../results/0bb9e51192d54a398d771bab611e9fb1.json) / [사후](../../results/posthoc/b9d1e1ea89bd4499989c6968dfc63957.json) |
| B75− | 75 | -2 | 0 | 미실행 | | | |
| B75+ | 75 | +2 | 0 | 미실행 | | | |

### 3.1 조건 검증과 품질·비용

6개 모두 Kaggle Tesla T4×2·FP32·global256·동일 model/data seed54·코드 `90d407c`에서20k를 완료했다. Python3.13.15/PyTorch2.11.0+cu128/Transformers4.45.2/CUDA12.8과 prepared manifest가 일치한다. 실제 학습 target은 각383,975,424개, validation은248,420개이며 test는 평가하지 않았다. 같은 M의 두 Config는 write dropout·출력/run 이름 이외에 동일하다. M8/20은 micro128×accum1×GPU2, M75는 micro64×accum2×GPU2다. M 사이의 RNG·수치 차이는 남지만 dropout 쌍 안의 배치는 같다.

| M | p.1−p0 best ΔNLL | p.1−p0 best ΔPPL | PPL 상대 변화 | 잠정 선택 |
|---:|---:|---:|---:|---:|
| 8 | +0.005099 | +0.1744 | +0.51% | 0 |
| 20 | +0.013814 | +0.4694 | +1.39% | 0 |
| 75 | -0.025941 | -0.8646 | -2.56% | 0.1 |

M8은 작은 차이로 p0을 유지하고, M20은 p0, M75는 p.1을 우선한다. A단계 안에서 가장 낮은 best PPL은 M75/p.1의32.8984다. **M20/p0의 best는19k(33.7436), 최종20k는33.7718**이므로 selected RTLA도19k 기준으로 정리한다. 나머지는 best=20k다. 최종20k끼리 비교해도 dropout 선택 방향은 같다.

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

학습 manifest의 train118,436,938 tokens를 global256으로 나누면 stream당462,644 tokens이고 epoch당6,169 updates다. 코드가6,169/12,338/18,507 이후 epoch를 시작할 때 state와 BPTT 이력을 초기화한다. 큰 전환이 그 직후 첫 평가와 맞물리므로, 학습 전환을 분화 발생 시점으로만 해석하지 않는다. 데이터 재노출·S0 재노출·state 초기화·LR 변화의 인과관계는 현재 로그만으로 분리되지 않는다.

![L75 dropout 비교: 품질·학습 전환·기본 RTLA](../figures/20261016_l75_dropout.png)

## 4. 분화 분석과 판단란

새 학습은 마지막 report 셀에서 각 best checkpoint를 자동 분석한다. 이미 완료된 run은 기존 Colab 사후 분석 노트북에 best checkpoint를 넣어 같은 helper로 분석할 수 있다. batch8·warmup3·측정8·horizon4·window0/16/32·random 반복3/seed54·slots/registers=true를 공통 사용한다. L/M은 checkpoint에서 읽는다. L75에서 warmup225/측정600 tokens이며, 이전 L150 probe와 절대 token 범위는 다르다. 이번 run끼리는 같은 조건으로 비교한다.

| 비교 | 전체 val 품질 / 학습 안정성 | 현재 기본 RTLA에서 보이는 차이 | 판단 / 확장 분석 |
|---|---|---|---|
| M8 dropout0 vs .1 | p0가0.1744 PPL 낮음; p.1의 후반 회복이 늦음 | p.1에서 write·cosine 감소, 품질 개선으로 이어지지는 않음 | p0 잠정; 열린8→2·actual update rank 감소, 품질 이득 없음 |
| M20 dropout0 vs .1 | p0가0.4694 PPL 낮음 | p.1에서 write 감소·cosine 증가 | p0 잠정; 열린4/중간형2→열린2, 극단 포화·갱신 rank 감소 |
| M75 dropout0 vs .1 | p.1이0.8646 PPL 낮음 | p.1에서 write·relative update 증가, cosine 감소 | p.1 잠정; 중간형5개에 문맥 기여 집중, 품질 개선과 일치 |
| M8 bias−2/0/+2 | 미실행 | | |
| M20 bias−2/0/+2 | best PPL32.6100 / 33.7436 / 33.0599 | −2는 문맥별 gate·갱신 다양성 증가, +2는 극단 이진 포화 | −2 잠정 우선; 전체 val 개선과 짧은 probe 악화를 구분(4.9) |
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

M8/p0는 write 평균1·cosine≈.992로 강한 포화/유사화가 의심되고, p.1에서는 write.25·cosine≈.801로 바뀌었다. M20에서는 반대로 p.1의 cosine이.825로 높아져 dropout이 항상 분화를 늘린다고 할 수 없다. M75/p.1은 p0보다 cosine이.912→.633, state norm이15.799→2.035로 낮고, write.040→.087·실제 update norm.362→.773은 커졌다. **평균 gate와 norm만으로 세 유형의 비율·동적 분화를 확정하지 않는다.** 학습 report에는 raw gate/logit·슬롯/그룹 개입·rank/CKA가 없으며, 이번 수신 사후 report로 아래4.3~4.8에 보완했다.

write 평균만으로 학습 전환을 설명할 수 없다. M8/p.1은12k→13k에 write.25를 유지하면서 PPL55.1371→36.9852로, M75/p.1도 write≈.0933인 채64.3067→35.9088로 바뀐다. 후자는 cosine.868→.635·reset ΔNLL−.212→+.282 변화도 동반한다. candidate state·backbone·읽기 등의 변화를 구분하려면 확장 분석이 필요하다.

### 4.2 Colab 자동 실행 범위 확인

| 경로 | 자동 실행 내용 | 이번 수신 확인 |
|---|---|---|
| `colab.ipynb` / `kaggle.ipynb`의 `job.train()` | RTLA on이면 validation마다 및 종료 후 best 복원 시 정상/reset/swap·gate/state/update 평균 | 6run 모두 step별 RTLA20건과 selected_rtla 포함 |
| 학습 노트북 마지막 `job.report(posthoc=True, ...)` | best.pt의 확장 분석 후 experiment_results.zip에 학습·사후 JSON 동봉 | 2026-10-16 연결; 이번에 별도 사후 JSON6개 수신 |
| 별도 `colab_analysis.ipynb`의 `job.run()` | 지정 checkpoint의 slot·retention·3window·logit·rank/CKA·group 개입을 일괄 실행 | 같은 helper의 사후 JSON6개 완료 확인; 사용 진입점은 파일만으로 구분하지 않음 |

학습 중 기본 RTLA는 `WikiTextRun` → `train_stream` → `adapter.log_analysis` → `analyze_stream(horizon=0, slots=False)`를 유지한다. 기존 수신6개 report도 horizon0·slot/retention 배열이 비어 있다. **2026-10-16부터 학습 노트북 마지막 셀에서 `job.report(posthoc=True, ...)`가 별도 `WikiTextPosthoc`을 호출해 전체 사후 분석을 실행한다.** 학습 JSON을 먼저 보존하고 두 결과를 ZIP 하나로 묶는다. 이후 수신한 별도 사후 JSON6개로 기존6개 best checkpoint의 확장 분석 완료를 확인했다. 근거는 [학습 루프](../../src/grt/stream_training.py), [공통 notebook helper](../../src/grt/utils/kaggle.py), [기본 분석](../../src/grt/memory_analysis.py)이다.

6조건의 best.pt 사후 분석을 아래에 통합했다. 추가 checkpoint는 기존 [Colab 분석 노트북](../../notebooks/colab_analysis.ipynb)에서 같은 옵션으로 분석하거나 학습 마지막 report 셀을 사용한다. M20/p0의 best는19k로 다른20k best와 step이 다름을 기록하고, 엄밀한 동일 step 비교가 필요하면 모든 조건의 last.pt/20k를 별도로 분석한다.

세 유형의 균등 분배나 높은 rank 자체를 성공 기준으로 삼지 않는다. threshold 민감도·동일 개수 무작위 그룹·단일 효과의 비가산성·reset과swap 차이를 함께 본다. 학습 중 RTLA로 전환 시점을 보고, 최종 checkpoint 사후 관측으로 학습 원인을 확정하지 않는다. 추가 seed·긴 학습·다음 구조 실험은 아래 사후 해석과 사용자 판단을 바탕으로 결정한다.

### 4.3 사후 분석의 연결·집계 기준

6개 모두 원래 학습 Config·prepared manifest·selected step·학습 중 best RTLA의 모든 scalar 지표가 정확히 일치한다. M20/p0는19k, 나머지는20k이며, 파일명 대신 checkpoint의 실제 Config로 M/dropout을 연결했다. checkpoint hash는 각 사후 원본에 보관한다. 학습 JSON 자체에는 hash가 없으므로 전달받지 않은 가중치를 별도로 재해시한 것은 아니다.

사후 실행은 코드 `fbb9a53`·clean checkout·동일30개 source hash(해당 분석 실행 간 일치)·Colab T4 단일 GPU·FP32/TF32 off다. Python3.13.15/PyTorch2.11.0+cu130/CUDA13.0/Transformers4.45.2가6개 모두 같다. 원래 학습의 PyTorch cu128/CUDA12.8과는 다르지만 첫 probe의 기존 RTLA는 전 항목 동일하게 재현됐다. 설정은 batch8·warmup3·측정8·horizon4·slots/registers true·window0/16/32·random3·seed54·손상 비율10/25/50/75%다.

**window0에서 그룹을 정의하고, 아래 별도 표의 대표값은 평가 window16·32의 동일 가중 평균으로 집계한다.** 각 window/lag에4,800 targets가 있어 token 가중 평균과 같다. 세 window는 각각 S0로 시작해225 tokens warmup 후600 tokens를 측정하며, 이어225 tokens까지 개입 branch를 추적한다. 연속적인 긴 문맥3개를 이어서 분석한 것이 아니다. 단일 슬롯 기여는 window0에서만 측정한다. random 범위는 같은3개 mask 각각의 두 평가 window 평균에서 min/max를 계산하며 신뢰구간이 아니다.

| ID | checkpoint step | 사후 T4 seconds | peak MiB | checkpoint SHA256 앞12자리 |
|---|---:|---:|---:|---|
| A8-0 | 20,000 | 61.36 | 337.71 | `5aa9d458db84` |
| A8-1 | 20,000 | 94.44 | 337.71 | `be6f249c6745` |
| A20-0 | 19,000 | 163.58 | 338.43 | `919e91014cda` |
| A20-1 | 20,000 | 104.86 | 338.43 | `d56a848c6426` |
| A75-0 | 20,000 | 112.75 | 341.71 | `1de8e1540d5c` |
| A75-1 | 20,000 | 178.46 | 341.71 | `9f53f8caa514` |

사후 분석은6개 합계715.44초(11.92분, 단일 T4 약0.199 GPU-hours)다. 보고된 분석 본체 시간이며 코드 설치·Drive 복사 등은 제외한다. 이번 고정 probe의 비용으로, 모델 크기·측정 길이·슬롯 수를 늘린 분석 시간으로 일반화하지 않는다.

### 4.4 Gate 분화와 포화: dropout의 방향이 M마다 다르다

closed는 calibration의 모든 관측에서 gate<=.01, open은 모두>=.99, other는 그 밖이다. **other는 관측 분류이며 자동으로 유용한 선택형 메모리를 의미하지 않는다.** 아래 포화율·logit·router 값은 두 평가 window 평균이다. |logit|>20은 극단 포화의 기술 지표이며 닫힘/열림 양쪽을 포함한다.

| ID | closed / open / other (window0) | 평가 window 유형 유지율 | \|logit\|>20 | median \|logit\| | router context norm / MLP hidden norm |
|---|---|---|---:|---:|---:|
| A8-0 | 0 / 8 / 0 | 100.00% / 100.00% | 100.00% | 59.24 | 87.09 / 272.11 |
| A8-1 | 6 / 2 / 0 | 100.00% / 100.00% | 100.00% | 153.25 | 91.24 / 396.98 |
| A20-0 | 14 / 4 / 2 | 95.00% / 100.00% | 35.16% | 18.08 | 44.36 / 108.06 |
| A20-1 | 18 / 2 / 0 | 100.00% / 100.00% | 100.00% | 276.33 | 160.74 / 778.41 |
| A75-0 | 72 / 3 / 0 | 100.00% / 100.00% | 100.00% | 88.29 | 134.57 / 484.43 |
| A75-1 | 68 / 2 / 5 | 98.67% / 97.33% | 4.34% | 10.63 | 6.29 / 37.54 |

M8/p0·M8/p.1·M20/p.1·M75/p0는 세 window와 threshold.001/.01/.05 모두 동일한 이진 그룹 분할이다. M8/p.1은6개 닫힘/2개 열림으로 나뉘지만 모든 logit이 극단 포화돼, 부드럽고 문맥에 따라 쓰기를 조절하는 슬롯이 늘어난 것은 아니다. M20/p.1도18/2/0으로 변하며 median |logit|이18.08→276.33, hidden norm이108.06→778.41로 커진다. dropout이 router를 항상 안정화하지는 않는다.

M75/p.1은 반대로 |logit|>20 비율100%→4.34%, context norm134.57→6.29, hidden norm484.43→37.54로 낮아진다. window0의68/2/5는 평가 window에서68/3/4·68/4/3으로 바뀐다. threshold.001/.01/.05에 따른 window0 그룹은67/0/8·68/2/5·68/4/3이므로 정확한 세 유형 비율은 threshold에 민감하다. 다만 기본 threshold에서 닫히지 않은7개 슬롯 묶음은 두 평가 window에서도 유지되고, 그 내부의 open/other 분류가 달라진다.

마지막 write bias의 학습 후 절댓값 최대는6run 모두.042 미만이다. 따라서 현재 극단 logit을 마지막 bias 값 자체만으로 설명할 수 없으며 context·hidden·출력 weight 경로의 기여가 크다. 이 관찰이 초기 bias의 최적화 경로 영향을 배제하지는 않으므로 예정된 bias 실험은 여전히 의미가 있다.

### 4.5 표현 다양성: cosine 감소와 실제 갱신의 분화를 구분한다

rank는 각 state의 slot×feature 행렬에서 squared singular value 에너지로 구한 entropy effective rank를 평균한 값이다. 문맥 평균 제거는 슬롯별 관측 평균을 빼고, actual update는 실제 S_next−S다. CKA는 같은64개 segment/example 관측의 off-diagonal 슬롯쌍 평균이며, 문맥에 따른 표현 변화 유사도의 기술 통계다.

| ID | raw state rank | 문맥 평균 제거 rank | 실제 update rank | 평균 slot CKA |
|---|---:|---:|---:|---:|
| A8-0 | 1.0468 | 1.2156 | 1.5821 | 0.9931 |
| A8-1 | 1.7987 | 1.7898 | 1.0486 | 0.9310 |
| A20-0 | 2.8119 | 3.0334 | 2.4098 | 0.9016 |
| A20-1 | 1.6723 | 1.8974 | 1.0508 | 0.9654 |
| A75-0 | 1.7026 | 2.0655 | 1.8039 | 0.9757 |
| A75-1 | 1.9522 | 2.2142 | 2.4346 | 0.9150 |

M8의 dropout은 state rank1.05→1.80·전체 CKA.993→.931로 표현 차이를 늘리지만, actual update rank1.58→1.05다. 두 open 슬롯 사이 CKA도.9986으로 매우 높다. 즉 닫힌 묶음/열린 묶음 사이의 차이가 커진 것과, 갱신하는 슬롯끼리 정보가 다양해진 것은 다르다.

M20은 문맥 평균 제거 rank3.03→1.90·update rank2.41→1.05·CKA.902→.965로, dropout 후 표현·갱신이 더 유사해지는 방향이다. open 내부 CKA도.844→.998로 높아진다. M75는 update rank1.80→2.43·CKA.976→.915로 개선되며 정상 validation 개선과 일치한다. 다만75개 슬롯에 비해 rank는 여전히 낮고, M20/p0가 문맥 평균 제거 rank는 더 높아도 최저 PPL은 M75/p.1이다. rank 최대화를 성능 목표로 삼지 않는다. M75/p.1에서도 전체 CKA는.915로 여전히 높고, open2개 내부는.9904, other5개 내부는.8509다. 낮아진 cosine만으로75개 슬롯이 독립적으로 분화됐다고 할 수는 없다. CKA는 관측64개/feature256의 제한과 작은 변동 진폭에도 높아질 수 있는 특성이 있어 gate·그룹 개입과 함께 본다.

### 4.6 그룹 개입: M75의 중간형은 기능적으로도 구별된다

아래는 각 calibration 그룹을 따로 reset/swap한 평가 window의 lag0 ΔNLL이다. 양수면 손상, 음수면 해당 probe에서 개선이다. **원래 group mask를 평가 window에서도 그대로 사용하며, 각 행의 효과를 합산하지 않는다.**

| ID | closed reset / swap | open reset / swap | other reset / swap | 전체 reset / swap |
|---|---:|---:|---:|---:|
| A8-0 | +0.0000 / +0.0000 | +0.4549 / +0.6553 | +0.0000 / +0.0000 | +0.4549 / +0.6553 |
| A8-1 | -0.0078 / -0.0031 | +0.3017 / +0.5824 | +0.0000 / +0.0000 | +0.3622 / +0.6191 |
| A20-0 | +0.0155 / -0.0028 | +0.1045 / +0.4575 | -0.0005 / +0.0246 | +0.3547 / +0.7480 |
| A20-1 | +0.0043 / -0.0018 | +0.2977 / +0.5880 | +0.0000 / +0.0000 | +0.3053 / +0.5932 |
| A75-0 | +0.0825 / -0.0007 | +0.2190 / +0.5938 | +0.0000 / +0.0000 | +0.4389 / +0.6804 |
| A75-1 | +0.0151 / -0.0007 | +0.0068 / +0.0397 | +0.1178 / +0.3678 | +0.3002 / +0.6373 |

M8/p.1과M20/p.1은 open2개에 swap 손상이 각각.5824/.5880으로 집중된다. 같은 크기의 무작위 그룹 평균은.0363/−.0009다. 전체 swap 손상.6191/.5932와 가깝지만 이를 기여율로 나누지는 않는다. 두 슬롯의 높은 CKA·update rank≈1을 함께 보면, 크기가 작은 동적 경로에 기능이 집중된 상태와 부합한다.

M20/p0의 other2개는 단독 reset−.0005·swap+.0246으로 작지만, 다른 슬롯을 모두 바꾼 상태에서 이 둘만 정상으로 복원하면 reset 손상을.2314, swap 손상을.2659 줄인다. 무작위 대조군3개 중 하나가 우연히 other2개와 완전히 같아 other 단독 효과는 대조군 최대와 같다. 따라서 이 조건의 other를 독립적으로 특별하다고 과장하지 않으며, 큰 **조건부 복원 효과와 상호작용**을 기록한다.

M75/p.1에서는 other5개가 두 평가 window 각각에서 swap+.3707/+.3649이며, 동일 크기 random 최대+.0081/+.0105를 크게 넘는다. 두 window를 평균한 비교는 다음과 같다.

| M75/p.1 그룹 | 슬롯 수 | reset ΔNLL | swap ΔNLL | 같은 수 random swap 평균 [min,max] | 전체 swap 후 해당 그룹 정상 복원 효과 |
|---|---:|---:|---:|---|---:|
| open | 2 | +0.0068 | +0.0397 | -0.0004 [-0.0004, -0.0004] | +0.2380 |
| other | 5 | +0.1178 | +0.3678 | +0.0026 [-0.0008, +0.0093] | +0.5929 |
| open+other | 7 | +0.2604 | +0.5974 | +0.0301 [+0.0225, +0.0452] | +0.6380 |

M75/p.1의 open2개보다 other5개를 교환했을 때 손상이 크고, other만 정상으로 복원하는 효과도.5929다. other의 swap 영향은 lag3에도.0305 남고 같은 크기 random 평균은−.0007이다. **M75에서는 dropout 후 생긴 중간형 묶음이 문맥에 유용한 정보를 담는다는 개입 근거가 있다.** 다만 서로 다른 두 모델을 비교한 것이므로 이 분화가 PPL 개선의 원인임을 확정하거나, 모든 중간형 슬롯이 같은 의미적 역할을 한다고 결론내리지는 않는다.

닫힌 슬롯도 곧 미사용은 아니다. M8/p.1 closed6개는 측정 중 update norm=0이지만 문맥별 state 변동 norm 평균12.88이고, M75/p0 closed72개도 각각0·8.29다. 측정 전 warmup에서 형성된 문맥 상태를 보존할 수 있기 때문이다. 특히 M75/p0의 closed reset은+.0825이지만 swap은−.0007로, S0와의 차이와 올바른 donor 문맥의 중요성이 다르게 나타난다. M75/p.1 closed는 변동 norm.0474·update norm.00084로 더 작아, 전체 슬롯 CKA만으로 닫힌 묶음의 정보량을 비교하면 오해하기 쉽다. reset/swap 분석은 슬롯을 삭제하거나 attention에서 제외한 실험이 아니므로 pruning 성능을 보장하지 않는다.

단일 슬롯 초기화 효과의 합계도 활용성 지표로 충분하지 않다. window0에서 M20/p.1의 단일 reset ΔNLL 합은−.1076이지만 전체 reset은+.3309, M75/p.1은−.0124이지만 전체 reset은+.3276이다. M8/p0도 합+.1655 대 전체+.4788이다. 그룹 단위에서도 M75/p.1은 전체 swap.6373과 세 그룹 단독 합.4068 사이 차이가.2305다. 음의 개별 효과나 작은 단독 효과를 근거로 불필요 슬롯을 확정하지 않는다.

### 4.7 Retention과 무작위 손상: 장기 보존 개선은 확인되지 않았다

한 번 전체 state를 swap한 뒤 정상 입력을 계속 주었을 때의 ΔNLL이다. lag0는 현재 segment, lag3은225 tokens 뒤 segment이며 분기 전체가 원래 trajectory와 다를 수 있다. 손상이 오래 남는 것은 유용한 정보의 장기 보존과 동의어가 아니다.

| ID | lag0 | lag1 | lag2 | lag3 |
|---|---:|---:|---:|---:|
| A8-0 | 0.6553 | 0.1770 | 0.0655 | 0.0287 |
| A8-1 | 0.6191 | 0.1566 | 0.0543 | 0.0234 |
| A20-0 | 0.7480 | 0.2533 | 0.1116 | 0.0597 |
| A20-1 | 0.5932 | 0.1590 | 0.0557 | 0.0226 |
| A75-0 | 0.6804 | 0.2377 | 0.1006 | 0.0497 |
| A75-1 | 0.6373 | 0.1847 | 0.0751 | 0.0392 |

세 M 모두 dropout 후 평균 swap 손상이 lag0~3에서 작아진다. M20은 lag3 .0597→.0226으로 특히 줄어든다. 정상 PPL도 악화되고 update rank도 낮아져 장기 문맥 활용이 약해졌을 가능성과 부합하지만, 서로 다른 모델의 손상 회복률이므로 원인을 단정하지 않는다. M75는.0497→.0392로 줄어도 정상 품질이 좋아진다. 따라서 dropout이 전반적으로 장기 기억을 늘렸다는 주장은 지지되지 않으며, 더 나은 표현·조건부 활용과 개입 강건성 변화로 나누어 해석한다.

무작위 슬롯50% 교환의 lag0 손상은 M8 .1438→.1247, M20 .1138→.1166, M75 .0994→.2084다. M75는 전체 swap 손상이 줄어도 부분 swap 손상은 오히려 커질 수 있다. 사용 슬롯의 위치·개수, 원래/교환 문맥이 섞이는 효과와 그룹 상호작용이 관여하므로 dropout을 보편적 손상 강건성 개선으로 부르지 않는다. 무작위 mask는3개뿐이며 손상 곡선의 작은 차이를 통계적 차이로 해석하지 않는다.

![L75 dropout 사후 비교: gate·표현·그룹 개입·retention·단일 슬롯 비가산성](../figures/20261016_l75_dropout_posthoc.png)

### 4.8 이번 사후 분석을 반영한 판단

**성능 기준 잠정 선택은 M8=0, M20=0, M75=.1로 유지한다.** M8의 dropout은 외형적 표현 차이를 늘려도 갱신 effective rank≈1·open2개에 집중되고 품질은 좋아지지 않는다. M20은 dropout 후 극단 포화·중간형 소실·갱신 rank 감소·PPL 악화가 함께 관측돼 이번 조건에서 부정적 근거가 가장 일관된다. M75는 극단 포화 완화·중간형의 무작위 대조군 대비 기여·갱신 rank 상승·PPL 개선이 함께 관측돼 가장 유망하다.

이번 결과는 “세 유형이 있다는 것”보다 **그룹의 비율·포화 정도·문맥별 표현·조건부 기여를 함께 봐야 한다**는 판단을 강화한다. 균등 비율이 최적이라는 근거는 없고, 같은 dropout도 M에 따라 반대 결과를 낸다. 분화 실험의 목표를 단순 cosine 감소나 other 수 증가로 두지 않고, validation 품질과 held-out 그룹 기여의 동반 개선으로 둔다.

당시 다음 단계로 정한 bias−2/0/+2 중 M20 결과를2026-10-17에 수신해4.9에 추가했다. M8/M75는 미실행이다. 각 M에서 새 초기화로 비교하고 위 지표와 gate/context scale을 같이 본다. 추가 확인의 우선순위는 warmup 길이 변화에 따른 그룹 안정성, epoch 경계 전후 checkpoint의 gate·표현 변화, 효과가 큰 M75의 추가 seed다. 현재 분석은 단일 seed·세 짧은 window·horizon4의 최종 관측이며 긴 문맥/완전 수렴·분화의 학습 원인·다른 seed 재현성을 검증한 것은 아니다. 새로운 학습이나 분석 코드는 이번 결과 정리에 추가하지 않았다.

### 4.9 M20 bias 실험: −2는 갱신 다양성과 전체 validation이 함께 개선됐다

2026-10-17 수신 학습2개·사후2개는 **L75/M20/bias−2 및+2**다. 기존 A20-0(bias0)을 대조군으로 연결했다. 세 조건 모두 dropout0·read off·seed54·T4×2·FP32·micro128×accum1×GPU2·20k이며, Config는 bias·출력/run 이름 외에 동일하다. 실제 학습383,975,424 targets·전체 validation248,420 targets·prepared manifest도 일치한다. **세 best가 모두19k/364,775,424 학습 targets**이므로 아래 best 사후 비교는 학습 step이 같다. 테스트는 평가하지 않았다.

학습 코드는 기존 `90d407c`, 신규 `56510e6`으로 다르지만 models·stream_training·config·wikitext·memory_analysis·rtla의 diff는 없다. 사후 코드는 기존 `fbb9a53`, 신규 `1d495c8`이며 모든 분석 source hash·프로토콜·실행 환경이 동일하다. 사후 Config·manifest·checkpoint step·기본 RTLA scalar가 각 학습 report의 selected 결과와 정확히 일치한다. 실행 환경은3.1/4.3과 같고 경고·metrics parse 오류는 없다. 새로운 두 학습의 wall time 합계13.120h(T4×2), 사후 분석 합계162.77초(단일 T4)다.

**품질과 학습 경로.**

| bias init | best19k NLL / PPL | 최종20k PPL | bias0 대비 best ΔNLL / ΔPPL | 15k~20k PPL 평균 |
|---:|---:|---:|---:|---:|
| -2 | 3.484620 / 32.6100 | 32.6552 | -0.034170 / -1.1336 (-3.36%) | 33.2183 |
| +0 | 3.518790 / 33.7436 | 33.7718 | +0.000000 / +0.0000 (+0.00%) | 34.0582 |
| +2 | 3.498322 / 33.0599 | 33.0679 | -0.020469 / -0.6837 (-2.03%) | 34.0192 |

−2는 best PPL이0보다3.36% 낮고,13k~20k의8개 평가점 모두0보다 좋다. +2도 best는2.03% 좋지만15k~20k 평균은0과 거의 같고19k에 큰 개선이 집중된다. 평균은6개 평가 checkpoint의 기술 통계이며 독립 반복의 평균이나 신뢰구간이 아니다. 최종20k 순위도−2 < +2 < 0이다. −2가 이번 L75 수신 run 중 최저 PPL이지만, M75/p.1과는 M·dropout이 함께 달라 bias 효과의 대조군으로 삼지 않는다.

−2의 write 평균은1k .3846 → 8k .2041 → 12k .2275 → 13k .3387 → 19k .3666으로 변한다. +2는1k~6k 약.400,7k 이후 약.350으로 거의 고정된다. 첫 기록은1k여서 step0의 실제 gate나 포화 발생 시점을 알 수 없다. −2도6k PPL187.4263(0은141.2861)으로 초기 불안정성이 있으며, 완전 수렴이나 전 구간 안정화가 확인된 것은 아니다. 전환은3.2의 epoch 경계와도 겹쳐 gate 변화만으로 원인을 확정하지 않는다.

![M20 bias 비교: 품질·gate·갱신·포화·개입 지속](../figures/20261017_m20_bias.png)

**Gate의 동적 동작과 포화.** 그룹은window0, 그 외 아래 지표는window16·32 평균이다. gate 표준편차는 슬롯별64개 segment/example 관측의 표준편차를 슬롯·window에 걸쳐 평균했다. sigmoid slope는 logit에 대한 국소 미분 g(1−g)이며 실제 parameter gradient의 측정값은 아니다.

| bias | closed / open / other | gate 평균 | 슬롯별 gate std | median \|logit\| | \|logit\|>20 | sigmoid slope |
|---:|---|---:|---:|---:|---:|---:|
| -2 | 0 / 0 / 20 | 0.3941 | 2.483e-01 | 3.49 | 0.39% | 6.865e-02 |
| +0 | 14 / 4 / 2 | 0.2293 | 1.158e-03 | 18.08 | 35.16% | 1.230e-02 |
| +2 | 13 / 7 / 0 | 0.3500 | 5.945e-08 | 94.25 | 95.00% | 4.866e-08 |

−2에서는 극단적 logit 비율이35.16%→0.39%로 줄고 gate가 문맥에 따라 크게 변한다. 같은 segment 안 example 간 gate std도.2168(0은.001146, +2는약5.83e−8)로, 시간에 따른 평균 변화만이 아니다. +2는13개 닫힘/7개 열림 분할이 세 window·threshold.001/.01/.05에서 같고 문맥별 gate 변화는 거의 없다. 현재 write는 sigmoid를 사용하는 연속·미분 가능한 경로다. +2에서는 극단 포화로 국소 미분이 작아지는 상태가 관측된다.

**−2의other20개가 모두 같은 중간값이거나 미분화됐다는 뜻은 아니다.** window0의 슬롯별 평균은.0062~.9993이다. 거의 항상 열리는 슬롯, 보통 닫혀 있다가 가끔 쓰는 슬롯, 평균.3826/std.3739로 크게 전환하는 슬롯이 공존한다. 모든 관측에서 닫힘/열림 임계값을 만족하지 않아서other로 분류된다. 평가 window에서는0/1/19이며, threshold를 바꾸면window0도0/0/20에서0/1/19로 달라진다. 전체20개other의 그룹 개입은 전체 개입과 같으므로 그 효과만으로 특별한 역할을 입증할 수는 없다.

학습 후 마지막 bias 범위는−2 조건에서−2.0333~−1.8428, 0 조건에서−.0101~+.0134, +2 조건에서1.9639~2.0039다. 초기값에서 크게 움직이지 않아 일시적인 초기화 효과와 최종 offset 효과는 분리할 수 없다. 한편 router context norm은13.99/44.36/64.43, MLP hidden norm은24.48/108.06/286.95다. 최종 bias의±2 차이만으로 logit 전체 차이를 설명하기도 어렵다. 학습 경로와 내부 scale의 상호작용이 시사된다.

**표현과 실제 갱신의 다양성.**

| bias | raw state rank | 문맥 평균 제거 rank | actual update rank | 평균slot CKA |
|---:|---:|---:|---:|---:|
| -2 | 2.7403 | 3.9955 | 3.8166 | 0.6172 |
| +0 | 2.8119 | 3.0334 | 2.4098 | 0.9016 |
| +2 | 2.4098 | 3.0794 | 2.2358 | 0.8152 |

−2는0보다 문맥 평균 제거rank3.03→4.00, actual update rank2.41→3.82, CKA.902→.617로, 문맥에 따른 변화와 실제 갱신이 더 다양하다. 반면 raw rank는 높아지지 않았고 window0의 raw slot cosine도.7064(0은.6933, +2는.7260)로0보다 높다. 단순 state 직교화나 cosine 감소가 이번 품질 개선을 설명하지는 않는다. 20개 슬롯의 effective rank가 약4이므로 독립적인20가지 기능이 생겼다고 볼 수도 없다. +2는 update rank가0보다 낮아도 PPL이 개선돼, 다양성 지표와 품질의 단조 관계도 성립하지 않는다.

**기능적 기여와 비가산성.** window0의 단일 슬롯 reset ΔNLL은−2에서20개 모두 양수(.00160~.03534), 합계.13136이지만 전체 reset은.45026이다. 0의 단일 합계.03126/전체.38491, +2는.00265/전체.38704로, 모두 개별 기여의 합으로 전체를 설명할 수 없다. 한 probe의 점추정이므로 작은 양수를 개별 슬롯의 유의한 기여로 주장하지 않는다.

+2의open7개 swap ΔNLL은 평가 window 평균.6362로, 같은 수random의 평균.0853·3mask 범위[.0100,.1770]보다 크다. closed13개의swap은.0035, 전체는.7074여서 문맥 정보에 대한 민감성이open 쪽에 집중된다. 전체swap 후open만 정상으로 복원하는 효과도.7039로 크다. 고정된 gate를 쓰는+2에서도 메모리는 기능한다. −2는 모든 슬롯이other여서 동일 개수random 비교가 자명해지고, 세 그룹 비교만으로 건전한 비율을 결론낼 수 없다.

**보존·손상 민감성과probe의 한계.** 아래는 평가 window 평균의 전체 개입ΔNLL이다. lag3은225tokens 뒤다.

| bias / 개입 | lag0 | lag1 | lag2 | lag3 |
|---|---:|---:|---:|---:|
| -2 / reset | 0.4478 | 0.4227 | 0.1796 | 0.0915 |
| -2 / swap | 0.6936 | 0.1836 | 0.0904 | 0.0597 |
| +0 / reset | 0.3547 | 0.2293 | 0.1066 | 0.0500 |
| +0 / swap | 0.7480 | 0.2533 | 0.1116 | 0.0597 |
| +2 / reset | 0.3919 | 0.2092 | 0.0933 | 0.0504 |
| +2 / swap | 0.7074 | 0.2309 | 0.1142 | 0.0658 |

−2의reset 손상은lag1에도.4227 남지만, swap의lag3은.0597로0과 거의 같다. reset 손상의 긴 지속을 유용한 장기 기억 향상으로 해석하지 않는다. 다음random 손상 값은 각 비율3mask·두 평가 window의 평균이다.

| 무작위swap 슬롯 수 | bias−2 ΔNLL | bias0 ΔNLL | bias+2 ΔNLL |
|---:|---:|---:|---:|
| 2 | 0.0525 | 0.0262 | 0.0141 |
| 5 | 0.1469 | 0.0522 | 0.0435 |
| 10 | 0.2203 | 0.1138 | 0.1856 |
| 15 | 0.4328 | 0.3612 | 0.4105 |

−2는 일부 슬롯만 손상해도 영향이 크다. 이는 넓게 분포한 활용 가능성과 손상 취약성을 모두 포함하며, 견고성 향상을 보여주지는 않는다.

window0 정상probe PPL은−2/0/+2에서36.7766/35.8678/36.2851로 전체validation과 순위가 다르다. window16도37.0595/36.6884/37.2799, window32도31.1266/30.3501/30.5784로 신규 두 조건이 모두0보다 높다. 각probe는S0에서225tokens warmup한 짧은 관측이므로 전체validation의 연속state와 문맥 조건이 다르다. warmup·관측window·장기state 의존성을 분리하는 추가 관측 전에는−2의 전체PPL 개선을 장기 기억 성능 개선으로 단정하지 않는다.

**잠정 판단.** 이번M20 조건에서는bias−2를 우선 후보로 둔다. 전체validation 개선·극단적 gate 포화 완화·문맥별 갱신 다양화가 함께 관측됐다. 다만+2도 고정gate로 품질이 개선됐고, 단일seed·유한 학습 예산·짧은probe라는 한계가 있다. 다음 확인은−2와0의 추가seed, 동일checkpoint의warmup/측정 길이 변경을 우선할 가치가 있다. M8/M75로의 일반화, 세 그룹의 최적 비율, 분화가 품질 개선의 원인이라는 주장, regularization의 필요성은 미확정이다.

사용자 판단란:

| 판단 사항 | 기입란 |
|---|---|
| M20의 잠정bias 선택 | |
| M8/M75 bias 실험과 M20 추가seed 중 우선순위 | |
| warmup·측정 길이를 바꾼 추가 분석의 우선순위 | |

### 4.10 L150의 bias−2 결과와 교차 확인

2026-10-17 학습 원본의 resolved Config와 사후 checkpoint Config를 대조했다. **L150 GRT M10/M40/M160은 모두 write bias init−2·write dropout0·read off·model/data seed54**다. NeoX/RMT baseline에는 GRT write gate가 없어 같은 bias 실험으로 세지 않는다. 아래는 [L150 실험](20261012_small_register_sweep.md)의 원본과 이번 M20 결과를 같은 집계 정의로 재계산한 비교다. 그룹은 window0, 포화·gate·표현은 window16/32 평균이다.

| L / M | bias init | 닫힘 / 열림 / other | 평균gate | 극단 포화(abs logit>20) |
|---|---:|---|---:|---:|
| 150 / 10 | -2 | 0 / 10 / 0 | 1.0000 | 100.00% |
| 150 / 40 | -2 | 7 / 33 / 0 | 0.8250 | 100.00% |
| 150 / 160 | -2 | 77 / 26 / 57 | 0.3221 | 3.79% |
| 75 / 20 | -2 | 0 / 0 / 20 | 0.3941 | 0.39% |
| 75 / 20 | +0 | 14 / 4 / 2 | 0.2293 | 35.16% |
| 75 / 20 | +2 | 13 / 7 / 0 | 0.3500 | 95.00% |

같은−2에서도 L150/M10은 전부 열림, M40은 고정 이진 분할, M160은 포화·가변 gate가 공존하고, L75/M20은 문맥별 gate 변화가 크다. 따라서 **−2 자체가 동적 분화를 보장한다는 일반화는 기존 실험에도 맞지 않는다.** L75/M20에서−2가0/+2보다 동적이라는 이번 관측은 유효하지만, 그 방향의 seed 재현성이나 다른 L/M에 대한 일반성은 별도다.

| bias−2 조건 | 전체val best PPL | router context norm | MLP hidden norm | 문맥 평균 제거rank | 실제update rank | 평균slot CKA |
|---|---:|---:|---:|---:|---:|---:|
| L150/M10 | 34.9897 | 121.54 | 365.38 | 1.054 | 1.195 | 0.999 |
| L150/M40 | 34.7458 | 156.27 | 551.06 | 1.833 | 2.403 | 0.801 |
| L150/M160 | 34.1511 | 16.43 | 50.97 | 9.571 | 11.278 | 0.500 |
| L75/M20 | 32.6100 | 13.99 | 24.48 | 3.995 | 3.817 | 0.617 |

CKA는 모든 off-diagonal 슬롯쌍의 평균으로 통일했다. L150 문서4.6의 CKA 중앙값 범위와는 다른 요약이다. M10/M40은 큰 context·hidden norm과 극단 포화가, M160과 L75/M20은 더 작은 scale과 높은 문맥별 표현 다양성이 함께 나타난다. 이는 bias 값과 내부 scale·학습 경로를 함께 봐야 한다는 해석을 강화하지만, norm 증가와 포화의 인과 방향은 확정하지 않는다. 서로 다른 M의 rank 상한도 달라 절대 rank를 효율 순위로 쓰지 않는다.

특히 **L150/M40과 L75/M20은 M/L=0.2667, bias−2로 같지만**, 극단 포화100% 대0.39%, update rank2.403 대3.817로 차이가 크다. 비율이 중요하다는 가설을 부정하지는 않지만, 같은 비율만으로 같은 분화가 재현된다고 예측할 수 없다. L과 M의 절대 크기, gradient 범위, write/optimizer 빈도 및 초기화 경로가 함께 달라진 비교다.

기능적 기여도 다르다. L150/M40은 open33개의 swap ΔNLL이.4398(전체.4638)로 열린 경로에 집중된다. M160의 other57개 swap은.1150으로 같은 수 random 평균.0534보다 크며, open과 other의 공동 swap.2848은 두 단독 효과 합.1566보다 크다. 반면 L75/M20−2의 other는 전체20개라 같은 그룹 비교로 기능 분화를 확증할 수 없다. 두 실험 모두 닫힌 그룹의 작은 swap과 개입 효과의 비가산성이 나타나므로, 세 유형 비율만으로 좋고 나쁨을 결론낼 수 없다.

**교차 비교의 통제 한계.** L150은10k, L75는20k로 완료 target 예산은 각383,917,824/383,975,424개로 거의 같다. 그러나 L75는 token당 write·optimizer update 빈도가 약2배이고 K3의 과거 gradient 범위는450→225tokens로 줄며, micro/accum도 M40의32×4에서 M20의128×1로 바뀐다. L150 best는10k, L75 best는19k여서 선택 checkpoint의 학습 target도 정확히 같지 않다. M 변경은 parameter shape와 초기화 RNG 경로를 바꾸므로 seed54라는 숫자가 같아도 동일 초기 가중치 비교는 아니다. M160은 학습 Python/PyTorch도 다르다. 따라서 PPL 차이를 bias 효과나 L 축소의 순수 효과로 귀속하지 않는다.

사후 분석은 같은 정의의 gate/logit·rank/CKA·그룹 개입 구현이며 source hash 차이는 utils/kaggle.py뿐이다. 다만 같은3warmup/8측정/horizon4여도 L150은450/1200tokens·lag3=450tokens, L75는225/600tokens·lag3=225tokens다. window 시작16/32의 token 위치도 다르므로 ΔNLL과 retention을 동일 문맥·동일 token 지연의 성능 순위로 비교하지 않는다.

현재 판단은 **고정 L75/M20에서 bias의 효과가 seed를 바꿔도 같은 방향인지 확인하는 것**이다. L150 자료는−2의 조건 의존성을 보여주는 추가 사례이며, L75 bias−2의 독립 seed 재현으로 세지 않는다.

## 5. 기존 노트북에서 실행

새 노트북이나 run별 YAML을 만들지 않는다. 기존 [Kaggle](../../notebooks/kaggle.ipynb) / [Colab 학습](../../notebooks/colab.ipynb)의 선택 셀을 사용한다.

```python
models = ('grt',)
segment_len = 75
num_registers = 8        # 8 / 20 / 75
budget = '20k'
router_hidden = 512
write_gate_bias = 0.0    # A: 0, B: -2 / 0 / +2
write_dropout = 0.0      # A: 0 / .1, B: 모든 M에서 0 고정
```

Read gate off는 model override에 명시한다. 값을 바꾸면 기존 공통 helper의 설정 hash가 달라져 별도 run이 된다. 같은 값·환경은 기존 last.pt를 재개하므로, 동일 조건을 새 폴더에서 재실행하려면 명시적 run_id를 사용하고, 독립 seed 검증은 seed를 바꾼다. GPU 수가 다른 환경으로 학습을 재개하지 않는다. 초기값·dropout·M·L을 바꾼 기존 checkpoint에서 이어 학습하지 않는다.

순서는 setup → 기존 prepared 데이터 복원 → preflight → train → report다. preflight의 실제 micro/accum·target tokens/update·예상 시간을 확인한다. 새 L/M의 시간을 이전 T4 실측에서 단순 환산하지 않는다. 예약 VRAM 목표95%는 정확한 점유율 보장이 아니다. 장시간 run/checkpoint·데이터는 세션 밖에 보존한다.

학습 노트북 마지막 `job.report(posthoc=True, download=True, posthoc_options=...)`의 **experiment_results.zip** 하나를 보내면 안의 학습·사후 JSON을 이 문서의 해당 ID에 반영한다. 마지막 셀에서 batch8·warmup3·측정8·horizon4·slots/registers true·window0/16/32·group_repeats3을 직접 바꿀 수 있다. 기존 학습 JSON과 별도 사후 JSON도 계속 받을 수 있다. 여러 환경 결과를 CSV로 누적하지 않는다. 미실행·미측정은 빈칸/상태로 남긴다.

## 6. 준비 검증

관련 CPU 검사55개가 통과했다. 기존 카탈로그에20k 예산을 추가하고, 실제 Kaggle/Colab 선택 셀에서 M3개×bias3개×dropout2개의 설정 전달·고유 run hash·read off·실질 배치256·384M nominal targets를 확인했다. 기존 gate/write 구현과 공통 utils를 사용하며 모델 구조·checkpoint 형식·기존 결과는 바꾸지 않았다. 2026-10-16 사용자 제공6개 report로20k GPU 학습·시간·기본 RTLA 완료를 확인했다. 같은 날 사후 JSON6개를 추가 수신해 gate 원값·threshold·3window·그룹별 token 가중 ΔNLL을 검증하고 위 표·그림에 반영했다. 기존 원본은 보존했다.

## 7. 원본과 재현성

| ID | report ID / 원본 | W&B |
|---|---|---|
| A8-0 | [ccf041592f3e4ebe8f287e69195205a5](../../results/ccf041592f3e4ebe8f287e69195205a5.json) | [run](https://wandb.ai/joyyoj1-korea-university/grt-research/runs/748b71bf) |
| A8-1 | [746ab4ab41e84ec0bdd0f3f0ed6bffdb](../../results/746ab4ab41e84ec0bdd0f3f0ed6bffdb.json) | [run](https://wandb.ai/joyyoj1-korea-university/grt-research/runs/edf5664c) |
| A20-0 | [86fb2481ac1e45f29da0164863b28d66](../../results/86fb2481ac1e45f29da0164863b28d66.json) | [run](https://wandb.ai/joyyoj1-korea-university/grt-research/runs/54d2c635) |
| A20-1 | [51bafa3f611d48c2808a7a55d732c91d](../../results/51bafa3f611d48c2808a7a55d732c91d.json) | [run](https://wandb.ai/joyyoj1-korea-university/grt-research/runs/48443a79) |
| A75-0 | [b4b4414578524de0a6e9de50851b91b2](../../results/b4b4414578524de0a6e9de50851b91b2.json) | [run](https://wandb.ai/joyyoj1-korea-university/grt-research/runs/feb72c65) |
| A75-1 | [06fe396a3e6b4e258b779481409de1ba](../../results/06fe396a3e6b4e258b779481409de1ba.json) | [run](https://wandb.ai/joyyoj1-korea-university/grt-research/runs/761c6ca9) |
| B20− | [3001e7328e87456395275745ada7b946](../../results/3001e7328e87456395275745ada7b946.json) | [run](https://wandb.ai/joyyoj1-korea-university/grt-research/runs/7a17fbd4) |
| B20+ | [0bb9e51192d54a398d771bab611e9fb1](../../results/0bb9e51192d54a398d771bab611e9fb1.json) | [run](https://wandb.ai/joyyoj1-korea-university/grt-research/runs/cc21f8cd) |

원본 JSON bytes를 그대로 보관했다. best·최종20k·selected RTLA를 구분해 검증했고, 학습 curve20개 평가점·기본 RTLA20건·환경·Config를 대조했다. 결과 정리 시 학습을 추가 실행하지 않았다. 이후 2026-10-16 사용자 요청으로 학습 노트북 마지막 셀의 확장 분석과 ZIP 전달을 연결했다. 기존 수신 원본은 수정하지 않았다.

| ID | 사후 원본 report ID | 전체 checkpoint SHA256 |
|---|---|---|
| A8-0 | [10b47cdf69034d8d91a842f60f226ede](../../results/posthoc/10b47cdf69034d8d91a842f60f226ede.json) | `5aa9d458db8495bb70197fd910576b640db270f16eeb62857845f2263f64daf6` |
| A8-1 | [17f1bc032be54c249e645fa01275bac2](../../results/posthoc/17f1bc032be54c249e645fa01275bac2.json) | `be6f249c6745d8e34c41b0ad69b2754fb31ef4395d6d678397fd6b06305eed27` |
| A20-0 | [5435927747a34aa895d5804c2b5772a6](../../results/posthoc/5435927747a34aa895d5804c2b5772a6.json) | `919e91014cda899df36e4346a952627beb258c00624a8340e8065dd3eda1281a` |
| A20-1 | [1ae377fe64e243749040cf61991c37f0](../../results/posthoc/1ae377fe64e243749040cf61991c37f0.json) | `d56a848c64267afb71a09ae951a12cd8397a5512f6986b332c399cdd35740a59` |
| A75-0 | [f5a1825cc5d84f6da400eb249ec409ac](../../results/posthoc/f5a1825cc5d84f6da400eb249ec409ac.json) | `1de8e1540d5cbd216acdf2e20d25e0b93e245ea7302f0c0b80c63ee594d71ed0` |
| A75-1 | [df797486d1cd4606acc91fc478535e33](../../results/posthoc/df797486d1cd4606acc91fc478535e33.json) | `9f53f8caa51467c29beee91f098dae758dcec9635c5336d267ac26cbcbb877e3` |
| B20− | [3cb39cb5ceaf40e4b2ac4790af009b6d](../../results/posthoc/3cb39cb5ceaf40e4b2ac4790af009b6d.json) | `9e5c3815d522be15e6068fbc14553966f3ebd0dc84876f2c941ca696ea6658dc` |
| B20+ | [b9d1e1ea89bd4499989c6968dfc63957](../../results/posthoc/b9d1e1ea89bd4499989c6968dfc63957.json) | `ffc7bfd5f8b614125a4bba0bbddf2e292df7d6aa9e872cfdb7d84056b75660c3` |

L75 사후 원본8개(기존6개+M20 bias2개)의 bytes를 그대로 보관했다. 원본 학습·사후 JSON과 기존 L150 결과는 수정하지 않았다. 이번 추가 분석은 받은 자료를 재집계한 것으로 GPU 추론·학습을 새로 실행하지 않았다.

2026-10-17 M20 bias 수신분: 학습2개·사후2개를 추가하고, 기존bias0과 Config·환경·예산·selected19k 연결 및 분석 source hash를 대조했다. raw gate 재분류와 원시배열의 token 가중 개입ΔNLL 재계산, 원본 bytes·문서 링크 검증을 수행했다. GPU 실행과 모델 코드는 변경하지 않았다.
