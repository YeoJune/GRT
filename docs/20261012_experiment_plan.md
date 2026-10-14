# small 기준 실험 계획과 결과표

갱신일: 2026-10-14 · 상태: B0/B1 baseline과 R0/R1 GRT 완료, R2/R3 용량 탐색 대기

**메모리 없는 NeoX와 RMT baseline을 먼저 학습한다. 이후 GRT의 레지스터 수만 탐색하고, 선택한 레지스터 수를 고정한 뒤 라우터 MLP hidden만 탐색한다.** 어텐션은 backbone에 맞춰 고정한다. 각 실험은 같은 Kaggle/Colab 노트북에서 값을 바꿔 하나씩 실행한다.

## 1. 공통 조건

| 항목 | 설정 |
|---|---|
| 데이터 / tokenizer | WikiText-103-v1 / 기존 GPT-2 BPE artifact 고정 |
| backbone | small: D256, 6 layers, 8 heads, FF1024, tied embedding |
| segment / 예산 | 150 BPE tokens / 10k optimizer updates, target 상한384M |
| 학습 | global batch256, Adam LR5e-4·beta2=0.999·decay0, cosine·warmup0 |
| 정규화 / 수치 | residual dropout0.1, attention dropout0, clip0.25, FP32 |
| BPTT | RMT/GRT: 이전3 segments; NeoX:0, 과거 메모리·재계산 없음 |
| 평가 / seed | validation batch10, 1000 updates마다 평가, seed54 |
| GRT 고정값 | read gate off, write bias=-2, gate dropout0; router attention D256/head8 |
| T4 배치 | auto_batch on, 예약 VRAM 목표95%; 가장 큰 허용 microbatch, accumulation으로 global256 유지 |
| 선택 지표 | best checkpoint의 validation PPL/NLL과 같은 checkpoint의 RTLA |

95%는 목표 상한이며 정확히 채운다는 뜻은 아니다. global batch와 허용 microbatch 약수 때문에 더 낮을 수 있다. 설정마다 preflight에서 실제 peak를 확인한다. 기본 batch는 Kaggle2GPU micro32×accum4, Colab1GPU micro32×accum8이며 자동 보정된다. 장치/평가 메모리가 맞지 않으면 결과를 실행 불가로 기록한다.

## 2. 진행 순서

1. **B0 → B1:** NeoX, RMT를 각각10k 학습한다. 같은 데이터·backbone·target 예산으로 기준값을 확보한다.
2. **R0 → R3:** GRT MLP hidden512를 고정하고 M=10/40/160/512를 한 조건씩 실행한다. M512가 T4에 맞지 않으면 M256/320 등으로 상한을 찾는다.
3. **레지스터 선택:** 정상 품질과 문맥 교환 ΔNLL, VRAM·시간을 함께 보고 M을 정한다. 큰 gate/norm만으로 활용성을 판단하지 않는다.
4. **H0 → H3:** 선택된 M에서 MLP hidden=128/256/512/1024를 한 조건씩 실행한다. 해당 M의 hidden512 결과는 R단계에서 재사용한다.
5. 이 표를 채운 뒤 다음 단계와 seed 반복을 결정한다. 이번 범위에서는 segment·게이트·어텐션·size를 sweep하지 않는다.

활용 지표는 `Δ_swap=NLL(다른 stream state)−NLL(정상 state)`, `Δ_reset=NLL(S0 초기화)−NLL(정상 state)`다. 양의 delta가 여러 checkpoint/window에서 유지되는지 확인한다. baseline에는 현재 GRT 전용 RTLA를 적용하지 않으며 미측정값을0으로 기록하지 않는다.

## 3. 노트북에서 바꾸는 값

사용 노트북: [Kaggle](../notebooks/kaggle.ipynb), [Colab](../notebooks/colab.ipynb). 두 노트북 모두 기본은 NeoX/small/10k다. 선택 셀에서 다음 값만 바꾼다.

```python
models = ('neox',)     # 다음 ('rmt',), 이후 ('grt',)
num_registers = 10     # R단계: 10 / 40 / 160 / 512
router_hidden = 512    # H단계에서만 128 / 256 / 512 / 1024
memory_fraction = 0.95
```

`num_registers`는 RMT/GRT에만, `router_hidden`은 GRT의 read/write MLP에만 적용한다. NeoX는 자동으로 M0/K0를 사용한다. MLP를 바꿔도 어텐션 차원256/head8은 바뀌지 않는다. 모델 여러 개를 지정하면 순차 실행되지만 첫 진행은 한 모델씩 권장한다.

실험 선택/준비 → `job.prepare_data()` → `job.preflight()` → `job.train()` → `job.report(download=True)` 순서다. preflight의 ETA/VRAM을 보고 실행하며 ETA는 RTLA·준비·저장 비용을 제외한다. 실험별 config 파일을 직접 만들 필요가 없고 값이 바뀌면 설정 hash의 새 run이 생성된다.

마지막 `job.report(download=True)`는 **이번 실험의 experiment_report.json 파일 하나**를 내보낸다. best checkpoint의 validation/RTLA, 최종 Config, 데이터 hash, GPU·소프트웨어 환경, 실제 batch·VRAM·시간, 학습 곡선·RTLA 요약, 상태·경고를 함께 담는다. 여러 모델을 선택했다면 이번에 선택한 모델들이 같은 파일에 들어간다. 다른 실험/환경 결과는 누적하지 않는다.

이 파일을 보내면 에이전트가 [results/](../results/README.md)에 `<report_id>.json`으로 보관하고 아래 결과표의 한 행씩 갱신한다. 같은 report_id의 재개 결과는 step을 확인해 갱신한다. 보고서를 반복 실행해도 한 파일과 같은 report_id를 유지한다. B0/B1 baseline과 R0/R1 GRT의 GPU 결과를 받아 원본과 요약을 기록했다.

## 4. 결과 기록

B0 NeoX/B1 RMT baseline과 R0/R1 GRT는 사용자 제공 GPU 보고서로 완료를 확인했다. 아직 받지 않은 결과는 빈칸으로 두고 상태만 갱신한다. interrupted/OOM/발산은 완료 결과와 구분한다. MLP단계의 `선택 M`은 R단계 완료 후 채운다.

| ID | 모델 | M | MLP hidden | best val PPL | swap ΔNLL | reset ΔNLL | VRAM GiB¹ | 시간 h | 상태 / 원본 보고서 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| B0 | NeoX | 0 | — | 37.9590 | — | — | 6.899 | 2.610 | 완료 · [원본](../results/0f4be0ee5d5c4a77b8ce50d591932c69.json) |
| B1 | RMT | 10 | — | 34.8754 | — | — | 11.049 | 5.366 | 완료 · [원본](../results/24b2e511de2e419eb0ac7f9b52769d89.json) |
| R0 | GRT | 10 | 512 | 34.9897 | 0.422916 | 0.294231 | 11.269 | 5.812 | 완료 · [원본](../results/22ab31a56cb546f993197e5ce937e5de.json) |
| R1 | GRT | 40 | 512 | 34.7458 | 0.524916 | 0.345704 | 6.538 | 7.009 | 완료 · [원본](../results/22cfd16b44964befbe42912c4aba28c5.json) |
| R2 | GRT | 160 | 512 | | | | | | 대기 |
| R3 | GRT | 512 | 512 | | | | | | 대기 |
| H0 | GRT | 선택 M | 128 | | | | | | R단계 후 |
| H1 | GRT | 선택 M | 256 | | | | | | R단계 후 |
| H2 | GRT | 선택 M | 512 | | | | | | 해당 R결과 재사용 |
| H3 | GRT | 선택 M | 1024 | | | | | | R단계 후 |

¹ 본 학습 peak allocated GiB를 적고 preflight reserved GiB는 JSON에서 함께 확인한다. 시간은 준비·분석 포함 여부가 다른 ETA와 구분한다.

### B0: NeoX baseline · 2026-10-13 수신

[원본 JSON](../results/0f4be0ee5d5c4a77b8ce50d591932c69.json) · [학습 코드](https://github.com/YeoJune/GRT/commit/e238ac356bbfcbf5766fc20466105617a7944354) · [W&B](https://wandb.ai/joyyoj1-korea-university/grt-research/runs/633dd52b)

| 항목 | 확인된 결과 / 조건 |
|---|---|
| 선택 checkpoint | best.pt, step10,000; 마지막 checkpoint와 validation 지표 동일 |
| 품질 | validation NLL **3.636507**, PPL **37.9590**; GPT-2 BPE token 기준 |
| 실제 학습량 | 10,000 updates, 383,917,824 target tokens (약383.918M) |
| 평가 범위 | validation target 248,420개, batch10; test 미평가 |
| 모델 | small D256/layers6/heads8/FF1024, 17,604,864 parameters; M0/K0, memory 없음 |
| 환경 / 배치 | Kaggle, Tesla T4×2, FP32; micro64×accum2×GPU2=global256 |
| optimizer / seed | Adam LR5e-4, beta2=.999, decay0, cosine/warmup0, clip.25, dropout.1; model/data seed54 |
| 소프트웨어 | Python 3.13.15, PyTorch 2.11.0+cu128, Transformers 4.45.2, CUDA 12.8 |
| 시간 | 9395.66초 = **2.610h**, 자원 환산 **5.220 GPU-hours** |
| VRAM | 본 학습 peak allocated **6.899 GiB/GPU**; preflight reserved 8.686 GiB, 예약 비율59.64% |
| preflight와 실측 | preflight 0.782초/update, ETA 2.186h; 1k→10k의 평가·저장 포함 경과시간 약0.940초/update |
| 보고서 상태 | completed, W&B finished/실패0/step10k; 경고·metrics 파싱 오류 없음 |

실행 시간은 학습·평가·checkpoint 저장 등을 포함하고 환경 준비와 preflight는 제외한다. GPU-hours는 경과시간×GPU 수의 계산값이며 Kaggle quota 차감을 뜻하지 않는다. VRAM95%는 설정 목표로, 이번에 측정된 예약 비율은59.64%다. RTLA 설정은 켜져 있었지만 NeoX는 분석 대상이 아니므로 메모리 delta는 미측정으로 둔다.

| update | validation NLL | validation PPL |
|---:|---:|---:|
| 1,000 | 4.504925 | 90.4616 |
| 2,000 | 4.130438 | 62.2051 |
| 3,000 | 3.927304 | 50.7699 |
| 4,000 | 3.831173 | 46.1166 |
| 5,000 | 3.763982 | 43.1198 |
| 6,000 | 3.706069 | 40.6935 |
| 7,000 | 3.671761 | 39.3211 |
| 8,000 | 3.653824 | 38.6221 |
| 9,000 | 3.639824 | 38.0851 |
| 10,000 | 3.636507 | 37.9590 |

후반 PPL 감소는 8k→9k **0.5369**, 9k→10k **0.1261**다. 현재 예산에서는 개선이 완만해졌지만 마지막 LR이0인 cosine 결과여서 장기 수렴을 확정하지 않는다. 이 값을 **10k 고정 예산의 메모리 없는 baseline**으로 사용하며, seed1개 결과의 작은 차이는 반복 검증 전까지 확정하지 않는다. 앞서 받은 Colab T4 단일 GPU 속도 로그와 이번 Kaggle2GPU 결과는 별도 실행 조건이다.

후속 RMT/GRT는 동일 BPE artifact, segment150/global256/10k/seed54 및 validation batch10을 확인한다. 데이터 식별값은 train `2e43bd7c329c`, validation `a2bc335cb8a1`, tokenizer `1fe93b615295` (앞12자; 전체 hash는 원본). B1 RMT M10/K3와 R0/R1 GRT 레지스터 sweep 결과는 아래에 별도로 기록한다.

### B1: RMT baseline · 2026-10-14 수신

[원본 JSON](../results/24b2e511de2e419eb0ac7f9b52769d89.json) · [학습 코드](https://github.com/YeoJune/GRT/commit/e238ac356bbfcbf5766fc20466105617a7944354) · [W&B](https://wandb.ai/joyyoj1-korea-university/grt-research/runs/92d08d99)

| 항목 | 확인된 결과 / 조건 |
|---|---|
| 선택 checkpoint / 품질 | best.pt, step10,000; validation NLL **3.551781**, PPL **34.8754** |
| 실제 학습 / 평가 | 10,000 updates, 383,917,824 targets; validation 248,420 targets/batch10, test 미평가 |
| 모델 | small D256/layers6/heads8/FF1024, RMT M10/K3, 17,607,424 parameters; router/gate 없음 |
| 환경 / 배치 | Kaggle Tesla T4×2, FP32; micro64×accum2×GPU2=global256 |
| optimizer / seed | Adam LR5e-4/beta2=.999/decay0, cosine/warmup0, clip.25/dropout.1; model/data seed54 |
| 소프트웨어 | Python 3.13.15/PyTorch 2.11.0+cu128/Transformers 4.45.2/CUDA 12.8 |
| 시간 / GPU-hours | 19315.84초 = **5.366h** / **10.731 GPU-hours** |
| VRAM | 본 학습 peak allocated 11.049 GiB/GPU; preflight reserved 13.064 GiB (89.71%) |
| preflight / 실측 | 1.751초/update, ETA 4.876h; 1k→10k의 평가·저장 포함 경과시간 1.932초/update |
| 보고서 상태 | completed, W&B finished/실패0/step10k, 경고/metrics 파싱 오류 없음 |

B0/R0/R1과 prepared 데이터·tokenizer hash, backbone, optimizer·schedule·FP32·global256·실제 target 예산·validation batch/범위가 일치한다. B0/R0와 학습 코드 commit도 동일하고, R0와는 M10/K3 및 실제 배치까지 동일하다. RTLA는 GRT 전용이라 RMT의 분석 결과는 없으며, 해당 값을0으로 대체하지 않는다. 시간에는 학습·평가·checkpoint I/O가 포함되고 준비/preflight는 제외한다.

| update | validation NLL | validation PPL |
|---:|---:|---:|
| 1,000 | 4.510919 | 91.0054 |
| 2,000 | 4.097579 | 60.1944 |
| 3,000 | 3.892089 | 49.0132 |
| 4,000 | 3.759391 | 42.9223 |
| 5,000 | 3.687632 | 39.9501 |
| 6,000 | 3.627745 | 37.6279 |
| 7,000 | 3.590725 | 36.2604 |
| 8,000 | 3.569755 | 35.5079 |
| 9,000 | 3.555373 | 35.0009 |
| 10,000 | 3.551781 | 34.8754 |

validation은 모든 평가 지점에서 개선됐으며 마지막1k PPL 감소는 **0.1255**다. 마지막 LR이0인10k cosine 결과로, 완전 수렴이나 여러 seed의 평균 성능을 뜻하지 않는다. RMT는 NeoX보다 PPL **3.0836** 낮다. GRT M10은 RMT보다 **0.1143 높고**, GRT M40은 **0.1296 낮다**. 두 GRT와의 차이는 작으므로 최종 품질 우열을 확정하지 않는다. 실행 시간은 RMT5.366h, GRT M10 5.812h, M40 7.009h이며, GRT 시간에는 현재 RMT가 실행하지 않는 RTLA 비용도 포함된다.

### R0/R1: GRT 레지스터 M10/M40 · 2026-10-14 수신

[R0 M10 원본](../results/22ab31a56cb546f993197e5ce937e5de.json) · [R1 M40 원본](../results/22cfd16b44964befbe42912c4aba28c5.json)

[R0 코드](https://github.com/YeoJune/GRT/commit/e238ac356bbfcbf5766fc20466105617a7944354) · [R0 W&B](https://wandb.ai/joyyoj1-korea-university/grt-research/runs/e284b89f) · [R1 코드](https://github.com/YeoJune/GRT/commit/c36920d0e4423ecf714b6d34e5785fe3a3beae72) · [R1 W&B](https://wandb.ai/joyyoj1-korea-university/grt-research/runs/710b2330)

두 run 모두 **Kaggle Tesla T4×2, small/segment150/K3/10k/FP32/seed54**, router MLP512·attention D256/head8, read gate off/write bias=-2/dropout0다. B0와 동일한 prepared 데이터·tokenizer hash, optimizer·schedule·global256·validation batch10을 사용했다. 실제 target tokens **383,917,824**, validation targets **248,420**으로 동일하고 test는 미평가다. Python3.13.15/PyTorch2.11.0+cu128/Transformers4.45.2/CUDA12.8도 동일하다. R1의 코드 commit은 c36920d로 R0의 e238ac3와 다르지만 그 사이 변경은 결과 문서/원본 JSON뿐이며 실행 코드는 동일하다.

| 항목 | R0: M10 | R1: M40 |
|---|---:|---:|
| best validation NLL | 3.555054 | 3.548059 |
| best validation PPL | **34.9897** | **34.7458** |
| 선택 checkpoint | best.pt · step10,000 | best.pt · step10,000 |
| parameter 수 | 18,669,588 | 18,708,048 |
| 실제 배치 | micro64×accum2×GPU2=256 | micro32×accum4×GPU2=256 |
| 실행 시간 / GPU-hours | 5.812h / 11.623 | 7.009h / 14.018 |
| 본 학습 allocated / preflight reserved | 11.269 / 13.156 GiB | 6.538 / 7.436 GiB |
| preflight 예약 비율 | 90.35% | 51.06% |
| preflight / 1k→10k 실측 초/update | 1.830 / 2.092 | 2.168 / 2.524 |
| RTLA 정상 probe PPL | 32.9314 | 32.4541 |
| RTLA reset / swap ΔNLL | 0.294231 / 0.422916 | 0.345704 / 0.524916 |
| write gate 평균 / slot cosine | 1.000000 / 0.999348 | 0.825000 / 0.759750 |

시간은 학습·평가·RTLA·checkpoint I/O를 포함하고 준비/preflight를 제외한다. M40의 allocated VRAM이 작은 것은 **microbatch가64에서32로 바뀐 조건**의 실측이므로 큰 memory 자체가 더 적은 VRAM을 쓴다고 해석하지 않는다. global256은 유지됐지만 microbatch 차이에 따른 dropout RNG/수치 차이까지 완전히 제거한 비교는 아니다. 두 보고서 모두 completed, W&B finished/실패0/step10k, 경고/metrics 파싱 오류 없음이다.

RTLA는 같은 best step10k에서 validation의 **고정 probe9,600 targets**(batch8, warmup3, 측정8 segments)를 사용했다. probe PPL과 전체 validation PPL을 구분한다. 두 run 모두 reset/swap ΔNLL이 양수이며 그 probe에서 정상 과거 state가 예측에 도움을 주는 근거다. 다른 window/seed, slot·지속 범위 개입은 아직 미검증이다. write gate 평균과 slot cosine은 동작 진단값이며 그 자체로 활용성/중복 저장을 확정하지 않는다.

| update | M10 val PPL | M40 val PPL | M10 reset ΔNLL | M40 reset ΔNLL |
|---:|---:|---:|---:|---:|
| 1,000 | 123.4827 | 208.9937 | -0.284458 | -0.948474 |
| 2,000 | 87.7662 | 432.5057 | -0.472108 | -1.470909 |
| 3,000 | 79.3278 | 727.0352 | -0.430206 | -1.889253 |
| 4,000 | 43.1153 | 43.8405 | 0.220558 | 0.252695 |
| 5,000 | 40.1375 | 40.5349 | 0.252012 | 0.297673 |
| 6,000 | 37.7818 | 37.8219 | 0.272346 | 0.317933 |
| 7,000 | 36.4037 | 36.3005 | 0.272470 | 0.315949 |
| 8,000 | 35.6394 | 35.4525 | 0.285898 | 0.332226 |
| 9,000 | 35.1090 | 34.9164 | 0.293418 | 0.342090 |
| 10,000 | 34.9897 | 34.7458 | 0.294231 | 0.345704 |

M40은 1k→3k에서 전체 validation PPL이 **208.99→432.51→727.04**로 악화됐고 두 run의 초기 reset ΔNLL도 음수였다. 4k에는 PPL이 크게 내려가고 reset 효과가 양수로 바뀐다. 초기 state의 악영향을 보여주는 관측으로 기록하며, 원인을 확정하거나 이를 계속되는 발산으로 분류하지 않는다. 짧은 pilot 결과만으로 이 후보를 배제하면 최종 경향을 놓칠 수 있다.

최종 M40은 M10보다 PPL **0.2439(0.70%)** 낮고 swap ΔNLL은 **0.422916→0.524916**로 커졌으나 시간도 **20.60%** 늘었다. 단일 seed의 작은 PPL 차이이므로 M40의 최종 우위나 최적 용량을 확정하지 않는다. 두 모델은 B0 NeoX보다 각각 PPL **2.9693/3.2132** 낮지만 K0/K3와 parameter·계산 비용도 다른 조건이다. 이후 받은 B1 RMT의 PPL은34.8754로 두 GRT와의 차이가 작다. 위 B1 비교를 참고하고 단일 seed에서 RMT 대비 품질 우위를 확정하지 않는다.

마지막1k PPL 개선은 M10 **0.1193**, M40 **0.1706**이고 LR은0이다. 두 값을 10k 고정 예산의 레지스터 비교 결과로 유지한다. 미기입 실험은 R2 M160/R3 M512이며, 선택 M과 MLP sweep은 아직 결정하지 않았다.

| 결정 / 메모 | 기록할 내용 |
|---|---|
| R단계 선택 M | |
| 선택 이유: 품질·활용성·비용 | |
| H단계 선택 hidden | |
| 다음 실험 ID / 변경할 값 | R2 · `models=('grt',)`, M160/MLP512/K3, small/segment150/10k/FP32 유지; 먼저 preflight |
| 실패 원인 / 추가 관측 | M40 초기1k~3k validation 악화 후4k 회복; 양 run 초기 reset 효과 음수. 원인 미확정 |

## 5. 운영과 검토 범위

- 탐색 중 test는 평가하지 않는다(`test_at_end=False`). 종료 시 best checkpoint의 validation과 GRT RTLA를 함께 재평가한다. 최종 선택 후에만 test를 별도로 평가한다.
- Kaggle 주30h는 baseline/본 학습, Colab 하루4~5h는 preflight·분석 또는 시간 내 완료 가능한 독립 run에 쓴다. 본수는 실측 ETA로 정한다. Vast.ai는 이번 sweep의 필수 자원으로 잡지 않고 무료 시간 부족 시 별도 예산으로 검토한다.
- 같은 Config·world_size·데이터의 last.pt만 재개한다. Kaggle2GPU → Colab1GPU 학습 재개는 불가하며 run·prepared 데이터와 결과 JSON을 외부에 보관한다. JSON에는 가중치/NPZ 배열이 없으므로 학습 재개와 추가 분석용 checkpoint는 별도로 보관한다. 10k를 늘리는 것은 새 schedule의 새 실험이다.
- 기존 `max_seconds`는 누적 상한이어서 매일4h씩 이어가는 session 제한으로 쓰지 않는다. 긴 run은 세션/저장 계획을 확인한 뒤 실행한다.
- 이번 구현은 메모리 없는 NeoX, 노트북 수동 설정, test 격리·동일 checkpoint 평가, 단일 JSON 내보내기와 Markdown 결과 기록을 포함한다. 장시간 GPU 학습·실제 T4 VRAM/수렴은 사용자 실행 단계다.

검증: 메모리 없는 동작·loss/gradient·재개, test 격리, 고정 attention의 수동 sweep, 동일 checkpoint 지표, 실험별 JSON 분리·재보고 ID 유지·중단 로그·다운로드 호출과 노트북 선택 셀을 CPU에서 확인했다. 기존 모델의 causal/prefix·replay 검사도 통과했다. CPU 두 프로세스 검사는 localhost 소켓 권한 제한으로 미완료이며, 사용자 제공 B0/B1/R0/R1 보고서에서 Kaggle T4×2/온라인 W&B 완료를 확인했다. 다른 용량/환경의 GPU 결과와 여러 seed의 수렴은 아직 미확인이다.

최종 검토 대상은 **위 공통 설정, R단계 용량 후보, R단계 뒤에 진행하는 MLP 후보**다. 더 넓은 탐색 아이디어는 [이전 검토 보고서](20261012_hyperparameter_search_report.md)에 보존한다.
