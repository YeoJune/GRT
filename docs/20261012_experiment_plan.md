# small 기준 실험 계획과 결과표

갱신일: 2026-10-13 · 상태: B0 NeoX baseline 완료, B1 RMT baseline 대기

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

이 파일을 보내면 에이전트가 [results/](../results/README.md)에 `<report_id>.json`으로 보관하고 아래 결과표의 한 행씩 갱신한다. 같은 report_id의 재개 결과는 step을 확인해 갱신한다. 보고서를 반복 실행해도 한 파일과 같은 report_id를 유지한다. B0 NeoX의 첫 GPU 결과를 받아 원본과 요약을 기록했다.

## 4. 결과 기록

B0 NeoX baseline은 사용자 제공 GPU 보고서로 완료를 확인했다. 아직 받지 않은 결과는 빈칸으로 두고 상태만 갱신한다. interrupted/OOM/발산은 완료 결과와 구분한다. MLP단계의 `선택 M`은 R단계 완료 후 채운다.

| ID | 모델 | M | MLP hidden | best val PPL | swap ΔNLL | reset ΔNLL | VRAM GiB¹ | 시간 h | 상태 / 원본 보고서 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| B0 | NeoX | 0 | — | 37.9590 | — | — | 6.899 | 2.610 | 완료 · [원본](../results/0f4be0ee5d5c4a77b8ce50d591932c69.json) |
| B1 | RMT | 10 | — | | — | — | | | 대기 |
| R0 | GRT | 10 | 512 | | | | | | 대기 |
| R1 | GRT | 40 | 512 | | | | | | 대기 |
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

후속 RMT/GRT는 동일 BPE artifact, segment150/global256/10k/seed54 및 validation batch10을 확인한다. 데이터 식별값은 train `2e43bd7c329c`, validation `a2bc335cb8a1`, tokenizer `1fe93b615295` (앞12자; 전체 hash는 원본). 다음 실험은 **B1: RMT M10/K3**이며, 그 결과를 받은 후 레지스터 sweep을 진행한다.

| 결정 / 메모 | 기록할 내용 |
|---|---|
| R단계 선택 M | |
| 선택 이유: 품질·활용성·비용 | |
| H단계 선택 hidden | |
| 다음 실험 ID / 변경할 값 | B1 · `models=('rmt',)`, M10/K3, small/segment150/10k/FP32 유지 |
| 실패 원인 / 추가 관측 | |

## 5. 운영과 검토 범위

- 탐색 중 test는 평가하지 않는다(`test_at_end=False`). 종료 시 best checkpoint의 validation과 GRT RTLA를 함께 재평가한다. 최종 선택 후에만 test를 별도로 평가한다.
- Kaggle 주30h는 baseline/본 학습, Colab 하루4~5h는 preflight·분석 또는 시간 내 완료 가능한 독립 run에 쓴다. 본수는 실측 ETA로 정한다. Vast.ai는 이번 sweep의 필수 자원으로 잡지 않고 무료 시간 부족 시 별도 예산으로 검토한다.
- 같은 Config·world_size·데이터의 last.pt만 재개한다. Kaggle2GPU → Colab1GPU 학습 재개는 불가하며 run·prepared 데이터와 결과 JSON을 외부에 보관한다. JSON에는 가중치/NPZ 배열이 없으므로 학습 재개와 추가 분석용 checkpoint는 별도로 보관한다. 10k를 늘리는 것은 새 schedule의 새 실험이다.
- 기존 `max_seconds`는 누적 상한이어서 매일4h씩 이어가는 session 제한으로 쓰지 않는다. 긴 run은 세션/저장 계획을 확인한 뒤 실행한다.
- 이번 구현은 메모리 없는 NeoX, 노트북 수동 설정, test 격리·동일 checkpoint 평가, 단일 JSON 내보내기와 Markdown 결과 기록을 포함한다. 장시간 GPU 학습·실제 T4 VRAM/수렴은 사용자 실행 단계다.

검증: 메모리 없는 동작·loss/gradient·재개, test 격리, 고정 attention의 수동 sweep, 동일 checkpoint 지표, 실험별 JSON 분리·재보고 ID 유지·중단 로그·다운로드 호출과 노트북 선택 셀을 CPU에서 확인했다. 기존 모델의 causal/prefix·replay 검사도 통과했다. CPU 두 프로세스 검사는 localhost 소켓 권한 제한으로 미완료이며, 사용자 제공 B0 보고서에서 Kaggle T4×2/온라인 W&B 완료를 확인했다. 다른 모델/환경의 GPU 수렴은 아직 미확인이다.

최종 검토 대상은 **위 공통 설정, R단계 용량 후보, R단계 뒤에 진행하는 MLP 후보**다. 더 넓은 탐색 아이디어는 [이전 검토 보고서](20261012_hyperparameter_search_report.md)에 보존한다.
