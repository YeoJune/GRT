# small · L50/L150 메모리 용량 비교 실험

실험 시작: 2026-10-29 · 갱신: 2026-10-29 · **설정 확정, L50 우선, M5 GPU preflight 수신·학습 결과 미수신**

[실행 안내](../workflow.md) · [모델 명세](../specification.md) · [선행 L150 용량 비교](20261012_small_register_sweep.md) · [L75 gate 안정화](20261018_l75_gate_stabilization.md) · [candidate 선택](20261019_l75_gate_paths.md) · [원본 보관](../../results/README.md)

## 1. 결정과 질문

사용자 지정 조건으로 다음 두 단계를 실행한다. **L50을 먼저 완료하고 L150으로 진행한다.** 앞선 M10/40/80/160·공통18억 targets 제안은 이번 조건으로 대체한다.

| 단계 | L | GRT·RMT 각각의 M | updates/run | 과거 BPTT K | baseline |
|---|---:|---|---:|---:|---|
| 1 · 먼저 실행 | 50 | 5, 10, 25, 50 | 100,000 | 3 | NeoX M0/K0·100k |
| 2 · 후속 | 150 | 10, 25, 75, 150 | 40,000 | 3 | NeoX M0/K0·40k |

각 단계는 GRT4개+같은 M의 RMT4개+NeoX1개=9개, 전체18개다. 질문은 같은 L·M·예산에서 GRT의 gated write가 RMT보다 예측 품질과 메모리 활용에 어떤 차이를 만드는지, M 증가가 품질·연산 비용에 어떤 영향을 주는지다. NeoX는 메모리 없는 공통 GPT 계열 backbone이며 사전학습 GPT-2가 아니다.

L75의 candidate·패널티 A·write dropout0 선택을 고정해 다른 L/M으로 전달성을 확인한다. 이전 L150은 bias−2·10k·다른 gate 조건이므로 결과를 새 조건의 대체 run으로 쓰지 않는다. L50/L150 사이에는 학습 target 수와 BPTT token 범위가 달라 L 자체의 효과만 분리한 비교로 해석하지 않는다.

## 2. 공통 비교 설계

| 항목 | 고정 조건 |
|---|---|
| 데이터·tokenizer | WikiText-103-v1·동일 GPT-2 BPE artifact, vocabulary50,257 |
| Backbone | small D256·6layers·8heads·FF1024·tied embedding·공통 NeoX |
| 학습 | global256, Adam LR5e-4·betas(.9,.999)·weight decay0·clip.25 |
| 정규화·수치 | residual dropout.1·attention dropout0·FP32 |
| BPTT | GRT/RMT 이전3segments 재계산·현재 segment targets만 loss; NeoX K0 |
| GRT | candidate·source detach false·MLP hidden512·bias0·read gate off·학습 가능한0 초기 S0 |
| Gate 안정화 | 패널티 A(threshold10, coef1e-4)·write dropout0, 모든 M 동일 |
| LR schedule | warmup5% 후 cosine·최소 LR 비율.01(최종5e-6) |
| 평가 | 전체 validation248,420 targets·batch10·매1k updates·최종 budget도 평가, test 미평가 |
| checkpoint | 전체 validation NLL로 best 선택; best/final PPL·NLL을 별도 보고 |
| seed | model/data54, 우선 단일 seed |
| GRT 사후 분석 | best checkpoint·batch8·warmup3·측정8·horizon4·window0/16/32·random3·FP32·slots/registers on |
| 실행 | Kaggle T4×2 우선, Colab 단일 GPU도 같은 global batch 유지 |

평가 간격은 선행 실험과 같은1k updates로 유지한다. 따라서 L50은약12.8M, L150은약38.4M 새 targets마다 평가하며 best 선택 기회 수가 다르다. 단계 사이에는 final-budget 지표도 함께 비교한다. 동일 seed가 모델별 초기화/dropout RNG까지 같은 표본을 보장하지 않는다. RMT/NeoX에는 GRT 전용 RTLA를 적용하지 않으며 사후 지표 미측정값을0으로 채우지 않는다.

Kaggle 초기 배치는micro32×accum4×GPU2=256, Colab은micro32×accum8×GPU1=256이다. 기존 노트북 auto-batch·reserved memory 목표.95를 유지하되 preflight의 실제 peak와 보정 배치를 기록한다. T4 약16GiB에서 모델·gradient·Adam 상태는 약0.27~0.29GiB이며 activation/logits/replay가 주요 추가 메모리다. L150/M10 과거 실측은micro64×accum2·peak 약11GiB/GPU다. 큰 RMT M에서는micro16×accum8×2도 후보이며 GPU 실측 전 안전한 최대 배치를 확정하지 않는다. Auto-batch에서도 effective256과 evaluation batch10은 유지한다.

## 3. 학습량과 원논문 관계

매 update의 새 targets는 `global×L`이다. K3은 이전3개 segment와 현재1개를 함께 재계산하지만 과거 target은−100으로 제외한다. 학습량에3이나4를 곱하지 않는다.

| 항목 | L50 · 먼저 | L150 · 후속 |
|---|---:|---:|
| updates | 100,000 | 40,000 |
| 새 targets/update · nominal | 12,800 | 38,400 |
| 총 새 targets · nominal | 1,280,000,000 | 1,536,000,000 |
| 기존 manifest와 stream tail 반영 예상 실제 targets | 1,279,982,080 | 1,535,671,296 |
| train118,436,938 BPE tokens 대비 corpus 노출 | 약10.8회 | 약13.0회 |
| 이전3segments / 현재 포함 최대 text 문맥 | 150 / 200 tokens | 450 / 600 tokens |
| replay 포함 명목 text 입력 상한 규모 | 약5.12B | 약6.144B |
| warmup updates | 5,000 | 2,000 |
| 정기 validation 횟수 · 1k마다 | 100 | 40 |

실제 targets는 기존 manifest와 global256의 stream partition으로 계산했으며 최종 수치는 실행 report를 따른다. Replay는 epoch 시작에서 짧아지며, 위 입력 상한에 memory slots는 포함하지 않았다. 계산 시간이 정확히4배라는 뜻도 아니다.

[RMT 원저자 공개 기록](https://github.com/booydar/LM-RMT/blob/master/experiment_results/results_wt103_enwik8_all_parameters.csv)의 WikiText 예산은200k updates, L50/global128=1.28B word tokens, L150/global60=1.8B word tokens다. 이번 L50은 nominal token 수가 같고 L150은약85.3%다. BPE와 word tokenizer·모델 규모가 달라 원문과 같은 corpus 노출/계산량/word PPL 재현으로 표현하지 않는다. 원문 M sweep 전체가 K3인 것은 아니며 이번에는 모든 GRT/RMT M을K3으로 통일한다.

## 4. 실행 순서와 노트북 설정

[Colab](../../notebooks/colab.ipynb)·[Kaggle](../../notebooks/kaggle.ipynb)의 시작값은GRT·L50/M5·100k다. 기존 선택 셀에서 다음 값만 바꾸고 setup→prepare_data→execute 순서로 실행한다. 새로운 실험용 노트북/YAML을 복사하지 않는다.

```python
models = ('grt',)       # 한 run에 grt / rmt / neox 하나
segment_len = 50
num_registers = 5       # 순서대로 5, 10, 25, 50
budget = '100k'
```

GRT/RMT K3은 공통 카탈로그의 `training.bptt_segments=3`을 사용한다. NeoX는 모델 preset이M0/K0으로 설정하므로 선택 셀의 num_registers 값은 적용되지 않는다. 두 모델에서 같은 M을 비교하며 큰 M을 사전에 우승 후보로 간주하지 않는다.

권장 실행 순서는L50 NeoX→GRT M5/RMT M5→GRT M10/RMT M10→GRT M25/RMT M25→GRT M50/RMT M50이다. 각 조건의 preflight로 정상구간 update 시간·peak memory를 확인한 뒤 학습한다. 이후 L150은`segment_len=150`, `budget='40k'`, M10/25/75/150으로 같은 순서를 따른다. L과M·예산 변경은 새 run이며 기존40k checkpoint의 schedule을 바꿔100k로 연장하지 않는다.

서버 실행은`execution_mode='new'`로 모델/job 하나씩 등록한다. 기존11시간 세션 기한을 유지하고 중단 뒤 서버에서 이어서 학습을 선택하여 새 세션에서`execution_mode='resume'`·`resume_job_id`로 원래 Config·배치를 복원한다. Resume에서는 셀의 새 선택값보다 원래 job 설정이 우선한다. 자동 등록·GPU 학습은 아직 실행하지 않았다.

공유할 파일은 각 run의`experiment_results.zip`이다. Prepared 데이터와 checkpoint는 별도 보관한다. Commit·source hash·실제 Config·dataset/tokenizer hash·GPU/라이브러리·micro×accum×GPU·실제 target 수를 report로 확인한다.

## 5. 예상 소요 시간 · GPU 미실측 조건은 외삽

T4 두 장 한 쌍에서 학습·정기 평가·checkpoint 저장을 포함한 wall time이다. 준비/preflight·별도 사후 분석·재개 대기는 제외한다. GPU-hours는표 시간의2배다. 조건별 보정 전 추정치이며, 특히 L50/큰 M의 RMT는 불확실성이 크다.

| 단계 | 모델 | M | 중심 추정 h | 계획 범위 h |
|---|---|---:|---:|---:|
| L50·100k | NeoX | 0 | 9 | 6~14 |
| L50·100k | GRT | 5 | 20 | 15~31 |
| L50·100k | RMT | 5 | 19 | 13~29 |
| L50·100k | GRT | 10 | 21 | 15~32 |
| L50·100k | RMT | 10 | 22 | 15~34 |
| L50·100k | GRT | 25 | 26 | 18~40 |
| L50·100k | RMT | 25 | 32 | 21~49 |
| L50·100k | GRT | 50 | 36 | 26~54 |
| L50·100k | RMT | 50 | 47 | 32~73 |
| L150·40k | NeoX | 0 | 10 | 8~14 |
| L150·40k | GRT | 10 | 23 | 19~30 |
| L150·40k | RMT | 10 | 21 | 17~28 |
| L150·40k | GRT | 25 | 26 | 20~35 |
| L150·40k | RMT | 25 | 25 | 18~35 |
| L150·40k | GRT | 75 | 35 | 26~48 |
| L150·40k | RMT | 75 | 38 | 26~53 |
| L150·40k | GRT | 150 | 51 | 39~69 |
| L150·40k | RMT | 150 | 57 | 39~82 |

중심 추정의 반올림 전 합계는L50 약232h(9.7일), L150 약286h(11.9일), 전체 약518h(21.6일)다. Seed54 한 번·같은 T4 두 장을 연속 사용하는 누적 계산시간이며 quota·재개 대기를 포함한 달력 일정은 더 길다.

추정 근거:

- L150: [과거10k 실측](20261012_small_register_sweep.md)의 NeoX2.610h, RMT M10 5.366h, GRT M10/40/160=5.812/7.009/13.150h를40k로4배 환산한다. GRT 미측정 M은 인접 M 사이를 선형 보간한다. 과거 bias−2·warmup0·write 경로와 현재 candidate/A 조건은 다르다.
- L50 GRT: [L75 candidate40k](20261019_l75_gate_paths.md)의 M8/20/75=12.323/12.488/21.730h를 기준으로 M_ref=1.5M(최솟값은 기존 M8)에서 wall time을 보간하고 `(50+M)/(75+M_ref)×100000/40000`으로 길이와 updates를 보정한다.
- RMT: L150/M10 기준 `5.366×(updates/10000)×(L+2M)/170`으로 외삽한다. NeoX L50은 과거 L150에서 targets 비율로 환산한다. 고정 update 비용·softmax·attention 제곱항·batch 변경을 완전히 나타내지 않으므로 추정 범위를 둔다.

실행 시 `남은 updates×정상구간 update초+남은 평가 횟수×전체 validation초+저장 여유`로 갱신한다. 짧은 preflight로 후보의 수렴 순위를 판단하지 않는다.

## 6. 결과와 분석 · 미수신

각 run의 best step/PPL, final PPL/NLL, 실제 targets, wall h, peak GiB, 원본 학습/사후 report 링크를 M별로 기록한다. 실측 전 값을0으로 채우지 않는다.

GRT는 선행 보고와 같은 공통6패널(validation PPL, probe write, 절대 logit ECDF, 슬롯 내 gate std, 문맥 중심화 state/update rank, reset/swap ΔNLL)을 기본으로 한다. 학습 곡선은 실제 누적 새 targets를 가로축에 놓고 학습 probe·사후 window·전체 validation을 구분한다. C/O/T는 calibration window0에서 고정하고 연속 사후 지표는 window16/32 동일 가중 평균으로 기록한다. Rank 절댓값을 M 간 효율 순위로 쓰지 않으며 단일 seed·유한 budget의 결론으로 다룬다.

결과 수신 후 이 문서에 조건·원본 링크·판단을 누적한다. 이전 L75/L150 보고서의 수치와 이번 미실행 조건을 섞지 않는다.

## 7. L50/M5 첫 서버 실행 · 학습 시작 전 중단

2026-10-29 사용자 전달 로그: job `job_mvt0ohj42d5669da3c60bf52bb55`, commit `4727470e5b92260a3216fe22a94872771d1920c9`, prepared manifest `675a7671fa308434d6bab1709fc5280d21f28a70e9664e156ce32d5bbe64b7a3`. GRT L50/M5·100k·T4×2, Python3.12/PyTorch2.10.0+cu128/Transformers4.45.2, seed54, FP32다.

GPU preflight는0.594s/update·micro128×accum1×GPU2=global256·peak7.17GiB/GPU·reserved8.918GiB/GPU(61.2%)로 완료됐다. 계산 전용 ETA는학습16.494h+validation.265h=16.758h이며 준비·저장·RTLA·재개 대기는 제외한다. 이 측정은 위 M5 계획 추정을 보정할 근거지만 실제 전체 wall time이나 학습 수렴 결과는 아니다.

본 학습 명령 출력 전에 worker가 종료되어 summary·학습 metrics는 미수신이다. 당시 코드가 최초 예외를 숨기고 서버에 `worker_failed: Worker could not prepare data or restore its authorized snapshot`을 기록했다. 코드 점검에서 queue train의 내부 정리 후 바깥 worker가 연결 부재를 준비 실패로 오분류하는 경로를 확인했다. Config/provenance 검증·업로드 등을 포함한 학습 시작 경로의 구체적 최초 오류는 원래 로그만으로 확정하지 못했다. 오류 보존/보고 수정과 재실행 절차는 [서버 실패 진단](../server.md#worker-실패-진단)을 따른다.

이후 사용자 전달 NeoX L50/M0 로그: job `job_mvt24p5x04e4210be081dffa0e7a`, commit `c4849f9f085e8d1ecd56d19fbd79fb6885c58e37`. 실패 위치는 `ServerRun.evidence()`의 config 업로드이며 저장소 PUT이403을 반환했다. Summary는 없으며 GRT와 NeoX에서 공통으로 관측된 업로드 경로 실패다. 당시 응답 본문을 폐기하여 S3의 실제 Code(AccessDenied/SignatureDoesNotMatch 등)는 전달 로그에 없다.

GRT `c4849f9`와 GRT_server `4bfa012`의 최신 코드를 대조했다. 운영 문서상 저장소는 S3 `grt-server/grt/`·서울 리전이다. 가짜 임시 자격증명을 사용한 AWS SDK presign과 Python requests PUT의 독립 SigV4 계산은 일치했다. 이는 기본 URL/헤더 조합의 오프라인 검증이며 운영 IAM·bucket policy·실제 발급 자격증명과 거절 사유는 검증하지 못했다. 읽기용 데이터 다운로드 성공은 쓰기 권한의 근거가 아니다. 알려진 S3 응답 코드만 보존하도록 client 진단을 보완했으며 운영 장애 원인을 추정으로 확정하지 않는다.


2026-10-30 사용자 전달 NeoX job `job_mvu6lfaw44f8f3bb68b92effb783`(GRT `811a8e4`)에서 `storage_code=AccessDenied`를 수신했다. 사용자는 같은 서버에서 기존 SDK 마이그레이션의 업로드가 성공했고 IAM 설정을 완료했다고 확인했다. 이후 사용자 지시에 따라 업로드를 실행기→인증된 EC2 API→서버 SDK→S3로 변경한다. 연구 조건은 유지하고 서버/client를 함께 업데이트한 새 job으로 실행한다. 기존 S3 데이터 다운로드는 유지하며, 운영 재실행 결과는 아직 미수신이다.


2026-10-31 사용자 보고: 업로드 경로 변경 후 이전403은 재발하지 않았으나 baseline 사후 분석 완료 검증과 W&B crashed/서버 running 불일치가 발생했다. 점검 결과 baseline 실행 스킵과 require_posthoc 등록이 불일치했고, lease 만료는 stale 표시만 남겼다. Baseline은 정상 skipped 완료, 프로세스 오류는 업로드와 독립적인 실패 보고, worker 유실은 lease 만료 interrupted로 정리한다. 실제 보고된 crash의 최초 stack trace는 미수신이며 이 기록을 특정 학습 오류의 원인 확정으로 해석하지 않는다.
