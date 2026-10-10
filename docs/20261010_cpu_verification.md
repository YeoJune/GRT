# 실제 진입점 CPU 검증 기록

2026-10-10, 현재 미커밋 작업 트리에서 실행했다. 단위 테스트 외에 사용자가 실행하는
`scripts/train.py`, `scripts/evaluate.py`, `scripts/analyze.py`를 별도 Python 프로세스로 호출했다.
별도 모델/학습 구현을 만들어 시험하지 않았다.

## 실행 조건

- Python3.12.14 / Torch2.14.1+cpu / Transformers4.45.2, FP32, 프로세스당 CPU thread1.
- RMT와 GRT 모두 `configs/base.yaml` + 해당 모델 YAML을 사용했다.
- **주 실험은 cpu.yaml을 사용하지 않고 D128 / 4층 / FF128 / memory32를 유지했다.**
  RMT435,456 / GRT847,552 parameters다.
- Remember key/value 길이1, train128 / validation64 / test64, 원본 random collator.
- microbatch16 × accumulation2 = effective batch32.
- 1쌍100 → 2쌍200 update, 평가 간격25. 실제 노출량은3,200/6,400 samples다.
- 모든 단계의 실행을 확인하기 위해 이 CPU 주 실험만 `stop_on_convergence=false`로 했다.
  T4 설정이나99% 수렴 기준을 변경하지 않았다.

실행 명령은 다음과 같다. `$MODEL`은 rmt/grt이며 `$ROOT`는 아래 결과 경로다.

```bash
python scripts/train.py --config configs/base.yaml configs/$MODEL.yaml "$ROOT/cpu_experiment.yaml" --output-dir "$ROOT/$MODEL" --device cpu
python scripts/evaluate.py --checkpoint "$ROOT/$MODEL/stage_02_pairs2_key1/best.pt" --output-dir "$ROOT/${MODEL}_evaluation_cli" --samples 64 --batch-size 16 --device cpu
python scripts/analyze.py --checkpoint "$ROOT/grt/stage_02_pairs2_key1/best.pt" --output-dir "$ROOT/grt_analysis_cli" --batch-size 4 --device cpu
```

## 주 실험 결과

| 모델 | 1쌍 fixed validation CE, step25→100 | 2쌍 fixed validation CE, step25→200 | best checkpoint test 생성 EM |
|---|---|---|---|
| RMT | 2.9606 → 1.7540 | 1.6394 → 1.3232 | 28.125% |
| GRT | 2.1796 → 1.2660 | 1.3635 → 0.6568 | 43.750% |

표의 CE 시작점은 처음 기록한 step25이며 초기화 직후의 값이 아니다.
CE는 value+EOS teacher forcing, EM은 실제 생성이다. `stop_on_convergence=false`에서는
원본 방식대로 random-length validation EM으로 best를 선택하므로 마지막 fixed 평가와
best test는 checkpoint·분포가 다르다. 두 모델의 sampling sequence도 초기화 RNG 때문에 다를 수 있다.

별도 평가 CLI의 출력은 학습 종료 평가와 정확히 일치했다. 두 모델은 모두300update를
완료했고 loss가 유한했으며, 주요 모듈의 저장된 AdamW moment가0이 아님을 확인했다.
각 run의 JSON/TXT와 stage JSONL/TXT 기록, best/last checkpoint도 확인했다.
두 주 실험의 wall time은 약101초/77초였으며 병렬 CPU 검사 시간이다. T4 시간 추정치가 아니다.

**99% 수렴 기준에는 미달했다. 이 검사는 실행·gradient·학습 진행을 확인하며,
완전한 과제 학습이나 전체 메모리 capacity 성공 판정은 아니다.**

## 실제 종료·재개

두 모델을 같은 기본 규모와 배치로 별도1쌍10 → 2쌍30 update 실행했다.
각 모델의 한 실행을2쌍step5 checkpoint 저장 후 SIGTERM으로 종료하고,
checkpoint를 수정하지 않은 채 동일 학습 CLI·설정·경로로 재개했다.

연속 실행과 재개 실행의 최종 model/optimizer/scheduler/RNG, step, loader cursor,
samples seen, best EM, validation과 완료 상태가 모두 정확히 일치했다.
이는 단순 완료 run 재호출이나 checkpoint tensor만 다시 읽는 검사가 아니다.

작은 `cpu.yaml` 기반의 별도 CLI 실행에서는99% 기준 미달 시 stage1에서 중지하고
stage2를 만들지 않는 것도 두 모델에서 확인했다.

## 선택 기능과 발견한 오류

- 실제 W&B SDK0.30.0을 설치해 두 모델의 작은 CLI 실행에 offline 모니터링을 켰다.
- 첫 실행에서 `wandb.util.generate_id`가 현재 SDK에 없어 모니터링 초기화가 실패했다.
  로컬 학습은 계속되고 오류가 기록됐지만, W&B 기능 성공으로 판정하지 않았다.
- ID 생성은 표준 라이브러리 UUID로 수정했다. 수정 후 별도의 새 run에서 SDK 초기화·기록·종료가
  오류 없이 완료됐다. 실제 `.wandb` 파일의 history를 읽어 global steps1/2/3/4,
  pairs1/1/2/2, 최종 test summary를 확인했다. 온라인 서버 업로드는 검사 범위에 없다.
- GRT에는 conditional read=true, write dropout=.2, rtla.enabled=true를 함께 적용했다.
  학습 중 NPZ/JSON/PNG4세트, 유한 trace 값과 실제 fact 수1/2를 확인했다.
  W&B history에도 GRT gate/register 통계가 저장됐다.
- 주 실험 checkpoint의 별도 GRT 분석 CLI도2fact ×32register trace와 PNG를 생성했다.
- 수정 후 기존 CPU 테스트22개가 다시 통과했다. SDK 내부 ID 생성 도구가 없어도
  모니터링 초기화·run ID 재사용 검사가 동작하도록 mock 검사도 갱신했다.

## 결과 위치

로컬 결과: `runs/cpu_cli_20261010_j_tprozf/`.
운영 설정에 CPU 예산을 추가하거나 기존 Colab 실험 결과를 덮어쓰지 않았다.

- `cpu_experiment.yaml`: 주 실험 override, `rmt/`·`grt/`: 실제 로그·checkpoint.
- `verification.json`: CLI 평가 대응, 기록, 실제 sample count, 모듈의 optimizer moment 점검.
- `rmt_resume_verification.json`, `grt_resume_verification.json`: 실제 종료·재개 대조.
- `rmt_features_fixed/`, `grt_features_fixed/`: 수정 후 실제 SDK·선택 기능 실행.
- `rmt_wandb_verification.json`, `grt_wandb_verification.json`: 실제 offline history 점검.
- `grt_analysis_cli/`: 별도 분석 결과. 초기 W&B 실패 기록도 원인 확인용으로 보존했다.

코드 변경은 W&B ID 생성 호환성 수정과 해당 회귀 검사다. commit/push는 수행하지 않았다.
