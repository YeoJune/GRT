# GRT

[2026-09-16 명세](documents/20260916_spec.md)에 따른 합성 메모리 벤치마크입니다.
GRT와 독립 RMT를 Copy, Reverse, Associative Retrieval (`passkey`)에서 비교합니다.
각 모델은 양방향 세그먼트 attention으로 마지막 세그먼트의 정답 위치를 병렬 복원합니다.

## 설치

Python 3.10 이상 및 PyTorch 2.2 이상을 사용합니다. GPU 실험 환경의 CUDA에 맞는
PyTorch를 먼저 설치하고 다음 명령을 실행하세요.

```bash
python -m pip install -e '.[dev]'
# 선택: W&B를 사용할 때만
python -m pip install -e '.[wandb]'
python -m pytest -q
```

## Kaggle / Colab 점검

간단한 진입점은 [notebooks/smoke_test.ipynb](notebooks/smoke_test.ipynb)입니다.
현재 수정본을 `/kaggle/working/GRT` 또는 `/content/GRT`에 업로드·압축 해제하거나,
수정본이 반영된 저장소를 clone하세요. 노트북 첫 코드 셀의 `REPO_DIR`를 실제 경로로
맞추고 위에서부터 실행합니다. Kaggle은 인터넷을 켜고 GPU accelerator를 선택하세요.
Colab은 런타임 유형에서 GPU를 선택하세요. W&B 계정은 필요하지 않습니다.

노트북은 CPU 계약 테스트, 표준 small GRT/RMT의 2-step 학습, 세 길이 평가,
체크포인트 재개 CLI, GRT NPZ/PNG 분석을 실행합니다. `TASK`를 `reverse` 또는
`passkey`로 바꾸어 다른 과제를 점검할 수 있습니다. 표준 모델·과제 규격은 유지하며
학습 예산과 평가 표본 수만 줄입니다. 수렴과 zero-shot 일반화 실험은 별도로 수행해야 합니다.
재개 후 다음 update가 같은지는 dropout CPU fixture 테스트가 확인합니다.

결과와 `pytest_output.txt`, `pytest.xml`을 `runs/notebook_<timestamp>/`에 저장하고
마지막 셀에서 ZIP을 만듭니다. 실패하면 해당 로그를 공유하세요.

## 학습 / 평가 / 분석

```bash
python scripts/train.py --config configs/base.yaml configs/grt.yaml configs/copy.yaml --output-dir runs/grt-small-copy
python scripts/train.py --config configs/base.yaml configs/rmt.yaml configs/copy.yaml --output-dir runs/rmt-small-copy
python scripts/train.py --config configs/base.yaml configs/grt.yaml configs/passkey.yaml configs/large.yaml --output-dir runs/grt-large-passkey

python scripts/train.py --resume runs/grt-small-copy/last.pt
python scripts/evaluate.py --checkpoint runs/grt-small-copy/best.pt
python scripts/analyze.py --checkpoint runs/grt-small-copy/best.pt --split test --segments 20
```

`copy.yaml`를 `reverse.yaml` / `passkey.yaml`로 교체하여 과제를 선택합니다.
RMT large도 `configs/rmt.yaml` 뒤에 `configs/large.yaml`을 적용합니다.
설정은 왼쪽에서 오른쪽으로 병합하며 목록은 교체합니다. 알 수 없는 키·타입과
표준 규격을 벗어난 설정은 오류입니다. RMT에 GRT router/register 설정을 넣을 수 없습니다.

학습은 `--max-steps`, `--batch-size`, `--grad-accum-steps`, `--mixed-precision`,
`--device auto|cpu|cuda`를 지원합니다. CLI 변경은 resolved config에 저장합니다.
재개 시 실행 설정 override는 허용하지 않습니다. 체크포인트의 모델·optimizer·scheduler·
scaler·RNG·다음 sample ID를 복원하고, run 경로는 checkpoint 파일의 현재 부모 폴더로 정합니다.
기존 run을 덮어쓰는 새 학습은 거절합니다. 이전 형식의 checkpoint는 자동 이관하지 않습니다.
PyTorch checkpoint는 pickle을 포함하므로 직접 생성한 신뢰할 수 있는 파일만 사용하세요.

정상 종료 시 validation과 `last.pt`를 저장하고, validation CE로 선택한 `best.pt`를
기본 길이와 두 확장 길이에서 평가합니다. 기본 표본은 validation 1024, 길이별 test 4096입니다.
학습 목표 CE≤0.05는 최초 도달 시점을 기록하며 자동 조기 종료하지 않습니다.
FP32가 기본이며 이 구현의 FP16/BF16 실행은 지원하는 CUDA 장치를 요구합니다.
FP16 overflow는 optimizer/scheduler step을 진행하지 않고 로그·진단을 남긴 뒤 중단합니다.

## 구조와 산출물

`src/grt/models/`에는 공통 interface/factory와 독립 GRT/RMT가 있습니다.
`data.py`, `metrics.py`, `trainer.py`, `evaluator.py`, `checkpoint.py`, `logger.py`가
공통 실행을 담당하고 `rtla.py`, `plots.py`가 GRT 분석을 담당합니다.
모델은 labels나 loss를 받지 않으며 매 forward에서 메모리를 초기화하고 세그먼트 간
gradient를 유지합니다. 실제 padding은 지원하지 않아 false attention mask에 오류를 냅니다.

각 run은 `resolved_config.yaml`, `metadata.json`, `metrics.jsonl`, `best.pt`, `last.pt`,
주기 `step_XXXXXX.pt`, `evaluation.json`, GRT의 `rtla/*.npz|json|png`를 저장합니다.
RTLA는 갱신 전 상태에서 샘플별 FP32 norm을 구한 뒤 배치 평균합니다.
conditional read가 꺼진 R은 inactive read head로 표시하며 의미적 슬롯 역할을 자동 분류하지 않습니다.
기존 trace 경로는 덮어쓰지 않습니다. 분석을 다시 실행할 때는 새 `--output-dir`를 사용하세요.

평가는 정답 토큰 수로 CE/token accuracy를 집계하고 sample 단위 exact match를 계산합니다.
별도 inference 측정에서 데이터 생성·전송·분석을 제외하고 input tokens/sec,
samples/sec 및 peak allocated GPU memory를 기록합니다. full logits 출력 비용은 포함됩니다.
GRT의 내부 세그먼트 길이는 N+M, RMT는 N+2M이며 실제 파라미터 수를 함께 보고합니다.

설계 문서는 보존했습니다. 기존 구현의 보존 사본은
`documents/legacy_implementation_20261002.zip`이며 새 CLI가 참조하지 않습니다.
작업 환경의 Git 연결이 깨져 있으면 metadata의 Git commit은 null이고 오류를 함께 기록합니다.

이 변경은 개발 환경에서 학습·테스트를 실행하지 않았습니다. 노트북 실행 결과로 검증해 주세요.
