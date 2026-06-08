GRT — 실행 및 테스트 가이드 (간단)

목적: Phase 1/2 구현을 로컬에서 빠르게 검증하기 위한 최소 실행 명령 모음입니다.

전제: Windows PowerShell 환경(사용자 OS: Windows). Python 3.10+ 권장.

1. 가상환경 생성 및 의존성 설치

PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
```

(만약 `pip install -e .`가 `torch` 설치 문제를 일으키면, 시스템/환경에 맞는 `torch`를 먼저 설치하세요.)

2. Sanity 체크 (Phase 1 검증)

```powershell
python scripts/sanity_check.py
```

3. 간단한 학습(스모크) 실행

- 작은 toy config로 빠르게 학습합니다 (예: 10 스텝)

```powershell
python scripts/train.py configs/experiment/toy_passkey.yaml --max_steps 10
```

4. 체크포인트로 평가 실행

- `train`이 저장한 체크포인트(예: `checkpoints/step_000000.pt`)를 사용합니다.

```powershell
python scripts/evaluate.py configs/experiment/toy_passkey.yaml --checkpoint checkpoints/step_000000.pt
```

5. RTLA trace 생성 및 시각화

```powershell
python scripts/run_rtla.py configs/experiment/toy_passkey.yaml --checkpoint checkpoints/step_000000.pt --input_text "Test trace"
# 출력은 기본적으로 configs.base.yaml에 설정된 rtla.output_dir (기본: traces/) 아래에 저장됩니다.
```

6. RTLA 패널 확인

- 생성된 패널 이미지: `traces/rtla_panels.png` 또는 `traces/step_XXXXXX_rtla_panels.png` (설정에 따라 다름)
- 로컬에서 열어 시각적으로 확인하세요.

7. (선택) 유닛 테스트 실행

```powershell
pytest -q
```

(현재 리포지토리에 테스트가 없을 수 있습니다 — 존재하면 실행됩니다.)

8. (선택) W&B 연동 스모크

- `configs/base.yaml`에서 `wandb.enabled: true`로 설정하고 W&B API 키를 설정한 뒤 `train`을 실행하세요.

```powershell
setx WANDB_API_KEY "<your_key>"
# 또는 PowerShell 세션에서
$env:WANDB_API_KEY = "<your_key>"
python scripts/train.py configs/experiment/toy_passkey.yaml --max_steps 10
```

문제 발생 시 체크리스트

- `ModuleNotFoundError` 또는 import 오류: `sys.path`에 `src`가 포함되도록 실행 스크립트를 사용하세요 (제공된 `scripts/*.py`는 이미 처리함).
- `torch` 설치 문제: CUDA/CPU 빌드 옵션을 확인하고 `pip` 명령을 환경에 맞게 조정하세요.
- ALU 입력 길이 오류: `configs/base.yaml`의 `model.segment_len`과 `model.num_registers`의 합이 ALU가 기대하는 값과 일치하는지 확인하세요.

끝.
