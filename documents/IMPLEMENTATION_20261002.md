# 2026-10-02 구현 기록

기준: `20260916_spec.md` v1.0. 기존 과제·패키지 구현을 얕은 `grt` 구조로 전환하고
기존 설계 문서 및 구현 ZIP을 보존했다. 결과·체크포인트 파일을 삭제하지 않았다.

## 구현 범위

- 공통 logits-only interface, 독립 GRT/RMT, tied embeddings, local positions, full BPTT.
- 고정 Copy/Reverse/passkey 규격, sample ID와 split 기반 독립 RNG, 길이 간 동일 payload.
- strict YAML/dataclass 설정, 표준 small/large 검증, 순차 병합 및 resolved config.
- token-count CE/정확도 집계, AdamW 분류, accumulation, precision, warmup/cosine, gradient clipping.
- versioned atomic checkpoint와 Python/NumPy/Torch/CUDA RNG 및 sample 진행 복원.
- best-validation 선택, 최종 validation/checkpoint, 모든 test 길이 및 별도 성능 측정.
- GRT의 갱신 전 FP32 배치 평균 RTLA, NPZ/JSON/PNG, 선택적 W&B.
- 학습/평가/분석 CLI와 Kaggle/Colab 점검 노트북.

## 검증 상태

사용자 요청에 따라 이 작업 환경에서는 모델·학습·pytest를 실행하지 않는다.
추가한 테스트는 실행되지 않았으므로 통과를 주장하지 않는다. 노트북에서 실행할 항목은
데이터 oracle, 결정성/worker 독립성, read/write 동작, 메모리 순환 및 gradient,
고정 batch overfit, dropout CPU resume 동일성, accumulation, 평가 집계,
mode/RNG 복원, RTLA 공식과 로컬 출력이다.

모듈을 import하거나 실행하지 않고 Python 소스 및 노트북 코드 셀의 AST 구문을 확인했고,
노트북 JSON과 기존 구현 ZIP의 무결성을 확인했다. 이는 런타임 검증을 대체하지 않는다.

표준 50,000-step 실험 및 CE≤0.05 달성 여부는 미검증이다. GPU backend 결정성,
FP16/BF16 실행, W&B 업로드도 실제 환경에서 확인해야 한다.

## 운영 사항

- Git worktree의 `.git`가 존재하지 않는 원본 경로를 참조한다. Git 연결은 수정하지 않았다.
- 기본 Python에 PyTorch/PyYAML/pytest가 없다. 설치 시도는 승인되지 않았고 의존성 설치를 진행하지 않았다.
- 구형 checkpoint 자동 복원은 지원하지 않는다. FP16 overflow는 skip 기록 후 진단과 함께 중단한다.
- CPU fixture는 내부 테스트 전용으로 작은 모델을 쓰며 CLI에서는 표준 규격 검증을 우회하지 않는다.
