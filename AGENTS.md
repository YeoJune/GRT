# GRT 작업 지침

## 프로젝트와 파일 구조

GRT와 독립 RMT를 Copy, Reverse, Associative Retrieval(`passkey`) 합성 메모리 과제에서 비교한다. Python 3.10+, PyTorch 2.2+ 기반이며 W&B는 선택 사항이다.

| 경로 | 역할 |
|---|---|
| `src/grt/models/` | 공통 `interface.py`·`factory.py`, GRT의 `grt.py`·`router.py`·`alu.py`, 기존 `rmt.py`와 공개 Copy용 `rmt_relative.py` |
| `src/grt/data.py`, `config.py`, `metrics.py` | 결정적 데이터 생성, YAML 병합·검증, masked CE·정확도 집계 |
| `src/grt/trainer.py`, `evaluator.py`, `checkpoint.py`, `logger.py` | 공통 학습·평가, 상태/RNG 복원, 실행 기록 |
| `src/grt/rtla.py`, `plots.py` | GRT 전용 trace 수집과 NPZ·PNG 분석 출력 |
| `scripts/train.py`, `evaluate.py`, `analyze.py` | 학습·재개, checkpoint 평가, GRT 분석 CLI |
| `configs/` | 공통 `base.yaml`, 모델·과제 설정, `large.yaml`, Colab T4 Copy override |
| `tests/` | 데이터·설정·모델·학습/재개·RTLA 검증 |
| `notebooks/smoke_test.ipynb` | 사용자 Colab에서 테스트와 두 모델의 짧은 실행 점검 |
| `notebooks/colab_t4_copy.ipynb` | Colab T4에서 RMT/GRT의 Copy 초기 수렴 비교 |
| `runs/` | 실행별 설정·로그·checkpoint·평가·분석 산출물; 기존 결과 보존 |
| `pyproject.toml`, `documents/` | 패키지 의존성·개발 설정, 명세와 연구 기록 |

## 구현과 실험 작업

- 작업 전 관련 코드·설정, `README.md`, `documents/`의 관련 설계·실험 기록을 확인한다. 문서의 적용 범위와 현재 요청을 함께 파악하고, 구현과 문서의 불일치는 명시한다.
- 현재 두 모델은 공통 logits-only 인터페이스와 학습·평가 코드를 사용하되 모델 구현은 독립적이다. loss 계산은 모델 외부에, GRT 분석은 RTLA에 둔다.
- 표준 `recovery` 과제는 양방향 세그먼트 attention으로 마지막 정답을 병렬 복원한다. 별도 `paper_copy` 프로필은 causal 상대 위치·Post-LN RMT로 공개 Copy의 자기회귀 생성 조건을 검증한다. 이 프로필의 GRT 적용은 RMT 결과 확인 후 진행한다. 데이터 생성·loss mask·상태 전달을 변경할 때 정답 누출과 재현성에 주의한다.
- 설정은 `base → 모델 → 과제 → override` 순으로 병합하고 목록은 교체한다. 실험의 실제 조건은 `resolved_config.yaml`과 실행 metadata로 확인한다. 설정을 바꾼 실험은 기존 결과를 덮어쓰지 않고 별도 run으로 기록한다.
- 모델 비교에서는 데이터·학습 예산·평가 조건의 차이를 밝힌다. 짧은 실행 점검, 기본 길이 수렴, 길이 확장 성능을 구분하며 gate 분포만으로 메모리의 의미적 역할을 단정하지 않는다.

## 협업과 검증 절차

1. **사용자 요청 → 에이전트 구현:** 관련 명세·코드·문서를 읽고 요청한 변경을 완료한다. 필요한 테스트 코드와 Colab 실행 안내도 함께 준비한다.
2. **변경 보고 → 사용자 검토:** 변경 내용, 수행한 CPU 검증과 결과, 미검증 항목, Colab 실험 방법을 간결하게 보고한다.
3. **사용자 승인 → 커밋:** 사용자의 명시적인 커밋 승인 후 커밋한다. 승인 전에 임의로 커밋하지 않으며, push는 별도 지시가 있을 때 수행한다.
4. **CPU 검증과 사용자 Colab 실행:** 에이전트는 변경에 필요한 짧은 CPU 테스트, 작은 fixture의 forward·학습/재개 점검 등 간단한 검증을 직접 수행할 수 있다. GPU 검증과 장시간 학습·수렴 실험은 사용자가 별도 Colab 환경에서 변경이 반영된 저장소를 pull한 뒤 수행한다. CPU 검증 결과를 GPU 동작이나 실험 수렴의 증거로 표현하지 않는다.
5. **결과 공유 → 에이전트 분석:** 사용자가 전달한 로그·지표·그림을 실행 commit, 설정, 하드웨어 조건과 함께 분석한다. 관측 사실과 가설을 구분하고 다음 수정·실험을 제안하며 같은 절차를 반복한다.

분석 자료는 목적에 맞게 `resolved_config.yaml`, `metadata.json`, `metrics.jsonl`, `evaluation.json`, 테스트 로그, RTLA NPZ/PNG, Colab의 `comparison.json`·`convergence.png`를 활용한다. 실행하지 않은 테스트의 통과나 실험 수렴을 주장하지 않는다.

## Colab 실험 안내

실험이 필요한 변경에는 사용할 노트북 또는 실행 명령, 주요 설정, 분석을 위해 공유할 결과를 간단히 안내한다. 기본적인 Colab 사용법은 생략하고 이번 변경에 필요한 내용에 집중한다.

Colab 실험은 T4 GPU(VRAM 약 16GB)를 기준으로 모델·activation·optimizer 메모리와 여유 공간을 계산해 배치 크기를 정하고, 실제 peak VRAM 결과로 조절한다. microbatch·gradient accumulation·effective batch를 구분하며, effective batch 변경이 학습 조건에 미치는 영향도 명시한다.
