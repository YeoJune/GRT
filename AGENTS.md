# GRT 작업 지침

## 프로젝트와 구조

현재 기준은 ARMT 논문의 Remember 데이터와 저자 RMT baseline이다. GRT도 동일한 배치·forward/generate 계약과 공통 trainer/evaluator로 실행한다.

| 경로 | 역할 |
|---|---|
| `src/grt/models/` | 모델 factory, 원본 RMT 구성, GRT router·ALU·write-back |
| `src/grt/vendor/armt/` | 원본 backbone·wrapper·데이터, 출처와 hash |
| `src/grt/config.py`, `data.py`, `metrics.py` | 설정, 교체 가능한 데이터 adapter, predictor-mask CE |
| `src/grt/trainer.py`, `evaluator.py` | 공통 학습·curriculum·실제 생성 평가 |
| `src/grt/checkpoint.py`, `logger.py` | 상태/RNG 복원, JSON/JSONL/TXT, 선택적 W&B |
| `src/grt/rtla.py`, `plots.py` | GRT fact update trace와 분석 |
| `scripts/` | `train.py`, `evaluate.py`, `analyze.py`, 작은 CPU 검사 `check_rmt.py` |
| `configs/` | `base.yaml` → 모델(`rmt/grt`) → 환경(`t4/cpu`) |
| `notebooks/` | `template.ipynb`와 최신 `colab.ipynb` 두 개만 유지 |
| `docs/` | 날짜가 붙은 설계 기준, `archive/`의 이전 기록, 문서 목록 |
| `tests/`, `runs/` | CPU 검사, 실행별 결과·checkpoint |

## 구현 원칙

- 작업 전 관련 코드·설정, README와 관련 문서를 읽는다. 문서의 날짜·적용 범위와 사용자 요청을 함께 확인한다.
- 원본 기준 코드와 GRT 설계, 학습 최적화를 구분한다. 원본 파일 변경 시 `SOURCE.json`과 대조 검사를 갱신한다.
- 모델은 raw labels와 predictor 위치의 mask를 받아 `.loss`를 반환한다. 외부에서 labels를 먼저 shift하지 않는다. 두 모델 모두 같은 데이터·생성 target을 사용한다.
- GRT의 router·ALU·write-back 구현을 유지한다. 양방향 fact 처리와 query prefix 계산을 구분하고 정답 누출, 중복 memory write, fact→query gradient를 확인한다. 모델 세부 설계는 GRT 실험에서 확정한다.
- 새 데이터셋은 adapter를 통해 같은 배치 계약에 연결한다. trainer/evaluator에 과제·모델별 분기를 추가하지 않는다. 인터페이스 변경은 데이터 경계·모델 factory에서 명시한다.
- 원본 collator는 전역 Torch RNG를 사용한다. 같은 dataset/seed라도 모델 초기화에 따라 길이·질의 선택 순서가 다를 수 있다. 이를 동일 배치 비교로 표현하거나 정리 작업 중 몰래 바꾸지 않는다.
- 설정은 순서대로 병합하고 목록은 교체한다. 실제 조건은 resolved config와 metadata로 확인한다. 설정이 바뀌면 새 run을 사용하고 기존 결과를 보존한다.
- W&B 기본은 false이며 비활성 시 import·연결하지 않는다. 모니터링이 모델 RNG에 영향을 주지 않게 한다. 로컬 TXT/JSON 기록을 항상 유지한다.
- 임시 문서는 제거하고 기록 문서는 `YYYYMMDD_이름`으로 보존하며 목록을 갱신한다. archive 문서는 현재 실행 안내로 사용하지 않는다.

## 협업과 검증

1. 사용자 요청을 바탕으로 구현과 필요한 간단한 CPU 검증을 완료한다.
2. 변경·검증 결과·미검증 항목과 간결한 Colab 실험 방법을 보고한다.
3. 사용자의 명시적인 승인 후 커밋하고, push는 지시가 있을 때 수행한다.
4. GPU·장시간 실험은 사용자가 Colab에서 실행한다. 에이전트는 작은 CPU 기능·gradient·학습/재개 검사를 할 수 있다.
5. 사용자가 전달한 결과를 commit·설정·하드웨어와 함께 분석한다. 관측 사실과 가설, CPU 기능 검사와 GPU 수렴을 구분한다.

Colab 안내는 사용할 노트북/명령, 주요 설정, 공유할 결과만 간결하게 제시한다.
T4(VRAM 약16GB)를 기준으로 모델·activation·optimizer 메모리와 여유를 계산해 배치를 정하고 실측 peak로 조절한다. microbatch·accumulation·effective batch를 구분한다. 실측 update 시간으로 전체 소요 시간을 계산하며 원본 조건·샘플 예산과 전체 논문 실험 재현을 구분한다.
