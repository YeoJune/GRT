# GRT 작업 지침

## 프로젝트와 구조

현재 기준은 `docs/specification.md`의 공통 NeoX 인코더·디코더와 Remember/WikiText-103이다. 현재 공통 NeoX만 실행하며 과거 모델/설정/checkpoint 호환 경로는 유지하지 않는다.

| 경로 | 역할 |
|---|---|
| `src/grt/models/` | 공통 NeoX factory/backbone, RMT memory, GRT router·write-back |
| `src/grt/vendor/armt/` | 원본 backbone·wrapper·데이터, 출처와 hash |
| `src/grt/config.py`, `data.py`, `metrics.py` | 설정, 교체 가능한 데이터 adapter, predictor-mask CE |
| `src/grt/trainer.py`, `evaluator.py`, `distributed.py`, `benchmark.py` | 공통 학습·평가, torchrun 분산 실행·사전 측정 |
| `src/grt/checkpoint.py`, `logger.py` | 상태/RNG 복원, JSON/JSONL/TXT, 선택적 W&B |
| `src/grt/rtla.py`, `memory_analysis.py`, `plots.py` | GRT trace, WikiText state 개입 분석과 그래프 |
| `src/grt/experiments.py`, `utils/` | preset 조합, 노트북 공통 실행·준비·측정·결과 확인 |
| `scripts/` | `train.py`, `evaluate.py`, `analyze.py`, 큐 실행기와 결과 집계 |
| `configs/` | `experiments.yaml`의 과제·크기·모델·tokenizer·예산·환경 카탈로그; 과거 YAML은 삭제 |
| `notebooks/` | 현재 Remember template, Kaggle·Colab 실행·사후 분석 노트북 |
| `docs/` | 현재 명세·workflow·서버 연결·연구 요약, `experiments/`의 실험별 기록, `archive/`의 이전 설계 |
| `tests/`, `runs/` | CPU 검사, 실행별 결과·checkpoint |

## 구현 원칙

- 작업 전 관련 코드·설정, README와 관련 문서를 읽는다. 문서의 날짜·적용 범위와 사용자 요청을 함께 확인한다.
- 원본 기준 코드와 GRT 설계, 학습 최적화를 구분한다. 원본 파일 변경 시 `SOURCE.json`과 대조 검사를 갱신한다.
- Remember decoder는 raw labels와 predictor mask, encoder는 masked slot에 정렬된 raw labels를 받는다. WikiText adapter만 명시적 `target_ids` 계약으로 다음 token을 정렬한다. loss shift를 중복 적용하지 않는다. 두 모델은 같은 과제 입력·target을 사용한다.
- GRT는 모든 layer의 token이 segment 시작 state를 읽는 구조만 구현한다. read gate도 시작 state만으로 계산하며 기본 off다. candidate 중간 register hidden을 token K/V로 노출하지 않으며 읽기 source 선택 옵션·과거 구현 분기를 두지 않는다. segment의 전체 stack 뒤 write-back 한 번을 유지하고 병렬/prefix 출력·종료 state 동등성과 fact→query gradient를 확인한다. prefix 재계산은 추론/검증용이며 학습 경로에는 넣지 않는다.
- 새 데이터셋은 adapter를 통해 같은 배치 계약에 연결한다. trainer/evaluator에 과제·모델별 분기를 추가하지 않는다. 인터페이스 변경은 데이터 경계·모델 factory에서 명시한다.
- 원본 collator는 전역 Torch RNG를 사용한다. 같은 dataset/seed라도 모델 초기화에 따라 길이·질의 선택 순서가 다를 수 있다. 이를 동일 배치 비교로 표현하거나 정리 작업 중 몰래 바꾸지 않는다.
- 설정은 `docs/workflow.md`의 역할·병합 순서를 따른다. 신규 크기/고정 비교 조건은 카탈로그에 추가하고 숫자 변경은 `--steps`/`--set`을 쓴다. 실험별 flat YAML과 노트북 helper 코드를 복사하지 않고 utils를 확장한다. 목록과 null은 교체한다. 실제 조건은 resolved config와 metadata로 확인한다. 설정이 바뀌면 새 run을 사용하고 기존 결과를 보존한다.
- CUDA 배치 자동 조정은 `training.auto_batch`/`memory_fraction`과 공통 `utils/batching.py`에서 관리한다. off/CPU에서는 config 배치를 유지하고, on에서도 microbatch×accumulation×world_size와 평가 batch를 바꾸지 않는다. 현재 run의 보정 배치를 재개 시 재사용한다.
- W&B 공통 기본은 false이며 Kaggle 실험은 true로 설정한다. 비활성 시 import·연결하지 않는다. 모니터링이 모델 RNG에 영향을 주지 않게 한다. 로컬 TXT/JSON 기록을 항상 유지한다.
- WikiText RTLA는 validation마다 rank0의 고정 probe에서만 실행한다. 정상 trajectory의 segment 시작 state를 기준으로 초기화·문맥 교환을 독립 비교한다. slot 개입·지속 범위는 사후 분석으로 분리하며 미래 state를 주입하지 않는다. candidate 변화와 실제 gated update를 구분하고 read off의 gate 값을 사용량으로 보고하지 않는다. RMT는 GRT RTLA를 생략한다.
- 현재 문서는 역할별 경로에서 최신 내용만 갱신한다. 실험 문서는 완료 여부와 관계없이 `docs/experiments/`에 실험 주제별 하나씩 유지하며 조건·결과·판단을 해당 문서에 갱신한다. 이전 설계·계획·일회성 검증 문서는 `docs/archive/YYYYMMDD_이름.md`로 보존한다. 임시 검증 코드·노트북은 작업 완료 후 제거하며 회귀 검사는 `tests/`에 최소한으로 유지한다. archive에는 Markdown 문서만 보존한다. config·notebook·zip·실행 코드를 보관하거나 archive를 읽는 실행/테스트 경로를 추가하지 않는다. 하위 호환성 코드는 유지하지 않는다.

## 협업과 검증

1. 사용자 요청을 바탕으로 구현과 필요한 간단한 CPU 검증을 완료한다.
2. 변경·검증 결과·미검증 항목과 간결한 Colab/Kaggle 실험 방법을 보고한다.
3. 사용자의 명시적인 승인 후 커밋하고, push는 지시가 있을 때 수행한다.
4. GPU·장시간 실험은 사용자가 Colab/Kaggle에서 실행한다. 에이전트는 작은 CPU 기능·gradient·학습/재개 검사를 할 수 있다.
5. 사용자가 전달한 결과를 commit·설정·하드웨어와 함께 분석한다. 관측 사실과 가설, CPU 기능 검사와 GPU 수렴을 구분한다.

Colab 안내는 사용할 노트북/명령, 주요 설정, 공유할 결과만 간결하게 제시한다.
T4(VRAM 약16GB)를 기준으로 모델·activation·optimizer 메모리와 여유를 계산해 배치를 정하고 실측 peak로 조절한다. microbatch·accumulation·effective batch를 구분한다. 실측 update 시간으로 전체 소요 시간을 계산하며 원본 조건·샘플 예산과 전체 논문 실험 재현을 구분한다.
