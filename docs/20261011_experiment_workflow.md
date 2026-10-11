# 실험 설정과 노트북 운영 규칙

현재 실행 규칙은 이 문서를 따른다. 모델/gradient/데이터 계약은 [통합 명세](20261011_current_spec.md)를 따른다. 구조 정리로 모델 차원, tokenizer, optimizer, batch, 평가, score_kind, checkpoint 형식을 변경하지 않는다.

## 설정의 역할과 확장

현재 설정은 `configs/experiments.yaml` 한 카탈로그에서 관리한다. 각 항목은 기존 typed Config 구조에 적용하는 partial override다.

| 영역 | 책임 |
|---|---|
| defaults | 공통 NeoX 구조, seed와 기본 logging |
| tasks | vocabulary/segment/memory, 데이터 objective, 과제별 optimizer·평가·학습 기본값 |
| sizes | D/layers/heads/FF와 router pool heads; base/small/tiny |
| models | RMT/GRT 이름과 router/register 계약 |
| tokenizers | WikiText word/GPT-2 BPE vocabulary와 prepared 경로 |
| budgets | smoke200 / 20k20000 / full200000, 평가·logging 주기 |
| runtimes | GPU/CPU microbatch·accumulation과 W&B 기본 활성화 |

적용 순서는 **defaults → task → size → model → tokenizer → budget → runtime → 명시 override**다. dict는 재귀 병합, list와 null은 교체한다. 모델 선택은 크기 뒤에 적용해 RMT의 router/register를 null로 정리한다. unknown key와 잘못된 값은 typed Config 검사가 거부한다.

동일 필드의 기본값은 표의 담당 영역에서 관리한다. 예외는 뒤에서 적용하는 명시 override로 표현한다. size에 학습 예산을 넣거나 runtime에 모델 차원을 넣지 않는다. 모델별로 backbone 차원을 달리 정의하지 않는다. `data.options` 내부 키는 과제 adapter가 검증한다.

WikiText named API/CLI에서 생략한 선택은 small/GPT-2 BPE/full/RMT이며 runtime은 자동 선택하지 않는다. Kaggle 노트북에는 기존 tiny/smoke/두 모델 선택을 명시했다. Remember는 budget/steps를 생략하면 기존 pair curriculum을 사용하고, 지정하면 curriculum을 비워 `data.train_segments-1` pairs의 고정 update 실험을 사용한다. `steps`는 budget 다음에 max_steps를 덮어쓰고, overrides가 마지막 우선순위다. 업데이트 수 변경은 cosine schedule 길이도 바꾼다.

새 크기는 sizes에 한 항목을 추가한다. 새로운 고정 비교 조건만 budgets에 이름을 추가한다. 단순히 updates/평가 주기/LR를 바꾸는 실험에는 새 YAML을 만들지 않고 `steps=30000`, `overrides={...}` 또는 CLI `--steps`/`--set`을 사용한다. 새 과제는 adapter와 tasks 항목을 추가하며 노트북에 task 분기를 복사하지 않는다. 모델 구조와 과제 조건은 최신 통합 명세를 기준으로 한다. 과거 실행 구조의 분기나 설정 경로 변환을 추가하지 않는다.

```python
from grt.experiments import load_experiment
cfg = load_experiment(task='wikitext103', model='rmt', size='small',
                      tokenization='gpt2_bpe', budget='20k', runtime='kaggle')
```

```bash
torchrun --standalone --nproc_per_node=2 scripts/train.py --task wikitext103 --model rmt --size small --tokenization gpt2_bpe --budget 20k --runtime kaggle --output-dir runs/small_rmt_20k_preflight --device cuda --benchmark-steps 10
# 새로운 숫자 예산: --steps 30000; 단일 값: --set evaluation.every_steps=1000
```

`configs/`에는 현재 `experiments.yaml`만 둔다. 과거 YAML은 삭제하며 config archive와 이전 경로 fallback을 두지 않는다. `--task`는 현재 카탈로그를 조합하고, `--config`는 실행 시 생성한 최종 Config YAML 하나만 읽는다. 옛 이름의 YAML 목록을 병합하는 실행은 지원하지 않는다.

`docs/archive/`에는 과거 설명·설계·검증 기록의 Markdown 문서만 둔다. notebook·zip·config·실행 코드와 archive 문서를 읽어서 작동하는 테스트는 두지 않는다. 현재 코드가 사용하는 vendor 블록·데이터 함수와 출처·라이선스는 별도 `src/grt/vendor/`에 유지한다.

모델 factory는 현재 공통 NeoX만 구성한다. 과거 모델/wrapper 선택, legacy checkpoint 변환과 읽기 경로는 제거한다. 현재 checkpoint 형식은 `memory_experiment/3` 하나다. 예전 형식은 거부하며 변환 코드는 두지 않는다. 현재 run의 재개는 같은 최종 Config·world_size·데이터로 지원한다.

## 노트북과 utils의 경계

Kaggle 노트북에는 저장소 bootstrap, 선택한 실험 설정, 다음 호출만 남긴다.

```python
from grt.utils.kaggle import WikiTextRun
job = WikiTextRun(repo_dir=repo, size='small', budget='20k', models=('rmt',))
job.setup()
job.prepare_data()
job.preflight()
job.train()
job.report()
```

`models=('rmt','grt')`는 둘을 순차 실행한다. budget에 따라 모델 수를 자동 변경하지 않는다. tiny/smoke 동작 확인도 동일 API를 사용한다. 3만 updates는 `steps=30000`으로 지정하고, read source 같은 조건은 `overrides={'model': {'register_read_source': 'fixed_state'}}`로 지정한다.

setup/다운로드/prepared hash 검사, torchrun 명령, 로그 tee/자식 종료, calibration, ETA, W&B 확인, summary/plot은 `src/grt/utils/kaggle.py`에서 관리한다. 추가 환경 기능은 utils를 확장한다. 노트북 cell 수나 복사한 helper 함수 수를 늘리지 않는다. 저수준 tokenizer 준비/다운로드 CLI는 계속 사용할 수 있다.

run 이름에는 size/tokenization/budget/seed가 들어간다. custom steps/override는 별도 이름으로 구분한다. 최종 full Config YAML은 모델당 하나만 run 폴더에 남긴다. 환경·W&B·calibration 때문에 사용자용 YAML을 여러 개 만들지 않는다. calibration은 benchmark 결과에서 읽어 최종 설정에 적용한다.

prepare_data는 기존 prepared 데이터와 tokenizer를 재사용한다. 다른 세션의 결과/데이터는 `resume_input`, `prepared_input`으로 원래 working 경로에 복사한다. 기존 출력의 Config가 달라지면 학습 전에 거부하며 checkpoint의 world_size/data manifest 검사도 유지한다. 현재 run의 보정된 microbatch/accumulation을 그대로 복원한다. 과거 실행 형식을 자동 변환하지 않는다.

`prepare_data(source=..., tokenizer=..., local_files_only=True)`로 로컬 원본/tokenizer를 사용할 수 있다. 새 HF 원본 다운로드 시 `dataset_revision`을 지정할 수 있다. 이미 준비된 데이터는 manifest와 hash로 식별하며 revision 변경만으로 자동 교체하지 않는다. CPU runtime은 모델을 자동 축소하지 않으므로 작은 검사는 명시적 크기/override를 사용한다.

W&B는 rank0만 생성하며 preflight에서는 생성하지 않는다. 본 학습의 주기별 metrics와 `wandb_status.json`은 유지한다. 온라인 모드에서는 종료 상태·실패 수·마지막 step·run URL을 확인한다. disabled/offline을 선택한 경우 그 모드대로 확인한다.

## 검증 범위

현재 카탈로그 선택·typed Config 검증, config 단일 파일 입력, 현재 checkpoint 형식, 작은 CPU 실행과 재개 조건을 확인한다. 과거 YAML/fixture/archive notebook과 대조하는 호환성 검사는 유지하지 않는다. 전체 학습과 GPU 수렴을 반복 검사하지 않는다.

현재 코드 정리 후 관련 CPU 검사 54개가 9.16초에 통과했다. 카탈로그 조합 36개, 단일 Config 파일/현재 checkpoint 형식, 짧은 LM preflight·2 updates, 현재 Remember/stream 실행·재개와 vendor hash를 확인했다. 전체 회귀·GPU 실험을 반복한 결과는 아니다.
