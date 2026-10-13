# 실험 설정과 노트북 운영 규칙

현재 실행 규칙은 이 문서를 따른다. 모델/gradient/데이터 계약은 [통합 명세](20261011_current_spec.md)를 따른다. 모델 구조와 실험 조건은 카탈로그와 명시 override로 관리하며 자동 조정은 확정된 실질 배치를 바꾸지 않는다.

2026-10-12 노트북 선택 기본은 small/10k/NeoX다. 첫 순서는 [baseline → 레지스터 → MLP와 결과표](20261012_experiment_plan.md)를 따른다. NeoX는 M0/K0, RMT/GRT는 기본 M10/K3이다. `WikiTextRun(num_registers=..., router_hidden=...)`는 각각 RMT/GRT의 메모리와 GRT MLP만 바꾸며 attention은 size의 차원/head를 유지한다. 노트북 VRAM 예약 목표는95%로 명시하며 카탈로그 기본80%는 유지한다. global batch는 보존한다.

`WikiTextRun`은 기본 `test_at_end=False`로 탐색 중 test를 생략한다. CLI는 `--set evaluation.test_at_end=false`를 사용한다. 학습 종료 시 best checkpoint의 validation과 RTLA를 재평가해 `selected_validation`, `selected_rtla`, `selected_step`에 기록한다. `job.report(download=True)`는 이번 실험만 담은 `experiment_report.json` 하나를 내보낸다. 최종 Config·환경 metadata·학습 곡선·RTLA 요약·경고를 포함하며 다른 환경의 결과를 누적하지 않는다. 받은 JSON은 `results/`에 보관하고 실험 계획의 Markdown 결과표를 갱신한다. test가 필요한 최종 run은 명시적으로 켜거나 별도 평가 CLI를 사용한다.

## 설정의 역할과 확장

현재 설정은 `configs/experiments.yaml` 한 카탈로그에서 관리한다. 각 항목은 기존 typed Config 구조에 적용하는 partial override다.

| 영역 | 책임 |
|---|---|
| defaults | 공통 NeoX 구조, seed, 기본 logging·배치 조정 정책 |
| tasks | vocabulary/segment/memory, 데이터 objective, 과제별 optimizer·평가·학습 기본값 |
| sizes | D/layers/heads/FF와 router pool heads; base/small/tiny |
| models | NeoX/RMT/GRT 이름과 router/register 계약, NeoX의 M0/K0 |
| tokenizers | WikiText word/GPT-2 BPE vocabulary와 prepared 경로 |
| budgets | smoke200 / 10k10000 / 100k100000, 평가·logging 주기 |
| runtimes | GPU/CPU microbatch·accumulation·메모리 조정 정책과 W&B 기본 활성화 |

적용 순서는 **defaults → task → size → model → tokenizer → budget → runtime → 명시 override**다. dict는 재귀 병합, list와 null은 교체한다. 모델 선택은 크기 뒤에 적용해 RMT의 router/register를 null로 정리한다. unknown key와 잘못된 값은 typed Config 검사가 거부한다.

동일 필드의 기본값은 표의 담당 영역에서 관리한다. 예외는 뒤에서 적용하는 명시 override로 표현한다. size에 학습 예산을 넣거나 runtime에 모델 차원을 넣지 않는다. 모델별로 backbone 차원을 달리 정의하지 않는다. `data.options` 내부 키는 과제 adapter가 검증한다.

WikiText named API/CLI에서 생략한 선택은 small/GPT-2 BPE/100k/RMT이며 runtime은 자동 선택하지 않는다. Kaggle/Colab 노트북은 small/10k/NeoX를 명시한다. Remember는 budget/steps를 생략하면 기존 pair curriculum을 사용하고, 지정하면 curriculum을 비워 `data.train_segments-1` pairs의 고정 update 실험을 사용한다. `steps`는 budget 다음에 max_steps를 덮어쓰고, overrides가 마지막 우선순위다. 업데이트 수 변경은 cosine schedule 길이도 바꾼다.

새 크기는 sizes에 한 항목을 추가한다. 새로운 고정 비교 조건만 budgets에 이름을 추가한다. 단순히 updates/평가 주기/LR를 바꾸는 실험에는 새 YAML을 만들지 않고 `steps=30000`, `overrides={...}` 또는 CLI `--steps`/`--set`을 사용한다. 새 과제는 adapter와 tasks 항목을 추가하며 노트북에 task 분기를 복사하지 않는다. 모델 구조와 과제 조건은 최신 통합 명세를 기준으로 한다. 과거 실행 구조의 분기나 설정 경로 변환을 추가하지 않는다.

```python
from grt.experiments import load_experiment
cfg = load_experiment(task='wikitext103', model='rmt', size='small',
                      tokenization='gpt2_bpe', budget='10k', runtime='kaggle')
```

```bash
torchrun --standalone --nproc_per_node=2 scripts/train.py --task wikitext103 --model rmt --size small --tokenization gpt2_bpe --budget 10k --runtime kaggle --output-dir runs/small_rmt_10k_preflight --device cuda --benchmark-steps 10
# 새로운 숫자 예산: --steps 30000; 단일 값: --set evaluation.every_steps=1000
```

`configs/`에는 현재 `experiments.yaml`만 둔다. 과거 YAML은 삭제하며 config archive와 이전 경로 fallback을 두지 않는다. `--task`는 현재 카탈로그를 조합하고, `--config`는 실행 시 생성한 최종 Config YAML 하나만 읽는다. 옛 이름의 YAML 목록을 병합하는 실행은 지원하지 않는다.

`docs/archive/`에는 과거 설명·설계·검증 기록의 Markdown 문서만 둔다. notebook·zip·config·실행 코드와 archive 문서를 읽어서 작동하는 테스트는 두지 않는다. 현재 코드가 사용하는 vendor 블록·데이터 함수와 출처·라이선스는 별도 `src/grt/vendor/`에 유지한다.

모델 factory는 현재 공통 NeoX만 구성한다. 과거 모델/wrapper 선택, legacy checkpoint 변환과 읽기 경로는 제거한다. 현재 checkpoint 형식은 `memory_experiment/4` 하나다. 고정 state 표준화 이전 config/checkpoint로 재개하지 않고 새 run을 사용한다. 예전 형식은 거부하며 변환 코드는 두지 않는다. 현재 run의 재개는 같은 최종 Config·world_size·데이터로 지원한다.

## 노트북과 utils의 경계

Kaggle 노트북에는 저장소 bootstrap, 선택한 실험 설정, 다음 호출만 남긴다.

```python
from grt.utils.kaggle import WikiTextRun
models = ('neox',)  # 이후 ('rmt',), ('grt',)
num_registers = 10
router_hidden = 512
job = WikiTextRun(repo_dir=repo, size='small', budget='10k', models=models,
                  num_registers=num_registers, router_hidden=router_hidden,
                  test_at_end=False,
                  overrides={'training': {'auto_batch': True, 'memory_fraction': 0.95},
                             'rtla': {'enabled': True}})
job.setup()
job.prepare_data()
job.preflight()
job.train()
job.report()
```

`models=('rmt','grt')`는 둘을 순차 실행한다. budget에 따라 모델 수를 자동 변경하지 않는다. tiny/smoke 동작 확인도 동일 API를 사용한다. 현재 노트북 기본은 small/10k/NeoX이며 크기와 예산을 명시적으로 바꿀 수 있다. 3만 updates는 `steps=30000`으로 지정한다. GRT의 segment 시작 state 읽기는 표준이며 별도 override가 필요 없다. 중간 register 읽기 구현과 register_read_source 선택 필드는 제거한다.

현재 WikiText 노트북은 segment150을 카탈로그에서 사용하고 `num_registers`·`router_hidden`만 숫자 변수로 노출한다. 두 값은 양의 정수이며 기존 typed Config의 memory/MLP 필드로 연결한다. num_registers는 RMT의 read/write 블록 각각 또는 GRT의 저장 slot 수다. NeoX에는 적용하지 않는다. router_hidden은 GRT read/write MLP에만 적용하고 attention 차원/head는 size를 따른다. Remember 템플릿의 segment_len/num_registers 변수는 별도 CLI override이며 기본4/32다.

길이를 바꿔도 배치·LR·updates·BPTT segment 수는 그대로다. WikiText target 토큰 예산 상한은 global batch×segment_len×updates, 이전 문맥의 BPTT 범위 상한은 bptt_segments×segment_len이다. ETA는 최종 Config의 segment 길이를 사용한다. 기본 run 이름의 hash에는 두 값도 포함되어 다른 조건의 결과를 구분한다. 직접 run_id를 지정했다면 조건 변경 시 새 이름을 사용한다.

RMT/GRT의 `training.bptt_segments` 기본3은 재계산할 이전 segment 수이며 현재 segment까지 최대4개다. 이번 노트북 sweep에서는 고정하고, 다른 문맥 실험에는 training override로 바꾼다. segment75에서 이전450 token의 gradient 범위를 원하면6을 명시한다. 자동 환산은 하지 않는다. 메모리 없는 NeoX는0으로 검증한다.

전체 segment 수는 입력/스트림 길이에 따라 정해진다. WikiText는 문장들을 연속 token stream으로 처리하며 문장 길이를 고정 segment 수로 제한하지 않는다. 마지막 짧은 segment도 학습·평가에 사용한다. BPTT 범위 밖 과거의 정보는 저장 state로 전달하며, epoch/split 경계에서 state를 초기화한다. batch_size로 스트림을 분할할 때 나누어떨어지지 않는 끝의 소수 token은 제외하고 metadata에 기록한다.

setup/다운로드/prepared hash 검사, torchrun 명령, 로그 tee/자식 종료, calibration, ETA, W&B 확인, summary/plot은 `src/grt/utils/kaggle.py`에서 관리한다. 추가 환경 기능은 utils를 확장한다. 노트북 cell 수나 복사한 helper 함수 수를 늘리지 않는다. 저수준 tokenizer 준비/다운로드 CLI는 계속 사용할 수 있다.

Colab은 `notebooks/colab.ipynb`에서 같은 WikiTextRun을 사용한다. `runtime='colab'`, `world_size=1`, `work_dir=Path('/content')`를 명시한다. colab runtime은 micro32×accum8×GPU1로 Kaggle과 동일한 global256을 유지하고 배치 자동 조정을 켠다. LR·예산·평가 batch·모델·RTLA 조건은 공유하며 학습 함수나 wrapper를 복사하지 않는다. GPU 런타임을 선택하고 Colab Secrets에 `WANDB_API_KEY`를 등록해 노트북 접근을 허용한다. 환경변수가 있으면 우선 사용하고, 없으면 Colab의 userdata에서 읽는다. prepared_input/resume_input으로 기존 데이터/결과를 복원할 수 있지만 world_size가 다른 Kaggle checkpoint를 재개하지는 않는다.

Colab W&B 인증은 [W&B 인증 페이지](https://wandb.ai/authorize)에서 API key 준비 → 왼쪽 열쇠 아이콘에서 이름 WANDB_API_KEY의 secret 추가 → 해당 secret의 Notebook access 활성화 → runtime='colab'·wandb=True에서 job.setup() 실행 순서다. SecretNotFoundError는 secret 이름/등록 여부, NotebookAccessError는 접근 스위치를 확인한다. utils 모듈 이름은 kaggle이지만 Colab은 google.colab.userdata, Kaggle은 kaggle_secrets.UserSecretsClient를 사용한다.

run 이름에는 size/tokenization/budget/seed가 들어간다. custom steps/override는 별도 이름으로 구분한다. 최종 full Config YAML은 모델당 하나만 run 폴더에 남긴다. 환경·W&B·calibration 때문에 사용자용 YAML을 여러 개 만들지 않는다. calibration은 benchmark 결과에서 읽어 최종 설정에 적용한다.

prepare_data는 기존 prepared 데이터와 tokenizer를 재사용한다. 다른 세션의 결과/데이터는 `resume_input`, `prepared_input`으로 원래 working 경로에 복사한다. 기존 출력의 Config가 달라지면 학습 전에 거부하며 checkpoint의 world_size/data manifest 검사도 유지한다. 현재 run의 보정된 microbatch/accumulation을 그대로 복원한다. 과거 실행 형식을 자동 변환하지 않는다.

`prepare_data(source=..., tokenizer=..., local_files_only=True)`로 로컬 원본/tokenizer를 사용할 수 있다. 새 HF 원본 다운로드 시 `dataset_revision`을 지정할 수 있다. 이미 준비된 데이터는 manifest와 hash로 식별하며 revision 변경만으로 자동 교체하지 않는다. CPU runtime은 모델을 자동 축소하지 않으므로 작은 검사는 명시적 크기/override를 사용한다.

W&B는 rank0만 생성하며 preflight에서는 생성하지 않는다. 본 학습의 주기별 metrics와 `wandb_status.json`은 유지한다. 온라인 모드에서는 종료 상태·실패 수·마지막 step·run URL을 확인한다. disabled/offline을 선택한 경우 그 모드대로 확인한다.

## GPU 배치 자동 조정

DDP는 유지한다. 설정은 `training.auto_batch`(공통 기본 false, Kaggle runtime true), `training.memory_fraction`(기본 0.8) 두 값으로 관리한다. off일 때 config의 batch_size/grad_accum_steps를 그대로 쓰며 OOM에서도 임의로 축소하지 않는다. CPU에서도 배치를 바꾸지 않는다. Kaggle에서는 overrides={'training': {'auto_batch': False}} 또는 CLI --set training.auto_batch=false로 끈다.

on이면 CUDA 사전 측정에서 local effective batch의 약수만 큰 순서로 시도하고, 모든 rank에서 목표 peak reserved VRAM 이내인 가장 큰 microbatch를 선택한다. local effective=batch_size×grad_accum_steps, global effective=local×world_size를 고정한다. global_batch_size가 명시되면 이 값과 일치해야 하며 불일치는 조정으로 숨기지 않고 거부한다. accumulation=local/microbatch이므로 계산 microbatch는 rank당 effective batch를 넘지 않는다. 목표는 PyTorch가 예약한 VRAM의 상한이며 GPU 연산 가동률이나 정확한 점유율을 보장하지 않는다. 평가 batch·sequence length·BPTT·LR·updates는 변경하지 않는다.

실제 forward/backward/optimizer와 평가로 측정하며 DDP bucket 메모리도 확인한다. 실패 후보는 모든 rank에서 함께 낮춘다. microbatch1도 목표/평가 메모리를 만족하지 못하면 오류로 보고한다. 실질 배치 상한이나 약수 간격 때문에 목표 점유율보다 낮을 수 있다.

```python
job = WikiTextRun(repo_dir=repo, size='small', budget='10k', models=('rmt',),
                  overrides={'training': {'auto_batch': True, 'memory_fraction': 0.8}})
```

CLI는 `--set training.auto_batch=true --set training.memory_fraction=0.8`을 사용한다. 신규 CUDA 학습은 필요한 사전 측정을 수행하고 같은 run의 측정 결과가 있으면 재사용한다. Kaggle의 기존 preflight 호출과 학습 사이에 동일 측정을 반복하지 않는다. 최종 batch/accumulation은 Config와 benchmark에 기록하며 현재 run 재개에서는 확정된 값을 유지한다. 다른 GPU에 맞춰 다시 조정하려면 새 run을 사용한다.

현재 Kaggle은 GPU2개, micro32×accum4의 global256으로 시작한다. rank당 실질 배치128의 약수를 찾으므로 micro128×accum1, micro64×accum2, micro32×accum4 등 후보가 모두 2의 거듭제곱이다. Remember curriculum도 선택한 runtime과 각 stage의 실질 배치를 유지한다.

## T4 × 2의 WikiText 학습 조건

base/small/tiny와 RMT/GRT는 global256, Adam LR5e-4를 공유한다. 배치60의 LR2.5e-4에 제곱근 scaling을 적용한 약5.16e-4를 반올림한 초기 설정이며 최적값을 실측한 것은 아니다. cosine/warmup0/clip0.25/dropout0.1/BPTT3/segment150/평가 batch10은 유지한다. auto_batch는 이 학습 조건을 바꾸지 않는다.

10k/100k의 validation은1000 updates, W&B logging은10 updates마다 수행한다. smoke는200 updates와 validation20/logging10을 유지한다. 학습 budgets는10k=10000/100k=100000이며 target 예산 상한은 smoke7.68M/10k384M/100k3.84B이다. PPL 비교에 배치/LR/token 수를 함께 기록한다. 과거 학습 budget 이름과 호환 경로는 유지하지 않는다.

기존 결과에 이어서 학습하지 않고 새 run_id를 사용한다. T4 GPU의 실제 메모리 적합성과 update 시간은 기존 preflight로 측정하고 ETA를 갱신한다. 기존 micro10 측정 시간을 새 microbatch/update 시간으로 단순 환산하지 않는다.

## RTLA: 학습 중 모니터링과 사후 분석

목표는 register의 변화뿐 아니라 과거 state가 다음 token 예측을 개선하는지 확인하는 것이다. 같은 checkpoint·token·segment 경계에서 정상 state와 개입 state의 NLL을 비교한다. 양의 delta NLL은 개입으로 예측이 나빠졌다는 뜻이며 메모리 사용의 근거다. 단독으로 메모리 없는 모델 대비 성능 우위를 증명하지는 않는다.

| 구분 | 학습 중 W&B 및 로컬 기록 | 사후 checkpoint 분석 |
|---|---|---|
| 조건 | 정상 / 학습된 S0로 초기화 / 다른 스트림의 state로 교환 | 같은 세 조건 + 선택적 slot별 초기화 |
| 지표 | probe NLL/PPL, 초기화·교환 delta NLL, 위치별 효과의 앞/뒤 요약 | token 위치별 delta NLL, segment별 결과, slot별 기여 |
| state 동작 | write gate, state norm, 실제 update norm/relative update, slot cosine | 같은 trace와 그래프 |
| 지속 범위 | 실행하지 않음 | 한 경계에서만 state 교환 후 정상 write-back을 이어가며 horizon별 delta NLL |

`rtla.enabled` 공통 기본은 false다. Kaggle 선택 셀은 `rtla_enabled`로 켜고 끄며 RMT는 분석을 생략한다. WikiText는 validation마다 rank0에서만 고정 probe를 실행한다. `rtla.batch_size=8`, `rtla.segments=8`, `rtla.warmup_segments=3`이 기본이며 모두 명시 override로 변경한다. batch는 문맥 교환을 위해2 이상이고, warmup은1 이상이어야 한다. 원래 validation NLL/PPL은 전체 stream 평가이고 RTLA 지표는 고정된 별도 probe이므로 구분한다. probe가 부족하면 segment 수를 가용 범위로 줄이되 warmup 뒤 측정 segment가 없으면 오류로 보고한다.

정상 state는 warmup을 포함한 실제 과거를 순서대로 처리해 만든다. 각 측정 segment의 초기화·교환·slot 개입은 정상 trajectory의 동일 시작 state에서 별도로 계산하고 다음 정상 segment에 반영하지 않는다. 교환은 같은 시점에서 batch를1칸 순환시켜 자기 자신의 state를 받지 않게 한다. 미래 state를 주입하지 않는다. 지속 범위 분석만 개입 state를 이후 segment로 이어가며 현재 token 입력은 정상과 동일하게 유지한다.

gate·norm·cosine은 동작 진단이며 인과적 기여의 증거는 개입 NLL 차이다. relative update는 실제 갱신 norm/(이전 state norm+candidate norm)이며 초기 state가0이어도 유한하도록 분모에 epsilon을 둔다. read gate off에서는 read gate 평균을 W&B에 보내지 않는다. write pooling attention은 register 읽기 attention이 아니다. 전체 head attention 저장·slot 조합 탐색·자연어 entity 분류는 이번 최소 구현에 포함하지 않는다.

```python
# 기존 notebook overrides의 rtla 항목
{'rtla': {'enabled': True, 'batch_size': 8, 'segments': 8, 'warmup_segments': 3}}
```

학습 중 결과는 `run/rtla/step_XXXXXX/summary.json`과 `arrays.npz`, 개입 scalar는 `grt/rtla/*`, state 동작 scalar는 기존 `grt/*` W&B metric과 metrics JSONL에 남긴다. trace는 batch 평균이며 위치별 값은 유효 token 수로 가중한다. Remember의 기존 trace 실행도 유지한다.
사전 측정 ETA에는 RTLA probe의 추가 시간이 포함되지 않으며 실제 학습 wall time에는 포함된다.

```bash
python scripts/analyze.py --checkpoint runs/wiki/grt/best.pt --output-dir runs/wiki/grt/memory_analysis --device cuda --split validation --segments 64 --slots --horizon 4
```

사후 분석의 batch/warmup은 checkpoint의 RTLA 설정을 기본으로 쓰고 `--batch-size`/`--warmup-segments`로 바꾼다. 기본 horizon은3이며0은 지속 범위 분석을 끈다. slot별 초기화는 `--slots`일 때만 실행한다. summary JSON/TXT, 원본 배열 NPZ, 위치·slot·지속 범위 그래프와 trace PNG를 저장한다. checkpoint/data 식별 정보와 실제 probe 크기도 함께 기록한다. 비용은 measured segments×(정상+초기화+교환+선택한 slots+후속 horizon) forward이며 학습을 수행하지 않는다.

## 검증 범위

현재 카탈로그 선택·typed Config 검증, config 단일 파일 입력, 현재 checkpoint 형식, 작은 CPU 실행과 재개 조건을 확인한다. 과거 YAML/fixture/archive notebook과 대조하는 호환성 검사는 유지하지 않는다. 전체 학습과 GPU 수렴을 반복 검사하지 않는다.

현재 코드 정리 후 관련 CPU 검사 54개가 9.16초에 통과했다. 카탈로그 조합 36개, 단일 Config 파일/현재 checkpoint 형식, 짧은 LM preflight·2 updates, 현재 Remember/stream 실행·재개와 vendor hash를 확인했다. 전체 회귀·GPU 실험을 반복한 결과는 아니다.

배치 자동 조정 추가에서는 관련 CPU 검사만 실행했다. 정책·예산 보존·보정 재사용과 Remember 실제 사전 측정 검사 15개가 1.98초에 통과했고, WikiText의 짧은 사전 측정·2 updates와 보정된 run 설정 재사용도 통과했다. CUDA OOM/rank별 제한은 모의 메모리·collective로 확인했으며 실제 GPU VRAM 조정과 성능은 아직 실측하지 않았다.

T4 global256 설정 변경에서는 현재 카탈로그의36개 조합과 보정 재사용·실질 배치 보존 검사 총40개가0.71초에 통과했다. 노트북 JSON과 코드 문법도 확인했다. CPU 학습과 GPU 실험은 반복 실행하지 않았다.

고정 state 표준화에서는 관련 CPU 검사13개만 실행했다(합계3.13초). read gate on/off의 미래 변경 불변성과 layer별 prefix/병렬 출력·segment 종료 write-back을 확인했고 Remember gradient/trace, replay gradient 및 현재 형식의 짧은 Remember·WikiText 재개를 확인했다. 이전 중간 register 읽기 구현을 실행하는 검사는 제거했다. GPU 실험은 아직 수행하지 않았다.

20261012 RTLA 추가에서는 관련 CPU 검사5개가6.01초에 통과했다. 메모리 의존/무시 예제의 개입 NLL, 짧은 tail의 token 가중·이후 회복, 실제 gate 적용 갱신량, 분석 on/off의 실제 GRT2-update 최종 가중치/RNG 동일성을 확인했다. W&B 전달은 모의 클라이언트로 확인했고 동일 checkpoint의 사후 CLI·JSON/NPZ/PNG 저장도 실행했다. Remember trace와 노트북 설정 전달을 확인했다. 실제 온라인 W&B 업로드와 GPU/DDP 분석 비용·실제 WikiText 사용 효과는 아직 실측하지 않았다.

Colab 추가에서는 runtime의 global256·동일 학습 조건과 Secrets 인증 분기 검사1개가0.40초에 통과했다. Kaggle/Colab 선택 cell을 직접 호출해 GPU2/1의 실질 배치와 모델·RTLA 설정을 확인했으며 전체 학습 검사는 반복하지 않았다. 실제 Colab GPU·Secrets 연결은 미실측이다.
