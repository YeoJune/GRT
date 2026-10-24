# GRT Server 연결

공통 `WikiTextRun`의 선택적 서버 연결을 설명한다. HTTP 계약은 [GRT_server API 가이드](https://github.com/YeoJune/GRT_server/blob/main/docs/api.md), 환경별 큐 실행·결과 회수·실험 관리는 [환경별 가이드](https://github.com/YeoJune/GRT_server/blob/main/docs/environments.md), 서버 운영은 [운영 가이드](https://github.com/YeoJune/GRT_server/blob/main/docs/operations.md)를 따른다.

**읽는 순서:** [자동 저장](#colabkaggle에서-자동-저장) · [큐 등록·실행](#큐-등록과-실행) · [데이터 캐시](#prepared-데이터-공유-캐시) · [중단 후 재개](#세션-중단-후-이어서-학습). Colab/Kaggle의 복사 가능한 worker 셀과 데스크톱 명령은 [환경별 가이드](https://github.com/YeoJune/GRT_server/blob/main/docs/environments.md)에 모은다.

서버는 job의 조건·attempt 이력·파일을 보관하고, 사용자가 연결한 worker가 작업을 요청해 학습한다. 한 worker는 순차 실행하며 여러 worker는 다른 작업을 병렬로 실행한다. 서버는 GPU 세션을 시작하거나 코드·prepared 데이터를 자동으로 준비하지 않는다.

## 실행 모드

| 모드 | 동작 |
|---|---|
| `off` | 기본 학습은 서버 모듈·HTTP client·secret reader 없이 실행. 명시적으로 호출한 캐시 helper는 별도 접속 |
| `upload` | 기존 노트북 조건으로 모델별 attached job을 등록하고 상태·파일을 자동 저장. 큐 작업을 가져오지 않음 |
| `queue` | `WikiTextRun.claim()` 또는 `scripts/server_worker.py`가 서버의 불변 Config를 실행 |

학습·제출·캐시 helper는 write 토큰을 검사한다. 읽기 전용 파일 수집은 서버 웹·관리 CLI·HTTP API를 사용한다.

현재 서버의 `generic/1`과 불변 schema `grt_wikitext_v1`을 사용한다. Config·파일·완료 step을 운반하고 검증하는 연결이며, 서버의 `grt/1` 과학적 완료 검증과는 구분한다. 연결 대상은 WikiText 공통 helper다. Remember CLI는 기존 로컬 실행을 유지한다.

## Colab/Kaggle에서 자동 저장

플랫폼 Secrets에 `GRT_SERVER_TOKEN`을 등록하고 노트북 접근을 허용한다. write 토큰은 서버 전체 변경 권한을 가지므로 개인 실행 환경에서만 사용한다. 선택 셀의 세 값을 설정한다. deadline은 실제 세션 종료 시각의 UTC ISO8601(`YYYY-MM-DDTHH:MM:SSZ`)이다. KST 18시는 UTC 09시이며 재접속 때 임의로 연장하지 않는다.

```python
server_mode = 'upload'
server_url = 'https://grt.grtserver.cloud'
server_deadline = '<실제 세션 종료 시각: UTC ISO8601>'
```

노트북은 `secret_source='colab'` 또는 `'kaggle'`을 명시한다. 일반 Python 환경은 `'env'`를 사용한다. 선택한 reader가 실패하면 다른 위치의 오래된 토큰으로 fallback하지 않는다. 주소는 `GRT_SERVER_URL` 환경변수로도 지정할 수 있다. 원격 주소는 HTTPS, 로컬 검사만 loopback HTTP를 허용한다.

기존 `setup → prepare_data → preflight → train → report` 순서는 유지한다. `setup()`에서 권한·버전을 확인하며, setup을 생략해도 preflight 전에 검사한다. 필요한 schema와 큐는 등록한다. 연결 옵션은 연구 Config나 결과 JSON에 넣지 않는다.

Online W&B run URL은 heartbeat와 완료 시 서버로 전달하며 실험 목록·상세에서 클릭해 연다. 기존 실행의 링크는 서버 실험 수정에서 입력할 수 있다.

학습 전에 Config/provenance를 전송한다. 서버 연결 시 5분 간격으로 update 경계에서 확정 checkpoint를 저장하고 helper가 snapshot하여 전송한다. `report()`는 원본 `experiment_report.json`, 실제 Config·metadata·checkpoint·로그와 선택한 posthoc/ZIP을 자동 저장한다. 원본 bytes는 변경하지 않고 서버 검증용 envelope는 별도로 만든다. `download=True`의 기존 로컬 다운로드도 유지한다.

학습 프로세스 종료만으로 성공을 확정하지 않는다. 최종 `report()`가 업로드·finish를 요청하고 서버가 실제 크기·SHA256과 완료 증거를 검증한 뒤에만 성공을 반환한다.

## 큐 등록과 실행

제출용 노트북은 `server_mode='off'`로 두고 코드와 prepared 데이터를 먼저 고정한다. `queue`는 일반 생성자의 설정값으로 넣지 않고 worker/claim으로 시작한다. `job.prepare_data()`로 데이터를 검증한 뒤 다음과 같이 제출 JSON을 만든다. `posthoc_options=None`이면 학습 결과만 요구하며, 분석 옵션을 지정하면 posthoc도 완료에 필수다.

```python
import json
from pathlib import Path
from grt.utils.server import ServerClient

# Colab: 'colab', Kaggle: 'kaggle', 일반 Python: 'env'
client = ServerClient('https://grt.grtserver.cloud', secret_source='env')
client.check()
client.ensure_schema()
client.ensure_queue(job.server_queue)
body = job.server_job('grt', posthoc_options={
    'batch_size': 8, 'segments': 8, 'warmup_segments': 3, 'horizon': 4,
    'slots': True, 'registers': True, 'window_starts': [0, 16, 32],
    'group_repeats': 3,
})
Path('experiment.json').write_text(json.dumps(body, ensure_ascii=False, indent=2))
```

이 셀은 JSON만 만들며 실제로 제출하지 않는다. JSON은 서버 GUI 또는 GRT_server checkout의 CLI(`jobs validate`, `jobs submit`)로 검증·제출한다. 토큰 설정·Python 직접 제출과 응답 유실 시 같은 키를 재사용하는 절차는 [큐 제출 가이드](https://github.com/YeoJune/GRT_server/blob/main/docs/environments.md#큐-실험-제출)를 따른다. 코드 버전은 full commit SHA와 소스 파일 hash, 데이터는 manifest hash로 고정한다. 미커밋 소스도 hash에 포함하므로 worker에는 같은 파일이 있어야 한다. 제출 후 조건을 바꾸려면 새 job을 만든다.

아래 명령은 **GRT checkout**에서 실행한다. 패키지를 설치하고 서버 주소·write 토큰을 환경으로 주입한다. 환경 주입의 전체 예시는 [데스크톱 worker](https://github.com/YeoJune/GRT_server/blob/main/docs/environments.md#데스크톱-큐-worker)를 따른다. 토큰을 명령행 인자에 넣지 않는다.

```bash
cd /path/to/GRT
python -m pip install -e '.[wandb,wikitext]'
python scripts/server_worker.py \
  --work-dir /path/to/work --prepared-dir /path/to/prepared \
  --deadline '<실제 UTC ISO8601 종료 시각>' --world-size 1
```

기본은 CUDA 1개·환경변수 secret이다. CPU 검사는 `--runtime cpu --device cpu`를 추가한다. DDP는 `--world-size`를 지정하며 helper가 기존 torchrun 경로를 사용한다. 서버는 runner·schema·GPU 수·label·남은 시간을 보고 배정하며, 배정 후 GRT worker가 code_ref·소스 hash·world size를 확인한다. 데이터 준비에서 manifest를 확인한다. 불일치는 실행 전에 거부하므로 실패한 attempt가 남을 수 있다. 직접 `WikiTextRun.claim()`을 사용하는 경우에도 setup·prepare_data·preflight·train·report·close 수명주기는 호출자가 관리해야 한다. 공통 worker 스크립트는 이 순서를 처리한다.

worker는 서버가 지정한 임의 코드·URL·shell을 내려받아 실행하지 않는다. 로컬 코드·데이터·world size를 확인하고, 불변 Config에서 환경별 경로와 같은 effective batch의 auto-batch 보정만 해소한다. 실제 Config는 별도로 보존한다. 작업은 순차 실행하고, 큐가 비었거나 pause/drain·호환 조건 때문에 배정할 수 없으면 종료한다. `--once`는 한 작업만 처리한다. Vast 임대와 과금 종료는 이 worker의 기능이 아니다.

## Prepared 데이터 공유 캐시

로컬 데이터 재사용이 기본이다. 다른 세션에서도 같은 데이터를 쓰려면 준비·검증 후 `job.cache_prepared()`를 명시적으로 실행한다. 토큰은 기존 server_url/secret_source를 사용하며 write 권한이 필요하다. 서버는 원본 manifest SHA256으로 조회하고, 같은 manifest·파일 hash의 캐시를 재사용한다. 캐시는 공유 dataset이므로 실험 삭제 시 유지한다.

```python
job.prepare_data()
cached = job.cache_prepared()
print(cached['extensions']['manifest_sha256'])
# 다른 세션: hash를 지정했을 때만 서버 다운로드
job.prepare_data(server_cache='<manifest SHA256>')
```

다운로드는 ZIP·manifest·각 파일 SHA256을 검사한 후 확정하며 다른 조건이나 손상된 파일을 조용히 덮어쓰지 않는다. 데이터·tokenizer·전처리 결과가 달라 manifest가 다르면 별개 캐시다. 모델·학습 예산만 달라지고 prepared 데이터가 같으면 재사용한다. 서버 캐시는 명시적으로 업로드한 것만 제공하며 자동 만료하지 않는다. ZIP 크기가 서버의 MAX_ARTIFACT_BYTES(기본 1GiB)를 넘으면 운영 한도·디스크를 조정해야 한다.

큐 worker는 `--prepared-dir` 대신 `--prepared-from-server`를 선택할 수 있다. 이 경우 job의 dataset_sha256과 일치하는 서버 캐시를 받아 사용하며, 없으면 실패한다. 제출자와 worker 모두 선택한 GRT commit을 사용한다.

## 세션 중단 후 이어서 학습

서버 연결 학습은 주기적 checkpoint에서 `grt.resume/1` snapshot ZIP을 전송한다. Last/best checkpoint, 실제 Config, microbatch 보정·preflight 정보, 기존 metrics와 W&B run ID를 포함한다. 모델·optimizer·scheduler·rank별 RNG·token stream 위치·메모리 state는 기존 checkpoint 계약으로 복원하며 `.server`의 lease·토큰은 복제하지 않는다. Heartbeat에 검증된 snapshot ID도 전송한다.

Kaggle/Colab 세션 종료 후 웹의 **이어서 학습** 또는 `npm run cli -- jobs resume JOB_ID`를 선택한다. Stale running 작업은 lease 만료 후 `--evidence '이전 런타임 종료를 확인한 근거'`가 필요하다. 서버는 최신 검증 snapshot을 선택해 같은 job의 새 attempt를 대기열에 넣는다. 처음부터 실행하려면 기존 `jobs retry`를 사용한다. `--checkpoint ARTIFACT_ID`로 같은 실험의 다른 보존 snapshot도 선택할 수 있다.

새 환경에서 **같은 GRT commit·소스·prepared manifest·world size**를 준비하고 큐 worker를 실행한다. 데이터는 로컬 입력 또는 `--prepared-from-server`를 선택한다. `preflight()`가 할당된 snapshot을 검증·다운로드하고 기존 best·batch 보정·학습 상태를 복원한다. GPU 메모리에 맞춰 연구 조건이나 microbatch를 다시 바꾸지 않으며 기존 배치가 들어가는 환경이 필요하다. 재개 시 기존 preflight 보정·측정치를 재사용하므로 새 하드웨어의 속도·peak memory를 다시 측정한 값으로 해석하지 않는다. 새 attempt는 별도 로컬 디렉터리에서 실행한다. 같은 attempt의 동일 로컬 journal을 이용한 재접속은 기존 복원본을 재사용한다.

서버 URL·write 토큰을 환경에 설정한 터미널에서 **GRT_server checkout**의 관리 CLI를 실행한다.

```bash
cd /path/to/GRT_server
npm run cli -- jobs resume job_123
```

그다음 **GRT checkout**의 새 worker를 실행한다. 아래 예는 원래 world size가 2이고 prepared 캐시가 이미 서버에 있는 경우다. 캐시를 올리지 않았다면 `--prepared-from-server` 대신 `--prepared-dir /path/to/prepared`를 쓴다. 노트북은 플랫폼 Secret을 자식 프로세스 환경에 전달하는 [worker 셀](https://github.com/YeoJune/GRT_server/blob/main/docs/environments.md#colab--kaggle-큐-worker)을 사용한다.

```bash
cd /path/to/GRT
python scripts/server_worker.py --work-dir /path/to/new-work \
  --prepared-from-server --deadline '<실제 UTC ISO8601 종료 시각>' \
  --runtime kaggle --device cuda --world-size 2
```

재개용 snapshot은 실험별 최근 2개와 배정·실행 중인 재개 원본을 보존하고 나머지 snapshot bytes를 정리한다. 정리된 snapshot의 hash·이력은 남으며 최종 원본 report·개별 best/last checkpoint와 공유 prepared 캐시는 이 정리 대상이 아니다. 보관된 snapshot만 재개에 선택할 수 있다.

이 기능 도입 후 생성한 snapshot이 필요하며 기존 raw last.pt 하나만으로는 자동 재개하지 않는다. 서버에 업로드·검증된 지점 이후의 진행은 강제 종료 시 유실될 수 있다. 종료 자체를 서버가 자동 확인하거나 새 Kaggle 세션을 생성하지 않는다. 라이브러리·GPU 변경 시 수치 동등성까지 보장하지 않으며 CPU 검사에서는 중단 없는 실행과 재개의 최종 가중치 일치를 확인한다. 중단 시 추가 validation은 기존 학습 동작으로 유지한다.

<a id="gate-transfer-kaggle"></a>
## M8/M75 전달성: 새 Kaggle Save에서 재개

[전용 노트북](../notebooks/kaggle_gate_transfer.ipynb)은 서버 캐시·snapshot 사용을 필수로 한다. 기존 범용 노트북의 서버 기본 `off`는 유지한다. Kaggle Internet·T4×2와 Secrets의 write 권한 `GRT_SERVER_TOKEN`·`WANDB_API_KEY` 접근을 켠다.

최초 Save에서 `execution_mode='new'`, `num_registers=8` 또는 `75`, `write_source='candidate'` 또는 `'independent'`를 고른 뒤 **Save Version → Save & Run All** 한다. 노트북은 데이터를 준비해 서버 캐시에 검증·보관하고, full commit·소스 hash·manifest·사후 분석 조건을 고정한 job 하나를 전용 큐에 등록한 뒤 worker를 실행한다. 출력된 job ID·commit·manifest hash를 보관한다. 셀 재실행 시 같은 제출 본문·멱등 키를 사용한다.

| 새 Save의 목적 | 선택값 | 서버에서 먼저 할 일 |
|---|---|---|
| 새로운 조건의 첫 실행 | `execution_mode='new'`, M·source 선택 | 없음 |
| 중단된 같은 실험의 재개 | `execution_mode='resume'`, `resume_job_id='job_...'` | 이전 Save 종료 확인 → 해당 job의 **이어서 학습** |

중단 후에는 **처음부터 재시도**나 새 job 제출을 사용하지 않는다. Stale running이면 이전 런타임 종료와 lease 만료를 확인하고 서버 웹에 종료 근거를 입력한다. 재개 노트북은 queued 상태와 `resume_artifact_id`를 검사하고, 해당 job의 `spec.code_ref`를 checkout한 뒤 서버 캐시와 snapshot을 복원한다. 로컬 작업 폴더가 없는 새 Save여도 되며 기존 파일을 수동 업로드할 필요는 없다. 위 M·source를 바꿔도 재개 job의 불변 조건은 변경되지 않는다. 같은 T4×2·world size2·기존 microbatch/accumulation을 유지한다.

첫 코드 셀 실행 시각을 로컬 파일에 고정하고 **11시간 뒤를 worker 종료 시각**으로 등록해12시간 강제 종료 전 여유를 둔다. 셀 재실행으로 이 시각을 연장하지 않는다. 학습 중5분 간격 snapshot을 전송하지만 강제 종료·통신 오류에서는 마지막 **verified snapshot** 이후 진행분이 유실될 수 있다. 종료 전에 서버 웹의 snapshot step을 확인한다. 서버는 새 Kaggle Save 자체를 자동 생성하지 않는다.

한 Save는 `--once` worker로 한 조건만 처리한다. M8/candidate, M8/independent, M75/candidate, M75/independent의 네 조건을 각각40k로 실행하며, 각 조건이 중단될 때마다 같은 job을 재개한다. 기존20k 안정화 실험은40k·warmup5% 기준을 대체하지 않는다. 같은40k 기준이 이미 있다면 해당 independent 실행은 생략한다. 완료 기준은40k 학습과 window0/16/32 사후 분석의 서버 검증을 마친 `succeeded`다. 수신 결과는 [기존 경로 비교 보고서](experiments/20261019_l75_gate_paths.md)에 추가한다.

## 장애와 재시도

- heartbeat는 학습과 별도로 수행한다. 취소·인증 만료/폐기·세션 deadline·약 10분의 통신 단절이 확인되면 stop 파일을 만들고 학습은 update 경계에서 checkpoint를 저장하고 중단한다. 진행 중인 update·평가가 끝나는 시간은 필요하며 강제 런타임 삭제에는 저장을 보장하지 않는다.
- 요청 ID·업로드 ID·lease는 run의 `.server/`, 큐 worker 식별 정보는 work 디렉터리의 `.grt-server/`에 0600 파일로 보존한다. API 토큰과 signed URL은 journal이나 결과에 저장하지 않는다. lease는 실행 권한이므로 이 디렉터리를 공개·복제하지 않는다.
- 전송 실패 후에는 같은 run에서 `report()`를 다시 실행한다. 기존 업로드를 확인하며 학습을 반복하지 않는다. 이미 succeeded인 job은 기존 결과를 재사용한다.
- W&B나 posthoc가 실패해도 로컬 학습 결과를 먼저 내보내고 가능한 파일을 전송한다. posthoc 실패는 단계 오류로 표시하며 같은 lease가 유효하면 `report(posthoc=True, ...)`로 분석만 재시도한다. 프로세스를 떠날 때는 `job.close()`로 heartbeat를 종료한다.
- 토큰 교체 후 같은 run으로 재연결하면 journal의 소유권을 확인한다. 같은 Config·manifest·world size의 로컬 checkpoint만 기존 복원 계약으로 재개한다. 원래 lease를 잃었거나 새 세대가 생겼으면 실행 권한을 복제하지 않고 종료 확인 후 서버 resume 또는 fresh retry를 선택한다. 다른 환경에서 이어서 실행하려면 아래의 명시적 서버 snapshot 재개를 사용한다.

## 결과 조회와 실험 관리

[서버 웹](https://grt.grtserver.cloud)에서 상태·진행 step·attempt·검증된 파일을 확인하고 W&B 링크를 연다. `finalizing`은 완료 증거를 검사하는 단계이며 `succeeded` 이후 성공으로 취급한다. 기본 heartbeat는 30초, 웹 조회는 약 7초다. Preflight ETA와 loss/PPL 시계열은 현재 서버에 자동 전송하지 않으므로 로컬 ETA·로그·W&B에서 확인한다.

개별 다운로드·결과 ZIP·Campaign별 누적 수집은 [결과 받기](https://github.com/YeoJune/GRT_server/blob/main/docs/environments.md#결과-받기)를 따른다. CLI `results collect`는 succeeded 작업만 수집하며 부분·실패 결과는 웹·파일 API를 사용한다. ZIP 실패 시 학습을 반복하지 않고 ZIP 생성만 다시 요청한다. 받은 원본 JSON의 저장소 반영은 [workflow](workflow.md#6-결과를-저장소에-반영)를 따른다. 서버가 실험 Markdown을 자동 작성·커밋하지 않는다.

이름·Campaign·W&B 링크는 수정할 수 있고 큐·우선순위는 대기 중에만 바꾼다. 학습 조건 변경은 새 실험으로 등록한다. 영구 삭제는 실험·attempt·결과 파일과 관련 ZIP을 제거하므로 실행 종료 후 명시적으로 수행한다. 공유 prepared 캐시·외부 다운로드·백업은 유지한다. 상세 절차는 [실험 관리](https://github.com/YeoJune/GRT_server/blob/main/docs/environments.md#실험-관리영구-삭제)를 따른다.

## 검증

```bash
python -m pip install -e '.[dev]'
GRT_TEST_SERVER_REPO=/path/to/GRT_server python -m pytest tests/test_server.py -q
```

GRT_server checkout에는 `npm ci`가 필요하다. 실제 Express 서버를 임시 SQLite·계정으로 실행해 작은 합성 token 데이터의 CPU 학습·posthoc·원본 다운로드·중복 방지·전송 응답 유실·큐·취소 저장을 검사한다. 서버 설치본이 없으면 외부 통합 검사는 skip된다. 일반 GRT 검사는 서버 없이 실행한다.

자동 검사는 CPU 기능 검증이다. GPU 학습 품질·수렴, 새 플랫폼·저장소·운영 구성은 각 환경에서 확인한다. 배포 검증 기록은 [서버 아카이브](https://github.com/YeoJune/GRT_server/blob/main/docs/archive/README.md)에 보존한다.
