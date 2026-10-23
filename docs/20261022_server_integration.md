# GRT Server 연결

배포 후 바로 확인하려면 [임시 테스트 노트북](../notebooks/tmp_server_acceptance.ipynb)과 [실행 가이드](https://github.com/YeoJune/GRT_server/blob/main/docs/20261023_aws_test_execution_guide.md)를 사용한다. 기본은 합성 token의 작은 실제 학습이며 WikiText 표본은 선택할 수 있다.

갱신: 2026-10-22. [API 설계](20261021_experiment_control_api_design.md)의 선택 연결을 공통 `WikiTextRun`에 구현한다. HTTP 계약은 [GRT_server API 가이드](https://github.com/YeoJune/GRT_server/blob/main/docs/api.md)를 따른다.

## 실행 모드

| 모드 | 동작 |
|---|---|
| `off` | 기본값. 서버 모듈·HTTP client·secret reader를 불러오거나 서버에 접속하지 않음 |
| `upload` | 기존 노트북 조건으로 모델별 attached job을 등록하고 상태·파일을 자동 저장. 큐 작업을 가져오지 않음 |
| `queue` | `WikiTextRun.claim()` 또는 `scripts/server_worker.py`가 서버의 불변 Config를 실행 |

현재 서버의 `generic/1`과 불변 schema `grt_wikitext_v1`을 사용한다. Config·파일·완료 step을 운반하고 검증하는 연결이며, 서버의 `grt/1` 과학적 완료 검증과는 구분한다. 연결 대상은 WikiText 공통 helper다. Remember CLI는 기존 로컬 실행을 유지한다.

## Colab/Kaggle에서 자동 저장

플랫폼 Secrets에 `GRT_SERVER_TOKEN`을 등록하고 노트북 접근을 허용한다. write 토큰은 서버 전체 변경 권한을 가지므로 개인 실행 환경에서만 사용한다. 선택 셀의 세 값을 설정한다.

```python
server_mode = 'upload'
server_url = 'https://research.example.com'
server_deadline = '<실제 세션 종료 시각: UTC ISO8601>'
```

노트북은 `secret_source='colab'` 또는 `'kaggle'`을 명시한다. 일반 Python 환경은 `'env'`를 사용한다. 선택한 reader가 실패하면 다른 위치의 오래된 토큰으로 fallback하지 않는다. 주소는 `GRT_SERVER_URL` 환경변수로도 지정할 수 있다. 원격 주소는 HTTPS, 로컬 검사만 loopback HTTP를 허용한다.

기존 `setup → prepare_data → preflight → train → report` 순서는 유지한다. `setup()`에서 권한·버전을 확인하며, setup을 생략해도 preflight 전에 검사한다. 필요한 schema와 큐는 등록한다. 연결 옵션은 연구 Config나 결과 JSON에 넣지 않는다.

학습 전에 Config/provenance를 전송한다. 서버 연결 시 5분 간격으로 update 경계에서 확정 checkpoint를 저장하고 helper가 snapshot하여 전송한다. `report()`는 원본 `experiment_report.json`, 실제 Config·metadata·checkpoint·로그와 선택한 posthoc/ZIP을 자동 저장한다. 원본 bytes는 변경하지 않고 서버 검증용 envelope는 별도로 만든다. `download=True`의 기존 로컬 다운로드도 유지한다.

학습 프로세스 종료만으로 성공을 확정하지 않는다. 최종 `report()`가 업로드·finish를 요청하고 서버가 실제 크기·SHA256과 완료 증거를 검증한 뒤에만 성공을 반환한다.

## 큐 등록과 실행

제출할 코드와 prepared 데이터를 먼저 고정한다. `job.prepare_data()`로 데이터를 검증한 뒤 다음과 같이 제출 JSON을 만든다. `posthoc_options=None`이면 학습 결과만 요구하며, 분석 옵션을 지정하면 posthoc도 완료에 필수다.

```python
import json
from pathlib import Path
from grt.utils.server import ServerClient

client = ServerClient(secret_source='env')
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

JSON은 서버 GUI 또는 기존 GRT_server CLI로 검증·제출한다. 코드 버전은 full commit SHA와 소스 파일 hash, 데이터는 manifest hash로 고정한다. 미커밋 소스도 hash에 포함하므로 worker에는 같은 파일이 있어야 한다. 제출 후 조건을 바꾸려면 새 job을 만든다.

worker에는 패키지를 설치하고 서버 주소·토큰을 환경으로 주입한다. 토큰을 명령행 인자에 넣지 않는다.

```bash
python -m pip install -e '.[wandb,wikitext]'
python scripts/server_worker.py \
  --work-dir /path/to/work --prepared-dir /path/to/prepared \
  --deadline '<실제 UTC ISO8601 종료 시각>' --world-size 1
```

기본은 CUDA 1개·환경변수 secret이다. CPU 검사는 `--runtime cpu --device cpu`를 추가한다. DDP는 `--world-size`를 지정하며 helper가 기존 torchrun 경로를 사용한다. 요구 GPU 수와 world size가 맞는 job만 배정받는다. 노트북에서 한 작업씩 실행하려면 `WikiTextRun.claim(repo, work_dir=..., deadline=..., runtime='colab', device='cuda', world_size=1, secret_source='colab')`을 사용한다.

worker는 서버가 지정한 임의 코드·URL·shell을 내려받아 실행하지 않는다. 로컬 코드·데이터·world size를 확인하고, 불변 Config에서 환경별 경로와 같은 effective batch의 auto-batch 보정만 해소한다. 실제 Config는 별도로 보존한다. 작업은 순차 실행하고, 큐가 비었거나 pause/drain·호환 조건 때문에 배정할 수 없으면 종료한다. `--once`는 한 작업만 처리한다. Vast 임대와 과금 종료는 이 worker의 기능이 아니다.

## 장애와 재시도

- heartbeat는 학습과 별도로 수행한다. 취소·인증 만료/폐기·세션 deadline·약 10분의 통신 단절이 확인되면 stop 파일을 만들고 학습은 update 경계에서 checkpoint를 저장하고 중단한다. 진행 중인 update·평가가 끝나는 시간은 필요하며 강제 런타임 삭제에는 저장을 보장하지 않는다.
- 요청 ID·업로드 ID·lease는 run의 `.server/`, 큐 worker 식별 정보는 work 디렉터리의 `.grt-server/`에 0600 파일로 보존한다. API 토큰과 signed URL은 journal이나 결과에 저장하지 않는다. lease는 실행 권한이므로 이 디렉터리를 공개·복제하지 않는다.
- 전송 실패 후에는 같은 run에서 `report()`를 다시 실행한다. 기존 업로드를 확인하며 학습을 반복하지 않는다. 이미 succeeded인 job은 기존 결과를 재사용한다.
- W&B나 posthoc가 실패해도 로컬 학습 결과를 먼저 내보내고 가능한 파일을 전송한다. posthoc 실패는 단계 오류로 표시하며 같은 lease가 유효하면 `report(posthoc=True, ...)`로 분석만 재시도한다. 프로세스를 떠날 때는 `job.close()`로 heartbeat를 종료한다.
- 토큰 교체 후 같은 run으로 재연결하면 journal의 소유권을 확인한다. 같은 Config·manifest·world size의 로컬 checkpoint만 기존 복원 계약으로 재개한다. 원래 lease를 잃었거나 새 세대가 생겼으면 실행 권한을 복제하지 않고 종료 확인·서버 retry가 필요하다. 서버 checkpoint 자동 복원은 제공하지 않는다.

## 검증

```bash
python -m pip install -e '.[dev]'
GRT_TEST_SERVER_REPO=/path/to/GRT_server python -m pytest tests/test_server.py -q
```

GRT_server checkout에는 `npm ci`가 필요하다. 실제 Express 서버를 임시 SQLite·계정으로 실행해 작은 합성 token 데이터의 CPU 학습·posthoc·원본 다운로드·중복 방지·전송 응답 유실·큐·취소 저장을 검사한다. 서버 설치본이 없으면 외부 통합 검사는 skip된다. 일반 GRT 검사는 서버 없이 실행한다.

CPU 기능 검증은 실제 WikiText 학습 품질·GPU 수렴·Colab/Kaggle secret 권한·AWS/S3 배포·장시간 장애 내성을 입증하지 않는다. 실제 환경 pilot에서 별도로 확인한다.
