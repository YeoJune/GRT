# 실험 제어 서버: 최소 API·Vast.ai·GUI/CLI 설계

작성: 2026-10-21 · 검토 기준: `3eafa13` · **설계만 작성, 미구현**

[원격 GPU 큐 보고서](20261021_remote_gpu_queue_design.md)의 후속 설계다. AWS에서 관리 서버를 호스팅하고 노트북·원격 컴퓨터가 같은 규약으로 작업을 가져간다. API·제어 패널의 구체적인 범위는 이 문서를 우선하며, 기존 보고서의 재개·보관·비용 원칙을 따른다. 아래 경로·JSON·CLI는 모두 제안이며 현재 실행 가능한 기능이 아니다.

## 1. 최소 구성과 경계

```text
웹 GUI / 로컬 CLI ── HTTPS ── AWS 관리 서버 ── Vast.ai API
Kaggle / Colab / 원격 runner ──┘     │
          │                      SQLite: 큐·상태·권한
          └── 서명 URL로 전송 ── S3: 입력·결과·checkpoint
W&B: 학습 모니터링             별도 예약 감시: 임대 deadline 정지
```

초기 구현 후보는 **EC2 한 대의 HTTPS 웹 서버＋단일 FastAPI 애플리케이션＋로컬 EBS의 SQLite＋비공개 S3**다. 정적 프론트엔드를 API와 같은 origin에서 제공한다. 큐 claim은 DB transaction으로 처리하고 별도 메시지 브로커·Kubernetes·분산 scheduler는 도입하지 않는다. 단일 관리 서버의 가용성 한계는 인정하며, DB의 정합적 snapshot을 S3에 백업한다. DB 교체는 아래 HTTP 계약에 영향을 주지 않는다.

Vast 임대가 있을 때만 독립적인 EventBridge Scheduler→Lambda 감시를 추가한다. 서버 DB와 별개로 S3에 저장한 allocation ID·deadline·종료 정책을 읽어 정지를 확인한다. 예약 호출은 중복될 수 있으므로 stop을 멱등 처리한다. 서버 장애가 무제한 GPU 가동으로 이어지지 않게 하며 API 장애 시 사용 가능한 최신 저장 checkpoint로 복구한다. [AWS 예약 호출](https://docs.aws.amazon.com/scheduler/latest/UserGuide/what-is-scheduler.html)

관리 서버는 **작업과 파일의 제어**를 담당한다. GPU 실행·노트북 세션 생성·W&B 시계열 보관은 담당하지 않는다. 실행기는 인터넷으로 서버에 HTTPS 접속하므로 inbound 포트·SSH·고정 IP가 필요하지 않다. 실행 환경 설치와 GPU 호환성은 runner의 책임이다.

## 2. 장기적으로 유지할 데이터 규약

### 핵심 리소스

| 리소스 | 의미와 고정 계약 |
|---|---|
| `job` | 하나의 불변 실험 정의. `id`, `name`, `queue`, `runner`, `spec_version`, `spec`, `requirements`, `policy`, `labels` |
| `attempt` | Job의 실행·재개 시도. Worker, lease 세대, 단계, 진행, 종료 사유, 환경, W&B URL 보관 |
| `artifact` | 입력·결과·checkpoint 파일. ID, 종류, 형식 버전, byte 크기, SHA256, 검증 상태, 저장 object version |
| `worker` | 실행 환경. 토큰에 묶인 ID, 지원 runner 버전, GPU/world size, 상태·마지막 접속 |

`job`이 기존 보고서의 experiment에 해당한다. 별도 `run` 리소스를 추가하지 않는다. Campaign은 `labels.campaign`으로 묶고 queue는 이름으로 구분한다. `allocation`과 비동기 `operation`은 공급자 제어에만 사용하며 핵심 실험 규약과 분리한다.

### 범용성은 작은 외피와 버전 있는 spec으로 확보

공통 API에는 GRT의 M·L·gate·optimizer 필드를 추가하지 않는다. `runner="grt/1"` 같은 등록된 adapter가 자기 `spec_version`을 검증한다. 서버는 adapter의 JSON schema로 제출을 검증하고 worker도 실행 전에 검증한다. 다른 연구는 새 adapter/schema를 추가한다. 임의 shell 문자열이나 서버 파일 경로 실행을 범용성의 수단으로 삼지 않는다.

```json
{
  "name": "candidate-m20-long",
  "queue": "research",
  "runner": "grt/1",
  "spec_version": 1,
  "spec": {
    "code_ref": "<고정한 전체 commit SHA>",
    "dataset_artifact_id": "art_dataset",
    "selection": {"task": "wikitext103", "model": "grt", "size": "small", "budget": "100k"},
    "overrides": {"model.segment_len": 75, "model.num_registers": 20},
    "analysis_profile": "grt_standard_windows/1"
  },
  "requirements": {"gpu_count": 1, "labels": ["fp32"]},
  "policy": {"max_attempt_seconds": 172800},
  "labels": {"campaign": "long-budget"}
}
```

이 예시는 API 외형 설명이며 완전한 연구 설정이 아니다. 서버의 고정 버전 schema/profile과 runner가 기본값을 해소하고, 실행 전 resolved Config·dataset hash·code/image hash를 artifact로 남긴다. Profile은 `/1`의 의미를 바꾸지 않고 수정 시 새 버전을 발급한다. Job 제출 뒤 spec 변경은 새 job이며 clone은 입력 복사만 한다. 기존 결과·checkpoint를 새 학습 조건에 덮어쓰지 않는다.

API는 `/v1`을 사용한다. 불투명 ID, UTC ISO8601 시각, 초·byte 단위, 소문자 hex SHA256, JSON, cursor pagination을 고정한다. 응답의 추가 필드는 무시 가능하게 하고 기존 필드의 의미·단위는 바꾸지 않는다. 잘못된 요청 필드는422로 거부하며 연구 확장은 `spec`과 문자열 `labels`에 둔다. 깨지는 변경은 새 API/runner 버전으로 분리한다. HTTP 규약의 안정성과 옛 모델·checkpoint 호환 코드 유지는 별개의 문제다.

## 3. 자체 API

### 3.1 인증과 공통 응답

CLI/노트북은 `Authorization: Bearer <token>`을 사용한다. 처음 관리자 계정만 서버 설치 시 일회성 bootstrap으로 생성하며 공개 가입은 없다.

| 주체 | 권한 |
|---|---|
| Viewer | 허용된 job·결과 조회·다운로드 |
| Operator/Admin | 실험 제출·취소·재시도. 토큰 발급·임대 조작은 별도 admin scope |
| Worker | 허용 queue의 claim, 자기 attempt 보고·관련 입력 읽기·출력 업로드만 가능 |

토큰은 충분한 난수로 만들고 한 번만 표시한다. DB에는 검증용 해시·scope·만료·폐기 상태만 저장한다. Worker 토큰 발급 시 worker ID와 queue를 결합한다. 예시 기본 만료는7일이며 긴 실험에는 시작 전 유효기간을 확인한다. 브라우저는 토큰으로 session을 만든 뒤 Secure/HttpOnly/SameSite cookie를 사용하며 localStorage에 토큰을 보관하지 않는다. 변경 요청은 Origin/CSRF 검증을 한다. Signed URL도 임시 자격이므로 로그·공유 화면에서 숨긴다.

`POST /tokens`는 기존 `worker_id`를 주면 같은 주체의 토큰을 재발급하고, 생략하면 새 worker를 만든다. 재발급으로 기존 권한을 자동 확대하지 않는다. 실행 중 인증이 만료되면 신규 claim을 멈추고 로컬 저장·안전 중단하며, 재인증 후 같은 worker/attempt의 업로드를 복구한다. 비밀 토큰 발급 응답을 잃으면 기존 발급 건을 폐기하고 새로 발급한다. 토큰 원문을 멱등 응답 캐시에 남기지 않는다.

AWS 접근은 EC2/Lambda IAM role로 제한하고 장기 AWS 키를 노트북에 주지 않는다. Vast API key는 Secrets Manager 등 서버 전용 암호화 저장소에 보관하며 GUI·worker에는 반환하지 않는다. EBS/S3 저장 암호화, S3 공개 차단, TLS, 인증 실패 제한, 조작 감사 로그를 기본으로 둔다. [EC2 IAM role](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/iam-roles-for-amazon-ec2.html)

요청 생성·claim·finish·업로드 확정에는 `Idempotency-Key`를 쓴다. 같은 키/본문은 같은 결과, 같은 키/다른 본문은409다. Claim의 빈 결과는 그 요청에만 해당하므로 다음 poll은 새 키를 쓴다. 키 범위는 주체＋경로이며 terminal 작업의 재시도에서도 중복 생성하지 않도록 보관한다. 오류는 `{code, message, request_id, retryable}`로 통일한다. 401/403은 인증/권한, 409는 상태·lease 충돌, 422는 검증, 429는 호출 제한이며 backoff를 적용한다.

### 3.2 최소 endpoint 목록

모든 경로는 `/v1` 기준이다. GUI·로컬 CLI·노트북이 같은 endpoint를 사용한다.

| Method / 경로 | 용도·주요 계약 |
|---|---|
| `POST /sessions`, `DELETE /sessions/current` | 브라우저 로그인·로그아웃 |
| `POST /tokens`, `GET /tokens`, `DELETE /tokens/{id}` | Admin의 토큰 발급·메타데이터 조회·폐기. Worker형 발급은 worker ID도 생성 |
| `GET /capabilities` | 지원 API/runner/schema/profile 버전·입력 한도 조회 |
| `POST /jobs/validate` | 저장 없이 schema·연구 조건·실행 요구 검증, 해소된 설정과 경고 반환 |
| `POST /jobs` | Job 저장·queue 등록. 검증을 서버에서 다시 수행 |
| `GET /jobs`, `GET /jobs/{id}` | 상태/queue/campaign 필터, attempt·검증된 artifact·W&B 링크 조회 |
| `POST /jobs/{id}/actions` | `cancel`, `retry`. 불변 spec 유지, 상태에 맞지 않으면409 |
| `GET /workers`, `GET /workers/{id}` | Worker·현재 작업·지원 환경·마지막 접속 조회. Worker 주체는 자기 정보만 |
| `POST /workers/{id}/actions` | Operator의 `drain`, `resume`. 서버도 drain 상태에서는 새 claim 금지 |
| `POST /workers/{id}/claim` | 지원 runner/GPU·남은 세션 시간으로 FIFO 호환 job 획득. 없으면204＋재조회 간격 |
| `POST /attempts/{id}/heartbeat` | Lease 갱신, stage/progress/ETA/checkpoint 보고. 중단 요청 반환 |
| `POST /attempts/{id}/finish` | Outcome·실행 manifest·artifact ID 제출. 서버 검증 후 확정 |
| `POST /artifacts/uploads` | 크기·SHA256·종류·연결 job을 등록하고 제한된 업로드 URL 발급. 기존 upload ID를 주면 같은 파일의 URL만 갱신 |
| `GET /artifacts/uploads/{id}` | 예약/검증 중/verified/failed 상태와 확정 artifact ID 조회 |
| `POST /artifacts/uploads/{id}/complete` | 실제 저장 파일 검증·artifact 확정. 미검증은 다운로드/완료 대상에서 제외 |
| `GET /artifacts/{id}/download` | 권한 확인 후 해당 object version의 단기 다운로드 URL 발급 |

첫 버전에는 실시간 terminal·job DAG·자동 모델 선택·사용자별 quota 체계를 넣지 않는다. 큐 일시정지는 worker의 `drain`으로 현재 작업까지만 마치는 방식으로 제공한다. Job별 일반 pause API는 안전한 checkpoint 중단 계약이 구현될 때 추가하며 cancel과 혼동하지 않는다.

### 3.3 Claim·재개·완료의 의미

Claim 응답은 `job`, `attempt_id`, `lease_token`, `lease_generation`, `lease_expires_at`, `deadline_at`, `resume_artifact_ids`를 반환한다. Worker는 자기 토큰과 lease를 함께 제출한다. Heartbeat는 초기안30초 주기·lease는3분으로 두며 서버 시각을 기준으로 한다. 한 worker는 처음에는 한 attempt만 점유한다. 서버는 transaction으로 claim하고 지원되지 않는 runner나 GPU 조건을 넘겨주지 않는다.

Heartbeat의 공통 진행률은 `progress={current,total,unit}`과 `eta_seconds`이며 미측정 total/ETA는 null이다. Unit은 `updates`, `tokens`, `samples`처럼 명시하고 모델별 지표는 별도 `metrics` 이름 공간에 둔다. Queue ETA는 이 값과 측정 이력으로 서버가 계산하며 API가 특정 학습 단위를 가정하지 않는다.

Job 상태는 `queued → running → finalizing → succeeded`, 예외는 `interrupted / failed / cancelled`로 최소화한다. 세부 단계는 attempt의 `preparing / preflight / training / posthoc / uploading`에 둔다. `stale`은 attempt의 통신 불명 상태이며 lease 만료만으로 다른 worker에게 재할당하지 않는다. 기존 실행 종료·instance stop 또는 notebook 세션 종료를 확인해야 retry가 새 세대를 얻는다. 이전 세대의 공개·완료 갱신은409로 거부한다.

만료 후 같은 worker가 돌아오면 heartbeat에서 소유권·현재 세대·중단 요청을 대조한다. 새 세대가 없고 소유가 유효하면 같은 attempt의 lease를 갱신하고, 그 외에는409와 안전 중단을 반환한다. 만료된 lease로 바로 finish하지 않고 먼저 갱신한다. Cancel은 durable한 중단 요청이며, 실행 중에는 heartbeat로 전달하고 안전 저장·종료 확인 후 cancelled로 확정한다. 통신 불명 상태를 즉시 종료 확인으로 간주하지 않는다.

`retry`는 같은 spec의 새 attempt이며 복원 가능한 마지막 단계를 선택한다. Posthoc만 실패했으면 checkpoint에서 분석만 다시 한다. GRT의 Config·manifest·확정 batch·world size가 다르면 학습 재개를 거부한다. 40k 완료run을100k로 바꾸는 것은 새 job이다.

Finish는 exit code0 선언이 아니다. Adapter가 예산 도달·필수 report/schema·선택 checkpoint 대응을 검증하고 서버가 필수 artifact의 검증 상태를 확인해야 `succeeded`가 된다. 오래 걸리면202와 `finalizing`을 반환하며 `GET /jobs/{id}`로 확인한다. 학습만 완료됐거나 분석 실패인 파일도 보관하지만 기본 완료 목록에는 넣지 않는다.

### 3.4 파일 입력·출력

입력 dataset/tokenizer도 같은 artifact API로 사전 등록한다. Worker는 권한이 있는 artifact ID로 내려받고 hash를 검증한다. 서버 로컬 경로나 임의 외부 URL을 입력으로 받아 가져오지 않는다. 학습 adapter의 code 저장소·실행 entrypoint도 등록된 범위로 제한한다.

출력은 `업로드 예약 → S3 직접 PUT → complete → 검증 → finish` 순서다. 서버가 생성한 object key·예상 크기·checksum에 URL을 묶고 전송 실패 시 같은 upload ID로 새 URL을 받는다. S3 checksum을 검증하거나 검증 작업이 직접 SHA256을 계산한다. ETag를 SHA256으로 대체하지 않는다. Bucket versioning을 사용하고 완료 artifact는 검증한 version ID를 가리켜 URL 재사용에 의한 덮어쓰기와 구분한다. 큰 파일의 multipart 여부는 upload 응답 내부 확장으로 처리한다. [S3 서명 업로드](https://docs.aws.amazon.com/AmazonS3/latest/userguide/PresignedUrlUploadObject.html)

기존 결과 JSON 형식과 원본 bytes를 유지하며 ZIP·checkpoint·로그를 구분한다. 기본 수집은 `GET /jobs?state=succeeded`의 결과 bundle만 받고 로컬에 ID/hash receipt를 남긴다. Checkpoint·부분 결과는 명시적으로 고른다. W&B는 링크와 attempt별 run ID만 저장하고 학습 시계열은 W&B와 원본 report에 둔다.

### 3.5 Kaggle/Colab과 같은 사용 규약

가벼운 client는 HTTP/JSON·파일 전송을 담당하고 CUDA·AWS SDK를 요구하지 않는다. 설치된 연구 adapter가 기존 학습 CLI/helper를 연결한다. 향후 노트북 사용 형태는 `connect(server_url, token) → run_one()` 또는 남은 시간 제한이 있는 `run_until(deadline)` 정도로 둔다. Secret에는 자기 worker 토큰만 넣으며 관리자·Vast 키를 넣지 않는다.

Heartbeat/중단 확인은 학습과 별도로 돌고, 세션 종료 전 저장·전송 여유를 남긴다. 한도가 촉박하면 claim하지 않는다. Notebook 세션·인터넷 제한을 우회하거나 종료 후 계속 실행된다고 가정하지 않는다. 갑작스러운 종료에는 마지막 외부 checkpoint까지만 복원할 수 있다. 기존 노트북은 그대로 쓰고 연결 셀/helper를 선택적으로 추가하는 후속 구현으로 둔다.

## 4. Vast.ai 제어용 API

### 4.1 자체 서버에 노출할 최소 기능

Vast 키는 서버 내부 adapter만 사용한다. 조회·offer 검색은 `providers:read`, 생성·시작·정지·삭제는 `providers:write` scope가 필요하며 worker에는 허용하지 않는다. Allocation 조회와 조작은 서버가 생성·등록해 소유를 확인한 자원만 대상으로 한다.

| 자체 API `/v1` | 입력·출력 |
|---|---|
| `POST /providers/vast/offers/search` | GPU 수/종류·VRAM·시간당 가격 상한·disk 조건 → 정규화된 offer와 견적 시각 |
| `POST /allocations` | `provider=vast`, offer ID, 등록된 image/profile, 가격 상한, hard deadline, 종료 정책 → allocation ID·operation ID |
| `GET /allocations`, `GET /allocations/{id}` | 공급자 상태·worker 연결·비용 추정·deadline·저장 상태 |
| `POST /allocations/{id}/actions` | `start`, `stop`, `destroy` → operation ID. Destroy는 저장 조건·권한 확인 |
| `GET /operations/{id}` | `pending / succeeded / failed / unknown`, 공급자 응답 정규화 |
| `POST /workers/enroll` | 일회 bootstrap code＋allocation 결합 확인 → 기존 예약 worker ID·토큰·만료 시각. 이 경로만 bootstrap 자격으로 인증 |

생성·변경은202를 반환하며 공급자 목표 상태를 확인해야 operation이 완료된다. Create/start 성공은 공급자 `running` 확인이며, 작업 수신 준비는 별도 `worker_ready`로 표시한다. Fixed image의 runner가 짧은 TTL·일회 bootstrap code를 교환해 allocation에 묶인 worker 토큰을 얻고 접속해야 작업을 전달한다. Enroll은 새 worker를 중복 생성하지 않는다. 교환 응답 유실 시 bootstrap을 재사용하지 않고 admin이 기존 토큰을 폐기·재발급한다. 일반 노트북은 이 교환 없이 사전 발급된 worker 토큰을 쓴다.

### 4.2 Vast 원본 API와의 매핑

2026-10-21 공식 문서 기준이다. 호스트는 `https://console.vast.ai`, 인증은 `Authorization: Bearer <VAST_API_KEY>`다. 자체 `/v1`과 Vast의 API 버전은 관련이 없다.

| 동작 | Vast REST method/path |
|---|---|
| Offer 검색 | `POST /api/v0/bundles` |
| 생성 | `PUT /api/v0/asks/{offer_id}/` |
| 목록 | `GET /api/v1/instances` — pagination 처리 |
| 단건 조회 | `GET /api/v0/instances/{id}/` |
| 시작·정지 | `PUT /api/v0/instances/{id}`＋`state=running / stopped` |
| 삭제 | `DELETE /api/v0/instances/{id}/` |

검색·생성 흐름은 [API Hello World](https://docs.vast.ai/api-reference/hello-world), 목록은 [최신 목록 reference](https://docs.vast.ai/api-reference/instances/show-instances), 시작·정지는 [상태 제어 reference](https://docs.vast.ai/api-reference/instances/manage-instance)를 따른다. 문서에도 v0/v1 경로가 혼재하므로 raw 응답·경로·오류·pagination은 `VastProvider` 내부에 가둔다. 노트북·GUI가 공급자 JSON에 의존하지 않게 한다.

### 4.3 최소 비용·장애 규칙

- 임대는 명시 요청만 허용하고 최초 버전은 on-demand·동시 allocation1개로 제한한다. 시간당 상한·최장 시간·정지 여유 없이는 생성하지 않는다. 최종 가격을 재확인하며 예산 추정과 공급자 최종 청구를 구분한다.
- Create 요청 전에 operation을 기록한다. Timeout이면 `unknown`으로 두고 저장한 상관 ID/label로 기존 임대를 조회한다. 결과 불명 상태에서 자동 재생성하지 않는다. 자체 Idempotency-Key가 Vast 쪽 멱등 생성까지 보장하지는 않는다.
- 큐 소진 시 `drain → 외부 보관 확인 → stop → 상태 확인 → destroy`를 수행한다. Stop 후 storage 비용은 남고 destroy는 데이터를 삭제한다. 미동기화 자산이 있으면 기본은 stop·보존이며 저장비 잔존을 표시한다. 완전 비용 상한 우선 삭제는 사전 정책으로만 허용한다.
- API 장애·실행기 무응답·deadline은 독립 감시가 정지를 재확인한다. 실패는 `stop_failed`와 재시도 상태로 표시하며 성공으로 숨기지 않는다. 물리 PC·Kaggle/Colab에는 공급자 반납 기능을 적용하지 않는다.

## 5. 개발자용 프론트엔드와 CLI

동일한 API client를 사용하는 정적 SPA를 같은 origin에서 제공한다. 테마는 단색 배경·읽기 쉬운 표·상태 badge·고정폭 설정 편집기 정도로 제한한다. 실시간 websocket 대신5~10초 polling으로 시작하며 통신 단절과 데이터 갱신 시각을 표시한다.

| 화면 | 최소 구성 |
|---|---|
| Jobs | Queue/campaign/status 필터, 실험명·단계·step·ETA·worker·W&B 링크, 신규/복제/취소/재시도 |
| Job 설정·상세 | Schema 기반 기본 GUI＋같은 draft의 JSON 편집, 검증 결과·최종 spec, attempt 이력·오류·artifact·완료 결과 다운로드 |
| Workers/Settings | Worker online/drain 상태, Vast offer·allocation·deadline·정지/삭제, 토큰 발급/폐기 |

GUI와 JSON은 같은 draft를 편집한다. Schema에서 해석할 수 없는 고급 설정도 JSON에서 보존하며 UI 전환으로 삭제하지 않는다. Submit 전 서버 validation과 해소된 설정을 보여주고, 완료된 job의 조건 편집 대신 복제를 제공한다. ETA unknown은 빈 숫자나0이 아니라 미측정으로 표시한다. 임대 생성·미보관 삭제는 대상·가격·보관 상태를 보고 명시적으로 실행한다.

제어 패널 아래에는 접을 수 있는 **관리 명령창**을 둔다. 로컬 CLI와 명령 문법·JSON 출력 옵션을 공유하고 허용한 API 동작만 실행한다. 서버 shell·SSH terminal·임의 Python 실행 기능은 넣지 않는다.

```text
jobs submit experiment.json
jobs list --queue research --state succeeded
jobs show job_123
jobs retry job_123
results collect --campaign long-budget --new
workers drain worker_123
allocations stop alloc_123
```

`results collect`는 로컬 CLI에서는 실제 파일을 내려받고, 브라우저 명령창에서는 같은 artifact 목록으로 다운로드를 시작한다. 브라우저의 `jobs submit`은 파일 선택 또는 JSON 입력을 사용한다. 임대·토큰 명령도 같은 서버 권한을 적용한다. Worker drain은 heartbeat 응답으로 전달해 현재 attempt 뒤 claim을 멈추며, resume은 다음 claim을 다시 허용한다. GUI만 가능한 숨은 상태 변경을 만들지 않는다.

## 6. 최초 범위와 변경 원칙

우선 구현 범위는 **단일 사용자 관리자·단일 API 서버·worker별 순차 실행·기본 토큰 권한·불변 job·완료 artifact·Vast1개 allocation·표 중심 GUI/CLI**다. 다른 환경의 worker 등록은 가능하지만 초기 유료 동시 임대를 늘리지 않는다. 연구 의존성은 독립 wave를 사람이 등록하는 것으로 처리한다.

고정할 것은 ID·인증·claim/lease·heartbeat·finish·artifact 검증의 의미다. 확장할 것은 runner/schema/profile·화면·provider adapter다. 모델별 field, AWS 저장 경로, Vast 응답을 공개 규약에 섞지 않는다. 후속 구현 시 OpenAPI/schema를 같은 API 정의에서 생성해 GUI·CLI·노트북이 공유하도록 한다.

이 문서는 보고서와 문서 목록만 변경한다. 기존 노트북·코드·설정·원본 결과는 그대로 유지하며 구현·CPU 테스트·배포·실제 임대·인증정보 발급은 수행하지 않는다.
