# 실험 제어 서버: API·노트북 연결·GUI/CLI 완결 설계

작성·갱신: 2026-10-21 · 검토 기준: `db9cc85` · **미구현 설계, 사용 시나리오를 이용한 자체 검토 완료**

2026-10-22 구현 현황: 서버 API는 [GRT_server](https://github.com/YeoJune/GRT_server/blob/main/docs/api.md), GRT의 선택 연결과 검증 범위는 [서버 연결 안내](20261022_server_integration.md)를 참고한다. 아래 문서는 목표 설계이며 모든 기능의 구현·실증을 뜻하지 않는다.

[원격 GPU 큐 보고서](20261021_remote_gpu_queue_design.md)의 구체적인 서비스 규약이다. 본문만으로 로그인→토큰 발급→기존 실행 연결/큐 실행→저장→조회·다운로드→실패 복구가 이어지도록 정의한다. 이전 초안의 세분화된 scope·worker 전용 권한 대신 **write/read-only 두 종류**를 사용한다. 아래 API·설정·화면은 제안이며 아직 사용할 수 있는 기능이 아니다.

## 1. 요구사항과 최소 구성

| 요구 | 설계의 대응 |
|---|---|
| Colab/Kaggle secret으로 토큰 사용 | 같은 secret 이름과 환경별 reader, 연결 전 인증 검사 (§3) |
| 기존 실행에서 서버 자동 저장 | `off / upload / queue` 모드, 공통 실행 helper의 저장 hook (§3) |
| 서버 로그인과 제어 | 단일 소유자 계정·비밀번호, 서버 관리 session (§2) |
| write/read-only 발급·만료 | 두 종류만 발급, 매 요청 서버 검증·폐기·만료 관리 (§2) |
| 설정·큐·결과·다운로드 편의 | 네 개의 작업 화면, 같은 API의 GUI/CLI (§7) |
| 새로운 실험 필드 지원 | 불변 schema와 확장 JSON, 일반 편집기·원본 보존 (§4) |

```text
GUI / CLI / Codex ── HTTPS ── AWS 관리 서버 ── Vast.ai API
Colab / Kaggle / 원격 실행기 ──┘    │
          └── 서명 URL로 전송 ── S3│
                        SQLite: 큐·상태·인증·schema
W&B: 학습 곡선                 독립 예약 감시: 임대 deadline 정지
```

EC2 한 대에서 HTTPS reverse proxy·단일 FastAPI 서비스·정적 SPA를 제공한다. SQLite는 암호화된 로컬 EBS에 두고 정합적인 snapshot을 S3에 백업한다. S3는 비공개·암호화·versioning을 적용해 입력/결과/checkpoint를 저장한다. 별도 메시지 브로커·Kubernetes·웹 terminal은 도입하지 않는다. DB나 프론트엔드를 교체해도 HTTP 계약은 유지한다.

실행기는 외부로 나가는 HTTPS만 사용한다. Inbound 포트·SSH·고정 IP가 필요하지 않으며 GPU·소프트웨어 설치는 실행기 책임이다. 서버는 노트북 세션을 생성하거나 플랫폼 사용 한도를 우회하지 않는다. Vast 임대에는 별도 EventBridge Scheduler→Lambda 감시를 두고 서버와 독립된 S3 allocation 정책으로 deadline 정지를 재확인한다. W&B는 학습 지표, 자체 서버는 실험 상태·파일·큐를 담당한다.

## 2. 로그인과 두 종류 토큰

### 2.1 브라우저 로그인

최초 설치 시 서버 관리 명령에서 **소유자 계정 한 개와 비밀번호**를 설정한다. 공개 회원가입·이메일 발송은 없다. 비밀번호는 Argon2id 해시로 저장하고 원문은 보관하지 않는다. 복구는 AWS 관리 접속으로 수행하는 비밀번호 재설정이며 재설정 시 기존 session을 모두 폐기한다. API 토큰 폐기 여부는 별도로 선택한다.

로그인 화면은 기본 `계정/비밀번호`, 보조 `API 토큰으로 접속` 두 탭이다. 서버가 검증 후 opaque session을 발급하고 Secure·HttpOnly·SameSite cookie에 저장한다. 기본 최대8시간·미사용30분으로 만료하며 로그아웃은 서버 session도 폐기한다. 비밀번호 로그인은 write, 토큰 로그인은 원래 토큰 권한을 그대로 따른다. 토큰 session은 원래 토큰의 만료·폐기도 매 요청 확인하고 이를 넘어 연장하지 않는다. Cookie에서 비밀번호 변경·민감한 전체 삭제를 요청하면 해당 로그인 수단으로 재인증한다. Bearer write 요청은 매번 토큰을 검증하며 별도 숨은 관리자 권한을 요구하지 않는다.

모든 통신은 HTTPS다. 같은 origin UI와 API, 변경 요청의 Origin/CSRF 검증, 로그인 속도 제한·감사 로그를 적용한다. 토큰 원문을 localStorage·URL·노트북 출력·로그에 남기지 않는다. Session은 브라우저 로그인 수단이며 세 번째 API 토큰 권한 종류가 아니다.

### 2.2 권한은 정확히 두 종류

| 자격 | 허용 | 금지 |
|---|---|---|
| `write` | 모든 조회·실험 설정·큐/실행기 관리·업로드·토큰 발급/폐기/만료 설정·schema 등록·임대 조작 | 비밀번호/기존 토큰/외부 서비스 key 원문 조회 |
| `read-only` | 실험·큐·schema·실행 상태·결과/로그·비밀값 없는 설정 조회, 다운로드 | 제출·claim·heartbeat·업로드·취소·순서 변경·토큰/임대 조작 |

읽기 작업인 offer 검색·설정 validate와 읽기용 export 생성은 POST여도 read-only에 허용한다. 실험·실행·권한·보관 정책을 바꾸는 작업은 반드시 write를 요구한다. 브라우저 버튼 숨김에 의존하지 않고 서버가 검사한다. 단일 소유자의 연구 공간 전체에 대한 권한이며 tenant·queue별 scope를 추가하지 않는다. **노트북에 넣는 write 토큰도 전 권한**이므로 신뢰하는 개인 실행 환경에만 제공한다. Lease나 일회 bootstrap은 작업 소유 확인용이며 별도의 제한된 API 권한 종류가 아니다.

토큰 필드는 `id, name, kind, prefix, created_at, expires_at, last_used_at, revoked_at`다. `kind`는 생성 후 불변이며 기본 만료7일, 발급자가 절대 만료시각을 선택한다. 서버는 난수 토큰을 한 번만 보여주고 검증용 해시만 저장한다. 만료가 필수이고 무기한 자동 갱신은 없다. 활성 토큰의 만료시각은 write가 수정할 수 있지만 이미 만료·폐기된 토큰은 복구하지 않고 새로 발급한다. 발급 응답 유실 시 해당 ID를 폐기·재발급하며 원문을 멱등 응답 캐시에 보관하지 않는다.

모든 인증 요청에서 서버 UTC 시각과 DB 상태를 확인한다. `now >= expires_at` 또는 폐기 시401, 유효한 read-only의 변경 요청은403이다. 기존 session·기존 lease가 권한을 연장하지 않는다. 실행 중 만료되면 신규 claim·서버 쓰기를 중단하고 로컬 checkpoint/전송 대기 목록을 남긴다. 새 write 토큰을 secret에 등록해 재연결하면 같은 job/attempt의 복구 여부를 서버가 판정한다. Worker ID는 token ID와 분리해 토큰 교체로 새 worker가 생기지 않는다.

다운로드/업로드 서명 URL은 최대5분과 현재 자격의 남은 유효기간 중 짧은 값으로 발급한다. 이미 발급한 URL은 토큰 폐기 직후에도 자체 만료까지 유효할 수 있으며 전송 중 파일을 회수할 수는 없다. API의 즉시 폐기와 이 짧은 URL 유효기간을 구분한다. AWS role로 S3에 접근하며 AWS/Vast key는 노트북에 전달하지 않는다. Vast key는 서버의 암호화된 비밀 저장소에 둔다.

## 3. Colab/Kaggle secret과 기존 실행 연결

### 3.1 동일한 연결 설정

공통 secret 이름은 **`GRT_SERVER_TOKEN`**, 공개 가능한 연결 주소는 **`GRT_SERVER_URL`**이다. Colab은 Secrets의 notebook 접근을 허용하고, Kaggle은 Secrets에 등록해 해당 notebook에 연결한다. 플랫폼 secret 조회 함수는 기존 W&B 연결과 같은 패턴을 사용한다.

| 환경 | 토큰 조회 계약 |
|---|---|
| Colab | `google.colab.userdata.get('GRT_SERVER_TOKEN')` |
| Kaggle | `UserSecretsClient().get_secret('GRT_SERVER_TOKEN')` |
| 원격 PC·CLI·Codex | 실행 환경 secret으로 주입한 `GRT_SERVER_TOKEN` |

플랫폼 API 근거: [Colab userdata](https://github.com/googlecolab/colabtools/blob/main/google/colab/userdata.py), [Kaggle secrets](https://github.com/Kaggle/docker-python/blob/main/patches/kaggle_secrets.py). Colab UI 없이 내보낸 실행에서는 userdata 조회가 실패할 수 있으므로 그 환경은 명시적으로 환경변수 reader를 사용한다.

토큰 읽기는 `secret_source=colab|kaggle|env`로 지정하며 다른 곳의 오래된 키로 조용히 fallback하지 않는다. URL/토큰은 공통 client만 읽고 명령행 인자·resolved 연구 Config·checkpoint·결과 JSON에는 넣지 않는다. `GET /v1/me`로 서버·권한·만료·버전 호환을 확인하며 모드가 켜졌는데 secret 누락/접근 거부/read-only이면 GPU 작업 전에 오류를 표시한다. 서버를 잘못 지정하지 않도록 연결 화면에 URL·인증된 token 이름/종류·만료만 보여준다.

### 3.2 세 가지 실행 모드

| `server_mode` | 설정의 기준 | 동작 |
|---|---|---|
| `off` | 기존 notebook/CLI | 기존 로컬 저장·수동 다운로드 유지, 서버 import/연결 없음 |
| `upload` | 기존 notebook/CLI | 현재 설정을 서버의 attached job으로 등록하고 결과를 자동 저장. 다른 queue job을 가져오지 않음 |
| `queue` | 서버의 불변 job spec | 서버에서 조건에 맞는 작업을 claim해 실행·저장. Notebook의 연구 조건으로 덮어쓰지 않음 |

연결 옵션 예시는 `server_mode='upload', secret_source='colab', server_url='https://research.example'` 정도다. Queue 모드는 추가로 queue 이름·세션 deadline·전송 여유 시간을 선택한다. 연구 조건과 연결 옵션을 분리하며 서버 연결 때문에 Config hash나 기존 학습 스케줄을 바꾸지 않는다. 플랫폼 런타임 경로·GPU 배치는 별도 실행 profile에서 해소하고 실제 Config를 남긴다.

기존 `setup → prepare_data → preflight → train → report` 순서는 유지한다. 공통 실행 helper를 확장해 notebook마다 업로드 코드를 복사하지 않는다. Upload 모드는 시작 전 worker 등록과 `dispatch=attached` job 생성·attempt 할당을 원자적으로 완료한다. Claim 대상 queue에 중복 등록하지 않으며 저장된 제출 키를 사용해 셀 재실행도 같은 job에 연결한다. Queue 모드는 `dispatch=queued` job만 가져온다. 여러 모델을 선택한 기존 helper는 모델별 job/attempt로 관리하고 campaign label로 묶는다.

같은 설정의 셀 재실행은 완료한 단계를 재사용하고 전송만 남았으면 업로드만 반복한다. 이미 succeeded인 job은 기존 결과를 반환하며 다시 학습하지 않는다. 설정이 달라졌으면 충돌을 표시하고 새 job을 요구한다. Schema 검증으로 해소한 실행 계획과 실제 Config를 비교한 뒤 학습을 시작한다.

자동 저장 시점은 학습 전 Config/provenance, 확정 checkpoint, 학습 직후 report, posthoc 직후 report/최종 ZIP, 오류 발생 시 partial report다. 사용자에게 전송을 위한 별도 버튼을 누르게 하지 않는다. Final `report()`는 posthoc 요구 여부에 따라 최종 묶음을 확정하고, `download=True`는 서버 자동 저장과 별개로 기존 브라우저 다운로드도 수행한다. 결과 JSON 형식과 원본 bytes는 유지한다.

현재 helper의 엄격한 W&B 검사나 posthoc 오류가 발생해도 로컬 학습 결과부터 export·전송하는 finally 경로를 설계에 포함한다. W&B 경고와 학습 실패를 구분한다. 이것은 후속 구현 변경점이며 현재 helper에 이미 자동 저장·안전 중단이 있다고 가정하지 않는다.

### 3.3 서버 장애·세션 종료

실행·heartbeat·전송은 분리하고, run 디렉터리에 `job_id/attempt_id/worker_id`, 파일 hash, upload ID와 단계별 전송 대기 목록을 저장한다. Secret은 저장하지 않는다. 네트워크 오류는 backoff로 재시도하고 기본 최대 offline10분 뒤 안전 저장·중단한다. 그 안에 계산이 끝나도 서버 검증 전에는 ‘로컬 완료·동기화 대기’다. Queue가 아닌 off 모드는 이 정책의 대상이 아니다.

Checkpoint 저장 목표는5~10분이며 확정 snapshot만 전송한다. 세션 종료 전 충분한 저장 시간을 남기고, 남은 시간이 부족하면 claim하지 않는다. 갑작스러운 runtime 삭제는 아직 외부에 저장하지 못한 파일을 잃을 수 있다. 마지막 서버 검증 checkpoint와 로컬 현재 step을 화면에서 분리한다. 토큰 교체·재연결 시 기존 전송을 재시도하고 학습은 재개 계약을 통과할 때만 이어간다.

## 4. 데이터와 새 필드의 확장 계약

핵심 리소스는 `job`(실험 정의), `attempt`(실행·재개), `worker`(실행 환경), `artifact`(파일)다. Queue는 이름·paused·우선순위 제어를 가진 작은 리소스로 두고 campaign은 label로 묶는다. Job은 `id, name, queue, priority, dispatch, runner, schema_id, spec, requirements, policy, labels, extensions`를 가진다. Spec은 제출 후 불변이고 이동/우선순위만 queued 상태에서 변경한다.

```json
{
  "name": "candidate-m20-long",
  "queue": "research",
  "dispatch": "queued",
  "runner": "grt/1",
  "schema_id": "schema_grt_2",
  "spec": {
    "code_ref": "<전체 commit SHA>",
    "dataset_artifact_id": "art_dataset",
    "selection": {"task": "wikitext103", "model": "grt", "size": "small", "budget": "100k"},
    "overrides": {"model.segment_len": 75, "model.num_registers": 20},
    "analysis_profile": "grt_standard_windows/1"
  },
  "requirements": {"gpu_count": 1, "labels": ["fp32"]},
  "policy": {"max_attempt_seconds": 172800},
  "labels": {"campaign": "long-budget"},
  "extensions": {}
}
```

예시는 형태 설명이며 완전한 연구 설정이 아니다. Schema와 profile은 코드/검증 버전·내용 hash를 고정해 등록한다. 알려진 예산·세계 크기·dataset 조건은 runner가 검증하고, 신규 연구 필드를 공통 API에 열로 추가하지 않는다. 불변 schema에 기본값·타입·단위·설명·GUI hint를 두고 서버와 실행기가 같은 schema ID를 사용한다.

| 신규 데이터 | 저장·표시·실행 규칙 |
|---|---|
| 새 metric·summary 필드 | JSON object로 원형 보존. 자동 키 발견, 일반 JSON tree와 선택 열 표시. 서버 DB migration 불필요 |
| 새 보조 metadata | `extensions`에 임의 JSON으로 보존. 이전 client는 모르는 필드를 무시하되 수정 시 삭제하지 않음 |
| 새 실행 parameter | 새 불변 schema와 해당 필드를 이해하는 runner/code를 등록. 지원하는 worker만 claim |
| 새 report format | Artifact에 format/version을 기록해 보관·다운로드. 전용 요약이 없으면 JSON 보기 사용. 필수 완료 report로 지정하려면 대응 validator도 등록 |
| 공통 필드 의미 변경 | API `/v2` 등 별도 버전. 기존 필드 의미나 단위를 재사용하지 않음 |

Unknown parameter를 조용히 버리거나 학습에 적용한 척하지 않는다. Schema가 허용한 추가 필드만 수용하며 미지원 실행 조건은422 또는 호환 worker 대기로 표시한다. 새로운 필드의 **운반·보관·GUI 편집에는 API 변경이 필요 없지만, 실제 연산에는 실행 코드의 지원이 필요**하다.

서버는 결과 원본을 artifact로 보존하고 공통 `summary`만 별도 index로 만든다. 새 필드는 generic table/tree에서 JSON Pointer로 선택하며 없는 값은 null, 단위가 다르면 같은 열의 직접 비교를 경고한다. 큰 배열·binary는 파일로 두고 API summary에 중복 넣지 않는다. 예시 제한은 JSON 요청/summary 각각1MiB이며 정확한 한도는 capabilities에 게시하고 초과 시413을 반환한다. 자유 문자열은 HTML로 실행하지 않는다.

API는 `/v1`, UTC ISO8601, 초·byte, 소문자 hex SHA256, 불투명 ID, cursor pagination을 사용한다. 응답 추가 필드는 허용하고 core 요청 오타는422로 거부한다. 자유 필드는 spec/schema·extensions·metrics에만 둔다. 서버가 해소한 고정 spec과 환경별 최종 Config/provenance를 모두 남겨 기본값이 나중에 바뀌어도 예전 실험의 의미가 바뀌지 않게 한다.

## 5. 자체 API와 상태 전이

모든 경로는 `/v1`, 인증은 session 또는 `Authorization: Bearer <token>`이다. Read/write는 HTTP method가 아니라 아래의 의미로 판정한다. 비밀값은 어떤 조회에도 반환하지 않는다.

| 경로 | 권한·계약 |
|---|---|
| `POST /sessions`, `DELETE /sessions/current`, `GET /me` | 비밀번호/토큰 로그인, 자기 로그아웃·인증 상태. Read-only도 자기 session 관리 가능 |
| `POST /account/actions` | Write로 `change_password`, `revoke_sessions`. Cookie 요청은 로그인 수단 재인증. 만료/분실 복구는 서버 관리 명령 |
| `GET/POST /tokens`, `PATCH/DELETE /tokens/{id}` | 메타데이터 조회는 read, 생성·활성 만료 수정·폐기는 write. 이름·종류·만료·원문1회 표시 |
| `GET/PATCH /settings` | 비밀값 없는 보관/연결/비용 정책 조회는 read, 변경은 write. 변경은 새 작업에 적용하고 기존 job/deadline을 묵시적으로 수정하지 않음 |
| `GET /capabilities`, `GET /schemas/{id}`, `POST /schemas` | 버전·한도·schema 조회는 read, 불변 schema 등록은 write |
| `GET/POST /queues`, `POST /queues/{id}/actions` | 조회는 read, 생성·pause/resume은 write. Pause는 신규 claim만 정지 |
| `POST /jobs/validate` | Read로 비저장 검증 가능. 해소된 spec·field 오류·호환성 반환 |
| `GET/POST /jobs`, `GET/PATCH /jobs/{id}` | 조회는 read, 제출·queued의 queue/priority 변경은 write. 상세에 attempt·summary·artifact 목록 포함. `dispatch=attached`는 등록된 worker에 attempt를 원자 할당 |
| `POST /jobs/batch` | Write. Queued job 전체 검증 후 최대50개를 한 transaction으로 제출. 오류면 전부 미등록 |
| `POST /jobs/{id}/actions` | Write. Cancel/retry; stale 복구에는 기존 실행 종료 증거와 확인 기록 필요 |
| `GET/POST /workers`, `GET /workers/{id}` | 조회는 read, worker 생성/기존 ID 재연결은 write. 같은 notebook 세션은 중복 생성 방지 |
| `POST /workers/{id}/actions` | Write. Drain/resume; drain은 현재 attempt 종료 뒤 신규 claim 금지 |
| `POST /workers/{id}/claim` | Write. 조건·deadline에 맞는 queued job 하나, 없으면204와 poll 간격 |
| `POST /attempts/{id}/heartbeat`, `POST /attempts/{id}/finish` | Write＋현재 lease. 진행/lease 갱신·명령 수신/종료 요청. 완료는 서버 검증 후 확정 |
| `POST /artifacts/uploads`, `GET /artifacts/uploads/{id}` | Write로 예약/URL 갱신, read로 상태 조회. 이름·크기·SHA256·format/version·연결 job |
| `POST /artifacts/uploads/{id}/complete` | Write로 파일 확정 요청. 검증 대기면202 |
| `GET /artifacts/{id}/download` | Read로 정확한 object version의 짧은 다운로드 URL 발급 |
| `POST /exports`, `GET /exports/{id}` | 완성 결과 묶음 요청/조회는 read. 원본을 바꾸지 않는 임시 ZIP 생성, 용량·동시 요청 제한 |

### Claim과 소유권

Claim은 transaction으로 FIFO/priority·queue pause·worker drain·runner/schema·GPU 수·입력 artifact 검증·남은 시간을 확인한다. 알려진 ETA＋저장 여유가 세션/임대 기한에 안 맞으면 넘기고, ETA 미측정 job은 preflight 가능한 시간과 checkpoint 중단 허용 정책이 있을 때만 제한적으로 배정한다. 반환은 `job, attempt_id, lease_token, generation, lease_expires_at, deadline_at, resume_artifact_ids`다. 같은 worker의 active attempt는1개다. Worker 재연결은 이미 점유한 attempt 정보를 반환하며 그 상태로 다른 job을 claim하지 않는다. 기존 lease를 가진 client의 인증 교체는 이어갈 수 있지만, lease를 잃은 새 프로세스에는 실행 허가를 복제하지 않고 종료 확인 후 retry한다.

Heartbeat 초기안30초, lease3분. 공통 payload는 `stage, progress={current,total,unit}, eta_seconds, metrics, summary, last_verified_checkpoint_id`다. Total/ETA 미측정은 null이며 모델별 metrics는 임의 JSON object다. 응답은 서버 시각·갱신 lease·`continue/cancel/drain` 명령을 포함한다. 유휴 worker는 claim poll로 접속 시각을 갱신한다. Queue ETA는 현재 잔여와 호환 실행 이력으로 추정하고 pause·미측정은 별도 표시한다.

Lease 만료는 stale이지 종료 증명이 아니다. 같은 worker가 복귀하고 새 세대·중단 요청이 없으면 소유 확인 후 같은 attempt를 재승인한다. 그 외에는409와 안전 중단이다. 새 attempt는 기존 process 종료·instance stop·노트북 세션 종료 확인 후에만 생성하며 근거를 감사 로그에 남긴다. 구세대는 최신 결과를 완료로 덮어쓸 수 없다. Write 전 권한이어도 이 실행 정합성 검사를 생략하지 않는다.

### 상태·재시도·취소

`queued → running → finalizing → succeeded`, 예외는 `interrupted / failed / cancelled`다. 세부 단계는 attempt의 preparing/preflight/training/posthoc/uploading에 둔다. Finish outcome은 `completed/interrupted/failed/cancelled`이며 마지막 단계·오류·manifest·필수 artifact ID를 보낸다. Exit code0·W&B finished·사용자 선언만으로 성공시키지 않는다.

서버 adapter가 예산·report schema·필수 posthoc·선택 checkpoint 대응을 검증하고 필수 artifact가 verified일 때만 succeeded다. 학습 완료·분석 실패·전송 대기는 구분해 남긴다. Retry는 같은 spec의 새 attempt이며 posthoc만 실패하면 분석만 재개한다. 실제 Config·manifest·batch·world size가 다르면 학습 복원을 거부한다. 40k를100k로 바꾸는 것은 clone한 새 job이다.

Finish 접수 시 필수 파일 목록과 세대를 고정하고202를 반환하면 이후 finalizing은 서버의 책임이다. 실행기가 heartbeat를 끝내도 이 검증을 stale로 재분류하거나 재배정하지 않는다. 실패 사유는 job 상세에 남긴다. 이미 접수된 검증은 토큰 만료 뒤에도 끝낼 수 있지만 그 토큰으로 새 요청을 할 수는 없다.

Queued cancel은 즉시 cancelled, 실행 중 cancel은 명령을 저장하고 안전 저장/종료 확인 후 확정한다. 연락이 끊기면 cancel pending/stale로 남긴다. 안전 checkpoint 중단은 후속 구현의 공통 요구이며 현재 trainer에 SIGTERM 안전 저장 기능이 이미 있다고 가정하지 않는다.

생성·claim·finish·complete·batch·임대 조작은 `Idempotency-Key`로 중복 방지한다. 같은 키/본문은 같은 결과, 다른 본문은409다. 빈 claim 뒤 다음 poll은 새 키를 쓰고, 토큰 교체 후에도 같은 client request ID를 조회해 이전 성공을 복구할 수 있다. 오류 형식은 `{code,message,request_id,retryable,field_errors}`이며 인증401·권한403·충돌409·크기413·검증422·호출 제한429를 구분한다. 재시도는 backoff와 상한을 가진다.

정확한 조회 경로는 `GET /requests/{client_request_id}`다. 비밀값을 제외한 작업 ID·접수/완료 상태를 반환하며 write 요청의 중복 키는 개별 token ID가 아닌 소유자＋경로에 결합한다. 따라서 토큰 교체가 같은 실험의 이중 등록을 만들지 않는다.

## 6. 파일 보관과 Vast 제어

입력 dataset/tokenizer는 artifact ID와 hash로 공유한다. 임의 서버 path·임의 외부 URL을 입력으로 받아 내려받지 않고 허용한 코드 저장소·entrypoint만 실행한다. 출력은 `예약 → S3 직접 전송 → complete → 실제 크기/SHA256 검증 → verified → finish`다. 업로드가 끊기면 같은 upload ID로 URL을 갱신한다. ETag를 SHA256으로 대체하지 않으며 검증한 object version을 고정해 URL 재사용 덮어쓰기와 구분한다.

결과 ZIP·원본 JSON은 장기 보관하고 checkpoint는 별도 보관·선택 다운로드한다. 미완료·실패·원시 report도 보존하지만 기본 결과 목록은 succeeded만 보여준다. Export는 선택한 verified 파일 목록/hash를 고정한 뒤 외부 CPU에서 ZIP을 생성하고 `preparing/ready/failed`를 반환한다. Poll 후 read 자격으로 다운로드한다. 같은 fingerprint는 재사용하고 준비 결과는 초기안24시간 후 지워도 원본은 남긴다. 개별 다운로드·Codex의 증분 수집도 같은 artifact ID/hash를 사용한다.

Write로 수행할 Vast 제어 API는 `POST /providers/vast/offers/search`, `POST /allocations`, `GET /allocations[/{id}]`, `POST /allocations/{id}/actions`, `GET /operations/{id}`다. 조회·offer 검색은 read, 생성/start/stop/destroy는 write다. 생성은 image/profile·가격 상한·hard deadline·보관/종료 정책을 필수로 받고202와 operation ID를 반환한다. API key는 서버 내부에만 있다.

| 공급자 동작 | Vast REST 경로: `https://console.vast.ai` |
|---|---|
| 검색 | `POST /api/v0/bundles` |
| 생성 | `PUT /api/v0/asks/{offer_id}/` |
| 목록·조회 | `GET /api/v1/instances`, `GET /api/v0/instances/{id}/` |
| 시작·정지 | `PUT /api/v0/instances/{id}`＋`state=running/stopped` |
| 삭제 | `DELETE /api/v0/instances/{id}/` |

공급자 경로·버전·오류는 VastProvider 내부에서 정규화한다. 생성/start 성공은 공급자 running 확인이고 실제 작업 가능은 별도 worker_ready다. 고정 image에는 짧은 일회 bootstrap을 전달한다. `POST /workers/enroll`이 allocation 결합을 확인해 **write 토큰**을 한 번 발급하며, 이는 세 번째 권한 종류가 아니다. 응답 유실은 폐기·재발급한다. 최초 생성과 재시작의 연결 준비는 allocation action의 일부로 처리하며 사용한 bootstrap을 재사용하지 않는다. 재주입이 불가능하면 연결 대기 상태와 수동 secret 재발급 경로를 표시하고 새 임대를 자동 생성하지 않는다. 일반 notebook은 secret 토큰과 worker 등록을 사용한다.

초기 동시 유료 임대는1개다. 생성 응답 timeout은 unknown으로 저장해 상관 ID/label로 조사하고 중복 임대하지 않는다. 빈 queue에서는 drain→필수 외부 보관→stop 확인→destroy로 종료한다. 미동기화 자산이 남으면 기본 stop·보존이며 storage 과금 잔존을 표시한다. 사전 선택한 총비용 우선 정책만 기한 후 손실을 수용해 destroy한다. 공급자 장애 시 정지 성공을 가장하지 않고 독립 감시가 유한 재시도한다. 플랫폼 무료 notebook에는 임대 API를 적용하지 않는다.

공급자 reference: [생성·조회·종료](https://docs.vast.ai/api-reference/hello-world), [목록](https://docs.vast.ai/api-reference/instances/show-instances), [상태 제어](https://docs.vast.ai/api-reference/instances/manage-instance). 기존 보고서의 비용·보관 원칙은 유지한다.

## 7. 프론트엔드와 CLI의 구체적인 작업 흐름

화면은 **실험 / 큐·실행기 / 결과 / 설정** 네 개와 로그인이다. 같은 API client·schema를 사용하며 회색 계열·표·작은 상태 badge·고정폭 JSON 편집기로 구성한다. 5~10초 polling과 마지막 갱신 시각·오프라인 상태를 표시한다. 시각은 기본 Asia/Seoul로 변환해 시간대를 함께 표시한다. Read-only 접속은 상단에 표시하고 변경 버튼을 비활성화하되 서버도403을 강제한다.

```text
[연구 서버]  실험 | 큐·실행기 | 결과 | 설정     [write/read-only] [로그아웃]
[신규 실험] [조건 필터] [campaign] [선택 결과 받기]
이름       큐/순서   상태·단계   step/목표   ETA   마지막 서버 저장   W&B
...
[선택 실험 상세: 설정 | 실행 이력 | 결과·로그]
[접을 수 있는 관리 CLI >]
```

### 7.1 실험: 만들기→검증→실행

신규 실험은 ① runner/schema·code/dataset/profile 선택 ② 모델/예산/seed 등 schema form과 추가 parameter 입력 ③ 요구 GPU·queue·우선순위·deadline ④ 서버 검증 결과·해소된 spec·경고 검토 ⑤ 등록 순서다. JSON 탭은 같은 draft를 편집하고 모르는 필드를 삭제하지 않는다. Schema/type 오류는 field 단위로 표시하며 미지원 worker 조건은 ‘실행기 대기’로 설명한다.

등록된 실험 상세는 spec 읽기 전용이며 변경은 복제다. 여러 seed/M 값의 묶음 생성은 모든 해소된 job을 미리 보여준 뒤 batch 제출한다. 하나라도 잘못되면 전체 미등록이다. Draft는 저장하지 않은 편집 상태이며 로그인 만료 전에 JSON으로 내보낼 수 있다. Token·비밀번호·signed URL은 draft에 넣지 않는다.

### 7.2 큐·실행기: 대기열을 실제로 제어

Queue별 queued/실행/미완료 수·pause/resume·ETA를 표시한다. 우선순위와 같은 우선순위 안의 등록순으로 실행한다. 이동/순서 변경은 queued에만 가능하고 변경 직전 상태가 바뀌었으면409로 새 상태를 보여준다. Queue pause는 새 시작만 멈추고, 실행 중단은 별도의 cancel이다. Worker drain은 현재 작업 뒤 멈춘다.

Worker 행에는 환경·GPU/world size·지원 schema·현재 attempt·마지막 접속·동기화 상태를 둔다. Queue는 compatible worker가 없으면 사유를 보여준다. Token 만료 예상이 job보다 이르면 새 claim을 막고 교체를 안내한다. Stale 복구는 ‘이전 실행 종료 확인’에서 대상과 근거를 기록한 뒤 retry하며 단순 heartbeat timeout만으로 실행을 복제하지 않는다.

Vast 패널은 같은 화면 안에서 offer 검색→가격/시간/보관 정책 확인→임대, 현재 allocation의 deadline/worker_ready/정지·삭제를 제공한다. 삭제 버튼은 외부 미보관 파일과 데이터 손실 여부를 먼저 보여준다.

### 7.3 결과: 확인→비교→다운로드

기본은 성공 실험만 표시하고 partial/실패 포함 토글을 제공한다. 공통 열은 best/final 지표·예산·시간·메모리·저장 상태이며 결과 JSON에 있는 새 metric은 ‘열 추가’에서 검색한다. 선택한 동일 단위 scalar는 비교 표로 보고 nested/배열은 generic tree 또는 원본 파일로 연다. Missing은 null, 미측정과0을 구분한다. W&B는 새 창 링크로 제공한다.

개별 파일은 report/ZIP/checkpoint/log 종류·크기·hash·검증 상태를 보고 받는다. 선택 다운로드는 `POST /exports`→진행 상태→ready 후 다운로드다. 선택 집합을 고정해 중간에 새 성공 결과가 끼지 않는다. API 접근이 가능한 Codex에는 read-only 토큰과 서버 주소만 주면 같은 목록/파일을 받아 분석할 수 있다.

### 7.4 설정: 연결·토큰·보안

토큰 화면은 이름/종류/만료/마지막 사용/상태 목록과 발급·만료 변경·폐기다. 발급 modal은 write/read-only, 이름, 만료일을 고르고 원문은 한 번만 표시한다. ‘Colab/Kaggle 연결’ 안내는 secret 이름·권한 부여 위치·모드별 설정 예시를 제공하며 원문을 포함한 코드를 로그나 다운로드에 저장하지 않는다. 서버 URL·연결/버전 검사도 여기서 제공한다.

소유자는 비밀번호 변경·다른 session 로그아웃·저장 기간·최대 임대 시간과 비용 정책을 관리한다. Read-only 접속에서는 비밀값 없는 정보만 보이고 변경은 불가능하다. 만료된 session은 로그인 화면으로 이동하되 편집 draft는 메모리에 보존하고 복귀한다. GPU의 실행 상태는 session과 독립적으로 서버에서 계속 관리한다.

### 7.5 같은 명령의 GUI와 CLI

```text
jobs submit experiment.json
jobs list --queue research
jobs clone job_123 --edit
queues pause research
workers drain worker_123
results collect --campaign long-budget --new
exports create --jobs job_123,job_456
tokens create --kind read-only --expires-at <UTC 시각>
```

내장 CLI는 허용한 관리 명령만 처리하고 shell·SSH·임의 Python 실행을 제공하지 않는다. 로컬 CLI는 실제 파일을 저장하고 브라우저는 file picker/JSON 입력과 download로 같은 동작을 수행한다. Token 발급 원문은 전용 일회 표시 UI로 처리한다. 명령·GUI 모두 검증·권한·상태 충돌·에러 메시지가 같으며 `--json`으로 기계 판독 결과를 받을 수 있다.

## 8. 자체 사용 시나리오 시뮬레이션과 요구 충족 결과

**문서의 요청·권한·상태·파일 전이를 작성자가 직접 따라간 사고 실험**이다. 서버/노트북을 실행한 검증이나 CPU 테스트가 아니며 보안·장애 내성을 실증한 결과로 표현하지 않는다.

| # | 사용 예시와 따라간 경로 | 설계상 판정 |
|---|---|---|
| 1 | 소유자 비밀번호 로그인→write 발급→Colab secret 연결→me 검사→upload 모드 실행→report 자동 전송 | 요구1/2/3/4 충족. Attached 등록으로 queue 중복 방지 |
| 2 | 같은 secret 이름을 Kaggle에 연결→queue 모드→서버 spec claim→학습·분석→verified artifact→finish | 요구1/2/5 충족. 플랫폼별 차이는 secret reader·실행 profile로 제한 |
| 3 | 서버 연결을 off로 두고 기존 notebook 실행·로컬 다운로드 | 기존 사용법 유지. 서버 장애/secret 누락의 영향 없음 |
| 4 | Read-only를 Codex/토큰 로그인에 제공→결과 조회·export·다운로드→claim/토큰 발급 시도 | 읽기 허용, 변경403. GUI와 API 모두 같은 권한 |
| 5 | 실행 중 write 만료/폐기→heartbeat401→로컬 저장·중단→새 secret→기존 worker/attempt 재연결 | 권한 우회 없음. 마지막 외부 checkpoint/전송 대기 기준 복구 |
| 6 | GUI에서 M/seed 묶음 검증→한 job 오류→수정 후 batch→queue pause/resume·queued 우선순위 변경 | 일괄 등록 원자성·새 시작 제어·실행 중 spec 불변 충족 |
| 7 | 네트워크 단절→로컬 전송 대기→URL 만료→재인증·같은 upload ID 갱신→finish 응답 재시도 | 파일 재학습 불필요, verified 이전 완료 표시 금지, 중복 공개 방지 |
| 8 | 새 summary 필드 `memory.retention_auc` 도착→열 추가·JSON tree·묶음 다운로드 | 요구6 충족. 원본 보존, DB/API 변경 불필요 |
| 9 | 새 실행 parameter 추가→새 schema/code 등록→구형 worker만 존재→새 worker 연결 | 미지원 상태로 대기하다 호환 worker만 claim. 필드 묵살 없음 |
| 10 | Posthoc 실패→partial report 표시→checkpoint로 분석만 retry→선택 ZIP export | 학습 결과 보존, 최종 성공/부분 결과 구분, 요구5 충족 |
| 11 | Heartbeat 만료 후 다른 worker retry 시도→종료 불명→기존 종료 확인 후 새 세대 발급 | 동시 중복 실행 방지. 구세대 완료 갱신 거부 |
| 12 | Vast 생성 timeout→operation unknown 조사→정상 연결→빈 큐→외부 보관 확인→stop/destroy | 중복 임대·사용자 다운로드 대기 과금 방지 |
| 13 | 토큰 폐기 직후 이미 발급된 다운로드 URL 사용→5분 이내 잔여 유효 | 즉시 회수 보장하지 않음을 명시. API 접근 차단과 URL 만료 분리 |
| 14 | GUI/내장 CLI/로컬 CLI에서 같은 cancel·읽기 요청→권한/현재 상태 비교 | 동일 API·오류/상태 의미, 숨은 GUI 전용 조작 없음 |
| 15 | 브라우저 session 만료→다시 비밀번호 로그인→토큰 만료 변경·queue 재개 | 서버의 학습은 session 수명과 분리. 화면 조작 모두 대응 API 존재 |
| 16 | Finish202 뒤 실행기 종료·토큰 만료→서버 파일 검증 완료→새 인증으로 조회 | Finalizing을 stale로 재배정하지 않고 완료를 보존 |

점검 중 보완한 구조적 누락은 **두 토큰 권한과 로그인 session의 구분, attached 실행의 원자 등록, 토큰 교체/셀 재실행의 중복 방지, 신규 실행 필드와 단순 결과 필드의 구분, 선택 export와 비동기 완료 경로, 폐기 후 signed URL 잔여 유효기간**이다. 여섯 요구사항은16개 시나리오에서 모두 대응 경로를 갖는다. 미실증 항목은 실제 secret 접근, 안전 중단·checkpoint 성능, 네트워크 장애/동시성, AWS/Vast 청구·정지 동작이며 후속 구현 검증에 남긴다.

## 9. 구현 경계와 문서의 적용

먼저 공통 client·schema·token/session·job/attempt/artifact 계약을 구현하고, 기존 helper에 선택 연결 hook을 추가한 뒤 GUI/CLI와 공급자 adapter를 연결한다. 연구 모델·trainer에 서버/환경별 분기를 흩뿌리지 않는다. 서버 배포·실제 토큰 발급·기존 notebook 수정은 별도 구현 작업이다.

장기적으로 고정할 것은 인증 두 종류, 불변 spec/schema, claim/lease, 상태·완료 검증, artifact ID/hash다. 새 연구는 schema/runner와 JSON 데이터로 확장하며 불필요한 API 버전 변경을 피한다. 다중 사용자·분산 scheduler·자동 모델 선택은 최초 범위에 넣지 않는다.

이번 변경은 본 설계서와 문서 안내만 포함한다. 구현·CPU/GPU 테스트·실제 로그인/토큰 발급·배포·유료 임대는 수행하지 않았다. 자체 점검은 사용 경로의 논리 검토이며 실행 검증과 구분한다.
