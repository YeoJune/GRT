# 실험 기록

실험 주제별 하나의 문서에 질문·조건·원본 출처·결과·판단을 누적한다. 완료 후에도 이 폴더에 유지하며 아카이브로 옮기지 않는다. 새 예산·seed·후속 결과는 같은 질문이면 기존 문서에 추가하고, 별개의 비교 질문일 때 새 문서를 만든다. 문서 이름의 날짜는 주제 시작일이며 갱신일과 현재 상태는 본문에 표시한다.

| 주제 | 기록 | 상태·범위 |
|---|---|---|
| L150 baseline·레지스터 용량 | [용량 비교](20261012_small_register_sweep.md) | 수신 학습·사후 결과와 비교 한계 |
| L75 gate 안정화 | [안정화 탐색](20261018_l75_gate_stabilization.md) | 탐색 완료, 패널티 A·dropout0 선택 |
| L75/M20 write gate 정보 경로 | [경로 비교](20261019_l75_gate_paths.md) | M20 candidate 선택; M8/M20 세 경로·M75 independent/candidate40k 수신, M75 separate_head 실험 중, 추가 seed 미검증 |

공통 선택·지표는 [연구 요약](../research.md), 최신 실행은 [workflow](../workflow.md), 원본 JSON 보존은 [results](../../results/README.md)를 따른다. 이전 구현의 계획과 일회성 CPU 기능 검증은 [아카이브](../archive/README.md)에 남긴다. 실험별 문서는 당시 조건과 판단 근거를 보존하며, 과거 실행 명령을 현재 사용법으로 안내하지 않는다.
