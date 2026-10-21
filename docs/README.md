# 현재 문서

| 역할 | 문서 |
|---|---|
| 모델·loss·BPTT 계약 | [현재 구현 명세](20261011_current_spec.md) |
| 설정·학습·사후 분석·파일 전달 | [실험 workflow](20261011_experiment_workflow.md) |
| 원격 GPU 임대·순차 큐·완료 결과 수집·비용/ETA 관리 설계 제안 | [원격 GPU 큐 설계 보고서](20261021_remote_gpu_queue_design.md) |
| Gate 안정화의 선택·결과·진행 상태 | [패널티·dropout, 이전 시도](experiments/20261018_l75_gate_stabilization.md) |
| L75/M20 gate 경로·M 확장: M20 최종 선택 candidate·기본값 반영, 다른 M 검증 예정 | [Write gate 정보 경로 비교](experiments/20261019_l75_gate_paths.md) |
| L150 baseline·레지스터 용량 | [small 용량 비교](experiments/20261012_small_register_sweep.md) |
| 수신 원본 보관 | [results/README](../results/README.md) |

실험 주제마다 결과 문서 하나를 유지한다. 공통 지표·그림은 [안정화 보고서](experiments/20261018_l75_gate_stabilization.md#2-공통-비교-기준)의 정의를 따른다. 새 결과는 같은 문서에 반영하고 과거 계획·오류 수정·검증 기록을 최신 실행 안내에 누적하지 않는다.

이전 설계와 정리 전 상세 기록은 [아카이브 목록](archive/README.md)에 있다. 실행 시작점은 [저장소 README](../README.md), [Kaggle](../notebooks/kaggle.ipynb), [Colab](../notebooks/colab.ipynb), [사후 분석](../notebooks/colab_analysis.ipynb)이다.
