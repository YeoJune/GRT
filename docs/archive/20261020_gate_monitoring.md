# 게이트 안정화 단계의 모니터링 이력

2026-10-20 종료 기록. 현재 실행은 [공통 workflow](20261023_experiment_workflow.md#rtla-학습-중-모니터링과-사후-분석)를 따른다. 이 문서는 과거 측정 정의만 보존하며 코드·config·노트북 복사본은 두지 않는다.

안정화 탐색에서는 training LM/total loss와 raw/weighted penalty 외에 write logit의 절댓값 평균·최대, penalty threshold 초과 비율, abs>20 비율, raw write 평균, sigmoid slope, write MLP hidden norm을 수집했다. Hidden norm은GELU 뒤·출력 Linear 직전 벡터의L2 norm이었다. 이 정의와 실제 수치는 [완료 보고서](../experiments/20261018_l75_gate_stabilization.md)와 수신 원본에 남아 있다.

당시 W&B에는 training·validation·RTLA payload 전체와 중첩된 최종 summary를 전송했다. 노트북도 summary 전체를 출력해 학습 품질·비용·개입 지표와 실험 전용 진단이 섞였다.

현재 코드는 네 그룹의 핵심 W&B 지표와 best/final·비용·사후 요약만 기본 화면에 표시한다. 안정화 전용 training logit 평균/최대·threshold 초과율·slope·hidden norm 계산과 전용 최대값 집계 경로는 제거했다. LM·tail penalty 수식과 dropout 동작은 유지하며, 상세 gate/logit·표현·그룹 분석은 사후 경로에서 계속 지원한다. 과거 코드 실행 옵션·호환 분기는 유지하지 않는다.
