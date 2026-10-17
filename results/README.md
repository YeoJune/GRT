# 실험 결과 원본

[L150 용량 비교](../docs/experiments/20261012_small_register_sweep.md)의 학습5개·확장 사후3개와 [L75 write gate 비교](../docs/experiments/20261015_l75_write_gates.md)의 dropout6개+M20 bias−2/+2 학습2개·확장 사후8개를 보관한다. [L75/M20 Refine 비교](../docs/experiments/20261017_l75_refine_gate.md)의 on 학습·사후 각1개를 추가해 총 학습 JSON14개·사후 JSON12개다. Off는 기존 bias0 결과를 재사용한다. L75는 gate 포화·표현·그룹 개입·retention까지 같은 실험 문서에 반영했다. 실행과 파일 전달은 [공통 workflow](../docs/20261011_experiment_workflow.md#posthoc-colab)를 따른다.

| 경로 / 형식 | 내용 |
|---|---|
| `<report_id>.json` · `grt.experiment_report/1` | 실험별 학습 Config·metadata·학습 곡선·best validation/RTLA·비용 |
| `posthoc/<report_id>.json` · `grt.posthoc_report/1` | checkpoint hash/step·분석 조건·gate/logit·표현·slot/group/retention·분석 배열 |

사용자가 보낸 JSON의 bytes와 숫자를 바꾸지 않는다. 같은 run·checkpoint·조건의 재보고는 ID와 step을 확인해 최신 기록으로 반영하고, 다른 실험·조건은 해당 실험 문서에 구분한다. 미완료·OOM·미측정을 완료나0으로 채우지 않는다. CSV로 여러 환경의 결과를 누적하지 않는다.

JSON에는 checkpoint 가중치 본체·optimizer·원본 token stream이 없다. 학습 재개와 추가 분석용 checkpoint·prepared 데이터는 실행 환경 밖에 별도로 보관한다. 실험 판단·수치 표는 실험 문서 한 곳에서 관리한다.
