# 실험 결과 원본

[L150 용량 비교](../docs/experiments/20261012_small_register_sweep.md)의 학습5개·확장 사후3개와 [L75 gate 안정화](../docs/experiments/20261018_l75_gate_stabilization.md)의 dropout6개·M20 bias2개·Refine/Uniform2개·logit tail5개·A/dropout 결합3개의 학습/사후 원본을 보관한다. [L75 gate 경로 비교](../docs/experiments/20261019_l75_gate_paths.md)의40k M20 세 경로와 M8 independent/candidate 학습·사후 각5개도 보관한다. 총 학습 JSON28개·사후 JSON26개이며, 같은 대조군은 중복 저장하지 않는다. 완료된 시도의 조건·결과·판단은 링크한 실험별 문서에 보존하며, 새 결과와 판단도 같은 실험 문서에 반영한다. 실행과 파일 전달은 [공통 workflow](../docs/workflow.md#posthoc-colab)를 따른다.

| 경로 / 형식 | 내용 |
|---|---|
| `<report_id>.json` · `grt.experiment_report/1` | 실험별 학습 Config·metadata·학습 곡선·best validation/RTLA·비용 |
| `posthoc/<report_id>.json` · `grt.posthoc_report/1` | checkpoint hash/step·분석 조건·gate/logit·표현·slot/group/retention·분석 배열 |

사용자가 보낸 JSON의 bytes와 숫자를 바꾸지 않는다. 같은 run·checkpoint·조건의 재보고는 ID와 step을 확인해 최신 기록으로 반영하고, 다른 실험·조건은 해당 실험 문서에서 구분한다. 미완료·OOM·미측정을 완료나0으로 채우지 않는다. CSV로 여러 환경의 결과를 누적하지 않는다.

JSON에는 checkpoint 가중치 본체·optimizer·원본 token stream이 없다. 학습 재개와 추가 분석용 checkpoint·prepared 데이터는 실행 환경 밖에 별도로 보관한다. 실험 판단·수치 표·출처는 [실험별 문서](../docs/experiments/README.md)에 유지하고, 공통 선택과 지표 정의는 [연구 요약](../docs/research.md)에서 관리한다.
