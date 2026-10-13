# 받은 실험 결과

사용자가 Kaggle/Colab의 `job.report(download=True)`에서 받은 `experiment_report.json`을 보내면 이 폴더에 원본을 보관한다. 현재 B0 NeoX baseline의 첫 완료 보고서를 보관했다.

- 저장 이름: `<report_id>.json`. JSON 안의 `run_name`, 환경, 모델/설정으로 식별한다.
- 같은 report_id를 다시 받으면 재개·재보고 여부와 step을 확인하고 기존 기록을 갱신한다. 다른 report_id는 별도 결과로 보존한다.
- [실험 계획과 결과표](../docs/20261012_experiment_plan.md)의 해당 행에 best validation/RTLA, 하드웨어·시간·VRAM, 상태와 원본 링크를 적는다.
- 같은 checkpoint의 품질·RTLA를 묶고 GPU/precision/global batch·target 예산·데이터 hash를 확인해 비교한다. 미측정값은 빈칸으로 둔다.
- 실행 미완료/실패는 완료 결과와 구분하고, 보낸 파일의 숫자를 변경하지 않는다. 재개 후 이전 파일을 교체할 때 step이 뒤로 가지 않는지 확인한다.

보고서에는 최종 설정·환경·학습 곡선·RTLA 요약·경고가 들어 있다. checkpoint 가중치, 원본 데이터, NPZ 배열과 그림은 들어 있지 않으므로 실제 재개/추가 분석용 run 폴더는 실행 환경에서 별도로 보관한다.

| 실험 | 환경 / 상태 | 원본 |
|---|---|---|
| B0 · NeoX small/segment150/10k | Kaggle T4×2 · 완료, val PPL 37.9590 | [0f4be0ee5d5c4a77b8ce50d591932c69.json](0f4be0ee5d5c4a77b8ce50d591932c69.json) |
