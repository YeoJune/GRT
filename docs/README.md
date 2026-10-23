# 문서

현재 문서는 역할별로 한 곳에서 갱신한다. 실험 기록은 완료 후에도 [실험 폴더](experiments/README.md)에 주제별 하나씩 유지한다. 이전 설계·계획·일회성 검증은 [아카이브](archive/README.md)에 보존한다.

| 역할 | 문서 |
|---|---|
| 모델·loss·BPTT·checkpoint 계약 | [모델 명세](specification.md) |
| 설정·데이터·학습·모니터링·사후 분석 | [실행 workflow](workflow.md) |
| off/upload/queue·인증·전송·안전 중단 | [GRT Server 연결](server.md) |
| 현재 연구 선택·공통 지표 정의 | [연구 요약](research.md) |
| 실험별 질문·조건·결과·판단 | [실험 목록](experiments/README.md) |
| 원본 JSON 보존·출처 | [results](../results/README.md) |

실행 시작점은 [저장소 README](../README.md), [Colab](../notebooks/colab.ipynb), [Kaggle](../notebooks/kaggle.ipynb), [사후 분석](../notebooks/colab_analysis.ipynb)이다. 현재 문서에는 최신 계약·사용법만 유지하고 과거 코드·호환 분기는 보관하지 않는다.
