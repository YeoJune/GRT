# 이전 Remember Kaggle 운영 기록

현재 실행 방법은 [실험 workflow](../20261011_experiment_workflow.md)를 따른다. 아래는 당시 구성에 관한 문서 기록이며 실행 노트북은 보존하지 않는다.

이전 노트북은 Remember curriculum을 두 GPU에서 실행했다. 모델별 사전 측정으로 배치를 조정하고, 두 모델의 공통 curriculum prefix를 세션 시간 예산 안에서 선택했다. 별도 watchdog이 자식 프로세스를 종료하고 모델별 사용 시간을 기록했다. 이후 WikiText 전용 노트북과 공통 utils로 실행 경로가 바뀌었다.

현재는 모델·크기·과제·업데이트 예산·환경을 단일 카탈로그에서 선택한다. archive에 있던 실행 노트북과 구현 zip은 제거했으며 과거 YAML 경로·예산 자동계획·wrapper 실행 기능을 지원하지 않는다.
