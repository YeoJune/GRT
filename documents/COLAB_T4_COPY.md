# Colab T4 Copy 초기 수렴 프로필

실행 진입점: `notebooks/colab_t4_copy.ipynb`.
병합 순서: `base.yaml`, `rmt.yaml` 또는 `grt.yaml`, `copy.yaml`, `colab_t4_copy.yaml`.

## 선택 근거

표준 small 백본(4층, 4 heads, D=256, FF=1024)과 V=1024/N=128/M=32/Copy T=4를 유지한다.
microbatch 4와 accumulation 2로 effective batch 8을 유지하며 두 모델에 동일 설정을 적용한다.
FP32를 선택해 초기 수렴 확인에서 FP16 overflow 변수를 분리한다.
실제 T4 peak memory와 속도는 아직 측정하지 않았으므로 메모리 적합성이나 완료 시간을 보장하지 않는다.

5,000 update는 40,000개의 서로 다른 결정적 train sample, 총 800,000개의 정답 토큰이다.
LR=3e-4, warmup 200 후 cosine을 적용한다. 2-step smoke보다 충분히 긴 추세를 관찰하기 위한
초기 예산이며, 특정 수렴 step에 대한 실험 근거가 있거나 목표 달성을 보장하는 값은 아니다.
검증 표본 256개를 매 250 update 평가하고, best checkpoint를 길이별 test 256개에서 평가한다.
축소된 표본과 예산의 pilot 결과를 표준 전체 실험 결과와 혼합하지 않는다.

## 코랩 저장과 재개

노트북은 기본적으로 Drive에 모델별 run을 저장하고 500 update마다 주기 checkpoint와
`last.pt`를 갱신한다. `best.pt`가 더 최신일 수 있어 재개 지점은 세 종류 중 가장 최신 step이다.
재개 전에 모델·데이터·학습·분석 설정이 현재 profile과 동일한지 검사한다.
run 경로만 현재 위치로 치환하며 다른 설정은 변경하지 않는다.
완료한 run과 `evaluation.json`이 있으면 학습/평가를 반복하지 않는다.

Drive I/O는 벽시계 시간에 영향을 준다. 두 모델의 추론 성능은 별도의 timed region에서 측정한다.
RMT/GRT는 서로 다른 프로세스로 실행하므로 한 모델이 사용하는 GPU 메모리를 다른 모델이 유지하지 않는다.

## 판정과 후속 작업

- 기본 길이 validation CE≤0.05에 최초 도달한 step을 기록한다. 자동 조기 종료하지 않는다.
- 미달이면 CE/정확도의 개선 추세와 최근 LR/gradient, NaN/overflow를 먼저 확인한다.
- 기본 길이에서 학습이 안 된 상태의 확장 결과를 장기 기억 능력의 증거로 해석하지 않는다.
- OOM이면 두 모델 모두 새 run에서 microbatch 2 × accumulation 4를 사용하고 평가 batch를 동일하게 낮춘다.
- 5,000 update 이후의 추가 학습이나 precision/LR 변경은 새 config/run으로 기록한다.

Colab GPU 자원 및 runtime 제한은 동적이므로 한 세션 내 완료를 가정하지 않는다.
관련 공식 자료: [Colab FAQ](https://research.google.com/colaboratory/faq.html),
[NVIDIA T4 GPU](https://developer.nvidia.com/blog/nvidia-t4-gpus-now-available-on-google-cloud/).

## 검증 상태

사용자가 기존 Colab smoke 실행에서 60개 CPU 테스트 통과와 두 모델의 학습/평가 실행을 보고했다.
그 결과는 2-step smoke에 대한 것이며 이 5,000-update 프로필의 수렴을 검증하지 않는다.
이번 변경의 주기 `last.pt` 보존 회귀 테스트를 추가했지만 로컬에서는 실행하지 않았다.
새 프로필의 학습·평가는 사용자가 Colab에서 실행하고 `comparison.json`, `convergence.png`,
각 모델의 `metrics.jsonl`을 공유하여 확인한다.
