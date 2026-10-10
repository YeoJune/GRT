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


## 5,000-update 미수렴 결과와 구조 진단

사용자 실행에서 RMT best validation CE=6.9298, GRT=6.1831이었다.
RMT는 균등 content-token 예측의 CE=ln(1020)=6.9276에 가깝다.
이 결과는 기존 프로필의 수렴 적합성을 뒷받침하지 않는다.
근거 없이 배치와 학습량을 늘린 후속 후보는 철회했으며 기존 설정은 비교 기록으로 유지한다.

### 용량과 접근 구조

- Copy payload의 엔트로피는 20×log2(1020)≈199.9 bit다.
- 메모리는 32×256=8,192 실수 좌표다. 정답의 단순 one-hot 표현도
  20×1020=20,400 좌표가 필요하지만, 토큰을 10-bit 이진 코드로 표현하면 200 bit로 충분하다.
  따라서 one-hot 크기를 근거로 메모리 부족을 단정할 수 없다.
  반대로 실수 좌표 수만으로 접근·순서 복원·학습 가능성을 보장할 수도 없다.
- GRT는 S0=0이고 ALU에 슬롯별 position/ID 표현이 없다. dropout=0인 첫 세그먼트에서
  모든 memory query가 동일하므로 candidate C의 모든 행이 같다.
  write-back은 S1[m]=W[m]×C이므로 첫 상태 행렬 rank≤1이다.
  이는 32개의 독립된 초기 저장 주소와 다르다. 이후 상태의 rank가 항상 1이라는 뜻은 아니다.
- RMT의 mem0는 슬롯마다 다르게 초기화되므로 같은 정확한 대칭은 없지만,
  seed 1234 초기 모델의 첫 write-memory stable rank는 약 1.013이었다.
  stable rank=||S||F²/||S||2²이며 공통 성분에 민감하다. 이것만으로 실제 정보량을 확정하지 않는다.

### 실제 small 모델 CPU 진단

표준 Copy T=4, 실제 small 모델, train sample ID 0/1의 고정 배치 2,
seed 1234, dropout=0, AdamW betas=(0.9,0.95), weight decay=0.01,
clip=1, 일정 LR로 20 update를 수행했다. 정규 학습 프로필이나 일반화 실험과 구분한다.

| 모델 | LR | 원본 배치 CE / 정확도 | 첫 target 20개 제거 후 정확도 | target 유지·noise 교체 후 정확도 |
|---|---:|---|---:|---:|
| RMT | 3e-4 | 2.470 / 32.5% | 32.5% | 5.0% |
| RMT | 1e-3 | 0.212 / 100% | 100% | 30.0% |
| GRT | 3e-4 | 1.967 / 92.5% | 95.0% | 50.0% |
| GRT | 1e-3 | 0.258 / 100% | 100% | 55.0% |

실제 모델이 작은 배치를 최적화할 수 있음은 확인했지만,
target 제거에도 예측을 유지하므로 이 성공을 Copy 메모리 습득의 증거로 사용할 수 없다.
noise 또는 출력 위치를 이용한 고정 샘플 암기와 구분해야 한다.
최종 loss에서 첫 세그먼트 embedding으로 전달되는 gradient는 두 모델 모두 0이 아니었다.
이는 경로가 연결된다는 확인이며, 올바른 정보가 저장되거나 학습된다는 증거는 아니다.

### 후속 판단

모델 크기 부족, 학습 불가능, 또는 특정 LR에서 수렴 가능하다는 결론은 아직 없다.
LR=1e-3은 고정 샘플 최적화를 가속했지만 온라인 랜덤 샘플 수렴 효과는 검증하지 않았다.
장시간 예산을 다시 늘리기 전에, 다양한 payload/noise에서 target 교체에 출력이 반응하는지와
slot 구분·memory 접근을 검증해야 한다. slot ID 추가나 초기화 변경은 설정 조정과 달리
모델 설계 변경이므로 별도 비교 조건으로 다룬다. 기존 실험을 새 구조 결과와 혼합하지 않는다.


## 세그먼트 간 학습 정체: 제어 실험

### 재현 조건

고정 샘플 암기를 피하기 위해 매 update 새로운 train sample ID 16개를 사용하고,
독립 validation namespace의 64개 샘플에서 평가했다. 모델은 CPU 진단용으로
D=64, 1층, 4 heads, FF=128, N=128, M=32, V=1024를 사용했다.
seed=1234, dropout=0, AdamW LR=1e-3, betas=(0.9,0.95), weight decay=0,
clip=1, 일정 LR, 300 updates(4,800 train samples)로 조건을 맞췄다.
이는 표준 small/GPU 실험의 재실행이 아니며 아래 변경 과제들은 표준 벤치마크 결과가 아니다.

### 결과

| 조건 | RMT validation CE / token accuracy | GRT validation CE / token accuracy |
|---|---|---|
| 원래 Copy, T=4 | 6.9028 / 0.234% | 6.9315 / 0% |
| source와 OUT를 같은 단일 세그먼트에 배치 | 0.00543 / 100% | 0.00290 / 100% |
| T=4 유지, content를 8종으로 축소 | 2.0881 / 10.39% | 2.0871 / 10.39% |
| T=4, content 8종, 정답 위치 1개만 감독 | 2.0902 / 10.94% | 2.0974 / 10.94% |
| 위치 임베딩 초기 크기 ×10 | 6.9340 / 0% | 6.9316 / 0% |
| 매 세그먼트 메모리에 learnable 슬롯 ID 주입 | 6.9315 / 0.078% | 6.9323 / 0% |
| 매 세그먼트 종료 시 oracle memory 주입 | 0.00628 / 100% | 4.0372 / 20.47% |
| 중간 noise 세그먼트의 메모리 갱신 차단 | 6.6854 / 0.469% | 6.6865 / 0.391% |

oracle memory는 labels를 읽지 않고 원래 source input 20개의 embedding+position을
메모리 첫 20슬롯에 직접 배치하고 나머지를 0으로 채운다. GRT는 이 개입에서
W=1로 강제한다. 이 조건은 학습된 저장 경로를 우회하는 진단이며 실제 모델 성능이 아니다.
중간 갱신 차단은 최초 write와 최종 read는 학습하게 두되 두 noise 세그먼트에서
이전 memory를 그대로 넘기는 별도 개입이다.

RMT에 oracle memory를 첫 세그먼트에서만 주입하고 이후 정상 갱신을 허용한 추가 조건은
CE=6.8270, 정확도=0.234%였다. 이 조건에서도 중간 갱신을 거친 정보 유지가 학습되지 않았다.
GRT의 첫 세그먼트 oracle 추가 조건은 모든 W=1을 함께 강제하므로 독립적인
'초기 write만 수정' 비교로 해석하지 않는다.

### 배제한 문제와 좁힌 실패 구간

- train/validation/test 총 300개 샘플의 source/labels/OUT 위치를 직접 대조했고,
  source에서 만든 oracle logits의 masked CE가 거의 0임을 확인했다.
  데이터 정답 불일치나 label shift 오류는 이 검사에서 발견하지 못했다.
- 같은 모델과 optimizer가 새로운 샘플의 같은 세그먼트 Copy는 해결했다.
  따라서 일반적인 token 분류·출력 위치 구분·optimizer 미작동만으로 정체를 설명할 수 없다.
- 8종 vocabulary, 정답 1개에서도 장거리 조건은 균등 예측 수준이다.
  단순 어휘 수나 20개 정답의 총 정보량만을 줄이는 것으로 해결되지 않았다.
- 정답을 계속 memory에 공급하면 RMT는 회복된다. 반면 최초 oracle write만으로는
  회복되지 않았고, 중간 갱신 차단만으로도 회복되지 않았다.
  따라서 RMT의 실패 구간은 최종 decoder 자체보다 학습된 source→memory encoding과
  noise를 통과하는 retention의 결합에 있다. 갱신 차단만으로 고쳐진다고 주장하지 않는다.
- GRT도 memory 공급으로 개선되지만 완전 회복하지 않는다. 저장/유지 외에
  memory→출력의 주소 구분 학습도 추가 병목 후보로 남는다.
- 초기 슬롯 대칭은 실제 존재하지만 슬롯 ID/position 크기 개입만으로 회복하지 않았다.
  따라서 앞서 관측한 rank=1을 단독 원인이나 해결책으로 확정하지 않는다.

### 현재 결론과 한계

관측된 균등 예측 정체는 '메모리 수치 용량이 부족하다'보다,
최종 위치의 CE만으로 write→retain→ordered read를 동시에 배우는 최적화 병목과 부합한다.
다만 어느 단일 파라미터나 코드 줄이 사용자 5,000-step checkpoint의 실패를 일으켰는지는
아직 확정하지 않았다. 작은 모델의 단일 seed/300-update 개입 결과를 실제 small/T4
checkpoint의 내부 상태 진단이나 모든 예산에서의 학습 불가능 증명으로 표현하지 않는다.
실제 checkpoint 없이 trained attention/gate/slot collapse를 확인했다고 주장하지 않는다.

후속 수정은 저장 경로와 유지 경로를 분리해서 검증해야 한다. 예산만 늘리거나
슬롯 ID 하나를 추가한 모델을 검증된 해결책으로 제시하지 않는다.
길이 curriculum이나 memory 저장 보조 감독은 비교할 후보지만, 기존 과제/학습 계약과
다른 조건이므로 명시적인 실험 변경으로 다룬다.

원본 진단 코드와 JSON 결과는 ignored `runs/cpu_memory_diagnosis/`에 보존했다.
이 CPU 진단 작업에서는 production 모델·CLI·표준 과제를 변경하지 않았다.
후속 공개 Copy 프로필의 변경 사항은 `RMT_PAPER_COPY.md`에 별도로 기록했다.

### 원본 baseline 비교에 따른 범위 정정

2022년 공식 구현은 `booydar/LM-RMT`이며, 현재 RMT와 위치 attention·정규화·
attention mask·Copy 생성 방식이 다르다. 비교 근거와 공개 실험 조건은
[RMT_BASELINE_REVIEW.md](RMT_BASELINE_REVIEW.md)에 정리했다.
위 진단은 현재 RMT 변형에 관한 결과이며 원본 논문 baseline 자체의 실패를 뜻하지 않는다.
원본도 초기 memory 슬롯을 동일하게 만들지만 상대 위치 attention으로 슬롯을 구분한다.
따라서 초기 대칭만으로 실패 원인을 설명할 수 없다. 후속 작업에서는 예산 증가보다
baseline 정의와 원본 대비 구조·과제 차이를 먼저 바로잡는다.
