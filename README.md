# GRT

[2026-09-16 명세](documents/20260916_spec.md)에 따른 합성 메모리 벤치마크입니다.
GRT와 독립 RMT를 Copy, Reverse, Associative Retrieval (`passkey`)에서 비교합니다.
각 모델은 양방향 세그먼트 attention으로 마지막 세그먼트의 정답 위치를 병렬 복원합니다.
현재 RMT 검증은 ARMT 논문의 Associative Retrieval Remember(`paper_ar`)로 진행합니다.
기존 `paper_copy`는 2022 LM-RMT 기반의 별도 Copy 과제로 보존합니다.

## 설치

Python 3.10 이상 및 PyTorch 2.2 이상을 사용합니다. GPU 실험 환경의 CUDA에 맞는
PyTorch를 먼저 설치하고 다음 명령을 실행하세요.

```bash
python -m pip install -e '.[dev]'
# 선택: W&B를 사용할 때만
python -m pip install -e '.[wandb]'
python -m pytest -q
```

## Kaggle / Colab 점검

간단한 진입점은 [notebooks/smoke_test.ipynb](notebooks/smoke_test.ipynb)입니다.
현재 수정본을 `/kaggle/working/GRT` 또는 `/content/GRT`에 업로드·압축 해제하거나,
수정본이 반영된 저장소를 clone하세요. 노트북 첫 코드 셀의 `REPO_DIR`를 실제 경로로
맞추고 위에서부터 실행합니다. Kaggle은 인터넷을 켜고 GPU accelerator를 선택하세요.
Colab은 런타임 유형에서 GPU를 선택하세요. W&B 계정은 필요하지 않습니다.

노트북은 CPU 계약 테스트, 표준 small GRT/RMT의 2-step 학습, 세 길이 평가,
체크포인트 재개 CLI, GRT NPZ/PNG 분석을 실행합니다. `TASK`를 `reverse` 또는
`passkey`로 바꾸어 다른 과제를 점검할 수 있습니다. 표준 모델·과제 규격은 유지하며
학습 예산과 평가 표본 수만 줄입니다. 수렴과 zero-shot 일반화 실험은 별도로 수행해야 합니다.
재개 후 다음 update가 같은지는 dropout CPU fixture 테스트가 확인합니다.

결과와 `pytest_output.txt`, `pytest.xml`을 `runs/notebook_<timestamp>/`에 저장하고
마지막 셀에서 ZIP을 만듭니다. 실패하면 해당 로그를 공유하세요.

## 원본 RMT 기준 경로와 CPU 검사

현재 기준 경로는 `configs/rmt_author_reference.yaml`과 `src/grt/reference/`입니다.
저자 저장소의 논문 시점 commit `24cbb9`에서 backbone, memory cell, recurrent wrapper를 가져왔고,
데이터 생성과 collator의 함수 본문도 보존했습니다. 원본의 raw `labels`/`labels_mask`를 넘겨
wrapper 내부 loss를 직접 backward합니다. 기존 logits-only adapter와 shifted labels는 사용하지 않습니다.
출처와 변경 내역은 [third_party/NOTICE.md](third_party/NOTICE.md), 파일별 hash는
`src/grt/reference/SOURCE.json`에 있습니다. 2024년 생성된 모델 JSON은 공개본에 없어
세부 backbone 설정은 이후 공개된 저자 설정 생성기를 기준으로 합니다.

CPU에서 실행·단계 전환·재개를 확인하는 짧은 명령:

```bash
python scripts/train.py --config configs/rmt_author_reference.yaml configs/rmt_author_cpu_smoke.yaml --output-dir runs/author-cpu-smoke --device cpu
python -m pytest -q tests/test_author_reference.py
```

동일 명령과 출력 경로로 재개합니다. 이전 실험의 checkpoint와 형식이 다르므로 새 run을 사용합니다.
원본 기준 YAML의 2,000/10,000 update는 원본의 첫 두 curriculum 수준입니다.
짧은 Colab 실험은 아래 T4 override를 사용합니다.
native runner는 원본 길이 샘플링 검증과 고정 최대 길이 검증을 함께 기록합니다.
GPU 실행 시 batch512 × accumulation1은 기존2쌍 실측 약2.49GiB를 기준으로 한 원본 effective batch이며,
native 경로의 T4 B1024 실측은2쌍 평균0.529초/update, peak4.83GiB였습니다.

별도의 짧은 수렴 진단은 다음 명령입니다. 원본 생성기로 만든32개 문맥을 학습하고,
각 문맥의 두 키를 모두 질의한64개 답을 실제 생성해 확인합니다.
학습 배치는 원본 collator의 길이·질의 샘플링을 사용합니다.

```bash
python scripts/check_rmt_reference_cpu.py --size reference --max-steps 1000 --output-dir runs/author-cpu-convergence
```

이 검사는 학습 집합 내 수렴 검사이며 일반화·논문 재현 검사가 아닙니다. 미수렴 시 결과를 저장하고 exit1을 반환합니다.
원본 크기(D128/L4/memory32)의 CPU 검사에서800 update에64개 생성 답이 모두 정답이었고, loss는0.0539였습니다.
축소 CPU 모델(D32/L2/memory4)도1,800 update에64개 모두 정답, loss0.0899로 수렴했습니다.
별도 `--fixture contrast`는 두 값의 집합을 고정하고 연결·순서만 교환하는8예제 검사입니다.
이 검사에서는 원본 크기 모델이600 update 후5/8 정답에 머물렀습니다. 모든 작은 과제가 수렴했다고 주장하지 않습니다.
현재 확인한 것은 원본 데이터·모듈·loss 경로의 작은 학습 집합 내 수렴이며, 전체 데이터 일반화는 미검증입니다.

## Colab T4: 원본 RMT의 1 → 2쌍 검사

수정본을 pull한 뒤 [colab_rmt_paper_remember.ipynb](notebooks/colab_rmt_paper_remember.ipynb)를 실행합니다.
`rmt_author_reference.yaml` + `rmt_author_t4.yaml`을 기존 `scripts/train.py`로 실행하며,
Drive의 새 `rmt_author_t4_b2048_samples_seed54` 경로에 저장합니다. 같은 설정·경로로 실행하면 자동 재개합니다.

microbatch2,048 × accumulation1, 1쌍 최대500 → 2쌍 최대2,500 update입니다.
원본 B512의2,000/10,000 update와 같은 명목 샘플 노출 예산이며,
train1,000,000 / validation1,000 / test10,000의 원본 random 데이터 규모를 사용합니다.
epoch 마지막의 작은 배치와 조기 종료로 실제 노출량은 최대 예산보다 적을 수 있습니다.
생성 exact match≥99%일 때 다음 단계로 진행하고, 고정 길이 정확도로 best를 선택합니다.
원본의 무작위 길이 검증도 기록합니다. 배치와 update/scheduler 길이는 원본과 다릅니다.

native T4 B1024 실측0.529초/update, peak4.83GiB를 기준으로 B2048 peak는 약9.65GiB,
같은 처리량이면 최대 학습 약51분이며 평가·저장 시간이 추가됩니다. B2048 속도·VRAM은 추정입니다.
LR3e-4는 유지하며 warmup/scheduler horizon도 update 예산에 맞춰 조정합니다.

Drive에서 각 stage의 `metrics.txt`로 진행을 확인하고, 최종 `summary.txt`, `evaluation.txt`를 읽습니다.
분석용 JSON/JSONL도 함께 저장합니다. 공유 결과는 `summary.json`, `evaluation.json`, `convergence.png`와
각 stage의 `metrics.jsonl`, `resolved_config.yaml`, `metadata.json`입니다.

## 이전 Colab T4: Remember 기능 점검

이전 controlled 실험은 `rmt_paper_remember.yaml`에
[rmt_remember_functional.yaml](configs/rmt_remember_functional.yaml)을 병합한 경로입니다.
현재 Colab 노트북은 위의 native author 경로를 실행합니다.
원본 RMT와 Remember의 입력·출력 형식을 유지하면서, 기능 점검용으로 **서로 다른 key/value**를 쓰고
같은 context의 모든 key를 각각 질의합니다. context는 fact 순서를 바꾼 경우까지 고려해 split 간 분리합니다.

| 단계 | train context / query 예제 | validation / test context | 최대 update |
|---|---|---|---|
| 1쌍 | 192 / 192 | 각각32 | 400 |
| 2쌍 | 12,288 / 24,576 | 각각512 | 1,500 |
| 3쌍 | 8,192 / 24,576 | 각각341 | 1,000 |

매 단계 고정 pair 수를 학습하고, 예제는 정해진 유한 집합을 반복합니다.
1쌍의 서로 다른 context는 총16×16=256개이므로 split을192/32/32로 나눕니다.
2쌍은 `C(16,2)×P(16,2)=28,800`개의 서로 다른 mapping 중 분리된 집합을 사용합니다.
값을 무시하고 EOS만 맞히거나 마지막 값만 출력하는 전략을 구분하기 위해
value exact match, 질의 위치별 정확도, **context의 모든 질의가 맞는 비율**을 기록합니다.
생성 exact match와 모든 질의 성공률이 모두99% 이상이어야 다음 단계로 넘어갑니다.
최종 성공 후3/5/8쌍을 평가합니다. 이는 논문 전체 실험 재현과 구분한 controlled Remember입니다.

T4 실측 batch512의1/2쌍 peak는1.68/2.49GiB, update 시간은0.246/0.307초였습니다.
이번에는 microbatch1024 × accumulation1을 사용합니다. 3쌍 peak는 약6.6GiB로 예상하며 실제 peak를 기록합니다.
같은 측정값을 batch·세그먼트 수로 선형 환산하면 최대 학습 시간은 약32분입니다.
새 배치의 처리량과 데이터 생성 비용은 미측정이므로 전체 wall budget35분을 두고 검증·저장은100 update마다 합니다.
최종 test는 별도이며, 진행 중인 update/검증만큼 시간 제한을 초과할 수 있습니다.
effective batch는512에서1024로 변경되며 LR3e-4는 유지합니다.
OOM이면 새 run에서 `--batch-size 512 --grad-accum-steps 2`로 effective batch1024를 유지하세요.

```bash
python -u scripts/train.py --config configs/rmt_paper_remember.yaml configs/rmt_remember_functional.yaml --output-dir runs/rmt-remember-functional --device cuda
```

노트북의 새 RUN_DIR을 사용하세요. 기존 checkpoint는 모든1쌍 context를 이미 학습했을 수 있어
이번 split의 holdout 검증에는 재사용하지 않습니다. 같은 명령으로 새 실험을 자동 재개합니다.
공유 자료는 `summary.json`, `evaluation.json`, `convergence.png`와 각 stage의 `metrics.jsonl`입니다.

## 이전: 원본 형식의 짧은 Remember pilot

설정은 [configs/rmt_paper_remember.yaml](configs/rmt_paper_remember.yaml)입니다.
원본 RMT의 read/write memory wrapping에 GPT-NeoX(4층·D128·FF128·4 heads·memory32)를 사용합니다.
435,456개 파라미터, V128 중 일반 기호16개, key/value 각1개입니다.
key-value 한 쌍마다 별도 세그먼트를 처리하고 마지막 query에서 **value와 EOS를 생성**합니다.

`1 → 2 → 3 → 5쌍` curriculum이며 이전 단계의 가중치를 전달하고 optimizer/scheduler는 새로 시작합니다.
각 단계는 고정 pair 수의 validation512개에서 **생성 exact match≥99%**를 확인해야 다음 단계로 진행합니다.
학습에서는 현재 단계 범위의 pair 수를 무작위로 선택합니다. 최종 단계 성공 시 test는5/10/15쌍입니다.
RMT 결과 확인 후 GRT에 동일 과제를 연결합니다.

T4 설정은 FP32, microbatch512 × accumulation1, AdamW LR3e-4·WD.001·linear scheduler·warmup입니다.
원본 스크립트처럼 scheduler horizon은 각 단계 update 상한의2배, warmup은 상한의10%입니다.
CPU saved-tensor storage를 batch512로 환산하면1쌍 약2.65GiB, 5쌍 약5.75GiB이고
CUDA/backward 작업 공간은 별도입니다. 실제 **학습 peak VRAM**은 stage 로그/checkpoint에 기록합니다.
OOM이면 새 run에서 `--batch-size 256 --grad-accum-steps 2`로 effective batch512를 유지하세요.
각 단계의 최대 update는1000/500/500/1000이며 생성 기준 달성 시 일찍 넘어갑니다.
전체 curriculum wall budget은20분이며 마지막 test 평가는 별도입니다. 검증·저장은100 update마다 수행합니다.
시간 제한은 update/validation 사이에서 확인하므로 진행 중인 한 작업만큼 초과할 수 있습니다.

```bash
python -u scripts/train.py --config configs/rmt_paper_remember.yaml --output-dir runs/rmt-remember-t4 --device cuda
```

같은 명령으로 자동 재개합니다. 설정이 다른 run과 기존 Copy checkpoint는 재사용하지 않습니다.
공유 자료는 루트의 `summary.json`, `evaluation.json`, `convergence.png`와
각 stage의 `metrics.jsonl`, `resolved_config.yaml`, `metadata.json`입니다.
EOS를 맞히고 value만 추측해도 teacher-forced token accuracy가 약53%가 될 수 있으므로
**생성 value exact match와 value+EOS exact match**를 함께 확인하세요.

근거는 [논문](https://arxiv.org/abs/2407.04841) 부록 C/E/I와
[원본 RMT curriculum](https://github.com/RodkinIvan/associative-recurrent-memory-transformer/blob/24cbb9aed62a5748c4045fd928f7f59899f03b24/scripts/associative_retrieval/finetune_rmt_ar-value_cur.sh)입니다.
논문의 전체200쌍·전체 학습 예산 재현과 구분한 짧은 검증입니다.
당시 생성된 backbone JSON이 없어 세부 설정은 저자 레포의 후대 config generator를 사용했습니다.
원본 backbone/wrapper에서 생성한 작은 fixture로 logits·memory·gradient·greedy 생성을 대조합니다.
이 검증이 GPU 수렴의 증거는 아닙니다. 출처·버전은 `third_party/NOTICE.md`에 기록합니다.

## 이전 Colab T4: 2022 Copy 조건 검증

[notebooks/colab_rmt_paper_copy.ipynb](notebooks/colab_rmt_paper_copy.ipynb)를 실행하세요.
설정은 [configs/rmt_paper_copy.yaml](configs/rmt_paper_copy.yaml)입니다.
V12·source 24개·답 48개·N=M24, 4층·D128·FF256의 causal 상대 위치 RMT를
T4용 batch512·LR1e-4·FP32로 학습합니다. Teacher forcing과 자기회귀 정확도를 함께 확인하며,
GRT 구현과 비교는 RMT 수렴 결과를 확인한 뒤 진행합니다.

100K train source를 반복하며 validation1024/test2048개, 자기회귀 평가는 각각64개입니다.
최대2K update의 pilot이며 validation CE≤.05·teacher-forced 정확도≥99%·자기회귀 정확도≥99%·
자기회귀 exact match≥95%를 모두 만족하면 종료합니다. 학습 로그는100 update마다,
검증·저장은125 update마다 수행하고 LR 감소 판단은750 update 간격입니다.
125 update에2분이라는 T4 실측 기준으로 최대 약32분입니다. 원본 전체 sample 예산을
사용하지 않으며 이 시간 내 수렴을 보장하지 않습니다. 예산 종료 시 수렴 여부와 추세를 함께 확인합니다.
노트북은 기존25K 설정의 b512 run도 같은 학습 조건이면 재개하여 총2K에서 멈춥니다.
노트북의 `summary.json`, `convergence.png`, `metrics.jsonl`, `evaluation.json`을 공유하세요.

```bash
python -u scripts/train.py --config configs/rmt_paper_copy.yaml --output-dir runs/rmt-paper-copy-b512 --device cuda
python -u scripts/train.py --resume runs/rmt-paper-copy-b512/last.pt --device cuda
```

기존25K run을 중단 후 재개할 때는 `--stop-after 2000`을 추가하면 optimizer/LR 상태를
유지하면서 총2K update에서 종료합니다.

## 기존 Colab T4: 병렬 복원 Copy pilot

아래 프로필은 공개 Copy와 다른 기존 병렬 복원 과제입니다.

[notebooks/colab_t4_copy.ipynb](notebooks/colab_t4_copy.ipynb)를 T4 런타임에서 실행하세요.
공통 override는 [configs/colab_t4_copy.yaml](configs/colab_t4_copy.yaml)입니다.

| 항목 | 두 모델 공통 설정 |
|---|---|
| 모델 / 과제 | 표준 small, 4층·D=256·FF=1024, Copy T=4 |
| 메모리 / 토큰 | M=32, N=128, V=1024, target 20개 |
| 배치 | microbatch 4 × accumulation 2 = effective batch 8 |
| 예산 | 5,000 optimizer updates, 학습 샘플 40,000개 |
| 정밀도 / LR | FP32, 3e-4, warmup 200 + cosine |
| 검증 / test | 각각 256개, validation 매 250 update |
| 저장 / 분석 | checkpoint 매 500, GRT RTLA 매 1,000 update |

FP32로 먼저 수렴을 관찰하며 FP16 overflow 변수는 별도 실험으로 둡니다.
5,000 update는 초기 탐색 예산이며 CE≤0.05 달성을 보장하지 않습니다.
표준 50,000-update/전체 평가 표본 실험과 구분하고 두 모델에 동일한 데이터·예산을 적용합니다.
테스트 T=4/10/20은 종료 시 best checkpoint로 평가하며 선택에는 validation만 사용합니다.

설치 셀을 실행한 Colab에서 직접 실행하려면 다음 셀을 사용하세요.

```python
%cd /content/GRT
!python -u scripts/train.py --config configs/base.yaml configs/rmt.yaml configs/copy.yaml configs/colab_t4_copy.yaml --output-dir runs/rmt-t4-copy-pilot --device cuda
!python -u scripts/train.py --config configs/base.yaml configs/grt.yaml configs/copy.yaml configs/colab_t4_copy.yaml --output-dir runs/grt-t4-copy-pilot --device cuda
```

노트북에서는 RMT와 GRT 학습을 별도 셀로 실행하고, 기본적으로 Google Drive에 run을 저장합니다.
최신 checkpoint의 설정이 현재 설정과 일치할 때 자동으로 재개하며 완료한 run은 반복하지 않습니다.
주기 저장 이후 중단했다면 최대 499 update를 다시 실행할 수 있습니다.
`best.pt`가 더 최신이면 노트북은 그 checkpoint를 재개 지점으로 사용합니다.
Colab 로컬 `/content`는 런타임 종료 후 보존된다고 가정하지 마세요.
Drive I/O 시간과 런타임 제한 때문에 완료 시간을 보장하지 않습니다.

노트북의 마지막 셀은 validation CE/정확도 곡선, `comparison.json`,
두 모델의 기본·확장 길이 결과를 생성합니다. 기본 길이에서 CE≤0.05 도달 여부를 먼저 보세요.
미달이면 개선 추세와 최근 gradient/LR 및 수렴 정체를 확인하고,
확장 길이 실패만으로 구현 오류를 결론 내리지 마세요.
OOM이면 두 모델 모두 새 run에서 `--batch-size 2 --grad-accum-steps 4`를 적용해
effective batch 8을 유지합니다. 평가 batch도 두 모델에 동일하게 낮춘 새 YAML을 사용하세요.

이 프로필의 실제 T4 메모리·속도·수렴은 로컬에서 실행하지 않았습니다.

## 학습 / 평가 / 분석

```bash
python scripts/train.py --config configs/base.yaml configs/grt.yaml configs/copy.yaml --output-dir runs/grt-small-copy
python scripts/train.py --config configs/base.yaml configs/rmt.yaml configs/copy.yaml --output-dir runs/rmt-small-copy
python scripts/train.py --config configs/base.yaml configs/grt.yaml configs/passkey.yaml configs/large.yaml --output-dir runs/grt-large-passkey

python scripts/train.py --resume runs/grt-small-copy/last.pt
python scripts/evaluate.py --checkpoint runs/grt-small-copy/best.pt
python scripts/analyze.py --checkpoint runs/grt-small-copy/best.pt --split test --segments 20
```

`copy.yaml`를 `reverse.yaml` / `passkey.yaml`로 교체하여 과제를 선택합니다.
RMT large도 `configs/rmt.yaml` 뒤에 `configs/large.yaml`을 적용합니다.
설정은 왼쪽에서 오른쪽으로 병합하며 목록은 교체합니다. 알 수 없는 키·타입과
각 프로필의 규격을 벗어난 설정은 오류입니다. RMT에 GRT router/register 설정을 넣을 수 없습니다.

학습은 `--max-steps`, `--batch-size`, `--grad-accum-steps`, `--mixed-precision`,
`--device auto|cpu|cuda`를 지원합니다. CLI 변경은 resolved config에 저장합니다.
재개 시 실행 설정 override는 허용하지 않습니다. 체크포인트의 모델·optimizer·scheduler·
scaler·RNG·다음 sample ID를 복원하고, run 경로는 checkpoint 파일의 현재 부모 폴더로 정합니다.
기존 run을 덮어쓰는 새 학습은 거절합니다. 이전 형식의 checkpoint는 자동 이관하지 않습니다.
PyTorch checkpoint는 pickle을 포함하므로 직접 생성한 신뢰할 수 있는 파일만 사용하세요.

정상 종료 시 validation과 `last.pt`를 저장하고, validation CE로 선택한 `best.pt`를
기본 길이와 두 확장 길이에서 평가합니다. 기본 표본은 validation 1024, 길이별 test 4096입니다.
기존 recovery 프로필의 학습 목표 CE≤0.05는 최초 도달 시점을 기록하며 자동 조기 종료하지 않습니다.
paper_copy는 validation CE·teacher forcing·자기회귀 정확도 기준을 함께 만족하면 조기 종료합니다.
FP32가 기본이며 이 구현의 FP16/BF16 실행은 지원하는 CUDA 장치를 요구합니다.
FP16 overflow는 optimizer/scheduler step을 진행하지 않고 로그·진단을 남긴 뒤 중단합니다.

## 구조와 산출물

`src/grt/models/`에는 공통 interface/factory와 독립 GRT/RMT가 있습니다.
`data.py`, `metrics.py`, `trainer.py`, `evaluator.py`, `checkpoint.py`, `logger.py`가
공통 실행을 담당하고 `rtla.py`, `plots.py`가 GRT 분석을 담당합니다.
모델은 labels나 loss를 받지 않으며 매 forward에서 메모리를 초기화하고 세그먼트 간
gradient를 유지합니다. 실제 padding은 지원하지 않아 false attention mask에 오류를 냅니다.

각 run은 `resolved_config.yaml`, `metadata.json`, `metrics.jsonl`, `best.pt`, `last.pt`,
주기 `step_XXXXXX.pt`, `evaluation.json`, GRT의 `rtla/*.npz|json|png`를 저장합니다.
`last.pt`는 주기 checkpoint 시점과 정상 종료 시 갱신합니다.
RTLA는 갱신 전 상태에서 샘플별 FP32 norm을 구한 뒤 배치 평균합니다.
conditional read가 꺼진 R은 inactive read head로 표시하며 의미적 슬롯 역할을 자동 분류하지 않습니다.
기존 trace 경로는 덮어쓰지 않습니다. 분석을 다시 실행할 때는 새 `--output-dir`를 사용하세요.

평가는 정답 토큰 수로 CE/token accuracy를 집계하고 sample 단위 exact match를 계산합니다.
별도 inference 측정에서 데이터 생성·전송·분석을 제외하고 input tokens/sec,
samples/sec 및 peak allocated GPU memory를 기록합니다. full logits 출력 비용은 포함됩니다.
GRT의 내부 세그먼트 길이는 N+M, RMT는 N+2M이며 실제 파라미터 수를 함께 보고합니다.

설계 문서는 보존했습니다. 기존 구현의 보존 사본은
`documents/legacy_implementation_20261002.zip`이며 새 CLI가 참조하지 않습니다.
작업 환경의 Git 연결이 깨져 있으면 metadata의 Git commit은 null이고 오류를 함께 기록합니다.

이 변경은 개발 환경에서 학습·테스트를 실행하지 않았습니다. 노트북 실행 결과로 검증해 주세요.
