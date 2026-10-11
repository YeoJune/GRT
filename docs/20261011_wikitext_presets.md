# WikiText 크기 선택과 짧은 동작 확인

사용자 승인: tiny D128/3층/4 heads/FF512, 200 updates, 20 updates마다 전체 validation. base와 small은 기존 차원을 보존한다. 크기와 실험 예산을 별개 preset으로 선택한다.

| 크기 | D | layers | heads | FF | 설정 |
|---|---:|---:|---:|---:|---|
| base | 410 | 16 | 10 | 2100 | `configs/wikitext_base.yaml` |
| small | 256 | 6 | 8 | 1024 | `configs/wikitext_small.yaml` |
| tiny | 128 | 3 | 4 | 512 | `configs/wikitext_tiny.yaml` |

모든 크기는 동일 NeoX/parallel residual/partial RoPE/tied head, L150/M10/BPTT 이전3, FP32, global batch60이다. GRT router hidden512도 유지한다. `wikitext103.yaml`의 기존 base 기본값은 이전 CLI와 호환되도록 보존하되 새 실행은 세 크기 파일 중 하나를 명시한다.

GPT-2 BPE 50257 기준 factory의 고유 Parameter 수:

| 크기 | 공통 backbone | RMT 전체 | GRT 전체 |
|---|---:|---:|---:|
| base | 59009230 | 59013330 | 61212784 |
| small | 17604864 | 17607424 | 18669588 |
| tiny | 7027968 | 7029248 | 7434900 |

| 실험 예산 | updates | validation | 학습 로그 | 의미 |
|---|---:|---:|---:|---|
| full | 200000 | 매4000 | 매50 + 첫/마지막 step | 기존 전체 조건 |
| smoke | 200 | 매20 | 매10 + 첫/마지막 step | 실행·loss·저장·W&B 확인 |

smoke는 `configs/wikitext_smoke.yaml`을 추가한다. cosine 기간도 200 updates인 별도 실험이다. vocabulary/BPTT/배치/optimizer/LR/dropout을 축소하지 않는다. target 상한은 모델당 1.8M BPE tokens다. 실제 WikiText의 전체 validation/test를 사용하며 수렴이나 성능 비교 결과로 해석하지 않는다. tiny가 full 예산을 반드시 써야 하는 것은 아니며 크기와 예산은 독립 선택이다.

Kaggle 기본은 `MODEL_SIZE='tiny'`, `EXPERIMENT='smoke'`이고 RMT/GRT 각각 200 updates를 실행한다. 전체 실험은 크기를 바꾸고 `EXPERIMENT='full'`로 선택하며 `TRAIN_MODEL` 하나씩 실행한다. 새 실행 이름에는 크기·tokenization·실험 예산이 들어가 이전 small preflight/학습 폴더를 재사용하지 않는다. prepared 데이터는 그대로 공유할 수 있다.

```python
MODEL_SIZE = 'tiny'   # base / small / tiny
EXPERIMENT = 'smoke'  # smoke / full
```

CLI도 동일한 preset 조합이다. 예를 들어 아래 tiny RMT 명령은 GPU 사전 측정이며, 본 학습에서는 출력 calibrated override를 추가하고 `--benchmark-steps`를 제거해 새 output 폴더를 쓴다. GRT는 모델 파일을 `neox_grt.yaml`로 변경한다.

```bash
torchrun --standalone --nproc_per_node=2 scripts/train.py --config configs/base.yaml configs/neox.yaml configs/wikitext103.yaml configs/wikitext_tiny.yaml configs/wikitext_bpe.yaml configs/neox_rmt.yaml configs/wikitext_kaggle.yaml configs/wikitext_smoke.yaml --output-dir runs/tiny_rmt_smoke_preflight --device cuda --benchmark-steps 10
```

## 제공된 Kaggle 실측

사용자가 제공한 small GPT-2 BPE / T4 × 2 / 10-update preflight 결과:

| 모델 | seconds/update | peak allocated | reserved | 200000 updates + 평가 계산 |
|---|---:|---:|---:|---:|
| RMT | 0.478867 | 2.063 GiB | 2.467 GiB | 26.663시간 |
| GRT | 0.480246 | 2.010 GiB | 2.441 GiB | 26.740시간 |

시간은 준비·checkpoint I/O 제외다. 10회 평균이며 장시간 throughput을 보장하지 않는다. 이 값으로 small 200 updates의 순수 학습은 약 96초다. tiny 시간은 크기 변경 후 preflight로 다시 계산한다. socket hostname 및 NCCL device 추정 경고가 있었지만 두 측정은 완료됐다. warning을 실제 실행 실패로 해석하지 않는다.

## W&B 기록

preflight에서 W&B 로그를 보내지 않는 것은 기존 의도다. WikiText 본 학습은 validation 시점에만 로그를 보냈고 `wandb.every_steps`를 적용하지 못하는 문제가 있었다. 이제 학습 loss/LR/target tokens/elapsed/peak VRAM을 해당 주기에 기록하며 평가 시 val NLL/PPL도 같은 step에 기록한다. local JSONL/TXT와 W&B가 같은 payload를 받는다. rank0만 W&B run을 생성한다.

`wandb_status.json`에서 연결 상태·mode·run ID·URL·마지막 기록 step·실패 원인을 확인한다. W&B 실패 시 로컬 로그는 유지하며 notebook 결과 cell이 상태와 warnings를 보여준다. run 이름에는 크기·tokenization·예산·모델명이 들어간다. 재개에서는 같은 ID를 사용한다. offline SDK 확인은 온라인 업로드 성공의 증거가 아니므로 실제 Kaggle run URL과 차트를 확인한다.

CPU 검증은 새 로깅 경로와 tiny 설정에 한정한다. 기존 전체 회귀·tokenizer·분산·causal 검사를 반복하지 않는다. 200-update 실제 WikiText GPU 실행은 Kaggle에서 진행한다.

## 이번 변경의 간단한 확인

실제 tiny/V50257/L150/K3를 사용한 RMT/GRT 각각 CPU2 updates와 실제 W&B offline SDK 확인 두 검사가 3.96초에 통과했다. CPU batch1, synthetic token stream, validation/test31 tokens이며 실제 WikiText 결과가 아니다. 첫 step의 training-only 로그와 두 번째 step의 validation 로그, local JSONL 일치, offline `.wandb` 파일, 종료/summary/마지막 step 상태를 확인했다. 초기 offline SDK 시도는 실행 샌드박스의 local socket/기본 cache 쓰기 제한으로 실패했으며, writable cache와 loopback 허용 후 통과했다. 온라인 업로드를 확인한 것으로 표현하지 않는다.

노트북의 실제 config builder에서 크기3 × 모델2의 smoke 조합을 읽어 updates200/eval20/log10/global batch60과 명령 routing을 확인했다. 초기화 실패 상태 기록은 별도 작은 검사 한 개로 확인한다. 전체 회귀나 기존 학습/재개/분산 실험을 반복하지 않았다.
