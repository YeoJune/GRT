> 과거 실행 안내/검증 기록입니다. 현재 실행 규칙은 [실험 workflow](20261023_experiment_workflow.md)를 따릅니다.

# Small WikiText-103: Kaggle T4 × 2

현재 D256/6층/8 heads/FF1024 small을 사용한다. 이 문서의 GPU 시간과 메모리는 계산 또는 측정 후보이며, 실제 T4 측정 결과는 아직 없다. CPU synthetic LM 결과를 GPU 시간이나 실제 WikiText 성능으로 외삽하지 않는다.
이 문서의 계산은 word-level이다. GPT-2 BPE와 최신 [Kaggle 노트북](../../notebooks/kaggle.ipynb)은 [BPE 안내](20261011_wikitext_bpe.md)를 따른다.

## 고정 조건과 측정 후보

| 항목 | 조건 |
|---|---|
| 공통 backbone | NeoX, parallel residual, partial RoPE 0.25, tied embedding/head |
| vocabulary | 원래 tokenized WikiText-103 v1, word-level 267735 |
| segment / memory | 150 / 10 |
| BPTT | 이전 3 segment + 현재 1 segment를 재생, 현재 target에만 loss |
| global batch | 60 streams = GPU당 microbatch 10 × accumulation 3 × GPU 2 |
| 학습 | FP32, Adam LR 2.5e-4, β=(0.9,0.999), ε=1e-8, decay 0 |
| scheduler / clipping | cosine, warmup 0 / norm 0.25 |
| dropout | residual/FF 0.1, attention 0 |
| 예산 | 200000 optimizer updates; 현재 target 최대 18억 tokens |
| 평가 | batch 10, 4000 updates마다 validation, best로 최종 test |

microbatch 10은 첫 측정 후보다. OOM 또는 peak 기준 초과 시 preflight가 더 작은 divisor와 accumulation을 제안한다. 예: 5 × 6 × 2 = 60. main 학습에는 출력된 override를 명시적으로 적용한다. 200000 updates와 cosine 기간은 유지하며 Kaggle 세션 길이에 맞춰 임의로 축소하지 않는다.

GRT 기본은 intermediate read / read gate off이며 `teacher_forced_noncausal`이다. RMT는 `autoregressive`다. 이 두 결과를 동일한 causal PPL로 해석할 수 없다. `fixed_state.yaml`은 별도 causal 조건이며 이번 환경 준비에서 기본 모델을 변경하지 않는다.

## 연산량과 시간

한 update의 현재 target은 60 × 150 = 9000 tokens다. replay는 stream당 600 token이다. GRT augmented 길이는 160, RMT는 170이다. dense matmul의 multiply-add를 2 FLOPs, forward+backward를 forward의 약 3배로 잡았다.

`head = 3 × 2 × 60 × 150 × 256 × 267735 = 3.701 TFLOP/update`

`backbone = 3 × 60 × 4 × 6 × [(8D² + 4D·FF)S + 4S²D]`

GRT 합계 약 4.90 TFLOP, RMT 약 4.98 TFLOP/update다. router, norm, softmax, optimizer, 통신 비용은 제외한 근사다. head가 약 75%를 차지한다. tied는 파라미터를 줄이지만 full-vocabulary projection 연산은 그대로다.

NVIDIA T4의 dense FP32 이론 성능을 GPU당 8.1 TFLOP/s로 가정하면, 두 장에서 약 0.30–0.31초/update, **200000 updates의 순수 학습만 최소 약 17시간/모델**이다. 이상적인 하한이며 실제 ETA가 아니다. 현재 FP32 조건에는 FP16 Tensor Core 성능 수치를 적용하지 않는다. 실제 kernel 효율과 PCIe/DDP 통신을 포함한 시간은 측정해야 한다.

| 측정 update 시간 가정 | 200000 updates 순수 학습 | 6.5시간 순수 학습 가능한 updates |
|---|---:|---:|
| 1초 | 55.6시간 | 23400 |
| 2초 | 111.1시간 | 11700 |
| 3초 | 166.7시간 | 7800 |
| 5초 | 277.8시간 | 4680 |
| 10초 | 555.6시간 | 2340 |

각 행은 예측 확률이 없는 시나리오다. 6.5시간은 이전 환경 안내와 비교하는 예시일 뿐 이번 실험의 시간 제한이 아니다. 두 모델을 순차 실행하면 각각의 시간이 합산된다. Kaggle의 현재 세션/주간 GPU quota도 별도로 확인한다.

일반적인 token 수 약 train 103.23M / validation 217.65K / test 245.57K로 계산하면 한 epoch 약 11470 updates, validation 146 segments, test 164 segments다. 정확한 값은 준비된 `manifest.json` 기준이다. 전체 시간은 다음으로 환산한다.

`200000 × update_seconds + 50 × validation_segments × eval_segment_seconds + test_segments × eval_segment_seconds + checkpoint I/O + 준비 시간`

현재 평가는 각 rank가 전체 split을 중복 평가하므로 GPU 두 장이라고 평가 시간을 반으로 나누지 않는다. 실제 세션 내 updates는 평가와 저장 시간 때문에 표보다 적다.

## 메모리와 저장 공간

공통 backbone 73,279,232 parameters, 전체 RMT 73,281,792 / GRT 74,343,956이다. FP32 weight + gradient + Adam 두 상태는 GPU당 약 1.09 / 1.11 GiB다. DDP gradient bucket은 추가 약 0.27 / 0.28 GiB가 필요할 수 있다. 두 GPU의 VRAM은 각 rank에 모델을 복제하므로 한 덩어리 32GB로 사용할 수 없다.

| GPU당 microbatch | accumulation | FP32 logits 한 텐서 |
|---:|---:|---:|
| 5 | 6 | 0.75 GiB |
| 10 | 3 | 1.50 GiB |
| 15 | 2 | 2.24 GiB |
| 30 | 1 | 4.49 GiB |

CE backward와 log-softmax가 추가 텐서를 필요로 한다. logits 관련 peak를 한 텐서의 2–4배, backbone activation을 microbatch 10에서 약 0.8–1.3 GiB로 잡으면, optimizer/DDP/CUDA workspace를 포함한 **microbatch 10의 측정 계획 범위는 대략 6–11 GiB/GPU**다. 이는 구현별 임시 텐서와 allocator에 따라 달라지는 휴리스틱이며 상한 보장이 아니다. microbatch 30은 logits/CE만 약 9–18 GiB라 첫 후보로 적합하지 않다.

preflight의 `peak_bytes_per_gpu`와 `reserved_bytes_per_gpu`를 모두 확인한다. 초기 OOM 조절 probe는 DDP wrap 전이므로 최종 DDP peak가 기준 80%를 넘는지도 확인해야 한다. evaluation batch 10은 stream 분할/문맥을 바꾸므로 메모리 조절 목적으로 자동 변경하지 않는다.

weights+Adam 상태 checkpoint는 모델당 약 0.82–0.83 GiB/파일이다. best+last는 모델당 약 1.64–1.66 GiB, 두 모델과 int64 prepared train 데이터는 합계 약 4.1 GiB 이상을 확보한다. validation마다 last와 개선 시 best를 덮어쓰므로 누적 write I/O는 크지만 파일 수가 계속 증가하지는 않는다.

## Kaggle 실행

Accelerator를 T4 × 2로 선택하고 저장소 파일과 원래 `wiki.train.tokens`, `wiki.valid.tokens`, `wiki.test.tokens`를 준비한다. 아래는 `/kaggle/working/GRT`에서 실행한다. 이번 추가 파일이 포함된 checkout이 필요하다. 최신 `notebooks/kaggle.ipynb`에서 `MODEL_SIZE='small'`, `EXPERIMENT='full'`, `TOKENIZATION='word'`를 선택하면 이 조건으로 실행한다. `configs/kaggle.yaml`과 archive의 이전 노트북은 Remember curriculum용이다.

```bash
python -m pip install -e '.[wandb]'
python -c 'import torch; assert torch.cuda.device_count()==2; print([(torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i).total_memory/2**30) for i in range(2)])'
python scripts/prepare_wikitext.py --source /kaggle/input/YOUR_DATASET/wikitext-103 --output-dir data/wikitext103
```

Kaggle secret `WANDB_API_KEY`를 설정한 뒤 다음 Python cell에서 로그인한다. 키를 출력하지 않는다. W&B를 사용하지 않을 경우 `wandb: {enabled: false}` override를 별도로 적용한다.

```python
from kaggle_secrets import UserSecretsClient
import wandb
wandb.login(key=UserSecretsClient().get_secret("WANDB_API_KEY"))
```

RMT 사전 측정:

```bash
torchrun --standalone --nproc_per_node=2 scripts/train.py --config configs/base.yaml configs/neox.yaml configs/wikitext103.yaml configs/wikitext_small.yaml configs/neox_rmt.yaml configs/wikitext_kaggle.yaml --output-dir /kaggle/working/wiki_small_rmt_preflight --device cuda --benchmark-steps 10 --memory-fraction 0.8
python scripts/estimate_wikitext_time.py --benchmark /kaggle/working/wiki_small_rmt_preflight/benchmark.json --manifest data/wikitext103/manifest.json
```

GRT는 `neox_rmt.yaml`을 `neox_grt.yaml`로 바꾸고 별도 `wiki_small_grt_preflight` 폴더를 쓴다. 두 모델을 각각 측정한다. preflight에는 W&B 학습 로그를 보내지 않는다.

출력 override를 저장한다. 아래 cell의 경로를 모델에 맞춰 변경한다.

```python
import json, yaml
from pathlib import Path
probe = Path('/kaggle/working/wiki_small_rmt_preflight')
result = json.loads((probe / 'benchmark.json').read_text())
Path('/kaggle/working/rmt_calibrated.yaml').write_text(yaml.safe_dump(result['calibrated_override']))
```

측정 결과와 세션 예산을 검토한 뒤 본 학습을 시작한다. 별도 출력 경로를 사용하며 다음 명령은 200000 updates 전체 실행이다.

```bash
torchrun --standalone --nproc_per_node=2 scripts/train.py --config configs/base.yaml configs/neox.yaml configs/wikitext103.yaml configs/wikitext_small.yaml configs/neox_rmt.yaml configs/wikitext_kaggle.yaml /kaggle/working/rmt_calibrated.yaml --output-dir /kaggle/working/wiki_small_rmt --device cuda
```

현재 `last.pt`는 매 validation(4000 updates)과 정상 종료에 저장된다. 강제 세션 종료 시 그 이후 작업이 유실될 수 있다. 동일 설정/출력 경로로 재개하며 checkpoint와 결과를 세션 밖으로 보존해야 한다. `max_seconds`는 재개 시 누적 시간 기준이라 세션별 자동 제한 용도로 사용하지 않는다. 기본 null을 유지했다. 세션 종료 대비 저장 주기를 조정하는 기능은 이번 환경 설정에 추가하지 않았다.

공유할 결과: 두 모델의 `benchmark.json`, `resolved_config.yaml`, prepared `manifest.json`, GPU 이름/메모리. 이 자료로 실제 ETA와 배치 적합성을 확정한다.
