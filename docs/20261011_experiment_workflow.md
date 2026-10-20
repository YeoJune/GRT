# 실험 실행과 결과 갱신

갱신: 2026-10-20 · 현재 실행 방법만 유지한다. 과거 운영·오류 수정 이력은 [아카이브](archive/README.md)에 있다.

## 1. 설정과 실행 진입점

[단일 카탈로그](../configs/experiments.yaml)는 task·size·model·tokenizer·budget·runtime의 기본값을 정의한다. `Config`의 typed 검증을 거치고 수동 override가 최종 적용된다. 숫자 변경 때문에 조건별 YAML·노트북을 새로 만들지 않는다.

- Task: Remember / WikiText-103. Size: base/small/tiny. Model: neox/rmt/grt.
- Budget: smoke200 /10k /20k /100k optimizer updates. 임의 값은 `steps=` 또는 `--steps`.
- Tokenizer: GPT-2 BPE 또는 word. Runtime: Kaggle/Colab/CPU.
- 노트북은 값 선택과 setup/prepare/preflight/train/report만 담당한다. 공통 동작은 `src/grt/utils/`에 둔다.

[Colab](../notebooks/colab.ipynb)·[Kaggle](../notebooks/kaggle.ipynb)의 현재 선택은 **GRT small·L75/M20·20k·seed54·bias0·read off·A(10,1e-4)·write dropout0**이다. M8/20/75×dropout0/.1 탐색은 완료했으며, 후속 실험은 모든 M에서 dropout0으로 통일한다. Dropout 효과는 더 긴 학습에서 재검증하며 결과와 판단은 [실험 보고서](experiments/20261018_l75_gate_stabilization.md#logit-tail-expansion)에만 관리한다.

```python
num_registers = 20         # 완료된 탐색 조건의 재현 선택
write_dropout = 0.0        # 후속 실험 공통: off, M별 예외 없음
write_logit_threshold = 10.0
write_logit_coef = 1e-4     # 0이면 off
```

후속 [gate 경로 실험](experiments/20261019_l75_gate_paths.md)은 먼저 M20에서 `write_source='independent'/'separate_head'/'candidate'`를 바꿔 실행한다. State 참조는 register attention으로 고정하고 첫 비교는 `write_detach_source=False`다. 2번에만 D→D gate linear를 추가하며 지표 선정→선택적 detach→다른 M 검증 순서다. A와 선행 실험에서 선택한 write dropout은 비교 안에서 고정한다. Config는 `model.router`의 같은 이름 필드로 기록된다.

Colab은 GPU와 `WANDB_API_KEY` secret의 노트북 접근을 켠다. Kaggle은 Internet·T4×2·같은 이름의 secret을 사용한다. `wandb_enabled=False`로 로컬 기록만 사용할 수 있다. Tokenizer만 다운로드하며 backbone weights는 새로 초기화한다.

## 2. 데이터·preflight·학습

```python
job.setup()
job.prepare_data()
job.preflight()
job.train()
```

Prepared 데이터는 manifest·train/validation/test.pt·tokenizer/vocabulary artifact를 보존한다. 기존 artifact를 쓸 때는 `job.prepare_data(prepared_input=..., resume_input=...)`에 위치를 준다. 동일 데이터는 hash로 확인한다.

Global batch는256, precision은FP32다. Kaggle 기본micro32×accum4×GPU2, Colab은micro32×accum8×GPU1이다. Auto-batch는 VRAM 목표(노트북.95, 카탈로그.8)에 맞춰 micro/accum을 보정하며 global batch·평가 batch는 바꾸지 않는다. 학습률도 자동 변경하지 않는다. 실제 보정값·peak memory·ETA는 preflight 결과를 확인한다.

L을 바꾸면 처리 token 예산과 BPTT token 범위가 달라진다. K3은 이전3segments의 gradient 범위이며 기억 수명의 상한이 아니다. Nominal targets=global×L×updates, 실제 수는 tail을 포함한 report를 따른다.

설정이 달라지면 새 hash/run 경로를 사용하고 기존 checkpoint에서 이어 학습하지 않는다. 같은 Config·world size·manifest의 last.pt만 재개한다. 세션 종료 전에 checkpoint·prepared 데이터를 외부에 보관한다. Report에는 가중치·optimizer가 들어 있지 않다.

## 3. CLI와 로컬 과제

CLI도 같은 카탈로그를 사용한다. 예를 들어 현재 A의 수치 조건은 다음과 같이 명시한다.

```bash
python scripts/train.py --task wikitext103 --model grt --size small --tokenization gpt2_bpe --budget 20k --runtime kaggle --set model.segment_len=75 --set model.num_registers=20 --set model.register.write_gate_bias_init=0 --set model.register.write_logit_threshold=10 --set model.register.write_logit_coef=0.0001 --output-dir runs/a20 --device cuda
```

Kaggle2GPU CLI는 `torchrun --standalone --nproc_per_node=2`로 같은 script를 실행한다. `--benchmark-steps`로 preflight를 수행한다. `--config`는 현재 형식의 최종 Config 하나만 읽고 과거 경로·설정 변환은 제공하지 않는다.

Remember 실행은 [README](../README.md)와 [template](../notebooks/template.ipynb)를 사용한다. 인코더는 `model.attention_mode=encoder`, `data.options.objective=parallel`을 함께 지정하고, 디코더는 next-token 규칙을 사용한다. WikiText 인코더 MLM은 지원하지 않는다.

<a id="rtla-학습-중-모니터링과-사후-분석"></a>
## 4. 핵심 모니터링과 최종 결과

새 W&B run은 아래 네 prefix로 자동 패널을 나눈다. 기존 원격 workspace는 변경하지 않는다. 정의·단위·노트북 결과 요약은 `src/grt/monitoring.py` 한 곳에서 관리하며, 정의 밖의 payload를 W&B로 전송하지 않는다.

| 패널 그룹 | WikiText의 기본 표시 | 범위·단위 |
|---|---|---|
| `01_quality` | train NLL, validation NLL/PPL | 순수 LM. 전체 validation으로 best 선택 |
| `02_optimization` | LR, clipping 전 gradient norm, 가중 gate penalty | Penalty는coef*R이며 LM NLL과 분리 |
| `03_memory` | training raw write·abs logit>20 비율, probe write·relative update, reset/swap ΔNLL | training은mask 이전, probe는dropout off. ΔNLL은nats/token |
| `04_runtime` | 누적 처리 targets/s, 경과 시간, peak VRAM | 이전 평가·저장 시간을 포함한 누적 throughput, hours/GiB |

Quality·optimization·memory의 기본 가로축은 실제 누적 target tokens다. Runtime은 optimizer step을 쓴다. Step·tokens·stage 정보는 숨긴 축/진행 변수로 기록한다. Remember는 optimizer step을 쓰고 random/fixed validation NLL·exact match를 표시하며 PPL을 만들지 않는다. 측정하지 않은 필드는0으로 채우지 않는다.

`rtla_enabled=True`는 GRT validation마다 고정 probe를 실행한다. NeoX/RMT에는 GRT RTLA를 적용하지 않는다. State norm·candidate distance·slot cosine·probe 정상 PPL·early/late 개입값은 로컬 RTLA 원본에 보존하되 기본 W&B 패널에서 제외한다. Training logit 평균/최대·threshold 초과율·sigmoid slope·hidden norm의 안정화 전용 상시 측정은 제거했다. Gate raw write/극단 비율은패널티를 사용하는 학습에서 기록한다. 상세 gate·logit·rank/CKA·그룹 개입은 사후 분석에 둔다.

노트북 마지막 셀은 summary 전체 대신 **Quality / Memory·Cost / Posthoc** 요약과 전체 validation NLL/PPL 두 그래프를 보여준다. `overview`는 best validation과 final-budget validation을 별도 step과 함께 기록하며, 선택 checkpoint probe와 사후 평가 window 평균을 구분한다. `experiment_report.json`에는 전체 Config·metadata·summary·학습 이력·로컬 RTLA와 요약을 보존하고, ZIP에는 별도 사후 JSON을 담는다. 원본 상세 배열을 W&B에 중복 업로드하지 않는다.

학습 통계와 validation probe, 사후 checkpoint 관측은 다른 범위다. 공통 집계·그림·해석은 [안정화 보고서의 지표 사전](experiments/20261018_l75_gate_stabilization.md#2-공통-비교-기준)을 따른다. 같은 그룹 크기의 random 비교와 horizon도 실험 간 고정한다.

<a id="posthoc-colab"></a>
## 5. 사후 분석과 전달 파일

학습 노트북의 마지막 셀을 실행한다.

```python
job.report(posthoc=True, download=True, posthoc_options=dict(
    batch_size=8, segments=8, warmup_segments=3, horizon=4,
    slots=True, registers=True, window_starts=(0, 16, 32), group_repeats=3,
))
```

이번 run의 GRT best.pt와 prepared 데이터를 자동 선택한다. 별도 Python 프로세스에서 지원 버전 `transformers==4.45.2`로 복원하고 단일 GPU·FP32·dropout off로 분석한다. 선택 step과 checkpoint가 다르면 거부한다. 사후 시간은 학습 ETA와 별도다.

**보내줄 파일은 `experiment_results.zip` 하나**다. 안에는 `experiment_report.json`과 `posthoc/grt/posthoc_report.json`이 있다. 학습 결과를 먼저 저장하므로 분석 실패로 학습 결과가 사라지지 않는다. 분석 로그는 run의 `grt_posthoc_console.txt`에 남고 마지막 셀을 재실행하면 완료된 같은 분석은 재사용한다. 학습 JSON만 필요하면 `job.report(download=True)`다.

다른 세션의 checkpoint는 [Colab 분석 노트북](../notebooks/colab_analysis.ipynb)으로 분석한다. Drive에 다음과 같이 배치하고 선택 셀의 경로를 지정한다.

```text
MyDrive/GRT_analysis/
  checkpoints/m75/best.pt
  wikitext103_gpt2_bpe/manifest.json
  wikitext103_gpt2_bpe/validation.pt
  wikitext103_gpt2_bpe/tokenizer/...
  outputs/
```

원본 checkpoint의 Config·manifest·tokenizer artifact가 필요하다. W&B key는 필요 없으며 설정은 checkpoint에서 읽는다. 마지막 `posthoc_report.json` 하나를 전달하면 기존 학습 결과에 연결한다.

## 6. 결과를 저장소에 반영

1. JSON의 Config·report ID·checkpoint step/hash·manifest·GPU/precision을 확인한다.
2. `results/<report_id>.json`, `results/posthoc/<report_id>.json`에 **bytes를 바꾸지 않고** 보관한다. 미완료·미측정을0으로 채우지 않는다.
3. 실험 주제마다 Markdown 하나에 조건·품질·공통 진단·추가 지표·판단·다음 상태를 갱신한다. 실행 설계의 과거 버전은 archive로 옮긴다.
4. Gate 비교는 `scripts/report_gate_stabilization.py`의 RUNS/GROUPS에 수신 원본을 등록하고 스크립트를 실행한다. 공통 표·현재8개 그림·집계 JSON을 함께 갱신하고 해석은 수동으로 검토한다.
5. Raw 결과 보존·수치·링크·실행 경로를 검증하고 GPU 실험과 CPU 기능 검증을 구분해 보고한다. 커밋·push는 저장소 협업 지침에 따른다.

CSV로 환경별 결과를 누적하지 않는다. 재현용 checkpoint·prepared 데이터는 사용자 실행 환경 밖에 따로 보관한다. 자세한 원본 규칙은 [results/README](../results/README.md)를 따른다.
