# GRT

GRT·RMT·메모리 없는 NeoX를 공통 GPT-NeoX backbone에서 비교한다. Remember 인코더/디코더와 WikiText-103 디코더를 지원한다.

[현재 문서](docs/README.md) · [모델 명세](docs/20261011_current_spec.md) · [실행 workflow](docs/20261011_experiment_workflow.md) · [과거 기록](docs/archive/README.md)

## 현재 gate 안정화 실험

현재 사용하는 장치는 **sigmoid write + logit tail 패널티 A + write dropout**이다. Small·L75·seed54·bias0·read off·20k·FP32·global256을 기준으로 한다.

- [Gate 안정화 보고서](docs/experiments/20261018_l75_gate_stabilization.md): 패널티·dropout을 먼저 설명하고, 공통6패널·지표로 모든 조건을 비교한다. Bias·Refine·Uniform은 아래 이력에 정리한다.
- [현재 진행 상태](docs/experiments/20261018_l75_gate_stabilization.md#logit-tail-expansion): M8/M20 A 완료, M75 A 다음에 M8/20/75 dropout.1 비교.
- [L150 baseline·용량 비교](docs/experiments/20261012_small_register_sweep.md): 별도 실험 주제의 결과다.

## GPU 노트북

[Colab 학습](notebooks/colab.ipynb) 또는 [Kaggle 학습](notebooks/kaggle.ipynb)에서 GPU·`WANDB_API_KEY` secret을 설정한다. Kaggle은 Internet·T4×2, Colab은 GPU1개를 사용한다. `wandb_enabled=False`면 로컬 기록만 남긴다.

기존 선택 셀에서 다음 값만 바꿔 한 run씩 실행한다. 새 notebook/config 파일은 만들지 않는다.

```python
num_registers = 75
write_dropout = 0.0          # 이후0.1
write_logit_threshold = 10.0
write_logit_coef = 1e-4      # 0은 off 대조
```

`setup → prepare_data → preflight → train → report` 순서다. 자동 배치는 VRAM 목표 안에서 micro/accum을 조절하며 global256·평가 batch·FP32는 유지한다. 설정이 달라지면 새 초기화 run, 같은 설정·world size·manifest면 last.pt를 재개한다. Tokenizer만 다운로드하고 backbone weights는 새로 초기화한다.

마지막 `job.report(posthoc=True, download=True, ...)`는 best checkpoint의 사후 분석까지 실행한다. **`experiment_results.zip` 하나**를 보내면 된다. ZIP은 학습·사후 JSON만 포함하므로 checkpoint·prepared 데이터는 따로 보관한다. 다른 세션의 checkpoint는 [Colab 사후 분석](notebooks/colab_analysis.ipynb)을 사용한다. 파일 배치는 [workflow](docs/20261011_experiment_workflow.md#posthoc-colab)에 있다.

## 설치·CLI·로컬 과제

Python3.10+와 PyTorch를 사용한다.

```bash
python -m pip install -e '.[dev,wandb,wikitext]'
python scripts/train.py --task remember --model grt --size tiny --budget smoke --runtime cpu --steps 2 --set model.attention_mode=encoder --set data.options.objective=parallel --output-dir runs/remember_encoder_cpu --device cpu
```

Remember 디코더는 두 `--set` 옵션을 제거하고, RMT/NeoX는 `--model`로 선택한다. [로컬 template](notebooks/template.ipynb)과 CLI는 같은 [카탈로그](configs/experiments.yaml)를 쓴다. Task/size/budget/tokenizer 선택·데이터 준비·배치·재개는 [workflow](docs/20261011_experiment_workflow.md)에만 관리한다.

## 결과와 보고서 유지

수신 JSON은 [results/](results/README.md)에 bytes를 보존한다. 실험 주제마다 Markdown 하나를 갱신하고 과거 설계는 archive에 둔다. CPU 기능 검증과 사용자 GPU 결과를 구분한다.

Gate 결과의 공통 표·그림·구조화 집계는 모델 실행 없이 다음 명령으로 함께 재생성한다. 실험별 해석은 유지된다.

```bash
python scripts/report_gate_stabilization.py
```

출처·라이선스는 [NOTICE](third_party/NOTICE.md)와 [vendor SOURCE](src/grt/vendor/armt/SOURCE.json)를 따른다.
