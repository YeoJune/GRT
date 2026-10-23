# GRT

GRT·RMT·메모리 없는 NeoX를 공통 GPT-NeoX backbone에서 비교한다. Remember 인코더/디코더와 WikiText-103 디코더를 지원한다.

[현재 문서](docs/README.md) · [모델 명세](docs/20261011_current_spec.md) · [실행 workflow](docs/20261011_experiment_workflow.md) · [과거 기록](docs/archive/README.md)

## 현재 gate 안정화 실험

현재 후속 실험 설정은 **sigmoid write + logit tail 패널티 A + write dropout off**다. Dropout은 더 긴 학습에서 재검증할 수 있도록 지원한다. 후속 gate 경로 실험은 Small·L75·seed54·bias0·read off·40k·FP32·global256, warmup5%·cosine 최소 LR 비율1%로 진행한다. 완료된 안정화 탐색의20k 결과는 선행 기록으로 보존한다.

- [Gate 안정화 보고서](docs/experiments/20261018_l75_gate_stabilization.md): 패널티·dropout을 먼저 설명하고, 공통6패널·지표로 모든 조건을 비교한다. Bias·Refine·Uniform은 아래 이력에 정리한다.
- [최종 결론·후속 방침](docs/experiments/20261018_l75_gate_stabilization.md#logit-tail-expansion): 탐색 완료. 후속 실험은 모든 M에서 A/dropout0을 사용하고, dropout 효과는 더 긴 학습에서 재검증한다.
- [Gate 경로 비교 최종 보고서](docs/experiments/20261019_l75_gate_paths.md): M20 세 경로40k 비교 후 candidate를 기본값으로 선택했다. 성능 차이가 작고 reset/swap 진단이 양호하며 구조·연산량이 단순한 점을 반영했다. Independent·separate_head 구현은 유지하며 다른 M의 전달성은 후속 검증이다.
- [L150 baseline·용량 비교](docs/experiments/20261012_small_register_sweep.md): 별도 실험 주제의 결과다.

## GPU 노트북

[Colab 학습](notebooks/colab.ipynb) 또는 [Kaggle 학습](notebooks/kaggle.ipynb)에서 GPU·`WANDB_API_KEY` secret을 설정한다. Kaggle은 Internet·T4×2, Colab은 GPU1개를 사용한다. `wandb_enabled=False`면 로컬 기록만 남긴다.

기존 선택 셀에서 다음 값만 바꿔 한 run씩 실행한다. 새 notebook/config 파일은 만들지 않는다.

```python
num_registers = 20
budget = '40k'              # '1m', '40000' 등 숫자와 k/m으로 자유 입력
warmup_ratio = 0.05         # 전체 updates의5%; 0이면 off
write_dropout = 0.0          # 모든 M의 후속 실험에서 off; 장기 효과는 추후 재검증
write_logit_threshold = 10.0
write_logit_coef = 1e-4      # 0은 off 대조
```

`setup → prepare_data → preflight → train → report` 순서다. 자동 배치는 VRAM 목표 안에서 micro/accum을 조절하며 global256·평가 batch·FP32는 유지한다. 설정이 달라지면 새 초기화 run, 같은 설정·world size·manifest면 last.pt를 재개한다. Tokenizer만 다운로드하고 backbone weights는 새로 초기화한다.

마지막 `job.report(posthoc=True, download=True, ...)`는 best checkpoint의 사후 분석까지 실행한다. **`experiment_results.zip` 하나**를 보내면 된다. ZIP은 학습·사후 JSON만 포함하므로 checkpoint·prepared 데이터는 따로 보관한다. 다른 세션의 checkpoint는 [Colab 사후 분석](notebooks/colab_analysis.ipynb)을 사용한다. 파일 배치는 [workflow](docs/20261011_experiment_workflow.md#posthoc-colab)에 있다.

선택 셀의 `server_mode='upload'`와 서버 주소·UTC 세션 종료 시각을 지정하면 GRT Server에 상태·원본 결과·checkpoint를 자동 저장한다. 기본 `off`는 기존 실행을 유지한다. 서버의 불변 Config를 실행하는 큐 worker와 인증·재연결 방법은 [서버 연결 안내](docs/20261022_server_integration.md)를 따른다.

서버 배포 후 빠른 점검은 [임시 테스트 노트북](notebooks/tmp_server_acceptance.ipynb) / [Colab 열기](https://colab.research.google.com/github/YeoJune/GRT/blob/main/notebooks/tmp_server_acceptance.ipynb)를 사용한다. 작은 실제 학습·파일 hash·큐·취소를 검사하고 결과 요약 JSON을 만든다. [실행 가이드](https://github.com/YeoJune/GRT_server/blob/main/docs/20261023_aws_test_execution_guide.md)의 순서를 따른다.

새 run의 W&B는 **Quality / Optimization / Memory / Runtime** 네 그룹의 핵심 지표만 표시한다. 마지막 셀은 best·final validation, 선택 checkpoint의 memory probe, 비용, 사후 평가 window를 구분해 요약한다. 상세 지표·단위는 [공통 모니터링 안내](docs/20261011_experiment_workflow.md#rtla-학습-중-모니터링과-사후-분석)를 따른다.

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
