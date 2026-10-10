# GRT

ARMT 논문의 Remember 과제에서 저자 RMT baseline과 GRT를 비교한다. 두 모델은 같은 데이터 adapter, 학습·생성 평가, curriculum과 checkpoint 경로를 사용한다. GRT의 router·ALU·gated write-back을 유지하며 query는 허용된 prefix만 처리한다.

## 실행

Python 3.10+와 PyTorch를 사용한다.

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/train.py --config configs/base.yaml configs/rmt.yaml configs/cpu.yaml --output-dir runs/rmt_cpu --device cpu
```

모델 선택은 `configs/rmt.yaml` 또는 `configs/grt.yaml`이다. `cpu.yaml`은 작은 실행 검사, `t4.yaml`은 T4 실험 override다.

```bash
python scripts/train.py --config configs/base.yaml configs/rmt.yaml configs/t4.yaml --output-dir runs/rmt_t4 --device cuda
python scripts/evaluate.py --checkpoint runs/rmt_t4/stage_02_pairs2_key1/best.pt --device cuda
```

같은 설정·출력 경로로 재실행하면 `last.pt`에서 재개한다. 설정이 바뀌면 새 경로를 사용한다. Colab은 [최신 노트북](notebooks/colab.ipynb)의 `MODEL`, `RUN_DIR`만 선택해 실행하면 된다. [템플릿](notebooks/template.ipynb)은 새 실험용이며 기본은 CPU 검사다.

현재 T4 설정은 D128 / 4층 / FF128 / memory32 / FP32, batch2048, 1쌍 최대500 → 2쌍 최대2500 update다. 데이터는 train100만 / validation1천 / test1만이며 고정 길이 생성 exact match99%에서 단계 전환한다. 논문 전체 capacity 실험 재현이 아닌 1→2쌍 기능 검사다. GRT의 GPU 메모리·처리량·수렴은 별도 실험에서 확인한다.

## 결과와 모니터링

실행 루트의 `summary.json/.txt`, `evaluation.json/.txt`, `convergence.png`(노트북 생성)와 각 stage의 `metrics.jsonl`, `metrics.txt`, `resolved_config.yaml`, `metadata.json`, `last.pt`, `best.pt`를 확인한다. CE는 value+EOS의 teacher forcing, exact match는 실제 생성 지표다. `train/loss`는 해당 update의 배치 값이다.

W&B 기본값은 `false`다. 사용할 때 `pip install -e '.[wandb]'` 후 `wandb.enabled: true`로 설정하거나 노트북의 `WANDB=True`를 선택한다. 한 curriculum을 하나의 run으로 기록하며 재개 시 저장된 run ID를 사용한다. 로그인 키는 환경에서 제공한다. 연결에 실패해도 로컬 로그를 유지한다.

GRT trace는 `rtla.enabled: true`로 학습 중 수집하거나 아래 명령으로 확인한다.

```bash
python scripts/analyze.py --checkpoint runs/grt_cpu/stage_02_pairs2_key1/best.pt --output-dir runs/grt_trace --device cpu
```

## 기준과 기록

[설계 기준](docs/20261010_design.md)에 데이터·모델 계약과 출처, 원본과의 차이를 기록했다. [문서 목록](docs/README.md)에서 날짜별 기록을 확인할 수 있다. 원본 코드와 hash는 `src/grt/vendor/armt/`, 출처와 라이선스는 [NOTICE](third_party/NOTICE.md)에 있다.

진행 중인 이전 RMT 실험은 `cbc3f36` checkout에서 마친다. 새 코드는 이전 native `author_rmt/1` checkpoint의 읽기·평가를 지원하며 기존 실행을 자동 변환해 재개하지 않는다. `scripts/evaluate.py --samples 64 --batch-size 8 --device cpu`로 짧게 평가할 수 있다. 구형 Copy 모델 checkpoint는 현재 경로에서 사용하지 않는다.
