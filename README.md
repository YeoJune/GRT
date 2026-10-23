# GRT

GRT·RMT·메모리 없는 NeoX를 공통 GPT-NeoX backbone에서 비교한다. Remember 인코더/디코더와 WikiText-103 디코더를 지원한다.

[문서 목록](docs/README.md) · [모델 명세](docs/specification.md) · [실행 workflow](docs/workflow.md) · [서버 연결](docs/server.md) · [연구 요약](docs/research.md)

## GPU 실행

[Colab 학습](notebooks/colab.ipynb) 또는 [Kaggle 학습](notebooks/kaggle.ipynb)을 연다. Colab은 GPU1개, Kaggle은 Internet·T4×2를 사용한다. `WANDB_API_KEY` Secret의 notebook 접근을 켜며, `wandb_enabled=False`면 로컬 기록만 남긴다.

현재 기본 선택은 **GRT small·L75/M20·40k·seed54·bias0·read off·candidate·패널티 A(10,1e-4)·write dropout0**이다. 예산·M·gate 수치는 기존 선택 셀에서 바꾼다. 새 조건은 새 run으로 실행한다. 선택의 근거와 미검증 범위는 [연구 요약](docs/research.md#현재-선택)에 둔다.

`setup → prepare_data → preflight → train → report` 순서로 실행한다. Auto-batch는 global256·평가 batch·FP32를 유지하며 microbatch와 accumulation을 조절한다. 실제 Config·메모리·ETA는 preflight에서 확인한다. 같은 Config·world size·manifest의 `last.pt`만 재개한다.

마지막 `job.report(posthoc=True, download=True, ...)`는 best checkpoint의 사후 분석까지 실행한다. 공유할 결과는 **`experiment_results.zip` 하나**이며, 가중치·checkpoint·prepared 데이터는 별도로 보관한다. 다른 세션의 checkpoint는 [Colab 사후 분석](notebooks/colab_analysis.ipynb)과 [파일 배치 안내](docs/workflow.md#posthoc-colab)를 사용한다.

## 서버 연결

선택 셀에서 `server_mode='upload'`, 서버 주소·UTC 세션 종료 시각을 지정한다. 플랫폼 Secrets에 `GRT_SERVER_TOKEN` write 토큰을 등록하면 상태·원본 결과·checkpoint를 자동 저장한다. 기본 `off`는 서버 없이 실행한다.

서버의 불변 Config를 순차 실행하는 큐 worker, 인증·재연결·안전 중단은 [서버 연결](docs/server.md)을 따른다. 서버 API·운영은 [GRT_server](https://github.com/YeoJune/GRT_server)를 참조한다.

## 설치·로컬 실행

Python3.10+와 PyTorch를 사용한다.

```bash
python -m pip install -e '.[dev,wandb,wikitext]'
python scripts/train.py --task remember --model grt --size tiny --budget smoke --runtime cpu --steps 2 --set model.attention_mode=encoder --set data.options.objective=parallel --output-dir runs/remember_encoder_cpu --device cpu
```

Remember 디코더는 위 두 `--set` 옵션을 제거한다. RMT는 `--model rmt`로 선택하고, 메모리 없는 NeoX는 WikiText에서 `--model neox`로 실행한다. [Template](notebooks/template.ipynb)과 CLI는 같은 [카탈로그](configs/experiments.yaml)를 사용한다. 전체 설정·데이터·학습·모니터링 계약은 [workflow](docs/workflow.md)에 관리한다.

## 검증과 결과 관리

```bash
python -m pytest -q
```

작은 CPU 검사와 실제 GPU 연구 결과를 구분한다. 서버 통합 검사 실행은 [서버 연결](docs/server.md#검증)을 따른다. 수신 JSON의 원본 bytes는 [results](results/README.md)에 보존하고, 최신 비교는 [연구 요약](docs/research.md), 완료된 설계·경과는 [아카이브](docs/archive/README.md)에 둔다. 실행 코드·config·notebook은 archive에 보관하지 않는다.

출처·라이선스는 [NOTICE](third_party/NOTICE.md)와 [vendor SOURCE](src/grt/vendor/armt/SOURCE.json)를 따른다.
