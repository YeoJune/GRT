# GRT

GRT와 RMT를 공통 GPT-NeoX backbone에서 비교한다. 인코더·디코더, Remember, WikiText-103을 지원한다.
현재 설계와 세부 설정은 [최신 통합 명세](docs/20261011_current_spec.md)를 따른다.
이전 설계·실험 기록은 [archive 목록](docs/README.md)에 보존했다.

## 설치와 CPU 검사

Python 3.10+, PyTorch를 사용한다.

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/train.py --config configs/base.yaml configs/neox.yaml configs/remember_encoder.yaml configs/neox_cpu.yaml configs/neox_grt.yaml --output-dir runs/grt_encoder_cpu --device cpu
```

위 명령의 마지막 모델 파일을 `configs/neox_rmt.yaml`로 바꾸면 동일 backbone의 RMT 인코더다.
Remember 인코더는 정답을 가린 slot에서 value+EOS를 병렬 예측한다. 정답을 입력에 넣지 않는다.
새 디코더 Remember는 `remember_encoder.yaml`을 빼고 실행한다.
설정은 왼쪽부터 병합하므로 모델 선택 파일을 크기 설정 뒤에 둔다.

small D256 그대로 직접 CPU 실험을 수행하는 명령과 결과는 [CPU 실험 보고서](docs/20261011_small256_cpu.md)에 있다.

## WikiText-103

Small의 T4 × 2 설정·메모리·시간 계산과 실행 명령은 [Kaggle 안내](docs/20261011_wikitext_kaggle.md)를 따른다.
GPT-2 BPE 50,257 옵션은 [BPE 안내](docs/20261011_wikitext_bpe.md)에 있다.
[Kaggle 노트북](notebooks/kaggle.ipynb)은 tiny BPE·200-update 동작 확인이 기본이며 word-level도 선택할 수 있다.
[크기·예산 preset 안내](docs/20261011_wikitext_presets.md)에 base/small/tiny와 W&B 기록 조건을 정리했다.

원래 tokenized v1 파일을 준비한다. `wikitext-103-raw`와 BPE는 이 설정과 다르다.

```bash
python scripts/prepare_wikitext.py --source /path/to/wikitext-103 --output-dir data/wikitext103
python scripts/train.py --config configs/base.yaml configs/neox.yaml configs/wikitext103.yaml configs/neox_grt.yaml --output-dir runs/grt_wiki_preflight --device cuda --batch-size 1 --grad-accum-steps 60 --benchmark-steps 3
```

base는 D410/16층/10 heads/FF2100, small은 D256/6층/8 heads/FF1024이다.
tiny는 D128/3층/4 heads/FF512다. `wikitext_{base,small,tiny}.yaml` 중 하나를
`wikitext103.yaml` 뒤, 모델 선택 앞에 추가한다. 기존 base 기본값은 CLI 호환을 위해 보존했다.
RMT는 마지막 모델 파일을 `neox_rmt.yaml`로 바꾼다. tied vocabulary를 포함한 실제 backbone은
base 148.18M, small 73.28M으로 두 모델에서 동일하다.

사전 측정의 `benchmark.json`에 update 시간·peak VRAM·`calibrated_override`가 저장된다.
그 override를 YAML로 저장해 학습 명령 마지막에 추가하고 `--benchmark-steps`를 제거한다.
본 학습은 사전 측정 폴더와 별개로 `--output-dir runs/grt_wiki` 등 새 경로에서 시작한다.
global batch60은 `microbatch × accumulation × world_size`로 유지해야 한다.
단일 GPU 위 예시는 1×60이다. GPU 용량에 맞춰 사전 측정으로 조절한다.
데이터 경로는 별도 YAML의 `data.options.path`로 덮어쓸 수 있다.

BPE는 `prepare_wikitext_bpe.py`로 같은 원본 파일을 별도로 준비하고,
`configs/wikitext_bpe.yaml`을 크기 설정 뒤에 추가한다. GPT-2 tokenizer만 사용하고 모델 weights는 새로 초기화한다.
small BPE의 공통 backbone은 17.61M이다. BPE-token PPL을 word-level PPL과 직접 비교하지 않는다.
base/small/tiny BPE backbone은 각각 59.01M/17.60M/7.03M이다.
노트북의 `MODEL_SIZE`로 크기, `EXPERIMENT='smoke'/'full'`로 200/200000 updates를 선택한다.
smoke는 두 모델을 각각 실행하며 평가20/학습 로그10 updates 주기다. full은 선택한 모델 하나씩 실행한다.
Kaggle 노트북은 기본으로 Hugging Face `Salesforce/wikitext`의 `wikitext-103-v1`과
`openai-community/gpt2` tokenizer를 내려받는다. CLI에서는 `pip install -e '.[wikitext]'` 후
`scripts/download_wikitext.py --output-dir data/wikitext103_hf_source`로 원본을 준비한다.

GRT는 `intermediate` register 읽기, read gate off가 기본이다. token-token만 causal하며
층간 register 경로로 미래 정보가 전달되므로 결과는 `teacher_forced_noncausal`로 기록한다.
엄격한 causal 비교 조건은 `configs/fixed_state.yaml`을 추가하고 read gate를 끈 상태다.
RMT decoder 결과는 `autoregressive`로 기록한다. WikiText 인코더 next-token 학습은 거부한다.

## 결과와 재개

같은 설정·출력 경로로 다시 실행하면 `last.pt`에서 재개한다. 설정이 바뀌면 새 경로를 쓴다.
Remember는 stage별 checkpoint/metrics, WikiText는 실행 루트에 checkpoint/metrics를 저장한다.
두 경로 모두 `resolved_config.yaml`, `metadata.json`, `summary.json/.txt`, `evaluation.json/.txt`를 기록한다.
Remember는 value+EOS exact match, WikiText는 token-weighted NLL/PPL과 score_kind를 보고한다.
W&B 기본은 off이며 연결 실패 시에도 로컬 결과를 유지한다.
WikiText는 첫 step과 `wandb.every_steps`마다 학습 loss/LR/tokens/time/VRAM을 기록하며 평가 step에는 NLL/PPL을 추가한다.
`wandb_status.json`에 연결 상태·URL·마지막 기록 step·실패 수를 남긴다. preflight는 W&B run을 생성하지 않는다.

```bash
python scripts/evaluate.py --checkpoint runs/grt_wiki/best.pt --device cuda
python scripts/evaluate.py --checkpoint runs/grt_encoder_cpu/stage_01_pairs1_key1/best.pt --samples 8 --batch-size 4 --device cpu
```

WikiText CPU 부분 검사는 `--tokens N`으로 지정한다. 평가 batch를 바꾸면 stream partition도 달라지므로
두 모델 비교에는 동일한 batch를 사용한다. 평가 명령의 결과는 기본으로 checkpoint 폴더에 기록된다.

새 Remember GRT의 gate/state trace도 `scripts/analyze.py`로 기록할 수 있다.
기존 `configs/rmt.yaml`, `grt.yaml`, `cpu.yaml`, `t4.yaml`, `kaggle.yaml`과
[노트북](notebooks/kaggle.ipynb)은 legacy 참조 실험용으로 보존했다.
legacy/native `author_rmt/1` checkpoint 읽기·평가와 vendor fixture를 유지하며 새 구조로 자동 변환하지 않는다.
원본 출처·hash는 `src/grt/vendor/armt/`, 라이선스는 [NOTICE](third_party/NOTICE.md)에 있다.
