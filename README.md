# GRT

GRT와 RMT를 공통 GPT-NeoX backbone에서 비교한다. Remember 인코더·디코더와 WikiText-103 디코더를 지원한다.
[통합 명세](docs/20261011_current_spec.md)는 모델·데이터 계약,
[실험 workflow](docs/20261011_experiment_workflow.md)는 설정·실행·확장 규칙을 정의한다.
archive에는 과거 Markdown 문서만 보존한다. 코드와 설정은 현재 명세만 지원한다.

## 설치와 실행

Python 3.10+, PyTorch를 사용한다.

```bash
python -m pip install -e '.[dev,wandb,wikitext]'
python scripts/train.py --task remember --model grt --size tiny --budget smoke --runtime cpu --steps 2 --set model.attention_mode=encoder --set data.options.objective=parallel --output-dir runs/remember_encoder_cpu --device cpu
```

Remember 인코더는 정답을 가린 slot에서 value+EOS를 병렬 예측한다. 디코더는 두 `--set` 옵션을 제거한다.
RMT는 `--model rmt`로 선택한다. [로컬 템플릿](notebooks/template.ipynb)도 같은 인터페이스를 사용한다.

## WikiText-103 · Kaggle T4 × 2

[Kaggle 노트북](notebooks/kaggle.ipynb)은 실험 선택 → 환경 준비 → 데이터 준비 → 사전 측정 → 학습 → 결과 확인만 담당한다.
반복 기능은 [Kaggle utils](src/grt/utils/kaggle.py)에 둔다. Internet과 T4 × 2를 켜고 `WANDB_API_KEY` secret을 등록한다.
기본은 tiny·smoke·RMT/GRT 각각 200 updates이며, small 2만 updates는 다음 선택으로 실행한다.

```python
job = WikiTextRun(repo_dir=repo, size='small', budget='20k', models=('rmt',))
```

| 선택 | 값 |
|---|---|
| size | base D410/L16/H10/FF2100 · small D256/L6/H8/FF1024 · tiny D128/L3/H4/FF512 |
| budget | smoke 200 · 20k 20000 · full 200000 updates |
| models | `('rmt',)` / `('grt',)` / `('rmt', 'grt')` 순차 실행 |
| tokenization | gpt2_bpe(기본, 50257) / word(267735) |

GPT-2 BPE backbone은 base 59.01M, small 17.60M, tiny 7.03M이며 router 크기는 별도다.
원본은 Hugging Face `Salesforce/wikitext`의 `wikitext-103-v1`, tokenizer는 `openai-community/gpt2`다.
Tokenizer만 다운로드하며 모델 weights는 새로 초기화한다. BPE-token PPL과 word-level PPL을 구분한다.

`steps=30000` 또는 `overrides={'evaluation': {'every_steps': 1000}}`로 변경할 수 있다.
이런 숫자 변경에는 새 YAML이나 노트북을 만들지 않는다. 모든 preset은 [단일 카탈로그](configs/experiments.yaml)에서 관리한다.
사전 측정은 global batch60을 유지하며 microbatch·accumulation을 보정하고, 최종 Config와 ETA를 모델별로 저장한다.
CLI도 동일 카탈로그를 지원한다.

```bash
python scripts/download_wikitext.py --output-dir data/wikitext103_hf_source
python scripts/prepare_wikitext_bpe.py --source data/wikitext103_hf_source --output-dir data/wikitext103_gpt2
# torchrun 2 GPU: 사전 측정 예시
torchrun --standalone --nproc_per_node=2 scripts/train.py --task wikitext103 --model rmt --size small --tokenization gpt2_bpe --budget 20k --runtime kaggle --output-dir runs/small_rmt_20k_preflight --device cuda --benchmark-steps 10
```

CLI 본 학습에는 benchmark의 calibrated_override를 반영한 최종 Config를 사용한다. 노트북 utils가 이를 자동 처리한다.
GRT 기본 intermediate/read off는 `teacher_forced_noncausal`, RMT decoder는 `autoregressive`다.
GRT의 엄격한 causal 조건은 `--set model.register_read_source=fixed_state`이며 read gate off를 유지한다.

## 결과와 재개

같은 최종 설정·world_size·데이터·출력 경로로 다시 실행하면 last.pt에서 재개한다. 조건이 바뀌면 새 run을 사용한다.
숫자 예산/override를 바꾸면 utils의 기본 run 이름에 설정 hash가 붙는다. 기존 run과의 설정 차이는 학습 전에 거부한다.
다른 Kaggle 세션의 데이터/결과는 `job.prepare_data(prepared_input=..., resume_input=...)`로 복원한다.

Config/metadata, summary/evaluation JSON·TXT, metrics JSONL·TXT, checkpoint를 기록한다.
Remember는 stage별 value+EOS exact match, WikiText는 token-weighted NLL/PPL과 score_kind를 보고한다.
W&B는 Kaggle에서 기본 활성화하며 첫 step·주기별·평가·마지막 step을 기록한다.
`wandb_status.json`에 URL·마지막 step·실패 수를 남기며 utils가 완료 상태를 확인한다. preflight는 W&B run을 생성하지 않는다.

```bash
python scripts/evaluate.py --checkpoint /kaggle/working/wiki_small_gpt2_bpe_20k_seed54/rmt/best.pt --device cuda
```

평가 batch가 달라지면 stream partition도 달라지므로 비교 실험에서는 동일하게 유지한다.
`--config`는 현재 실행용 최종 Config YAML 하나를 읽는다. 이전 config 경로·모델 구현·checkpoint 형식은 지원하지 않는다. checkpoint는 `memory_experiment/3`만 읽으며 현재 형식으로 시작한 run을 재개한다. 출처·hash는 `src/grt/vendor/armt/`, 라이선스는 [NOTICE](third_party/NOTICE.md)에 있다.
