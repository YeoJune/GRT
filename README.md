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
기본은 tiny·smoke·RMT/GRT 각각 200 updates이며, small 1만 updates는 다음 선택으로 실행한다.

```python
segment_len = 150
num_registers = 10
bptt_segments = 3
job = WikiTextRun(repo_dir=repo, size='small', budget='10k', models=('rmt',),
                  overrides={'model': {'segment_len': segment_len,
                                       'num_registers': num_registers},
                             'training': {'bptt_segments': bptt_segments}})
```

| 선택 | 값 |
|---|---|
| size | base D410/L16/H10/FF2100 · small D256/L6/H8/FF1024 · tiny D128/L3/H4/FF512 |
| budget | smoke 200 · 10k 10000 · 100k 100000 updates |
| models | `('rmt',)` / `('grt',)` / `('rmt', 'grt')` 순차 실행 |
| tokenization | gpt2_bpe(기본, 50257) / word(267735) |
| segment_len | segment당 토큰 수, 기본150; RMT/GRT 공통 |
| num_registers | 메모리 개수, 기본10; RMT는 같은 개수의 read/write 블록을 각각 사용 |
| bptt_segments | gradient를 되돌릴 이전 segment 수, 기본3; 0 이상 |

WikiText는 문장별 길이 제한 없이 연속 스트림으로 처리한다. 전체 segment 수는 스트림 길이와 segment_len에 따라 정해지며 마지막 짧은 segment도 사용한다. bptt_segments는 과거 토큰의 재계산·gradient 범위만 제한하며, 그 이전 문맥의 정보는 저장 state로 이어진다. 스트림 분할 시 batch_size로 나누어떨어지지 않는 끝의 소수 토큰은 제외하고 metadata에 기록한다.

GPT-2 BPE backbone은 base 59.01M, small 17.60M, tiny 7.03M이며 router 크기는 별도다.
원본은 Hugging Face `Salesforce/wikitext`의 `wikitext-103-v1`, tokenizer는 `openai-community/gpt2`다.
Tokenizer만 다운로드하며 모델 weights는 새로 초기화한다. BPE-token PPL과 word-level PPL을 구분한다.

`steps=30000` 또는 `overrides={'evaluation': {'every_steps': 1000}}`로 변경할 수 있다.
이런 숫자 변경에는 새 YAML이나 노트북을 만들지 않는다. 모든 preset은 [단일 카탈로그](configs/experiments.yaml)에서 관리한다.
Kaggle은 global batch256, microbatch32×accumulation4×GPU2, Adam LR5e-4로 시작한다. 자동 조정은 Kaggle에서 기본 on이며 목표 VRAM80% 안에서 microbatch128→64→32…를 선택해 global256을 유지한다. off는 config 배치를 그대로 쓴다. 사전 측정은 최종 Config와 ETA를 모델별로 저장한다.
10k/100k은 validation1000/logging10 updates이며 segment150의 target 예산 상한은 각각384M/3.84B token이다. segment 길이 변경 시 토큰 예산은 global batch×segment_len×updates로 달라지며 LR·updates·BPTT segment 수는 자동 변경하지 않는다. 두 변수는 양의 정수이며 변경 시 기본 run 이름의 hash도 바뀐다. 설정 변경 시 새 run을 사용한다.
CLI도 동일 카탈로그를 지원한다.

```bash
python scripts/download_wikitext.py --output-dir data/wikitext103_hf_source
python scripts/prepare_wikitext_bpe.py --source data/wikitext103_hf_source --output-dir data/wikitext103_gpt2
# torchrun 2 GPU: 사전 측정 예시
torchrun --standalone --nproc_per_node=2 scripts/train.py --task wikitext103 --model rmt --size small --tokenization gpt2_bpe --budget 10k --runtime kaggle --output-dir runs/small_rmt_10k_preflight --device cuda --benchmark-steps 10
```

공통 `training.auto_batch` 기본은 false이며 Kaggle runtime에서 true로 켠다. 노트북은 `overrides={'training': {'auto_batch': False}}`, CLI는 `--set training.auto_batch=false`로 끈다. 목표는 `training.memory_fraction`(기본0.8)으로 지정한다. on이면 신규 CUDA 학습에서 필요한 사전 측정을 수행하고, 기존 측정/현재 run의 보정 배치를 재사용한다. 실질 배치와 평가 batch는 유지하며 CPU와 off에서는 자동 조정하지 않는다.
GRT는 segment 시작 state를 읽는 구조가 유일한 표준이며 read gate는 기본 off다. gate를 켜도 segment 시작 state만으로 계산하므로 GRT/RMT decoder는 모두 `autoregressive`다. candidate와 write gate는 전체 segment로 계산해 stack 종료 후 다음 state에만 반영한다.
이전 중간 register 읽기 방식은 병렬 학습에서 미래 정보 leakage가 발생할 수 있어 제거했다. 설계 이유는 [통합 명세](docs/20261011_current_spec.md)에만 기록한다. 읽기 source 선택 옵션과 이전 실행 설정의 호환 경로는 없다. 변경된 모델은 새 run으로 실행한다.

## 결과와 재개

같은 최종 설정·world_size·데이터·출력 경로로 다시 실행하면 last.pt에서 재개한다. 조건이 바뀌면 새 run을 사용한다.
숫자 예산/override를 바꾸면 utils의 기본 run 이름에 설정 hash가 붙는다. 기존 run과의 설정 차이는 학습 전에 거부한다.
다른 Kaggle 세션의 데이터/결과는 `job.prepare_data(prepared_input=..., resume_input=...)`로 복원한다.

Config/metadata, summary/evaluation JSON·TXT, metrics JSONL·TXT, checkpoint를 기록한다.
Remember는 stage별 value+EOS exact match, WikiText는 token-weighted NLL/PPL과 score_kind를 보고한다.
W&B는 Kaggle에서 기본 활성화하며 첫 step·주기별·평가·마지막 step을 기록한다.
`wandb_status.json`에 URL·마지막 step·실패 수를 남기며 utils가 완료 상태를 확인한다. preflight는 W&B run을 생성하지 않는다.

```bash
python scripts/evaluate.py --checkpoint /kaggle/working/wiki_small_gpt2_bpe_10k_b256_seed54/rmt/best.pt --device cuda
```

평가 batch가 달라지면 stream partition도 달라지므로 비교 실험에서는 동일하게 유지한다.
`--config`는 현재 실행용 최종 Config YAML 하나를 읽는다. 이전 config 경로·모델 구현·checkpoint 형식은 지원하지 않는다. checkpoint는 `memory_experiment/4`만 읽으며 현재 형식으로 시작한 run을 재개한다. 출처·hash는 `src/grt/vendor/armt/`, 라이선스는 [NOTICE](third_party/NOTICE.md)에 있다.
