# WikiText-103 GPT-2 BPE와 Kaggle 노트북

사용자 선택: GPT-2 기존 byte-level BPE, vocabulary 50257. 기본 다운로드는 Hugging Face `openai-community/gpt2` tokenizer와 `Salesforce/wikitext`의 `wikitext-103-v1`이다. 새 tokenizer 학습은 하지 않는다. 기존 word-level 267735 경로는 보존한다.

## 데이터·모델 계약

- Hugging Face의 tokenized v1 text rows를 내려받아 `wiki.train.tokens`, `wiki.valid.tokens`, `wiki.test.tokens`로 내보낸다. 기존 개행은 보존하고 없는 경우 LF를 붙이며 빈 row는 빈 줄로 내보낸다. raw 데이터로 자동 대체하지 않는다. BPE를 적용해도 원본 tokenized 데이터에서 이미 사라진 정보는 복원되지 않는다.
- GPT-2 tokenizer만 불러온다. GPT-2 pretrained model weights를 사용하지 않으며 NeoX는 기존 seed로 초기화한다.
- 각 줄에서 CR/LF만 제거하고 나머지 공백·대소문자를 그대로 BPE 인코딩한다. `add_prefix_space=False`, truncation/padding/BOS 없음. 줄마다 EOS 50256을 한 번 덧붙이며 빈 줄에도 EOS를 넣는다. EOS에서 memory를 reset하지 않는다.
- tokenizer vocabulary/merges와 직렬화 파일, 실제 tokenizer SHA256, source/token stream SHA256, split별 token 수를 manifest로 남긴다. 두 모델이 같은 prepared 데이터·tokenizer를 공유한다. 로컬 GPT-2 tokenizer 경로도 지원한다.
- BPE 설정은 `wikitext103.yaml` 뒤에 `wikitext_bpe.yaml`을 추가한다. adapter 계약은 기존 int64 1-D stream과 next-token target 그대로이며 task 전용 trainer 분기를 추가하지 않는다.
- base/small/tiny는 [preset 안내](20261011_wikitext_presets.md)에서 선택한다. L150/M10/이전3 segment BPTT, FP32, global batch60, optimizer/LR/dropout은 유지한다. full은 기존 200000 updates, smoke는 별도 200-update cosine schedule이다. L150은 BPE token 150개라 word-level과 문맥의 단어 수가 다르다. full의 18억 target 상한도 BPE tokens 기준이다.
- PPL은 BPE-token PPL이다. word-level PPL과 직접 비교하지 않는다. GRT intermediate의 `teacher_forced_noncausal`와 RMT의 `autoregressive` 표시는 유지한다.

small embedding/head는 12,865,792 parameters, 공통 backbone은 17,604,864이다. 전체 RMT는 17,607,424, GRT는 18,669,588이다. full-softmax 연산은 줄지만 vocabulary 비율만큼 전체 학습이 빨라진다고 보장할 수 없다. 기존 근사식의 dense 연산은 GRT 약 1.90 / RMT 약 1.98 TFLOP/update, microbatch 10 logits 한 텐서는 약 0.281 GiB다. 정확한 T4 시간과 peak는 다시 preflight한다.

## 실행

```bash
python -m pip install -e '.[wikitext]'
python scripts/download_wikitext.py --output-dir data/wikitext103_hf_source
python scripts/prepare_wikitext_bpe.py --source data/wikitext103_hf_source --output-dir data/wikitext103_gpt2
torchrun --standalone --nproc_per_node=2 scripts/train.py --config configs/base.yaml configs/neox.yaml configs/wikitext103.yaml configs/wikitext_small.yaml configs/wikitext_bpe.yaml configs/neox_rmt.yaml configs/wikitext_kaggle.yaml --output-dir runs/rmt_bpe_preflight --device cuda --benchmark-steps 10 --memory-fraction 0.8
python scripts/estimate_wikitext_time.py --benchmark runs/rmt_bpe_preflight/benchmark.json --manifest data/wikitext103_gpt2/manifest.json
```

네트워크 없는 환경에서는 `--tokenizer /path/to/saved_gpt2_tokenizer --local-files-only`를 추가한다. 모델 weight 다운로드는 필요 없다. 기본 remote tokenizer는 `openai-community/gpt2`다. 재개에서는 새로 준비하지 않고 기존 manifest와 tokenizer 파일을 함께 보존한다.

## Kaggle

최신 `notebooks/kaggle.ipynb`는 base/small/tiny WikiText용이며 기본은 tiny·GPT-2 BPE·smoke다. `TOKENIZATION='word'`로 기존 word-level도 선택한다. 기존 Remember 노트북은 `docs/archive/20261011_kaggle_remember.ipynb`에 보존한다. 유지하는 notebook은 `template.ipynb`와 `kaggle.ipynb` 두 개다.
첫 코드 셀이 저장소가 없으면 GitHub `main`을 `/kaggle/working/GRT`에 자동 clone한다. 기존 git checkout은 로컬 수정이 없을 때만 fast-forward로 갱신한다. 로컬 수정이 있으면 갱신을 생략하며 오프라인에서는 `UPDATE_REPO=False`를 사용한다. 저장소를 수동으로 준비해야 했던 첫 셀의 assertion을 제거했다.

설정 cell에서 크기와 실험 예산을 선택한다. 기본 `DOWNLOAD_WIKITEXT=True`는 데이터를 Hugging Face에서 다운로드하며 별도 Kaggle 입력 dataset은 필요 없다. 기존 로컬 `.tokens` 파일을 쓰려면 false로 바꾸고 SOURCE_DIR을 지정한다. T4 두 장 확인/W&B 로그인 → 다운로드·데이터 준비 → RMT/GRT 각각 preflight와 ETA → calibrated override 저장 → 본 학습/재개 → 결과 확인 순서다. 기본 학습 cell은 두 모델 각각200 updates, full에서는 선택한 모델200000 updates를 실행한다. 세션 종료 시 저장 폴더를 Kaggle output/dataset으로 보존하고, 재개 cell에서 이전 output을 writable working 경로로 복사한다.

세션 quota에 따라 schedule을 자동 축소하지 않는다. 명시한 smoke/full schedule을 사용한다. checkpoint는 validation(smoke20/full4000 updates)과 정상 종료 시 저장된다. 강제 종료 시 최근 진행분이 유실될 수 있다.

## 검증과 미검증

전체 CPU 회귀 검사 54개가 통과했다. 추가 검사는 모든 byte와 GPT-2 ID 범위를 갖는 로컬 BPE fixture를 사용한다. 실제 GPT-2 merges를 사용하는 fixture는 아니다. 공백·Unicode·빈 줄 EOS·긴 줄 무손실 처리, tokenizer artifact 변조 거부, 노트북의 실제 helper/config/preflight/train cell을 통한 RMT/GRT CPU 학습·재개·PPL 단위 기록을 확인했다. 실행 검사는 작은 D32 stack, 1 worker, 2 updates이며 GPU 성능/수렴 결과가 아니다. 모델 수는 factory에서 small 설정을 직접 생성해 확인했다.

사용자 요청으로 GPT-2 tokenizer.json과 WikiText-103 v1 train parquet에 각각 한 번 다운로드를 시도했으나 두 요청 모두 프록시 `403 Forbidden`으로 차단됐다. 위54개 통과 기록은 다운로드 경로 변경 전이다. 이후 사용자가 Kaggle small BPE/T4 두 장의 preflight 완료 결과를 제공했다. 실제 시간·VRAM과 최신 tiny/로깅 검증은 [preset 안내](20261011_wikitext_presets.md)에 있다. 200-update 실제 WikiText GPU 실행과 온라인 W&B 업로드는 Kaggle에서 확인한다.
