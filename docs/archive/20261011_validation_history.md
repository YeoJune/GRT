# 이전 구현 검증 기록

> 당시 코드에 대한 기록이며 현재 실행/검증 기준이 아니다. 과거 구현·config·fixture는 보존하지 않는다.

## 9. 검증 결과와 실행 전 남은 항목

공통 구조 구현 시 `python -m pytest -q`: **51 passed**, 93.96초(CPU). 새 경로 검사 18개를 포함한다.
GPT-2 BPE와 WikiText Kaggle 노트북 추가 후 전체 CPU 회귀는 **54 passed**, 108.49초다.
후속 Hugging Face 기본 다운로드 경로 변경에서는 사용자 지시에 따라 추가 학습·회귀 검사를 실행하지 않았다.
tiny/smoke와 주기별 W&B 로깅 변경은 실제 tiny CPU2 updates·offline SDK 두 검사(3.96초), 실패 상태 기록 한 검사와 가벼운 preset routing 확인에 한정했다. 전체 회귀는 반복하지 않았다.
추가 검사 범위와 실제 tokenizer/GPU 미검증 조건은 [BPE 안내](20261011_wikitext_bpe.md)의 검증 절을 따른다.

- vendor 원본 hash와 저자 RMT loss/gradient/generate fixture 회귀.
- 두 모델 backbone 초기 weight 일치, tied Parameter 공유 및 save/load 후 유지.
- Remember 정답 제외 입력, value_size=1/2, slot-aligned CE, fact→query gradient,
  독립 sampling RNG, segment별 stack/write-back 횟수, read off의 출력 불변.
- fixed_state/read=false와 RMT의 미래 token 변경 불변 및 nonzero incoming memory에서
  가변 길이 prefix 불변. intermediate의 register 우회 경로 영향 확인.
- replay 이전 detached state에는 gradient 없음, replay의 이전 3 segments에는 gradient 전달.
- encoder Remember 학습과 정확한 재개, WikiText 두 모델의 dropout 포함 정확한 재개.
- WikiText boundary target 수/NLL 정렬/PPL, CPU 2-rank 학습·state checkpoint·preflight,
  microbatch×accumulation×world_size 예산 유지, 인코더 LM 목적의 명시적 거부.

별도 CLI로 새 GRT 인코더의 4 update 학습, checkpoint 평가와 NPZ/JSON/PNG trace 생성을 확인했다.
기능 확인용 결과는 `runs/neox_encoder_check/`, `runs/neox_encoder_trace/`에 있다.
4 update의 exact match는 0으로, 이 검사는 학습 경로 확인이며 수렴 결과가 아니다.
`git diff --check`와 Python compile 검사도 통과했다.

GPU 수렴과 실제 WikiText-103 전체 데이터 학습은 아직 검증하지 않았다. 사용자가 제공한 small BPE/T4 두 장의
preflight 시간·VRAM은 [preset 안내](20261011_wikitext_presets.md)에 기록했다. 사용자가 전달한 tiny/smoke GPU 실행은 두 모델 모두 200 updates 완료했고 온라인 W&B도 failures=0, last_logged_step=200으로 종료했다. tiny/full 수렴과 시간은 아직 확인하지 않았다.
기본 intermediate의 teacher-forced_noncausal PPL을 엄격한 autoregressive RMT PPL과
동일 조건으로 해석해서는 안 된다. 정확한 RMT 2022 Transformer-XL/adaptive-softmax 재현도 아니다.

후속 small D256 변경과 실제 인터페이스 호출 결과: [CPU 실험 보고서](20261011_small256_cpu.md).
Remember 4조건과 full-vocabulary 합성 LM 2조건을 실행했다. 본 WikiText 데이터 결과와 구별한다.

