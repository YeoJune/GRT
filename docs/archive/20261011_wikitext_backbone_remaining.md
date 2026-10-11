# WikiText-103: 공통 NeoX backbone 확정 기준과 남은 문제

2026-10-11 사용자 지시 반영. 이전 30M 후보 명세보다 이 문서를 우선한다.
확정: GPT-NeoX, parallel residual, RoPE, tied input/output embedding.
두 모델 backbone 구조/규모를 동일하게 맞춘다. 30M 예산과 router 크기 제한은 폐기한다.
Causal 설계는 사용자와 별도 논의하며 이번 작업에서 변경하지 않는다.
이는 실험 명세 업데이트다. 기존 Remember 운영 코드/설정과 vendor 원본은 변경하지 않았다.

후속 합의: [인코더](20261011_grt_encoder_spec.md)와 [디코더](20261011_grt_decoder_spec.md)의
register 읽기는 intermediate 기본/fixed_state 옵션으로 확정했다. 아래 causal 선택 대기 설명은 후속 명세가 우선한다.

## 남은 문제

1. 기준 RMT의 식별: 2022 WikiText 논문은 Transformer-XL 기반이다. 현재 프로젝트는
ARMT 저자 GPT-NeoX RMT이다. 확정 구성은 후자에 맞춘 새 WikiText 실험이며
2022 모델의 정확한 재현이 아니다. tied는 현재 프로젝트 RMT의 untied에서 명시적으로 바뀐다.

2. Backbone 수치: layers/D/FF/heads는 아직 선택되지 않았다. V, activation, norm eps,
initialization, dropout, rope pct/base/scaling/max positions도 공통 config에서 하나로 고정해야 한다.
현재 RMT 값은 GELU, eps1e-5, init .02, rope pct .25/base10000/max positions2048이다.
이 값은 확인된 기존값이지 이번 LM의 확정값이 아니다. rotary dimension의 짝수 조건을 확인한다.
현행 config는 head dimension이8의 배수인 조건을 둔다.

3. Tied 실제 연결: 현재 make_rmt는 config false를 강제하고 validator도 tied를 거부한다.
공통 LM factory/validation에서 tied를 지원하고 실제 embed_in/embed_out Parameter 공유,
optimizer 중복 없음, save/load 후 공유 보존을 검증해야 한다.
기존 Remember 체크포인트는 untied이고 embedding/head가 일반적으로 서로 다르므로
새 tied 모델로 동일 동작을 유지하며 이어 학습할 수 있다고 가정하면 안 된다.
기존 baseline/fixture는 그대로 보존하고 새 LM 실험 경로로 분리한다.

4. GRT backbone 교체: ALU를 NeoX layer 경로로 바꾸고 learned absolute position을 제거한다.
NeoX final layer norm/LM head를 포함하는 경계를 RMT와 동일하게 정의한다.
GRT router/write-back은 보존하지만 메모리 후보를 final norm 전/후 어디서 취하는지 명시해야 한다.
RMT MemoryCell은 최종 hidden_states[-1](NeoX final norm 후)을 사용한다.
공통 backbone 초기 weights를 복제할지, 같은 분포만 사용할지도 미정이다.

5. RoPE 위치 배치: RMT는 read+tokens+write, 기존 GRT는 tokens+register 배치이다.
자동 position_ids라면 M 때문에 본문 위치가 달라진다. 본문/메모리 위치 정책,
segment마다 reset 또는 global offset, memory가 다음 segment에서 받는 position을 정의해야 한다.
동일 RoPE라는 사실만으로 동일 position 배치를 보장하지 않는다.

6. Causal(별도 논의): NeoX attention은 내부 triangular mask를 강제하고 현재 public
attention_mask는2D padding mask로 처리한다. 따라서 GRT의 양방향 처리를 이식하거나
원문 RMT memory block 내부 양방향 접근을 추가하려면 별도 block-mask 지원이 필요하다.
full-segment router/read gate, token->register->token 경로의 미래 누출을 함께 논의해야 한다.
이번에는 처리 방식을 선택하거나 구현하지 않는다.

7. Tokenizer/output: 예산 제한이 사라졌으므로 원문 word vocabulary 유지도 후보로 가능하다.
word267735 또는 공통 subword, 데이터 버전, EOS/document 처리, 출력 full/adaptive softmax를 정한다.
NeoX 기본 head는 full vocabulary linear이고 원문은 adaptive softmax이다.
원문 word PPL과 비교하려면 tokenizer와 평가 protocol을 일치시켜야 한다.
full logits의 메모리/시간은 vocab에 크게 좌우된다.

8. State/BPTT: 현재 양 모델 forward는 호출 시작에서 memory를 reset한다.
연속 LM에는 state 입출력, stream ID, document/split/epoch reset, detach 규칙이 필요하다.
vendor manage_gradients는 로컬 변수에 memory_state.detach()를 대입하고 caller에
반환하지 않는다. 코드상 caller state를 교체하지 않으므로 k2만으로 truncated BPTT를
구현했다고 볼 수 없다. 작은 gradient 검사로 원하는 경계를 확인해야 한다.
이것은 코드 검토 발견이며 아직 재현 테스트/수정을 하지 않았다.

9. Loss/evaluation/trainer: raw-label 내부shift를 유지하고 segment boundary next-token
예측과 tail/padding을 누락/중복 없이 처리한다. NLL sum과 target count로PPL을 계산한다.
현재 trainer의 Remember stage/EM, AdamW/value clip/linear schedule은 새 LM용으로 확장해야 한다.
원문 공개 실행 기준 Adam/cosine/norm clip을 따를지, 별도 조건을 사용할지 미정이다.
학습 token 예산/global batch/segment M/BPTT/seed/GPU시간이 남아 있다.

10. 비교 해석: backbone 동일이어도 memory layout, router, prefix 재계산에 따라
전체 파라미터/FLOPs/VRAM/throughput이 다르다. 크기를 억지로 맞추지 않고 실제값을 기록한다.
공통 데이터 순서와 target budget을 고정하려면 모델 RNG와 데이터 RNG를 분리한다.

## 후속 검증

실제 공유 backbone 명세와 고유 Parameter 수, tied save/load, 동일 초기화 옵션,
RoPE position_ids, causal 미래변경 불변성(논의 후), memory write 횟수,
BPTT gradient 경계, stream reset, segment boundary loss, token-weighted PPL,
두 모델 peak VRAM/처리량. 이번에는 문서와 코드 검토만 완료했다.
