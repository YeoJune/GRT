# GRT LM 아키텍처 합의 정리

2026-10-11. 20260608 spec v0.3.0의 router/ALU/write-back을 기준으로 사용자 논의를 반영했다.
범위: 새 WikiText LM 설계. 기존 Remember 구현/실험은 변경하지 않았다.
상태: 아래 합의는 확정. register 읽기 정책은 intermediate 기본/fixed_state 옵션으로 확정했다. 세부 구현은 미완료.

후속 정리: [인코더 명세](20261011_grt_encoder_spec.md), [디코더 명세](20261011_grt_decoder_spec.md).
공통 추가 합의: 전체 Transformer stack 종료 후 segment당 write-back 1회, 위치별 prefix 재계산 제거.

## 확정 사항

- RMT/GRT 공통 backbone: GPT-NeoX, parallel residual, RoPE, tied embedding/LM head.
- backbone의 층수/D/FF/heads 및 세부 옵션은 두 모델에서 동일하게 맞춘다. 수치는 미정.
- 30M 예산은 적용하지 않는다. GRT router 크기나 전체 parameter 수를 맞추기 위해 backbone을 바꾸지 않는다.
- router는 ALU 출력이 아닌 입력 X와 이전 state S를 받는다.
- read gate는 conditional_read 옵션으로 on/off, 기본 false.
- write gate는 항상 write-back에 사용한다.
- ALU 입력은 [X; S_read], 출력은 [Y; C]이다.
- register_read_source=intermediate를 기본으로 하고 fixed_state를 옵션으로 둔다. 두 버전에 공통 적용한다.
- 토큰->토큰 attention은 causal로 만든다.
- 저장 state S는 모든 Transformer 층 계산이 끝난 뒤 한 번 write-back하며 S_next는 다음 segment에서 사용한다.
- 최종 state 갱신을 층마다 수행하거나 현재 segment 예측에 S_next를 되먹이지 않는다.

## 수식과 순서

X = embedding(ids). RoPE는 attention Q/K에 적용하며 learned absolute position은 새 NeoX 경로에서 사용하지 않는다.
(u, alpha) = AttentionPool(q_pool, X)
v = RegisterCrossAttention(u, S, S)
g = concat(u, v)
R = sigmoid(MLP_read(g)); W = sigmoid(MLP_write(g))
S_read = R*S if conditional_read else S
[Y; C] = NeoX_ALU([X; S_read], block_attention_mask, position_ids)
logits = tied_lm_head(Y)
S_next = (1-W)*S + W*C

R/W는 register별 scalar [B,M,1], state/candidate는[B,M,D].
read=false에서 read branch를 계산할지 생략할지는 구현 선택이며 모델 예측에는 반영하지 않는다.
read branch에만 있는 파라미터는 기본 task loss gradient를 받지 않는다.
초기 state, write bias/dropout, router hidden 등의 수치는 20260608 값을 참고할 수 있지만 이번 LM에서 확정한 값이 아니다.

## 합의와 구분해야 할 causal 조건

최종 S_next가 다음 segment에만 쓰이는 것은 정상적인 과거 정보 사용이다.
그러나 [X;register]를 층마다 함께 변환하면 저장 state와 별개로 register 중간 표현 H_R^(l)이 생긴다.
register가 전체 X를 읽고 다음 층 token이 H_R^(l)을 읽도록 허용하면,
미래 X -> H_R^(1) -> 과거 token^(2) 경로가 가능하다. read=false는 이 경로를 제거하지 않는다.
따라서 token-token triangular mask만으로 전체 causal임을 보장했다고 표현하지 않는다.
사용자 합의로 intermediate를 기본 채택했다. 각층 token은 이전층 register hidden state를 읽는다.
fixed_state 옵션에서는 token이 segment 시작 S_read를 모든층에서 읽고 candidate 중간표현은 읽지 않는다.
기본 intermediate는 token-token causal이며 register 경로까지 완전한 causal이라고 주장하지 않는다.

read=true에서 full-segment router로 R을 계산해 현재 token에 적용하면 별도 미래 정보 경로가 생긴다.
read=true LM 실험은 gate의 정보 범위까지 정한 후 수행한다. 기본 false는 해당 경로를 제거한다.
W는 현재 segment 전체를 보아도 다음 state에만 적용하면 이 이유로 미래 누출이 되지 않는다.

## 미확정 실행 조건과 검증

fixed_state read/candidate 경로의 세부 구현, memory candidate final norm 위치,
RoPE position IDs, tokenizer/output softmax, layers/D/FF/heads,
segment/memory/BPTT/stream reset, optimizer/schedule/token budget/GPU/precision/seed는 미정.
새 NeoX 공통 경로, tied parameter/save-load, intermediate의 미래정보 경로 기록 및 fixed_state/read=false의 미래-token 변경 불변성,
최종 write-back 한번, state gradient 경계, segment boundary target과 token-weighted PPL을 검증한다.
이번 작업은 설계 기록이며 새 LM 모델 구현/학습 검증은 하지 않았다.
