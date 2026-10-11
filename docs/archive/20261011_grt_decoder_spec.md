# GRT 디코더 버전 명세

작성일: 2026-10-11. 상태: 설계 문서, intermediate 기본/fixed_state 옵션 확정, 미구현.
사용자 합의: 토큰 부분 causal, read 기본 off, 전체 Transformer 처리 후 write-back,
위치별 prefix 재계산 제거. 기존 Remember 운영 코드와 결과는 변경하지 않는다.

## 공통 구조와 갱신 시점

기준은 20260608 spec의 입력 기반 router, register bank, gated write-back이다.
새 backbone 기준은 GPT-NeoX 계열 layer, parallel residual, RoPE, tied embedding/출력 head다.
인코더/디코더 명칭은 attention 방식의 구분이며 encoder-decoder 두 stack을 뜻하지 않는다.
두 버전의 layers/D/FF/heads 및 backbone 세부 옵션을 동일하게 맞춘다. 수치는 아직 미정이다.
30M 예산과 router 크기 제한은 적용하지 않는다.

```text
X_t = E[input_ids_t]                           # [B,N,D]
(u_t, alpha_t) = AttentionPool(q_pool, X_t)
v_t = RegisterCrossAttention(u_t, S_t, S_t)
g_t = concat(u_t, v_t)
R_t = sigmoid(MLP_R(g_t)); W_t = sigmoid(MLP_W(g_t))  # [B,M,1]
S_read = R_t*S_t if conditional_read else S_t
H_0 = concat(X_t, S_read)                     # [B,N+M,D]
H_K = TransformerStack(H_0, mask, position_ids)
Y_t = H_K[:, :N]; C_t = H_K[:, N:]
logits_t = Y_t @ E.T
S_next = (1-W_t)*S_t + W_t*C_t
```

read gate는 on/off 가능하고 기본 conditional_read=false. write gate는 항상 적용한다.
Router는 ALU 출력이 아니라 입력 token embedding과 이전 register를 사용한다.
RoPE는 Q/K에 적용하며 learned absolute position을 중복 추가하지 않는다.
공통 final norm 및 candidate 추출 위치는 RMT와 함께 확정한다.

### write-back 계약

여기서 'Transformer 블록이 끝난 시점'은 개별 layer마다가 아니라,
해당 segment의 전체 K-layer Transformer stack 계산이 완료된 시점을 뜻한다.
S_t는 stack 처리 중 저장 state로서 고정한다. 최종 C_t가 나온 뒤 segment당 write-back을 한 번 수행한다.
S_next는 다음 segment에서만 사용하며 현재 segment 예측을 위해 되먹이지 않는다.
마지막 segment도 state 출력이 필요한 경우 동일하게 갱신한다.
state의 층간 중간 표현 H_R^(l)과 실제 저장 state S_t는 구분한다.

### prefix 제거 계약

학습/평가에서 위치별 prefix를 잘라 router/ALU를 반복 호출하는 경로를 제거한다.
segment당 router와 Transformer stack을 한 번 호출하고 전체 token 출력 Y_t를 사용한다.
prefix cell 후보를 임시 생성하거나 prefix별 write-back을 수행하지 않는다.
Autoregressive generation의 cache/증분 계산 방식은 이 문서에서 확정하지 않는다.
'prefix 제거'가 미래 정답을 생성 입력에 넣어도 된다는 뜻은 아니다.

### Register 읽기 정책

사용자 확정: 인코더와 디코더 모두 register_read_source의 기본값은 intermediate이다.
- intermediate (기본): layer l의 token attention은 이전 layer의 register 중간 표현 H_R^(l-1)을 읽는다.
  layer1에서는 H_R^(0)=S_read이며 token/register hidden states를 함께 다음 layer로 전달한다.
- fixed_state (옵션): 모든 layer의 token attention이 해당segment 시작의 S_read를 memory source로 읽는다.
  각layer의 normalization 및 K/V projection은 적용할 수 있으나 source는 갱신된 H_R로 대체하지 않는다.
  candidate 계산에 필요한 register 중간 표현은 별도로 유지하고 token의 memory 읽기에는 노출하지 않는다.
  단순히 register hidden state 전체를 매layer 초기화하여 candidate 경로까지 없애는 방식은 아니다.

두 정책 모두 최종 candidate C로 stack 종료 후 write-back을 한 번 수행한다.
이 옵션은 read gate on/off와 독립이다. 기본 조합은 intermediate + conditional_read=false.
fixed_state는 detach를 뜻하지 않는다. S_read로의 gradient 전달/BPTT는 별도 정책이다.
옵션 이름은 문서상 명세이며 현재 운영 config에 구현된 필드가 아니다.
fixed_state의 candidate/read 경로 분리를 위한 세부 layer 구현은 후속 구현에서 명시한다.

### 미확정 값

N/M/K/D/FF/heads, tokenizer, position IDs와 RoPE pct/base/scaling,
초기 memory, router hidden, write bias/dropout, backbone dropout/init/norm eps,
BPTT/reset, 학습/평가 예산은 미정이다. 기존 spec의 수치를 자동 채택하지 않는다.

## 디코더 attention: 확정 범위

토큰 i는 token j<=i만 읽는다. token-token block에 lower-triangular mask를 적용한다.
read=false에서는 전체segment router가 계산한 R을 ALU에 적용하지 않는다.
W는 segment 전체를 볼 수 있다. W와 최종 C는 다음state에만 적용한다.

| Query → Key | Tokens | Registers |
|---|---|---|
| Tokens | causal: j<=i | 전체 register; intermediate 기본, fixed_state 옵션 |
| Registers | 전체 tokens | 전체 registers |

토큰만 causal이라는 합의를 전체 [tokens;registers] 단순 triangular mask로 대체하지 않는다.
현재 순서에서 전체 triangular mask를 쓰면 token이 뒤쪽 register를 전혀 읽지 못한다.
공통 NeoX layer와 별도로 token/register block mask를 구현해야 한다.

## 저장 state와 중간 표현

최종 write-back은 stack 완료 후 수행하며 S_next는 다음 segment에서 사용한다.
이 시점 자체는 정상적인 과거 정보 저장이다.
그러나 각 layer에서 register 위치를 갱신하면 H_R^(l)이 생긴다.
Registers->Tokens를 전체 허용하고 Tokens->Registers도 전체 허용하면
미래token -> H_R^(1) -> 과거token^(2) 경로가 가능하다.
read off와 최종 write-back 지연만으로 이 층간 경로가 제거되지는 않는다.
따라서 이 문서는 'token-token causal 버전'으로 정의하며 완전한 causal 동작은 아직 보장하지 않는다.
사용자는 인코더와의 일관성을 위해 intermediate를 기본으로 확정했다.
fixed_state를 비교 옵션으로 둔다. read=false와 함께 사용하면 위 register 중간표현 경로를 막는 계약이다.
fixed_state에서도 token causal mask와 generation/loss 경계 검증은 필요하다.

read=true 옵션은 보존한다. fullsegment R을 현재 token 계산에 적용하는 상태는
causal LM 실험으로 해석하지 않는다. 활성화할 때 gate의 정보 범위를 별도로 정의해야 한다.

## LM loss와 generation

모든 유효 token 위치의 next-token CE/NLL을 기록할 수 있다.
다만 intermediate 기본값의 teacher-forced NLL/PPL은 완전한 autoregressive decoder baseline의
PPL과 동등하게 해석하지 않는다. fixed_state 옵션은 causal 조건을 검증한 뒤 AR PPL로 평가한다.
raw labels를 내부에서 한번 shift하며 boundary next-token target을 누락/중복하지 않는다.
평가는 NLL sum/target count 후 exp로PPL을 계산한다.
Tokenization이 다르면 원문 word PPL과 직접 비교하지 않는다.

Teacher forcing은 segment당 한번의 stack 계산으로 처리하고 prefix 재계산은 하지 않는다.
실제 생성은 미래token이 없는 입력으로 실행해야 한다.
KV cache/segment 누적/generation에서 write-back을 수행할 boundary는 후속 API 명세에서 결정한다.
불완전한 segment를 token마다 영구 memory에 write하지 않는다.

## 검증과 미확정 조건

최소2층에서 미래token 변경 검사를 한다. intermediate에서는 register 우회에 따른 과거logits
변화를 허용/기록하고, fixed_state + read=false에서는 과거logits 불변을 요구한다.
fixed_state의 fullsegment teacher-forced 예측과 정답 미제공 생성의 일치는 후속generation 경로와 비교한다.
intermediate에는 동일성을 보장하지 않는다.
Router read off 영향, segment당 write-back1회, state gradient/BPTT,
position IDs, boundary target, token-weighted PPL을 검증한다.
현재 NeoX의 내부 causal mask와2D padding mask API만으로 임의block mask를 구현했다고 가정하지 않는다.
실험 기록에 register_read_source와 conditional_read를 함께 남기며 intermediate 결과를
엄격한 causal baseline과 동일 조건의 AR 성능으로 표시하지 않는다.
