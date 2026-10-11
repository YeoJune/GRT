# GRT 인코더 버전 명세

작성일: 2026-10-11. 상태: 설계 문서, 미구현.
사용자 합의: 전체 Transformer 처리 후 write-back, 위치별 prefix 재계산 제거.
기존 Remember 운영 코드와 결과는 변경하지 않는다.

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

## 인코더 attention

padding을 제외하면 [tokens; registers]의 모든 위치가 모든 위치를 읽는다.

| Query → Key | Tokens | Registers |
|---|---|---|
| Tokens | 양방향 전체 | 전체 |
| Registers | 전체 | 전체 |

기본 intermediate 정책에서는 각 층에서 token/register 중간 표현을 함께 변환한다.
fixed_state 옵션에서는 token의 register 읽기 source만 S_read로 고정한다.
전체 입력을 이용한 router read gate는 인코더의 양방향 정의와 충돌하지 않는다.
최종 register 후보는 세그먼트 전체를 압축하고 write gate로 다음 state에 반영한다.
인코더는 현재 segment 전체가 관측된 과제의 contextual representation을 만드는 버전이다.

## 출력과 loss 해석

Y_t는 양방향 contextual representation이다. tied head의 logits를 낼 수 있지만
동일 입력의 shifted next-token CE를 그대로 쓰면 정답 token이 attention 입력에 있어
표준 autoregressive LM/PPL 비교가 되지 않는다.
적합한 task/loss는 별도 선택한다: 예를 들어 입력에서 target을 숨기는 masked modeling,
분류, 관측 context 인코딩 후 별도 예측 경로 등이 가능하지만 어느 것도 채택하지 않았다.
MLM을 선택하면 masking/corruption protocol과 head를 명시해야 한다.
현재 문서에는 자동 생성이나 RMT autoregressive PPL과의 직접 비교를 확정하지 않는다.

## 구현상 변경과 검증

NeoX 기본 causal mask를 해제할 수 있는 attention 경로가 필요하다.
전체segment 입력에서 router/stack/write-back 각1회인지 확인한다.
S_next의 갱신식, read off시 gate 값 변경에 대한 출력 불변성,
state 전달/gradient/reset, token/register shape와 tied save/load를 검증한다.
미래token 변경에 과거 representation이 달라지는 것은 인코더에서 허용된 동작이다.
