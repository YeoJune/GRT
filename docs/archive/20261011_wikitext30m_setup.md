# WikiText-103 약 30M 공통 backbone 실험: 검토용 명세

후속 사용자 지시로 30M 예산은 폐기했다. 현재 기준은 [공통 NeoX backbone와 남은 문제](20261011_wikitext_backbone_remaining.md)다. 아래는 이전 검토 기록이다.

작성일: 2026-10-11. 상태: 사용자 선택 대기. 실행 가능한 확정 config가 아니다.
사용자 요구: 약 30M small 모델, RMT/GRT 동일 크기 backbone, 세부 조건을 임의 확정하지 않음.
기존 Remember 모델·config·실험 결과는 변경하지 않았다.

## 1. 확정 요구와 미정 선택

확정: WikiText-103, RMT 대 GRT, small급 약 30M, backbone 크기 통제.
미정: 30M에 embedding/head/router를 포함할지, 허용 오차, 공통 backbone 종류,
tokenizer, causal GRT 처리, 메모리 크기, segment 길이, BPTT, GPU/시간, 학습 예산.
동일 backbone은 같은 모듈 명세와 크기를 뜻하며, 학습 중 두 모델이 하나의 weight tensor를 공유한다는 뜻은 아니다.
공통 초기 가중치를 복제할지 또한 선택 항목이다.

## 2. 현재 코드에서 확인한 차이

| 항목 | 현재 RMT | 현재 GRT |
|---|---|---|
| 본체 | vendored GPT-NeoX | torch TransformerEncoderLayer ALU |
| norm/residual | NeoX parallel residual | Pre-LN sequential residual |
| position | rotary_pct=.25 | learned absolute embedding |
| attention | causal | 양방향, 예측마다 prefix 재계산 |
| input/output weight | untied | tied |
| memory | read+write 양쪽 배치, 마지막 층 write 출력 | register+router+gated write-back |
| 초기 memory | 학습 가능한 random memory | 학습 가능한 zero S0 |
| read | attention | conditional_read=false 기본 |

따라서 층수/D/FF/heads만 맞추면 크기 통제만 가능하고 동일 backbone은 아니다.
공통화하려면 기존 Remember와 분리된 LM 모델 경계에서 새 구현을 준비해야 한다.
backbone 변경과 GRT causal 설계는 사용자 결정 후 구현한다.

## 3. 30M 규모 후보 계산

아래는 후보 예산표이며 채택한 설정이 아니다.
가정: standard sequential Transformer, FF=4D, bias 포함, LayerNorm 2개/층,
input/output tied, output bias 없음. position/final norm/memory/router는 제외.
층당 식: 4D² + 2D·FF + 9D + FF = 12D² + 13D.
공통 subtotal: V·D + N·(12D²+13D).

| 후보 | V | 층 | D | FF | heads 후보 | 블록 | embedding | subtotal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 32000 | 4 | 512 | 2048 | 8 | 12,609,536 | 16,384,000 | 28,993,536 |
| A2 | 32768 | 4 | 512 | 2048 | 8 | 12,609,536 | 16,777,216 | 29,386,752 |
| B | 32000 | 10 | 384 | 1536 | 6 | 17,744,640 | 12,288,000 | 30,032,640 |
| B2 | 32768 | 10 | 384 | 1536 | 6 | 17,744,640 | 12,582,912 | 30,327,552 |
| C | 16000 | 6 | 512 | 2048 | 8 | 18,914,304 | 8,192,000 | 27,106,304 |

GRT router가 현재 구현이면 추가 파라미터는 다음 식이다.
2개의 MHA: 8D²+8D; pooling query: D;
read/write MLP 합: 4DH+2H+2HM+2M.
초기 register MD, position PD, final norm 등은 별도 합산한다.
예: D512/H512/M32 router=3,184,192, register=16,384.
따라서 A의 GRT total은 position/final norm 전에도 약32.19M이다.
D384/H512/M32 router=2,003,392이며 B는 약32.05M+이다.
같은 backbone이면 router 때문에 total은 다르다. 전체도 엄격히30M이면 backbone 여유를 두거나
router 크기를 조절해야 하고, GRT만 backbone을 줄이면 동일 backbone 조건을 깨뜨린다.

untied이면 V·D가 추가된다. A는45,377,536으로 증가한다.
원문 V267735는 D128 tied embedding만34,270,080이어서 standard D384/512 전체30M과 양립하지 않는다.
원문 word tokenizer를 유지하려면 작은 embedding 차원+projection/adaptive input 등을 별도 설계한다.
최종 선택 후 실제 생성 모델의 numel/고유 weight 기준 count를 확인한다. 이 표는 측정치가 아니다.

## 4. tokenizer와 데이터의 선택

선택1: 두 모델 공통 subword tokenizer. V16K/32K/32768 및 BPE/Unigram/byte-level을 결정한다.
train split만 tokenizer 학습에 사용하고 validation/test는 vocabulary 학습에 사용하지 않는다.
선택2: 원문 word tokenizer V267735 유지. embedding projection/adaptive output 설계를 결정한다.
subword PPL과 논문의 word PPL은 직접 비교할 수 없다. token NLL/PPL에 tokenizer 이름·hash를 기록한다.
모델 간 tokenizer, normalization, special IDs, BOS/EOS, newline/document 처리는 정확히 공유한다.
raw와 pretokenized WikiText 버전은 혼용하지 않으며 dataset revision/file hash를 고정한다.

연속 stream 또는 독립 document packing 중 선택한다. 원문은 ordered stream이다.
stream ID, 순서, segment boundary, document boundary reset, split reset, epoch reset을 명시한다.
DDP는 stream을 rank에 고정하고 accumulation 동안 slot/state 대응을 유지한다.
단순 무작위 DataLoader와 배치 간 memory carry를 결합하면 다른 문서의 memory가 섞인다.

## 5. causal GRT: 가장 중요한 미정 설계

현재 양방향 ALU와 global router에 full segment를 넣어 그 segment의 next token을 예측하면 미래 누출이다.
후보1: 기존 prefix 재계산을 유지한다. 각 위치 t는 prefix<=t와 이전 segment memory만 사용한다.
완료 segment의 memory write는 한 번만 수행한다. 의미 보존은 쉽지만 L번 ALU 호출이 필요해 비용이 크다.
후보2: 토큰은 causal, memory 쓰기는 segment 완료 후에만 수행하는 공통 backbone 경로를 설계한다.
이때 prefix마다 달라지는 pooling/read gate를 어떻게 정의할지 선택해야 한다.
단순 triangular mask만 추가하면 register와 full-segment router를 경유한 누출을 막지 못할 수 있다.
segment-level router를 다음 segment에만 적용할지, causal prefix pooling을 만들지 등은 사용자 선택이다.
이는 GRT 의미 변화이므로 임의로 정하지 않는다.
두 후보 모두 미래 token 변경시 과거 logits 불변, boundary logits 정합, write 횟수,
query/예측에서 memory 갱신 여부와 gradient 범위를 검증한다.

## 6. 공통 backbone에서 맞출 항목

층수/D/FF/heads/head dim, activation, Pre/Post-LN, eps, final norm,
sequential/parallel residual, attention/FF/projection bias, positional encoding와 position IDs,
embedding scale, tied weight, output head, dropout 위치/확률, 초기화/std,
attention backend 및 precision을 동일 명세로 고정한다.
메모리 관련 mask/layout, router/gate/write-back은 실험 대상 차이로 명시한다.
RMT input은 L+2M, GRT 현행 ALU는 L+M이므로 동일 backbone/slot 수여도 FLOPs는 동일하지 않다.
원문 RMT의 memory block 내부 양방향 mask와 현재 NeoX causal mask 차이도 기록해야 한다.

## 7. 학습 조건 후보와 비교 예산

원 논문 공개 실행 기록의 기준점: Adam lr2.5e-4, beta(.9,.999), eps1e-8,
weight decay0, cosine to0, warmup0, global norm clip.25, dropout.1, attention dropout0,
FP32, 200000 updates, L150/B60 또는 L50/B128, validation 매4000 updates.
이는 원문의151M 모델 조건이며 small 모델에 최적이라고 확인된 값이 아니다.
현재 Remember는 AdamW, value clipping, linear warmup/schedule이므로 그대로 재사용할 수 없다.

원문 조건 유지 또는 small용 공통 tuning 중 선택한다.
공통 tuning이면 동일 search 후보/횟수/validation 기준을 주고 test로 설정을 선택하지 않는다.
실험 축 후보: L50/150 또는 subword L128/256, M10/25/32,
BPTT 이전 segment K0/1/3. 조합 전부 실행할지는 예산 확인 후 정한다.
K는 이전 segment 수이며 현재 포함 K+1; state carry 길이와 gradient 연결 길이를 구분한다.
loss를 현재 segment에만 걸지 unroll 전체에 걸지, detach 시점과 optimizer update 주기를 고정한다.

학습 예산 기준 선택: 동일 target token 수/updates를 주 비교로 할지, 동일 wall-clock를 할지.
시간 제한에서 각 모델을 따로 끝까지 돌리면 학습 token 수가 달라질 수 있다.
공통 global batch와 tokens/update를 고정하고 microbatch/accumulation만 GPU에 맞춰 조절할 수 있다.
각 update의 유효 target 수로 loss를 가중한다. padding 또는 tail 길이 차이를 sample 평균으로 처리하지 않는다.
seed 개수/값 미정. 공통 모듈 초기 tensor 복제와 독립 모델 RNG/데이터 RNG를 선택한다.
단순 동일 seed만으로 서로 다른 모델 초기화 후 동일 배치/dropout을 보장할 수 없다.

## 8. loss와 경계 인터페이스

현재 계약은 raw labels+predictor mask, 내부에서 한 번 shift다.
각 segment 마지막 토큰이 다음 segment 첫 token을 예측하는 target을 빠뜨리지 않는다.
중복 target, boundary 손실, 마지막 split에 존재하지 않는 target을 명시적으로 제외한다.
raw-label 계약을 유지하면서 segment 간 target을 포함하는 방식은 구현 명세에서 확정한다.
state API에는 state 입력/출력, stream/reset mask, detach policy가 필요하다.
현재 RMT wrapper와 GRT forward는 호출 시작에서 state를 초기화하므로 배치 간 carry API가 없다.
학습에 memory position loss/보조 loss를 추가할지 미정; 추가하면 두 모델 조건과 별도 보고한다.

## 9. 평가/기록

공통 목표: total_nll / valid_target_count, PPL=exp(mean_nll).
DDP에서는 NLL 합과 token 수 합을 reduce한 뒤 계산한다. batch PPL의 평균을 쓰지 않는다.
validation NLL 최소 checkpoint 선택, 최종 test는 선택 완료 후 평가하는 방식은 검토 후보다.
평가 L과 memory reset/carry 조건을 학습과 함께 고정한다.
공통 tokenizer에서만 RMT/GRT PPL 직접 비교; 원문 word PPL은 참고값으로 분리한다.

기록: tokenizer/dataset/code hash, resolved config, 공통/추가/전체 parameter count,
unique weight tying, target tokens seen, updates, BPTT context, memory slots/scalars,
peak allocated/reserved VRAM, wall time, tokens/sec, seeds, reset 정책, best/last NLL,
실제 평가 token 수, precision/backend, 중단 이유.

## 10. 자원과 구현 준비 범위

30M FP32 parameters 약120MB, gradients 약120MB, Adam 두 moments 약240MB:
모델+gradient+optimizer 약480MB(십진), activation/logits/임시 buffer 등 제외.
full vocab logits는 B·L·V·bytes; B60/L150/V32000 FP32는1.152GB/segment이다.
BPTT와 GRT prefix 경로는 추가 activation/연산이 크므로 총 VRAM/시간은 실측해야 한다.
GPU/precision/시간을 선택한 뒤 두 모델의 forward/backward peak와 update 시간을 사전 측정한다.

현재 config는 vocab/tied/precision/과제 validation이 Remember에 고정돼 있고,
trainer/evaluator는 pair curriculum 및 generation EM에 묶여 있다.
LM에는 데이터 adapter뿐 아니라 공통 training/evaluation capability 확장이 필요하다.
모델명으로 task 분기를 넣지 않고, 별도 LM schema/stream 상태/metric 경계를 설계한다.
현재 train CLI에 넣어 실행할 확정 YAML은 만들지 않았다.

## 11. 출처와 검증 상태

논문: https://papers.neurips.cc/paper_files/paper/2022/file/47e288629a6996a17ce50b90a056a0e1-Paper-Conference.pdf
원문 실행 기록: https://github.com/booydar/LM-RMT/blob/master/experiment_results/results_wt103_enwik8_all_parameters.csv
원문 모델: https://github.com/booydar/LM-RMT/blob/master/pytorch/mem_transformer.py
현재 코드: src/grt/models/{rmt,grt,alu,router}.py, config.py, data.py, trainer.py, evaluator.py.
이번 준비는 코드 검토와 파라미터 수식 계산이다. 모델 생성/학습/VRAM/수렴 검증은 하지 않았다.
