# GRT 하이퍼파라미터 탐색 검토 보고서

작성일: 2026-10-12 · 코드 기준: `b617aa3` · 상태: **검토 제안, 실험 조건 미확정**

후속 사용자 지시를 반영한 최신 순서와 자원 계획은 [메모리 활용성 우선 실험 계획](20261012_experiment_plan.md)에 정리했다. 아래 지표·구현 검토는 참고하되, 초기 탐색 순서와 후보 범위는 후속 계획을 따른다.

이 문서는 사용자가 탐색 방향을 결정하기 위한 보고서다. 아래 후보값·예산·trial 수는 실행 승인을 받은 설정이 아니다. 학습 코드, config, 의존성은 변경하지 않았으며 새로운 실험도 실행하지 않았다. 현재 구현 계약은 [통합 명세](20261011_current_spec.md), 실행 방법은 [실험 workflow](20261011_experiment_workflow.md)를 따른다. 코드의 정식 명칭인 **RTLA**를 사용한다.

빠른 검토는1절 권장안과14절 결정표부터 읽으면 된다. 지표·데이터·baseline 판단은3~6절, 실제 탐색 계획은7~12절, 실행 전 필요한 구현은13절에 정리했다.

## 1. 우선 제안하는 방향

**small을 주 모델로 두고, 동일 backbone의 메모리 없는 decoder·RMT·GRT를 비교한다. PPL을 품질 기준으로 삼고, RTLA의 문맥 교환 delta NLL을 메모리 사용 지표로 함께 평가한다.** 전체 크기·데이터셋·학습률·segment·register·BPTT를 한꺼번에 섞은 sweep은 피하고, 먼저 baseline과 측정 조건을 고정한 뒤 메모리 구조를 탐색하는 편이 해석하기 좋다.

| 판단 항목 | 우선 권장안 | 대안과 선택 이유 |
|---|---|---|
| 주 모델 | small: D256, 6 layers, 8 heads, FF1024 | tiny는 기능/실행량 점검, base는 최종 확장 확인 |
| 주 데이터 | WikiText-103-v1 + 기존 GPT-2 BPE | 저비용 예비 탐색은 WikiText-103의 고정된 문서 subset; WikiText-2는 별도 소규모 데이터 실험 |
| baseline | 동일 NeoX의 메모리 없는 decoder + 현재 RMT | 필요시 학습부터 과거 state를 사용하지 않는 static-register control 추가 |
| 주요 품질 지표 | 전체 validation의 token-weighted NLL, 표시값은 PPL | test는 최종 선택이 끝난 뒤 사용 |
| 주요 메모리 지표 | 정상 대비 문맥 교환 delta NLL | 초기화 delta, slot 기여, 지속 범위는 진단·검증용 |
| 구조 후보 | BPE segment 150/75/50, register 5/10/20 | 더 강한 병목이 필요하면 segment25, 용량 포화 확인에는 register40 |
| 자동화 | 소규모 구조 grid로 시작; 자동 제안은 Optuna TPE + 기존 W&B | 단일 품질 목표와 관리 편의가 우선이면 W&B Sweeps |
| 최종 선택 | 품질이 충분한 후보의 Pareto 비교 + 여러 seed 확인 | PPL과 RTLA를 임의 가중합 하나로 합치지 않음 |

이 경로의 의미는 **PPL 개선과 메모리 사용을 함께 확보할 수 있는가**를 묻는 것이다. 메모리에 크게 의존하게 만드는 것 자체는 좋은 언어 모델을 보장하지 않는다.

## 2. 현재 출발점과 확인된 사실

### 2.1 현재 코드와 사용자 관측

- 공통 backbone은 GPT-NeoX, parallel residual, RoPE, tied embeddings다. GPT-2 tokenizer만 사용하며 모델 weights는 처음부터 학습한다.
- GRT는 모든 layer에서 segment 시작 state를 읽으며 write-back은 stack 종료 후 한 번이다. read gate는 시작 state만으로 계산하고 기본 off다. 이 구조·AR 계약은 탐색 중 고정한다.
- WikiText의 기본값은 segment150, register10, 이전 BPTT3, global batch256, Adam LR5e-4, cosine, warmup0, clip0.25, residual dropout0.1, attention dropout0, FP32다.
- 사용자는 small·segment150에서 RMT/GRT의 PPL이 모두 약34까지 내려갔다고 보고했다. 이는 출발점으로 유용하나 seed·학습 token 수·checkpoint를 함께 기록해야 정량 baseline으로 사용할 수 있다.
- 사용자가 train 전체에서 측정한 word→BPE 비율은1.14734다. word150 구간의 BPE 길이는 평균172.10, 중앙값170, 10~90% 분위수160~186이다. 현재 BPE150은 평균 word130.74 분량이다.

**RMT와 GRT의 PPL이 비슷하다는 사실만으로 메모리가 사용되지 않는다고 판단할 수는 없다.** 두 모델이 모두 메모리 도움을 받고 있을 수도 있다. 메모리 없는 동일 backbone과, 정상 state를 교란한 결과가 필요하다.

### 2.2 원 RMT 실험에서 참고할 부분

RMT 논문은 WikiText-103에서 word segment150과50을 비교했다. 짧은 segment50을 사용한 목적은 recurrent dependency를 더 처리하도록 만드는 것이라고 설명한다. 따라서 짧은 segment에서 register 기여를 조사하는 방향은 논문의 문제의식과 맞는다. [RMT 논문, §4·Table2](https://papers.neurips.cc/paper_files/paper/2022/file/47e288629a6996a17ce50b90a056a0e1-Paper-Conference.pdf)

| 원 논문 조건 | memory | word segment | 보고된 PPL |
|---|---:|---:|---:|
| 메모리 없는 baseline | 0 | 150 | 29.95 |
| RMT BPTT-3 | 10 | 150 | 25.04 |
| 메모리 없는 baseline | 0 | 50 | 39.05 |
| RMT BPTT-3 | 10 | 50 | 26.37 |

이 수치는 문맥 제한을 강화했을 때 메모리 비교가 더 분명해질 수 있음을 보여주는 참고다. **우리 BPE PPL34와 직접 비교하는 목표값은 아니다.** 논문은 Transformer-XL 계열·word-level·다른 출력층/학습 조건을 사용하며, 현재 구현은 NeoX·GPT-2 BPE·full softmax다. 위 수치는 논문의 표를 참조한 값이다.

RMT의 word150에 문맥량을 맞추는 BPE 길이는 평균적으로170~172다. 메모리 병목을 강화하려고75/50으로 줄이는 실험과, 원 논문의 텍스트 양을 맞추는 실험은 구분해서 기록한다.

## 3. 두 평가 변수를 어떻게 정의할 것인가

### 3.1 첫 번째 변수: 언어 모델 품질

동일한 evaluation protocol에서 다음 값을 사용한다.

`L_val = validation의 NLL 합 / 유효 target token 수`

`PPL_val = exp(L_val)`

자동 탐색의 수치 목표는 NLL이 편하다. PPL과 순위는 같고, 손실 차이와 제약조건을 표현하기 쉽다. 최종 보고에는 독자가 읽기 쉬운 PPL도 함께 제시한다. BPE-token PPL과 word-token PPL은 서로 다른 척도다.

### 3.2 두 번째 변수: 메모리에 담긴 과거 문맥의 기여

현재 RTLA에서 직접 사용할 수 있는 지표는 다음과 같다.

`Δ_reset = NLL(S0로 초기화) − NLL(정상 state)`

`Δ_swap = NLL(다른 스트림의 같은 시점 state) − NLL(정상 state)`

우선 **Δ_swap을 문맥 특이적인 state 사용의 지표**로 삼고 Δ_reset을 함께 확인하는 안을 권한다. 초기화는 state 분포를 크게 바꿀 수 있다. 교환은 실제로 만들어진 state를 주입하므로, 정상 문맥과의 대응이 필요한지를 더 직접적으로 묻는다. 교환도 완전한 in-distribution 개입은 아니므로 단독 증거로 결론을 확정하지 않는다.

| 관측 | 우선 해석 | 추가 확인 |
|---|---|---|
| 낮은 PPL, reset/swap 모두 예측 악화 | 유용한 메모리 사용 후보 | 여러 문서·seed, static/GPT baseline에서 확인 |
| 낮은 PPL, 두 delta가 거의0 | 해당 probe에서 과거 state 의존이 작음 | 다른 문맥 길이·문서·충분한 warmup에서 재확인 |
| reset만 큼, swap은 작음 | 동적 state의 일반적 분포/크기가 도움이 될 가능성 | 문맥 내용과 정적인 보조 표현을 구분 |
| 높은 PPL, 큰 delta | 메모리에 취약하게 의존할 수 있음 | 품질 기준을 통과하지 못하면 선택하지 않음 |
| 큰 gate/norm 변화, delta는 작음 | state는 변하지만 예측 기여는 작을 수 있음 | 변화량을 목적함수로 최대화하지 않음 |

예를 들어 delta0.01 nats/token은 개입 PPL이 정상 PPL보다 약1% 높은 수준에 대응한다. 이를 실용적 효과 크기의 초기 예시로 사용할 수는 있지만, **0.01을 보편적인 성공 기준으로 고정하지 않는다.** pilot에서 문서·seed별 변동을 측정하고 최소 효과 크기를 사용자가 정하는 편이 좋다.

### 3.3 큰 delta만 최적화하면 생기는 문제

교란된 state에서만 극단적으로 나쁜 예측을 하도록 학습된 모델은 delta가 커질 수 있다. 그렇다고 정상 PPL이 개선되거나 메모리 없는 모델보다 좋은 것은 아니다. 작은 segment를 선택하기만 해도 메모리 의존도가 높아질 수 있다. 따라서 raw delta를 무제한 최대화하는 sweep은 권하지 않는다.

우선 적용 가능한 선택 규칙은 세 가지다.

| 방법 | 선택 규칙 | 적합한 단계 |
|---|---|---|
| 품질 중심 + 메모리 확인 | PPL을 최적화하고 메모리 효과가 재현되는 후보를 선별 | 첫 탐색, 적은 trial |
| 품질 제약 아래 메모리 최적화 | `L_val ≤ L_ref + log(1+ε)`를 만족하는 후보에서 메모리 효과 비교 | 허용 PPL 손실을 정한 뒤 |
| 두 목적 Pareto 탐색 | NLL 최소·Δ_swap 최대의 비지배 후보를 보존하되 품질 기준 적용 | 충분한 trial과 안정된 probe 확보 후 |

ε는 허용 상대 PPL 악화율이다. 예를 들어1~3%라는 범위는 검토 후보이며 확정값이 아니다. baseline 자체의 seed 변동보다 작은 차이를 절대적인 우열로 판정하지 않는다. 메모리 사용의 신뢰성을 중시한다면 반복 probe의 평균뿐 아니라 불확실성도 함께 비교한다.

권장 순서는 **품질 중심 탐색 → 메모리 효과 검증 → 살아남은 후보의 Pareto 비교**다. 처음부터 noisy RTLA와 PPL을 같은 비중으로 최적화하면 학습이 늦게 진행되는 모델을 조기에 놓칠 수 있다.

## 4. 데이터셋 선택

### 4.1 WikiText-2와 WikiText-103의 역할

원 데이터의 word-level train token 수는 WikiText-2 약2.09M, WikiText-103 약103.23M이다. 두 데이터는 **validation/test에 같은 원본 기사들을 사용하며 vocabulary가 다르다.** 이 때문에 WT2에서 선택한 후보를 WT103의 표준 validation으로 다시 평가해도 완전히 독립적인 검증 데이터로 볼 수 없다. [Pointer Sentinel Mixture Models, §4·Table1](https://arxiv.org/pdf/1609.07843)

| 데이터 | 장점 | 제한 | 제안 역할 |
|---|---|---|---|
| WikiText-2 | 준비가 작고 한 epoch가 짧음; pipeline·과적합 경향 확인 | 많은 token을 학습하면 같은 작은 corpus를 반복; WT103로의 순위 전이 불확실 | 보조 실험, 저예산 pilot |
| WikiText-103 전체 | 현재 결과와 직접 연결; 다양한 문서·희귀 항목·긴 문맥 | trial당 충분한 학습은 비용이 큼 | 주 결과와 최종 후보 확인 |
| WT103의 고정 문서 subset | tokenizer·전처리·도메인을 그대로 공유; 예비 실험을 관리하기 쉬움 | subset 편향과 과적합; 전체로 순위가 바뀔 수 있음 | 권장 예비 탐색 |
| Remember/간단한 기억 과제 | register의 실제 정보 전달을 명확히 검사 | 자연어 PPL을 예측하는 지표가 아님 | 기능 진단, 필요시 추가 |

**WT2를 사용해도 같은 모델·같은 학습 token 수로 학습하면 GPU 계산량이 자동으로 줄지는 않는다.** 빨라지는 것은 corpus를 한 바퀴 처리하는 데 필요한 양이다. 작은 corpus에서 더 짧은 학습으로 후보를 고를 수 있을 때 시간 절약이 생긴다. 예비 탐색의 예산을 별도로 낮추어야 한다.

### 4.2 tokenizer와 raw/non-raw 문제

현재 실험은 `Salesforce/wikitext`의 **wikitext-103-v1**을 GPT-2 BPE로 다시 토큰화한다. 이 원문에는 vocabulary 밖 항목이 이미 `<unk>`로 치환된 부분이 있다. BPE를 적용한다고 소실된 원문이 복구되지는 않는다. raw/non-raw의 차이는 공식 dataset card에 명시돼 있다. [WikiText dataset card](https://huggingface.co/datasets/Salesforce/wikitext/blob/main/README.md)

선택지는 다음과 같다.

- 현재 결과의 연속성이 우선이면 **103-v1 + 기존 GPT-2 BPE**를 유지한다.
- BPE 자체의 자연스러운 처리와 희귀 이름의 기억을 연구하려면 **raw-v1 + GPT-2 BPE**를 별도 실험 계열로 고려할 수 있다. 이 경우 baseline도 다시 학습하고150↔172 환산도 다시 측정해야 한다.
- WT2-v1과 WT103-v1은 원본 평가 기사가 겹쳐도 `<unk>` 치환으로 평가 token이 달라질 수 있다. split 이름·행 수가 같다고 동일 evaluation이라고 가정하지 않는다.

본 보고서의 우선 제안은 현재 전처리를 유지하고 WT103 subset을 활용하는 것이다. raw 전환은 데이터셋 연구 질문을 함께 바꾸는 선택이므로 사용자가 결정한 뒤 별도 계열로 진행한다.

### 4.3 subset과 평가 분할

예비 subset은 **10~20M BPE tokens 정도의 전체 문서 묶음**을 검토할 수 있다. 크기는 제안값이다. 앞쪽 token을 단순히 잘라 쓰기보다 문서 단위로 고정 seed를 사용해 선택하고 원래 문서 내부 순서를 유지한다. 문서 사이 연결 방식은 모든 모델에 동일하게 적용한다.

표준 validation을 HPO에 사용하는 것은 일반적인 프로토콜이지만, 많은 탐색이 누적되면 그 데이터에 선택 편향이 생긴다. 더 엄격한 확인이 필요하면 train에서 문서 단위로 별도 개발 holdout을 미리 만들고, 표준 validation은 확인용으로 남기는 안이 있다. 이때 checkpoint 선택도 개발 holdout에서 해야 표준 validation이 독립적으로 남는다.

test는 최종 후보와 hyperparameter가 확정된 뒤만 평가한다. subset hash, 문서 목록, 데이터 revision, tokenizer artifact hash를 결과에 남긴다. WT2의 BPE token 수는 별도 측정해야 하며 WT103에서 구한1.147 비율을 그대로 대입하지 않는다.

## 5. 모델 크기 선택

현재 GPT-2 BPE50257 기준 parameter 수다. GRT/RMT 값은 register10의 현재 구성 기준이다.

| 크기 | D / layers / heads / FF | backbone | RMT 전체 | GRT 전체 | 권장 용도 |
|---|---|---:|---:|---:|---|
| tiny | 128 / 3 / 4 / 512 | 7.03M | 7.03M | 7.43M | 도구·진단·극단적 병목 확인 |
| small | 256 / 6 / 8 / 1024 | 17.60M | 17.61M | 18.67M | 주 탐색과 여러 seed 확인 |
| base | 410 / 16 / 10 / 2100 | 59.01M | 59.01M | 61.21M | 선정된 구조의 크기 확장 확인 |

수치 출처: [현재 parameter 명세](20261011_current_spec.md). backbone 비교는 동일하게 유지하고 router parameter는 별도로 기록한다. 이전에 합의한 대로 GRT의 총 parameter를 맞추기 위해 backbone을 임의로 줄이지 않는다.

tiny는 vocabulary embedding이 전체 parameter의 대부분을 차지하고, small보다 transformer depth와 계산 용량이 작다. 총 parameter만 보고 tiny에서 얻은 memory 설정의 순위가 small에서도 유지된다고 기대하기 어렵다. 이미 small의 학습이 정상적으로 진행된 관측이 있으므로 **tiny에서 전부 탐색한 뒤 small로 옮기는 경로보다 small을 직접 주 모델로 쓰는 안**이 낫다.

모델 크기를 첫 sweep의 자유 변수로 넣으면 큰 모델을 고르는 효과와 memory 설계 효과가 섞인다. 먼저 size별로 분리한 study를 운영하고, 최종 후보를 base에 옮겨 PPL·memory 효과·비용 변화가 함께 유지되는지 확인한다. base의16층 증가와 D 증가가 동시에 일어나므로 순수한 width 효과라고 해석하지 않는다.

## 6. baseline 설계와 사전 탐색

### 6.1 먼저 갖추어야 할 비교군

| 비교군 | 답하는 질문 | 현재 지원 상태 |
|---|---|---|
| NeoX causal decoder, recurrent memory 없음 | 현재 segment만으로 어느 정도 예측하는가 | **추가 구현 필요** |
| 현재 RMT | 같은 backbone의 recurrent memory와 비교하면 어떤가 | 지원 |
| 현재 GRT | router/write-back이 RMT와 어떻게 다른가 | 지원 |
| GRT static-register, 학습부터 state를 과거로 갱신하지 않음 | learned static registers와 과거 기억을 구분할 수 있는가 | 추가 control 구현 필요 |
| 같은 NeoX의 더 긴 token context | 압축 memory 대신 과거 원문에 직접 접근할 때의 기준은 어떤가 | baseline·입력 window 구현 필요; 비용도 별도 비교 |

여기서 GPT baseline은 **동일 NeoX backbone의 메모리 없는 decoder를 처음부터 학습하는 것**을 뜻한다. pretrained GPT-2의 PPL을 가져오면 weights의 사전학습량·구조·데이터가 다르므로 공정한 주요 baseline이 되지 않는다. pretrained 모델은 외부 참고로만 구분한다.

현재 config에서는 RMT/GRT만 허용하고 register 수는 양수여야 한다. `num_registers=0`을 지정하거나 기존 RMT의 state를0으로 만드는 것으로 메모리 없는 GPT가 구현된다고 보면 안 된다. inference 때의 state 초기화는 학습된 GRT에 대한 개입이며, 메모리 없이 학습한 모델의 성능을 대체하지 않는다.

### 6.2 baseline hyperparameter를 먼저 찾는 방법

권장 순서는 메모리 없는 NeoX와 RMT에 **같은 데이터·size·token 예산·LR 범위**를 주고, LR 중심으로 먼저 안정적인 품질 기준을 만드는 것이다. 그 결과를 GRT의 초기 LR 범위에 활용한다. GRT에는 적은 범위의 LR 재탐색도 허용해 baseline의 최적 LR을 강제로 물려주는 불이익을 피한다.

두 종류의 비교를 함께 보고한다.

1. **공통 학습 조건 비교:** 같은 LR·dropout 등으로 architecture 효과를 통제한다.
2. **모델별 튜닝 후 비교:** 각 모델에 합의한 탐색 예산을 주고 가장 좋은 조건을 비교한다.

GRT만 여러 조건을 탐색한 뒤 RMT 기본값 한 번과 비교하면 튜닝 양의 차이가 결과에 섞인다. RMT의 memory size도 필요시 같은5/10/20 범위에서 탐색한다. trial 수와 GPU-hours를 둘 다 기록한다. 동일 trial 수와 동일 GPU-hours는 모델당 비용이 다르면 동시에 만족하지 않을 수 있다.

현재 약34라는 값은 초기 참고다. baseline을 다시 측정하면 그 validation NLL을 `L_ref`로 사용한다. 기존 로그와 tokenizer·budget·checkpoint가 같은지 확인하기 전에는 절대적인 성공선으로 사용하지 않는다.

### 6.3 긴 context baseline을 넣는 경우

예를 들어 현재 segment L과 과거 gradient 범위 R을 합친 `C=L+R`만큼 원문을 읽는 NeoX를 추가할 수 있다. 이 경우 **현재 L개의 target만 loss로 사용**해야 같은 optimizer update에서 supervision 양을 비교하기 쉽다. 모든 context token을 target으로 추가하면 학습량도 바뀐다.

이 비교는 압축 memory가 원문 접근 대비 얼마나 정보를 유지하는지 묻는 참고다. attention 계산과 VRAM이 달라지므로 순수한 비용 동등 비교라고 표현하지 않는다. 첫 탐색에서는 메모리 없는 local NeoX·RMT·GRT를 우선하고, 긴 context 비교는 최종 후보에 제한하는 편이 효율적이다.

## 7. segment·batch·BPTT·학습 예산의 결합

기호를 다음처럼 정의한다.

- L: segment의 BPE token 수
- M: register 수
- K: 재계산하는 이전 segment 수, 현재 segment는 별도로1개
- B: global batch의 stream 수
- T: loss 대상 target token의 누적 학습량. corpus 반복을 포함하며 서로 다른 원문 token 수라는 뜻은 아니다.

상한 기준으로 다음 관계를 갖는다.

`target tokens/update = B × L`

`과거 BPTT 범위 = K × L`, `재계산 입력 범위 = (K+1) × L`

`updates ≈ T / (B × L)`

실제 짧은 tail과 stream/epoch 경계는 현재 `target_tokens_seen`으로 기록한다. replay token은 loss 대상 학습량과 별도로 계산 비용에 포함된다.

### 7.1 같은 update 수로 segment를 비교하지 않기

현재 B256에서 L150을75로 줄이고 updates를 그대로 두면 supervision token 수가 절반이다. 구조 효과와 학습량 감소가 섞인다. **주 비교에서는 B256을 고정하고 T를 맞추는 안**을 권한다.

다만 T를 맞추더라도 L이 작으면 updates는 많아지고 한 update의 target 수는 적어진다. optimizer noise와 update 횟수까지 완전히 같아지는 것은 아니다. LR schedule·evaluation 시점은 가능한 한 token 진행률로 맞추고, 최종 후보에서 LR 민감도를 재확인한다.

동일 target tokens/update와 batch의2의 거듭제곱을 동시에 원한다면 L128/64/32 같은 별도 계열에서 B256/512/1024를 사용할 수 있다. 하지만 stream 수를 바꾸면 데이터 partition과 state history도 바뀐다. 또한150/75/50에서 같은 token batch를 맞추려면 일부 B가2의 거듭제곱이 아니게 된다. 이 대안은 현재 B256 고정 비교와 구분해야 한다.

### 7.2 BPTT를 고정할 것인가, 과거 token 범위를 고정할 것인가

| L | 기존 K=3에서 과거 범위 | 과거450 tokens를 맞출 때 K |
|---:|---:|---:|
| 150 | 450 | 3 |
| 75 | 225 | 6 |
| 50 | 150 | 9 |
| 25 | 75 | 18 |

두 실험은 질문이 다르다.

- K3 고정: 같은 recurrence unroll 수에서 짧은 segment가 어떤 조건을 만드는가.
- 과거 token 범위 고정: gradient가 전달되는 문맥량을 맞추고 segmentation을 비교하면 어떤가.

빠른 스크리닝은 K3을 사용하되 결과를 **L과 gradient 범위가 함께 바뀐 설정**으로 명시할 수 있다. 최종 후보에는450-token 범위를 맞춘3/6/9 비교를 추가하는 안을 권한다. 처음부터450을 맞추는 안은 더 통제된 비교지만 짧은 segment의 replay 비용이 커진다. L25·K18은 초기 필수 조건으로 두지 않는다.

같은 T라도 replay token 양은 대략(K+1)T이므로, 짧은 segment에서 K를 크게 늘리면 계산이 증가할 수 있다. attention 길이가 줄었다는 이유만으로 전체 학습이 빨라진다고 가정하지 않는다.

### 7.3 현재 GRT에서 K=0의 특별한 문제

현재 WikiText 학습은 다음 segment로 넘어갈 때 state를 detach하고, 현재 segment의 token loss만 계산한다. K0이면 다음 segment의 loss가 이전 write-back까지 돌아가지 않는다. fixed-state 구조에서는 현재 token이 현재 write candidate를 읽지도 않는다.

따라서 **read gate off의 GRT에서 K0은 write router에 학습 gradient가 전달되지 않는 조건**이다. read gate on이면 공유 router 일부가 read 경로로 학습될 수 있어도 write head의 다음 segment credit 경로는 여전히 끊긴다. K0을 보통의 memory hyperparameter 후보로 넣어 성능이 낮다는 이유로 구조를 배제하면 해석이 잘못된다.

첫 GRT 탐색의 K는1 이상으로 두는 안을 권한다. K0은 credit assignment를 제거하는 별도 negative control로만 다룬다. 이 판단은 현재 [stream 학습](../src/grt/stream_training.py)과 [cell/state 경로](../src/grt/models/neox.py)를 확인한 논리적 분석이며 추가 실험 결과를 주장하는 것은 아니다.

## 8. 변경할 하이퍼파라미터와 제안 범위

아래 표는 **선택 후 사용할 후보 공간**이다. 모든 조합을 실행하자는 뜻은 아니다.

| 변수 | 1차 후보 | 탐색 목적·주의 | 현재 설정 가능 여부 |
|---|---|---|---|
| learning rate | 2.5e-4 / 5e-4 / 1e-3 | baseline 안정화; 실패 경계 확인 후 주변을 log-scale로 세밀 탐색 | 가능 |
| residual dropout | 0 / 0.1 / 0.2 | 데이터 양·과적합과 상호작용; LR 뒤에 제한적으로 | 가능 |
| segment L | 150 / 75 / 50 | 직접 token 문맥과 memory 의존 변화 | 가능 |
| register M | 5 / 10 / 20 | 기억 용량·압축·읽기/갱신 비용 | 가능 |
| write gate bias init | -4 / -2 / 0 | 초기 유지/갱신 성향과 학습 속도 | 가능 |
| 이전 BPTT K | 우선3; 통제 비교는3/6/9; 추가 K≥1 | temporal credit와 replay 비용 | 가능 |
| router MLP hidden | 128 / 256 / 512 | router 용량이 실제로 필요한지; backbone은 그대로 | 가능 |
| read gate | 기본off, 이후 on/off 비교 | state-only 계약 유지; LayerNorm 위치 영향 먼저 확인 | 가능 |
| clip norm | 기본0.25; 필요시0.5/1.0 | clipping이 빈번한 경우에 한해 | 가능; 현재 clip 빈도 metric은 없음 |
| write gate dropout | 기본0; 후속0.05/0.1 | write learning을 약화시킬 수 있어 늦게 탐색 | 가능 |
| Adam beta2 | 우선0.999 고정; 필요시0.98/0.999 비교 | LR·token batch·수렴 곡선과 결합하므로 후순위 | 가능 |
| warmup 비율 | 기본0; 필요시1~5% | 초기 LR 불안정이 확인될 때 | **현재 stream scheduler가0으로 고정, 구현 필요** |
| weight decay / AdamW | 첫 단계는현재 Adam·decay0 유지 | 현재 Adam의 decay는 AdamW와 다름; 비교 규칙 먼저 결정 | Adam decay는 가능, AdamW 선택은 구현 필요 |

고정할 항목은 backbone 구조·tokenizer·AR mask·fixed-state 읽기·write-back 타이밍·FP32·global batch·evaluation partition이다. heads/layers/D/FF는 size 선택에 포함하고 첫 HPO에서 독립적으로 섞지 않는다. attention dropout은 우선0으로 유지한다.

write bias의 입력 항을 무시하면 -4/-2/0은 sigmoid 기준 약2%/12%/50%의 초기 갱신 성향에 대응한다. 실제 gate는 입력과 학습된 가중치의 영향을 받는다. 작은 초기 gate가 메모리 효과를 늦게 만들 수 있으므로 매우 이른 RTLA 순위만으로 제거하지 않는다.

### 8.1 중요한 상호작용

L/M/K를 독립적인 숫자 목록으로만 탐색하면 놓치는 부분이 있다.

| 결합 | 왜 중요한가 | 확인 방법 |
|---|---|---|
| L × M | 짧은 segment에서 같은 M은 현재 token 대비 큰 memory 용량이며 계산 비중도 증가 | M 고정 비교와 M/L 비율 비교를 구분 |
| L × K | 같은 K라도 gradient가 전달되는 token 범위가 달라짐 | 제7절의 고정 K/고정 과거 범위 비교 |
| L × write gate | 같은 gate라도 token당 갱신 횟수가 증가 | 초기 bias와 실제 갱신량·persistence를 함께 관찰 |
| M × router capacity | 많은 slot을 두어도 독립적인 기여가 작거나 router가 구별하지 못할 수 있음 | slot별·묶음 개입, hidden 용량의 후속 비교 |
| L × LR/beta2 | target 수/update와 update 횟수가 달라져 optimizer의 token 기준 동작도 달라짐 | 최종 구조 주변에서 좁은 재탐색 |

GRT의 직접 residual 경로만 보면 일정한 write gate w에서 이전 state의 계수는 k번 갱신 뒤 `(1−w)^k`다. L을 줄이면 같은 원문 길이에 더 많은 갱신이 발생한다. 다만 candidate도 이전 state의 정보를 보존할 수 있으므로 이 계수가 전체 모델의 실제 기억 수명은 아니다. gate 평균만으로 forgetting을 판정하지 말고 RTLA의 한 번 개입 후 영향을 측정한다.

현재 구현의 입력 위치 수는 GRT에서 L+M, RMT에서 L+2M이다.

| L / M | GRT 위치 수 | RMT 위치 수 |
|---|---:|---:|
| 150 / 10 | 160 | 170 |
| 75 / 10 | 85 | 95 |
| 50 / 10 | 60 | 70 |
| 50 / 20 | 70 | 90 |

위치는 실제 FLOPs와 동일하지 않다. GRT의 fixed-state K/V projection과 candidate 계산, replay, vocabulary softmax도 비용에 포함된다. 큰 M은 parameter 증가가 작아도 attention·state 비용이 증가할 수 있다. 효과를 M으로 단순히 나눈 `delta/M`도 선형적인 효율 지표로 가정하지 않는다.

### 8.2 read gate 탐색 전에 확인할 구현상의 의미

현재 token이 읽는 state는 `source = read × S`이고, 각 layer에서 이 source를 LayerNorm한 뒤 K/V를 만든다. read는 slot마다 하나의 양의 scalar다.

`LN(aS) = γ(S−μ) / sqrt(σ² + ε/a²) + β`, 단 a>0

분산이 epsilon보다 충분히 크면 scalar 크기 변화가 LayerNorm에서 대부분 상쇄된다. 그래서 read gate 값이0.5라는 사실이 token의 memory contribution을 절반으로 만든다는 뜻은 아니다. gate가 매우 작거나 state 분산이 작은 경우는 달라질 수 있다. 또한 candidate의 초기 residual에는 source가 들어가므로 이후 write-back에 대한 영향도 남는다.

이는 곧바로 read gate가 무효라는 결론이 아니라 **현재 on/off가 어떤 경로를 바꾸는지 먼저 확인할 필요**가 있다는 뜻이다. 동일 incoming state에서 현재 logits와 다음 state를 각각 비교하는 진단 후 탐색 우선순위를 정한다. 현재 설계의 off 기본값은 유지하고, 이 보고서에서 gate 위치를 변경하지 않는다.

## 9. 탐색 순서와 규모의 예시

### 9.1 권장 단계

| 단계 | 수행할 일 | 통제 조건 | 결과를 보고 내릴 결정 |
|---|---|---|---|
| 0. 측정 준비 | 메모리 없는 baseline·test 격리·probe 정렬·trial 실행 계약 준비 | 현재 AR 구조 고정 | sweep을 시작할 수 있는가 |
| 1. baseline calibration | small의 local NeoX·RMT를 LR 중심으로 제한 탐색; GRT 기본값도 비교 | 먼저 L150/M10/K3, 동일 token 예산 | 품질 reference와 안정 LR 범위 |
| 2. 구조 스크리닝 | GRT의 L150/75/50 × M5/10/20; RMT의 대응 용량·local baseline 확인 | shared LR/품질 budget, K 정책을 명시 | segment·memory의 효과와 후보 지역 |
| 3. 선택 후보 확대 학습 | PPL 상위와 memory 효과가 좋은 후보를 함께 더 학습 | 동일 데이터·미리 정한 schedule | 초반 순위가 유지되는가 |
| 4. router·credit 탐색 | write bias, K/과거 범위, 제한적 LR 재탐색; 필요시 router hidden | 단계2의 상위2~3 구조 주변 | GRT 고유 요소가 품질/기억에 기여하는가 |
| 5. 확인 | 후보2~3개, 비교군, 모델 seed3개 이상, 더 넓은 RTLA | 같은 checkpoint에서 품질과 분석 | 차이가 재현되는가 |
| 6. 확장 | base 또는 다른 데이터 크기에서 소수 후보만 확인 | 같은 비교군·평가 protocol | 효과가 small에만 국한되는가 |

단계2의 GRT 구조 grid는9개다. 동일한 L/M grid를 RMT에도 주고 local NeoX를 L별로 비교하면18+3개의 구조 조건이다. baseline pilot과 같은 조건/예산인 run은 재사용할 수 있다. 모든 조건을 여러 seed로 처음부터 반복하지 않고, 불확실하거나 유망한 조건에 반복 예산을 배분한다.

모든 변수를3값씩 곱하면 trial 수가 급증한다. 대신 LR는 log-scale random/TPE, 의미가 분명한 L/M는 작은 grid, on/off는 유망한 영역에서 paired 비교를 사용한다. random search가 넓은 연속 공간에서 효율적인 기본 탐색이라는 근거는 있지만, 의미 있는3×3 구조 grid까지 피할 필요는 없다. [Bergstra & Bengio, 2012](https://jmlr.org/papers/v13/bergstra12a.html)

### 9.2 token 예산의 검토안

수치는 계획 규모를 판단하기 위한 예시다. 최종 예산은 현재 조합의 GPU preflight와 learning curve를 보고 결정한다.

| 용도 | 예시 supervision 예산 | 목적 |
|---|---:|---|
| 기능 점검 | 기존 smoke | 실행·기록·재개 확인; HPO 순위 근거로 쓰지 않음 |
| 예비 스크리닝 | 24M tokens | 실패 영역·거친 후보 지역 확인 |
| 중간 비교 | 72M tokens | 초반에 늦었던 memory 후보도 확인 |
| 후보 확인 | 216M~384M tokens | small의 충분한 학습과 여러 seed 비교 |

24/72/216M을 사용하면 예산을3배씩 늘리는 rung 예시가 된다. 이를 진짜 promotion으로 쓰려면 최초부터 동일한 최종 schedule을 정해놓고 중간에 정지·재개해야 한다. 짧은 budget의 cosine을 끝까지 소진한 checkpoint에서 max_steps만 늘리는 방식은 다른 학습 곡선을 만든다.

subset에서 고른 후보를 전체 corpus로 옮기는 것은 새로운 실험이다. 데이터 manifest가 바뀌므로 현재 exact-resume 경로로 이어 붙이지 않는다. 작은 데이터의 많은 epoch와 큰 데이터의 적은 epoch를 동일한 일반화 실험으로 해석하지 않는다.

### 9.3 pruning 원칙

처음에는 발산·OOM·명백히 불리한 품질 후보를 주로 제외한다. gate가 작은 후보나 memory delta가 늦게 생기는 후보를 너무 빨리 제거하지 않는다. 초기 여러 checkpoint와 최종 순위의 관계를 pilot에서 확인한 뒤 aggressive pruning을 사용한다.

품질 상위뿐 아니라 **품질 허용 범위 안에서 memory 효과가 유망한 후보**도 일정 수 보존한다. 단일 PPL 순위로만 모든 후보를 줄이면 두 번째 목표를 조사할 여지가 사라질 수 있다. 전체 keep 수와 허용 품질 차이는 사용자 판단 항목으로 남긴다.

## 10. 탐색 도구 비교

### 10.1 도구 선택

| 도구/방법 | 강점 | 현재 프로젝트의 부담·제한 | 제안 |
|---|---|---|---|
| 작은 grid/random runner + 기존 W&B | 가장 적은 변경; paired 구조 비교가 명확; 실행 방식 유지 | 자동 제안·분산 pruning은 제한 | 첫 구조 탐색에 적합 |
| W&B Sweeps | 기존 dashboard·cloud agent; grid/random/bayes와 Hyperband | 기본 sweep metric은하나; raw PPL+RTLA의 진짜2목적 검색과는 다름; 현재 Monitor 연결 조정 필요 | 품질 중심 sweep의 편의가 우선일 때 |
| Optuna TPE + W&B logging | mixed/conditional 공간, trial DB, 명시적 다목적 탐색 가능 | training 결과/진행량 연결 필요; 다목적의 표준 report/prune 제한 | **두 목표를 자동 탐색하려면 우선 검토** |
| Ray Tune + ASHA/BOHB | 많은 worker·resource 관리·checkpoint scheduler | notebook 단일 세션에는 orchestration 부담; 기존 DDP와 자원 배치 연결 필요 | 다수 GPU/서버로 확장할 때 |
| Ax/BoTorch 계열 | 비싼 실험의 noisy multi-objective Bayesian optimization | 모델링·제약·지표 noise 추정과 구현이 더 필요 | 실험이 매우 비싸고 탐색 공간이 작아진 후 |

W&B의 공식 sweep schema는 하나의 metric 이름/방향을 설정한다. 두 지표를 기록하고 사후 Pareto를 그리는 것은 가능하지만, 기본 Bayes sweep이 두 목표를 동시에 최적화한다고 설명하면 안 된다. Hyperband의 iteration은 metric을 기록한 횟수와 연결되므로 token milestone을 맞추는 작업이 필요하다. [W&B 공식 schema](https://github.com/wandb/sweeps/blob/master/src/sweeps/config/schema.json), [W&B sweep configuration](https://github.com/wandb/gitbook/blob/master/sweeps/configuration.md)

Optuna는 `directions`로 다목적을 명시하고 TPE나 NSGA-II 등을 선택할 수 있다. 작은 trial 예산과 conditional search에서는 TPE를 우선 검토한다. 다목적 study에 표준 `Trial.report()`·`should_prune()`를 그대로 사용할 수 없다는 제약이 있으므로, **다목적 TPE와 일반 pruning을 자동으로 함께 적용할 수 있다고 가정하지 않는다.** 처음에는 단일 품질 목적의 pruning 단계와 최종 다목적 단계를 분리하는 안이 단순하다. [Optuna multi-objective](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/002_multi_objective.html), [Trial API 제한](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html), [TPE](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html)

Ray ASHA는 보고된 resource/time attribute를 기준으로 중간 성능을 비교하도록 연결할 수 있다. 여기서는 optimizer step보다 target token 누적량을 resource로 쓰는 편이 낫다. ASHA/Hyperband 자체가 우리가 원하는 memory 지표의 품질이나 조기 순위의 신뢰성을 보장하지는 않는다. [Ray schedulers](https://docs.ray.io/en/latest/tune/api/schedulers.html), [Hyperband 논문](https://arxiv.org/abs/1603.06560), [ASHA 논문](https://arxiv.org/abs/1810.05934)

BoTorch는 noisy multi-objective acquisition을 지원하지만 적은 실험으로 surrogate를 구축해야 하는 문제와 구현량이 있다. 첫 단계의 필수 도구로 두지 않는다. [BoTorch multi-objective](https://botorch.org/docs/multi_objective)

### 10.2 현재 규모에 맞는 두 실행안

**최소 변경안:** 동일 조건의 작은 L/M grid를 순차 실행하고 W&B에서 PPL·delta·비용을 비교한다. 다음으로 LR/router의 좁은 random sweep을 붙인다. 자동 다목적 제안이 없어도 사용자가 trade-off를 판단하는 데 필요한 자료는 얻을 수 있다.

**자동 다목적안:** Optuna가 후보를 제안하고 기존 trainer/Monitor를 사용해 일반 W&B run을 기록한다. 먼저 품질 단일 목적·안전한 early stop으로 후보를 줄이고, 최종 예산에서 NLL과 memory delta의 두 목적 study 또는 Pareto 분석을 진행한다. Optuna는 제안·trial 상태 관리, W&B는 모니터링·비교 역할을 맡는다.

첫 구현부터 Optuna·Ray·Hydra·W&B agent를 모두 중첩하지 않는다. PBT는 LR 등을 학습 도중 바꾸는 데 유용할 수 있으나 M/size 변경은 parameter shape가 달라지고 L 변경은 stream/state 의미가 달라진다. 현재 immutable Config/exact-resume 계약과도 맞지 않아 초기 탐색에서 제외하는 편이 좋다.

## 11. RTLA probe를 HPO용으로 보완하는 방법

### 11.1 현재 probe의 비교 제한

현재 기본 RTLA는 batch8, 이전 warmup3 segments, measured8 segments다. L150에서는 stream당450 tokens를 warmup하고1200 tokens를 측정하지만, L50에서는150/400 tokens가 된다. **L을 sweep하면서 이 기본값을 유지하면 서로 다른 과거와 평가 token을 비교한다.**

권장되는 probe 계약은 다음과 같다.

| 항목 | 권장 계약 |
|---|---|
| 데이터 | 동일 split·같은 prepared token stream·같은 hash |
| partition | evaluation batch와 RTLA batch를 각각 고정; 두 PPL이 같은 데이터 범위를 뜻하지 않음을 명시 |
| warmup | segment 수 대신 token 수로 맞춤 |
| 측정 범위 | 같은 stream ID와 절대 token offset·같은 token 수 |
| 개입 위치 | 각 L의 실제 segment 경계; 경계가 다르다는 구조 효과는 별도로 해석 |
| 문맥 교환 | 같은 시점의 다른 stream; 여러 고정 permutation으로 결과 민감도 확인 |
| 반복 | 고정된 여러 문서/window를 사용하고 model seed도 반복 |
| checkpoint | PPL과 RTLA가 반드시 같은 checkpoint에서 계산됨 |

L150/75/50/25의 공통 배수에 맞는450-token warmup과1200-token 측정 범위는 정렬 예시다. 이 경우 warmup segment 수는3/6/9/18, 측정 segment 수는8/16/24/48이 된다. 실제 위치를 고정하면 초기 비교를 맞추기 쉽다. HPO 전체에서 필요하면 다른 문서/window로 확장한다.

고정 absolute token 범위는 같은 target에서의 품질 효과를 비교한다. 별도로 segment 상대 위치/앞뒤 절반은 같은 역할의 위치에서 memory가 어떻게 쓰이는지 보여준다. 두 분석을 혼동하지 않는다.

### 11.2 도중 분석과 사후 분석의 역할

| 학습 중 | 후보 확정 전 사후 |
|---|---|
| 작고 고정된 probe, 정상/reset/swap NLL | 여러 window/문서·더 긴 warmup·넓은 측정 범위 |
| 실제 update norm·gate·slot cosine은 동작 진단 | slot별 개입과 persistence, 필요시 slot 묶음 개입 |
| 같은 token milestone에서 추세 확인 | 정상의 동일 checkpoint·평가 token에서 재측정 |
| 너무 이른 delta0으로 후보를 자동 제거하지 않음 | seed·window 반복으로 효과 크기와 변동 확인 |

현재 arrays는 stream별 loss를 batch 평균으로 줄여 저장한다. 따라서 세부 문서/stream별 불확실성을 계산하려면 per-stream 또는 문서/block 집계 보존이 추가로 필요하다. 개별 token을 모두 독립 표본으로 취급해 지나치게 작은 신뢰구간을 만들지 않는다. 연속 문맥의 상관을 고려해 문서/긴 block 단위 반복이나 bootstrap을 사용한다.

같은 evaluation 문서를 HPO 중 계속 관찰했다면, 최종 test에서의 delta 확인은 선택된 모델만 대상으로 한다. 지속 범위 분석은 여러 origin의 후속 target이 겹칠 수 있으므로 그 count를 독립적인 새 평가 token 수로 해석하지 않는다.

### 11.3 현재 trace에서 목적함수로 쓰지 않을 것

read gate off의 `r_gates`, router의 write pooling attention, raw state norm, candidate distance는 memory 활용 점수가 아니다. slot cosine도 비슷한 방향을 가진 표현이 동일한 정보를 담는다는 증거가 아니다. slot별 delta가 작아도 분산·중복 저장일 수 있으므로 모든 slot이 불필요하다고 결론 내리지 않는다.

현재 EOS는 원본 **줄마다** 붙는다. EOS가 문서 끝을 뜻한다고 가정하면 문서 경계 분석이 틀릴 수 있다. 문서 기반 분석에는 원본 기사 경계 metadata가 필요하다. state의 현재 reset 규칙은 split/epoch 경계이며 모든 비교 모델에서 동일하게 유지한다.

## 12. 자원·중단·재현성과 코드 간결성

### 12.1 Kaggle/Colab 자원 배치

Kaggle T4×2에서 한 trial이 두 GPU의 DDP를 사용하면 동시에 실행할 수 있는 같은 종류의 trial은1개다. sweep worker를2개 띄운다고2개의2-GPU 실험을 병렬 실행할 수는 없다. 각 GPU에1개의 single-GPU trial을 배치하는 대안도 있지만 accumulation과 runtime을 명시적으로 바꾸고 global batch·조건·성능을 다시 확인해야 한다.

Colab의 기본 trial은 GPU1개, global256이며 실제 microbatch는 현재 자동 조정을 사용한다. GPU 모델·runtime·world_size·정밀도를 cost record에 남긴다. 실제 update 시간과 전체 ETA는 **선택된 L/M/K**에서 재측정한다. 과거0.48초/update 측정을 현재 batch·BPTT 조합에 비례 환산하지 않는다.

기록할 비용은 최소한 wall time, GPU 수, GPU-hours, peak VRAM, target tokens, replay 범위다. GPU-hours와 elapsed hours는 다르다. validation·RTLA·저장 시간도 실제 wall time에 포함한다. 현재 preflight ETA에는 RTLA 추가 시간이 포함되지 않는다.

### 12.2 trial 중단과 promotion

현재 run은 Config·world_size·dataset manifest가 같은 경우에만 exact-resume한다. max_steps 변경도 config 변경으로 거부될 수 있다. 외부 HPO 도구를 붙였다고 안전한 token rung 정지·checkpoint·resume가 자동으로 생기지 않는다.

promotion이 필요하다면 최종 schedule/Config는 고정한 채 execution budget에서 token milestone에 도달했을 때 checkpoint를 저장하고 정지하는 작은 인터페이스가 필요하다. 강제 kill만 사용하면 마지막 저장 이후의 학습을 잃거나 잘못된 trial 상태를 남길 수 있다. notebook session 종료·network 실패·OOM은 낮은 품질과 구분해 실패 상태로 기록한다.

L/M/size/data를 바꾸는 것은 새 trial로 처음부터 시작한다. 다른 GPU에서 기존 보정 batch를 그대로 복원하면 적합성을 보장할 수 없으므로 현재 resume 정책을 따라 처리한다. study 상태와 run 결과를 세션 밖 저장소에 보존하는 운영은 별도로 필요하다.

### 12.3 구현을 단순하게 유지하는 방법

후보 제안과 실행 관리만 담당하는 **하나의 HPO entrypoint**를 두고, 기존 load_experiment→typed Config→trainer/Monitor 경로를 재사용한다. 후보별 YAML·새 notebook·모델 wrapper를 복사하지 않는다. trial별 최종 Config와 결과는 기존 run 폴더에 저장한다.

W&B Sweeps를 선택하면 agent가 주는 parameter/run identity와 현재 Monitor가 생성하는 identity를 연결해야 한다. 바깥 sweep에서 하나, trainer에서 또 하나의 W&B run을 만들지 않는다. 현재 CLI는 일반적인 `--lr=...` sweep 인자를 직접 받지 않으므로 `--set`/typed override로 변환하는 작은 adapter가 필요하다.

Optuna를 선택하면 sampling과 DB 관리만 바깥에 두고 trainer의 계산을 복제하지 않는다. progress/pruning 연결은 명시된 token milestone에서만 동작하게 한다. 처음에는 pruning 없이 같은 예산의 trial을 실행하는 것이 구현과 해석 모두 가장 단순하다.

## 13. 탐색 시작 전에 필요한 코드 작업

| 항목 | 현재 상태 | 우선순위 |
|---|---|---|
| small/base/tiny, L/M/K override | 지원 | 바로 활용 |
| GRT RTLA reset/swap·slot·retention | 지원 | HPO용 probe alignment 보완 |
| 원격 W&B 기록·local metrics·재개 | 지원 | 새로운 trial/run identity 계약 확인 |
| 메모리 없는 동일 NeoX baseline | 미지원 | **필수** |
| WT103 fixed subset 생성/manifest | 별도 준비 필요 | subset 경로를 선택하면 필요 |
| WikiText-2 preset/download 선택 | 미지원 | WT2를 선택하면 필요; 공통 adapter 재사용 |
| trial 종료 시 test 평가 생략 | 현재는 매 run 종료 시 test 평가 | **HPO 전에 필수** |
| checkpoint에 맞춘 PPL/RTLA 재평가 | CLI 분석 가능하나 자동 trial selection 연결 없음 | **필수 계약** |
| token milestone의 evaluation/stop/promotion | update 기반·max_steps 고정 중심 | multi-fidelity를 선택하면 필요 |
| 자동 sweep driver | 미지원 | 도구 선택 후 작은 entrypoint 추가 |
| warmup·AdamW·clip 빈도 metric | 일부 미지원 | 해당 항목을 실제로 탐색할 때만 |
| RMT의 같은 state 개입 분석 | 현재 RTLA 개입은 GRT decoder만 지원 | 메모리 효과 자체를 RMT와 비교할 때 필요 |

특히 현재 [stream 학습 종료 코드](../src/grt/stream_training.py)는 `best.pt`를 로드해 **test를 평가하고 summary 최상위 loss/perplexity에 그 결과를 넣는다.** 이를 sweep 목표로 사용하면 test로 후보를 선택하게 된다. 탐색에서는 validation metric을 명시하고 test 실행을 최종 단계로 분리한다.

또한 현재 summary의 validation은 마지막 validation이고, test는 best checkpoint에서 계산한다. 학습 중 RTLA는 각 step의 현재 모델에서 측정한다. 최종 best checkpoint의 PPL에 마지막 step의 RTLA를 붙이면 같은 모델의 두 변수가 아니다. 선택된 checkpoint에서 두 값을 함께 다시 측정해야 한다.

이 표는 구현 계획을 결정하기 위한 것이며, 본 보고서 작성 중에 해당 코드를 변경하지 않았다. 우선 선택된 방법에 필요한 항목만 구현하고 도구 사용을 위해 불필요한 호환성/중복 실행 경로를 추가하지 않는다.

## 14. 사용자가 결정할 항목과 권장 시작 묶음

| 결정 항목 | 권장 시작점 | 다른 선택이 더 나은 경우 |
|---|---|---|
| 연구의 1차 주장 | 같은 backbone에서 품질과 메모리 기여를 함께 확보 | 메모리 메커니즘 자체만 분석한다면 품질 허용 폭을 넓게 설정 |
| 데이터 | 현재 WT103-v1 BPE, fixed subset 예비 탐색 + 전체 확인 | 소규모 corpus 일반화가 주 관심이면 WT2 별도 계열 |
| size | small 주 탐색 | 매우 낮은 예산이면 tiny pilot 후 small에서 순위 재검증 |
| 핵심 grid | L150/75/50 × M5/10/20 | 더 강한 병목/용량 포화가 필요하면25/40 확장 |
| BPTT 정책 | 빠른 K3 스크리닝 후450-token 통제 비교 | 비교의 통제가 최우선이면 처음부터3/6/9 |
| 품질 reference | local NeoX + 튜닝된 RMT의 실제 validation | 기존 PPL34 run을 reference로 쓰려면 metadata를 먼저 고정 |
| RTLA 목적 | Δ_swap 주 지표, Δ_reset과 baseline 차이로 검증 | 독립적인 문맥 진단이 확보되면 신뢰구간 기반 제약 추가 |
| 도구 | 작은 grid + W&B → 필요하면 Optuna TPE | cloud 단일 metric 관리가 우선이면 W&B Sweeps |
| 예산 | 예비24M, 중간72M, 확인216~384M의 검토안 | 현재 GPU preflight와 learning curve에 따라 조정 |
| 반복 | screen은1 seed, 최종 후보·baseline은3 seeds 이상 | 작은 차이를 주장하려면 더 많은 seed/문서 반복 |
| 최종 선택 | 품질 허용 후보의 Pareto와 재현성으로 판단 | 명확한 deployment 목표가 있으면 wall time/VRAM 제약을 추가 |

가장 먼저 확정할 것은 **데이터 계열, size, baseline 종류, 품질 허용 폭, BPTT 통제 정책, 사용 가능한 trial/GPU 예산**이다. 이 항목이 정해지면 검색 공간과 도구를 그 범위에 맞춰 작게 구현할 수 있다.

## 15. 근거 자료와 관측의 범위

외부 자료는2026-10-12에 확인했다. 도구 API는 실제 도입 시 버전을 고정하고 그 버전의 지원 범위를 다시 확인한다.

- [RMT 2022 논문](https://papers.neurips.cc/paper_files/paper/2022/file/47e288629a6996a17ce50b90a056a0e1-Paper-Conference.pdf): word150/50 비교와 memory/BPTT 실험.
- [Pointer Sentinel Mixture Models](https://arxiv.org/pdf/1609.07843): WikiText 규모와 공유된 validation/test 기사, vocabulary 차이.
- [Salesforce WikiText dataset card](https://huggingface.co/datasets/Salesforce/wikitext/blob/main/README.md): raw/non-raw 구분과 HF configuration.
- [Random Search for Hyper-Parameter Optimization](https://jmlr.org/papers/v13/bergstra12a.html): random search의 기본 근거.
- [W&B 공식 sweep schema](https://github.com/wandb/sweeps/blob/master/src/sweeps/config/schema.json), [configuration 설명](https://github.com/wandb/gitbook/blob/master/sweeps/configuration.md): sweep metric·방법·Hyperband의 기록 단위.
- [Optuna multi-objective](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/002_multi_objective.html), [TPE](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html), [Trial API](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html): 다목적 sampler와 report/prune 제약.
- [Ray Tune schedulers](https://docs.ray.io/en/latest/tune/api/schedulers.html), [Hyperband](https://arxiv.org/abs/1603.06560), [ASHA](https://arxiv.org/abs/1810.05934): multi-fidelity 자원 배분.
- [BoTorch multi-objective](https://botorch.org/docs/multi_objective): noisy 다목적 Bayesian optimization.

PPL34와 word→BPE 측정은 사용자가 제공한 관측이다. 모델 parameter와 현재 지원 기능은 로컬 코드·명세를 확인했다. 제안한 HPO 순서·후보 공간·예산은 이 자료를 바탕으로 한 설계 판단이며, 탐색을 실행해 얻은 성능 결과가 아니다.
