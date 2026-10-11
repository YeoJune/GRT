# Remember의 새 인코더 지원 검토

2026-10-11. 코드/데이터 계약 검토이며 새 모델 구현 전 기록이다.

## 결론

Remember는 새 인코더에서 prefix 재계산 없이 지원할 수 있다.
완성 fact는 관측된 key/value이므로 양방향 처리 후 stack 종료 시 write-back1회가 적절하다.
최종 query에 정답 value/EOS를 입력으로 넣은 채 shifted CE를 적용하는 기존 teacher-forcing 계약은 사용할 수 없다.
새 인코더가 정답을 볼 수 있기 때문이다. loss mask는 입력 attention을 차단하지 않는다.

## 현재 배치

fact: [key, SEP100, value, EOS102], 각fact가segment1개.
query train: [query_key, GEN101, target_value, EOS102].
query generation: [query_key, GEN101].
labels=input_ids, labels_mask는predictor 위치, loss는내부shift.
value_size1에서GEN->value, value->EOS 두CE의평균과 value+EOS 생성EM이다.

## prefix 없는 인코더 선택지

A. Retrieval classification: fact들을1회씩 처리하여state를 만들고 query=[key,GEN]를1회 처리한다.
GEN 출력의기존 tied vocabulary head로 target_value를분류한다. 정답은label로만 제공한다.
query 종료 후 최종 write-back은1회지만 이후 segment가없으면 해당state는 loss/readout에사용하지 않는다.
현재value_size1의검색문제를 그대로풀 수 있다. value CE와value EM을사용하고 EOS생성은평가에서제외한다.
기존value+EOS loss/EM과정확히같은평가가아니므로 별도metric 이름/결과로구분한다.
value_size>1의일반화는출력위치/target 규칙을추가설계해야한다.

B. Parallel answer slots: query=[key,GEN,MASK_value...,MASK_EOS]를1회처리하고
각slot의tied head로value+EOS를동시에예측한다. 정답은labels에만존재한다.
양방향intermediate register, read on/off 모두정답을읽지않는다.
value_size>1도고정길이slot으로가능하고value+EOS exact match의대상은기존과같다.
다만MASK token,slot위치,non-shift loss,고정답길이를정해야한다.
원본autoregressive generation과학습목적은달라진다.

현재승인만으로A/B중하나를임의선택하지않는다. 인코더목표의확정이필요하다.
현재decoder intermediate의query에도teacher-forced정답을넣으면register경유누출이가능하며,
단순token-token causal로이검토를생략할수없다.

## 유지할 공통 조건

원본fact/query-key 선택, 데이터split/seed/curriculum은재사용가능하다.
새encoder query view는data adapter에서만들고 labels를embedding입력에섞지않는다.
새batch/metric계약은명시적으로구분하며trainer에token ID 기반분기를추가하지않는다.
기존author RMT/Remembercheckpoint/fixture는보존한다.
공정한encoder비교가목적이면RMT encoder에도동일query/target/loss를적용한다.

## 구현 검증

segment당stack/write-back1회,prefix 재계산없음,정답label 변경시입력/logits 불변,
fact value변경시query예측에영향가능,fact->querygradient,
동일query view에서학습/평가일치,원본baseline회귀검사를한다.
이번작업은계약검토로학습/수렴검증은하지않았다.
