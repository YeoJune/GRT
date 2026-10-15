# Colab 사후 RTLA 분석과 결과표

작성일: 2026-10-14 · 대상: 현재 형식의 WikiText GRT checkpoint · validation 전용

[Colab에서 열기](https://colab.research.google.com/github/YeoJune/GRT/blob/main/notebooks/colab_analysis.ipynb) · [노트북 파일](../notebooks/colab_analysis.ipynb) · [학습 결과표](20261012_experiment_plan.md) · [M10/M40/M160 중간 비교](20261014_register_comparison.md)

## 1. 어디에 무엇을 넣는가

학습 환경에서 해당 모델의 `grt/best.pt`와 prepared 데이터의 아래 파일을 Google Drive에 복사한다. `last.pt`나 별도 보존한 중간 checkpoint를 사용해도 된다. checkpoint 이름 대신 내부 step과 SHA256으로 결과를 구분한다.

```text
MyDrive/GRT_analysis/
├── checkpoints/
│   ├── m10/best.pt
│   ├── m40/best.pt
│   └── m160/best.pt
└── wikitext103_gpt2_bpe/
    ├── manifest.json
    ├── validation.pt
    └── tokenizer/             # manifest에 기록된 원본 artifact 전부
```

현재 Kaggle 기본 prepared 경로는 `/kaggle/working/wikitext103_gpt2_bpe`다. 원래 run의 `grt/best.pt`가 필요하며 `experiment_report.json`에는 가중치가 없다. Drive 웹에서 폴더·파일을 업로드해도 되고 기존 Drive 보관본을 사용해도 된다. 원본 checkpoint는 자신이 학습한 현재 `memory_experiment/4` 형식을 사용한다.

**`train.pt`와 `test.pt`는 필요 없다.** manifest는 원본 전체를 그대로 복사하고 validation과 manifest에 열거된 tokenizer 파일만 가져오면 된다. 분석 helper는 checkpoint의 dataset manifest와 복사본이 같은지, validation·tokenizer 파일의 hash가 맞는지 검사한다. 데이터가 다르면 분석 전에 거부하며 새로 준비한 데이터로 조용히 대체하지 않는다. 로컬 tokenizer artifact를 검증해 사용하므로 HF 데이터·tokenizer를 다시 다운로드하지 않는다.

3k→4k 학습 전환을 나중에 분석하려면 당시 checkpoint가 따로 있어야 한다. `best.pt`와 `last.pt`만 남아 있고 둘 다10k이면 그 파일에서3k state를 복원할 수 없다. 당시 `rtla/.../arrays.npz`가 남아 있다면 기존 gate trace를 확인하는 자료로 별도 보관한다.

## 2. 실행 순서와 바꿀 값

1. 위 Colab 링크를 열고 GPU 런타임을 선택한다. 단일 T4를 권장한다.
2. 코드·라이브러리 설치와 Drive 연결 셀을 실행한다. W&B 인증은 필요 없다.
3. `checkpoint`, `prepared_dir`, `output_dir` 경로를 지정한다. M/MLP/backbone은 checkpoint에서 읽으므로 다시 입력하지 않는다.
4. `job.inspect()` 출력에서 실제 checkpoint step·M·MLP, hash 검증과 forward 횟수를 확인한다.
5. `job.run()` → 그림 확인 → `job.report(download=True)`를 실행한다.

전달받은 코드 묶음을 사용하는 경우 `colab_analysis.ipynb`를 Colab의 **파일 → 노트북 업로드**로 열고, `grt-colab-analysis.zip`은 왼쪽 파일 탭의 `/content`에 올린 뒤1번 셀부터 실행한다. 업로드된 ZIP이 있으면 해당 코드로 설치하고 없으면 GitHub main을 사용한다. ZIP은 checkpoint/data와 별도이며 runtime 코드·문서만 포함한다. ZIP에는 이 안내서도 `docs/20261014_posthoc_analysis.md`로 들어 있다.

| 변수 | 기본값 | 의미 |
|---|---:|---|
| batch_size | 8 | 고정 validation stream 수; 변경하면 probe token도 달라짐 |
| segments | 8 | 측정 segment 수 |
| warmup_segments | 3 | 시작부터 정상 state를 누적할 segment 수 |
| horizon | 4 | swap을 한 번 적용하고 현재부터 이후3 segments까지 추적; 0이면 생략 |
| analyze_slots | False | True면 각 슬롯을 하나씩 학습된 초기 state S0로 되돌려 평가 |

먼저 기본값으로 reset/swap·retention·gate 분포를 확인하고, 슬롯별 분석까지 필요하면 `analyze_slots=True`로 바꿔 선택 셀부터 다시 실행한다. M160·측정8 segments에서는 슬롯 개입만1,280회 forward를 추가한다. 실제 총 횟수는 warmup·reset/swap·retention까지 더해 inspect에서 계산하며 실행 중 진행률과 대략 남은 시간을 출력한다. GPU별 시간을 미리 확정하지 않는다.

**M10/M40/M160 비교는 batch·segments·warmup·horizon을 같게 유지한다.** 기본 batch8/warmup3/측정8은 학습 중 기본 probe와 맞는다. 다른 window는 warmup을11 등으로 변경해 같은 설정으로 각 M을 분석한다. warmup은 validation 시작부터 누적하므로 단순히 중간 state를 임의로 가져오지 않는다. probe PPL은 전체 validation PPL과 구분한다.

분석은 FP32·TF32 off·dropout off·gradient off이며 optimizer를 실행하지 않는다. Kaggle2GPU checkpoint를 Colab1GPU에서 분석할 수 있다. 학습 재개의 world_size 조건과 별도다. 모델 구조·가중치·학습 설정 파일은 변경하지 않고 런타임 내 데이터 경로와 probe batch만 분석용으로 지정한다.

## 3. 결과와 재실행

출력은 `output_dir/m<M>_step<step>_<signature>/` 아래에 저장한다. 기본 output_dir는 Drive의 `GRT_analysis/outputs`여서 세션 종료 후에도 보존된다. 입력 checkpoint·데이터·분석 조건·주요 분석 코드·실행 라이브러리/장치가 같은 완료 결과는 재사용한다. 옵션을 바꾸면 별도 폴더를 사용하고 결과를 합산하거나 이전 결과를 덮어쓰지 않는다. 중단된 분석은 재개 state를 저장하지 않으므로 처음부터 다시 실행한다.

| 파일 | 내용 / 용도 |
|---|---|
| **posthoc_report.json** | 에이전트에게 보낼 파일 하나. 아래 식별값·요약·분석 배열을 포함 |
| summary.json / summary.txt | probe 조건·정상/reset/swap·slot/retention의 로컬 요약 |
| arrays.npz | 모든 분석 배열의 로컬 보관본 |
| memory.png | 위치별/segment별 개입 효과, slot 초기화, retention |
| trace.png | slot별 batch 평균 gate/state/update와 write pooling attention |

JSON에는 checkpoint 이름·hash·내부 step·학습 world_size, 원래 학습 Config, 데이터 manifest, 실제 분석 조건·코드 식별값·라이브러리·GPU·시간·VRAM을 담는다. 정상/reset/swap NLL/PPL·early/late, slot별 delta, retention과 lag별 target 수, example별 raw gate·분포·slot별 통계 및 분석 배열도 함께 담는다. 미측정 tail은 JSON null이며 미측정 slot/retention은 빈 값으로 남긴다. checkpoint 가중치·optimizer·원본 token stream은 넣지 않는다.

현재 checkpoint를 고정한 probe에서 gate<=0.01 / >=0.99 비율, histogram, slot별 평균·분산·segment 변화·동일 segment 내 example 차이를 확인한다. **학습 step별 gate 변화가 아니다.** 기존 학습 trace의 batch 평균만으로는 보이지 않던 example별 포화를 별도로 보존한다. read gate off의 값을 읽기량으로 쓰지 않고, pooling attention을 token→register 읽기 attention으로 해석하지 않는다.

slot 개입은 독립적인 단일 슬롯 초기화이며 기여도를 합산해 전체 기억량으로 해석하지 않는다. retention은 한 번 잘못된 state를 주입한 뒤의 영향·회복으로, 정확한 기억 수명이나 일반적인 장기 기억 능력을 확정하지 않는다. 유효 rank와 token→register read attention 수집은 이번 노트북 범위에 포함하지 않는다.

## 4. 무엇을 보내고 어떻게 정리하는가

**실험별 `posthoc_report.json` 하나만 보내면 된다.** 여러 환경에서 만든 JSON을 노트북에서 누적하지 않는다. 파일명이 같아도 내부 report_id·checkpoint hash·step·조건으로 구분한다. 다운로드가 막히면 출력된 Drive 폴더에서 직접 받는다. PNG·NPZ는 기본 전달에 필요 없지만 Drive에 보존한다.

에이전트는 원본을 `results/posthoc/<report_id>.json`에 보관하고 아래 표를 채운다. M·checkpoint step/hash·validation/tokenizer hash와 probe 조건을 확인한 뒤 같은 조건끼리 비교한다. 슬롯 분석 on/off·다른 window·다른 step은 별도 행으로 기록한다. 같은 report_id 재전달은 중복 결과로 세지 않는다. 학습 experiment report에는 checkpoint hash가 없으므로 run 경로·step·설정·데이터 식별값을 대조해 연결하고, 불확실한 연결은 추정으로 표시한다.

M10/M40/M160의 품질·reset/swap, gate 포화/입력 의존성, slot 기여 집중/분산, retention을 표·그림으로 정리하고 [용량 비교 보고서](20261014_register_comparison.md)에 사후 관측을 추가한다. 초기 중간 결과와 사용자의 bandwidth·압축·write 가설은 관측 사실과 구분한다. 사후 분석만으로 학습 전환의 인과관계를 확정하지 않는다.

| M | checkpoint step / hash | probe / 정상 PPL | reset / swap ΔNLL | gate near0 / near1 | slot / retention 관측 | 상태 / 원본 |
|---:|---|---|---|---|---|---|
| 10 | | | | | | 미수신 |
| 40 | | | | | | 미수신 |
| 160 | | | | | | 미수신 |

| 사용자 판단 | 기록 |
|---|---|
| bandwidth·압축 가설을 지지/수정하는 관측 | |
| gate 포화 또는 선택적 갱신의 해석 | |
| 선택 M과 추가로 확인할 조건 | |

## 5. 구현·검증 범위

노트북은 bootstrap·경로·분석 선택과 helper 호출만 담당한다. 실행·검증·단일 JSON 내보내기는 `grt.utils.posthoc.WikiTextPosthoc`, 개입은 기존 `grt.memory_analysis.analyze_stream`을 재사용한다. example별 gate 수집과 진행 callback은 사후 분석에서만 선택적으로 켜며 학습 중 기본 RTLA의 기록·계산 경로는 유지한다. GPU 실행 시간·M160 분석은 실제 Colab에서 확인한다.

작성 시 Python 문법·노트북 모든 코드 셀의 컴파일·빈 실행 출력·문서 링크를 확인했다. CPU 회귀 테스트는 `tests/test_posthoc.py`와 기존 memory analysis 테스트에 추가했다. 2026-10-15 의존성 설치 후 Python3.12.14/PyTorch2.14.1+cpu에서 사후 report roundtrip·checkpoint 보존·데이터 불일치 거부·example별 gate 포화·기존 memory analysis 및 baseline의 단일 프로세스 검사15개가 통과했다. NeoX2-worker CLI 검사도 로컬 소켓을 허용한 환경에서 통과해 총16개를 확인했다. GPU·실제 small checkpoint의 Colab 실행은 아직 미검증이다. 재검사는 `python -m pytest tests/test_posthoc.py tests/test_memory_analysis.py -q`로 수행한다.
