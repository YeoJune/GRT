# 문서 목록

## 공통 기준과 실행

- [통합 명세](20261011_current_spec.md): 모델 구조·데이터·학습 계약.
- [실험 workflow](20261011_experiment_workflow.md): 설정·Kaggle/Colab 학습·사후 분석·파일 전달·새 실험 기록 규칙.
- [원본 JSON 보관 규칙](../results/README.md): 결과 파일 형식과 보관 경로.

## 실험별 보고서

| 실험 | 현재 상태 | 문서 |
|---|---|---|
| L75 small write gate 비교 | 실행 준비: M8/20/75·20k, dropout 선택 후 bias−2/0/+2 | [계획·결과·판단](experiments/20261015_l75_write_gates.md) |
| small 레지스터 용량 비교 | NeoX/RMT baseline, GRT M10/M40/M160의10k 학습·다중 window/그룹 사후 분석 완료; 선택 M 미정 | [계획·결과·해석·사용자 판단](experiments/20261012_small_register_sweep.md) |

실험 주제마다 위 문서 하나에 조건·run/seed·사후 분석·결론을 갱신한다. MLP·segment·size 비교를 시작하면 새 행과 문서를 추가한다. 실행 시작점은 [저장소 README](../README.md), [Kaggle](../notebooks/kaggle.ipynb), [Colab 학습](../notebooks/colab.ipynb), [Colab 사후 분석](../notebooks/colab_analysis.ipynb)이다.

## 이전 기록

이전 설계·검토·검증 기록은 `archive/`의 Markdown 문서로만 보존한다. 실행 코드와 config는 보관하지 않는다.
아래 문서는 과거 기록이며 최신 실행 설정으로 사용하지 않는다.

| 기록 | 문서 |
|---|---|
| 이전 Remember Kaggle 운영 기록 | [20261011_kaggle_remember.md](archive/20261011_kaggle_remember.md) |
| 20260608_experiment1 | [20260608_experiment1.md](archive/20260608_experiment1.md) |
| 20260608_proposal | [20260608_proposal.md](archive/20260608_proposal.md) |
| 20260608_run_tests | [20260608_run_tests.md](archive/20260608_run_tests.md) |
| 20260608_spec | [20260608_spec.md](archive/20260608_spec.md) |
| 20260916_spec | [20260916_spec.md](archive/20260916_spec.md) |
| 20261002_colab_t4_copy | [20261002_colab_t4_copy.md](archive/20261002_colab_t4_copy.md) |
| 20261002_implementation | [20261002_implementation.md](archive/20261002_implementation.md) |
| 20261010_cpu_verification | [20261010_cpu_verification.md](archive/20261010_cpu_verification.md) |
| 20261010_design | [20261010_design.md](archive/20261010_design.md) |
| 20261010_kaggle | [20261010_kaggle.md](archive/20261010_kaggle.md) |
| 20261011_grt_decoder_spec | [20261011_grt_decoder_spec.md](archive/20261011_grt_decoder_spec.md) |
| 20261011_grt_encoder_spec | [20261011_grt_encoder_spec.md](archive/20261011_grt_encoder_spec.md) |
| 20261011_grt_lm_architecture | [20261011_grt_lm_architecture.md](archive/20261011_grt_lm_architecture.md) |
| 20261011_remember_encoder_review | [20261011_remember_encoder_review.md](archive/20261011_remember_encoder_review.md) |
| 20261011_wikitext30m_setup | [20261011_wikitext30m_setup.md](archive/20261011_wikitext30m_setup.md) |
| 20261011_wikitext_approved_setup | [20261011_wikitext_approved_setup.md](archive/20261011_wikitext_approved_setup.md) |
| 20261011_wikitext_backbone_remaining | [20261011_wikitext_backbone_remaining.md](archive/20261011_wikitext_backbone_remaining.md) |
| 이전 small CPU 검사 | [20261011_small256_cpu.md](archive/20261011_small256_cpu.md) |
| 이전 WikiText Kaggle 안내 | [20261011_wikitext_kaggle.md](archive/20261011_wikitext_kaggle.md) |
| BPE 추가 기록 | [20261011_wikitext_bpe.md](archive/20261011_wikitext_bpe.md) |
| 크기·예산·W&B 추가 기록 | [20261011_wikitext_presets.md](archive/20261011_wikitext_presets.md) |
| 이전 구현 검증 기록 | [20261011_validation_history.md](archive/20261011_validation_history.md) |
