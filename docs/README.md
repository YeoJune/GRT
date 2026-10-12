# 문서 목록

현재 운영 기준은 다음 두 문서다.

- [통합 명세](20261011_current_spec.md): 모델 구조, 데이터·학습 계약.
- [실험 workflow](20261011_experiment_workflow.md): 설정 카탈로그, CLI, 노트북/utils, 확장 규칙.

검토 보고서:

- [메모리 활용성 우선 실험 계획](20261012_experiment_plan.md): small·segment150·10k baseline에서 시작하는 단계별 탐색, 무료/유료 자원 배분, WB·극대 메모리 후보와 추가 결정 사항. 사용자 후속 지시를 반영한 최신 계획이다.
- [GRT 하이퍼파라미터 탐색](20261012_hyperparameter_search_report.md): PPL·RTLA 지표, 데이터/size/baseline, 탐색 순서와 도구 비교. 제안 단계이며 실행 조건은 미확정이다.

실행 시작점은 [저장소 README](../README.md), [Kaggle 노트북](../notebooks/kaggle.ipynb), [Colab 노트북](../notebooks/colab.ipynb)이다.

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
