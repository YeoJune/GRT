# 아카이브

이전 설계·계획·검토·일회성 검증 기록이다. 실험의 최종 기록은 [실험 폴더](../experiments/README.md)에 주제별 하나씩 유지한다. 현재 실행·명세·선택은 [문서 목록](../README.md)을 따른다. archive에는 Markdown만 두며 실행 코드·config·notebook·원본 결과 파일은 넣지 않는다. 원본 결과는 `results/`, 그림·표 이력은 `docs/figures/history/`, `docs/tables/history/`에 보존한다.

| 기록 | 문서 |
|---|---|
| 실험 workflow: 역할별 문서 정리 전 스냅샷 | [20261023_experiment_workflow.md](20261023_experiment_workflow.md) |
| 모델 명세: 역할별 문서 정리 전 스냅샷 | [20261023_current_spec.md](20261023_current_spec.md) |
| 서버 연결: 초기 구현·검증 안내 | [20261022_server_integration.md](20261022_server_integration.md) |
| 원격 GPU 큐·임대 목표 설계 | [20261021_remote_gpu_queue_design.md](20261021_remote_gpu_queue_design.md) |
| 실험 제어 API 목표 설계 | [20261021_experiment_control_api_design.md](20261021_experiment_control_api_design.md) |
| 게이트 안정화 단계의 모니터링 이력 | [20261020_gate_monitoring.md](20261020_gate_monitoring.md) |
| L75 small: gate 안정화 실험 | [20261019_gate_stabilization_history.md](20261019_gate_stabilization_history.md) |
| Write loss regularization으로 레지스터 분화를 유도하는 설계 검토 | [20261016_write_regularization.md](20261016_write_regularization.md) |
| WikiText 크기 선택과 짧은 동작 확인 | [20261011_wikitext_presets.md](20261011_wikitext_presets.md) |
| Small WikiText-103: Kaggle T4 × 2 | [20261011_wikitext_kaggle.md](20261011_wikitext_kaggle.md) |
| WikiText-103 GPT-2 BPE와 Kaggle 노트북 | [20261011_wikitext_bpe.md](20261011_wikitext_bpe.md) |
| WikiText-103: 공통 NeoX backbone 확정 기준과 남은 문제 | [20261011_wikitext_backbone_remaining.md](20261011_wikitext_backbone_remaining.md) |
| WikiText-103 승인된 구현 설정 | [20261011_wikitext_approved_setup.md](20261011_wikitext_approved_setup.md) |
| WikiText-103 약 30M 공통 backbone 실험: 검토용 명세 | [20261011_wikitext30m_setup.md](20261011_wikitext30m_setup.md) |
| 이전 구현 검증 기록 | [20261011_validation_history.md](20261011_validation_history.md) |
| Small D256 직접 CPU 실험 결과 | [20261011_small256_cpu.md](20261011_small256_cpu.md) |
| Remember의 새 인코더 지원 검토 | [20261011_remember_encoder_review.md](20261011_remember_encoder_review.md) |
| 이전 Remember Kaggle 운영 기록 | [20261011_kaggle_remember.md](20261011_kaggle_remember.md) |
| GRT LM 아키텍처 합의 정리 | [20261011_grt_lm_architecture.md](20261011_grt_lm_architecture.md) |
| GRT 인코더 버전 명세 | [20261011_grt_encoder_spec.md](20261011_grt_encoder_spec.md) |
| GRT 디코더 버전 명세 | [20261011_grt_decoder_spec.md](20261011_grt_decoder_spec.md) |
| 실험 설정과 노트북 운영 규칙 | [20261011_experiment_workflow.md](20261011_experiment_workflow.md) |
| GRT/RMT/NeoX 최신 실험·구현 명세 | [20261011_current_spec.md](20261011_current_spec.md) |
| Kaggle T4×2 Remember 실험 | [20261010_kaggle.md](20261010_kaggle.md) |
| Remember 공통 실험과 코드 정리 기준 | [20261010_design.md](20261010_design.md) |
| 실제 진입점 CPU 검증 기록 | [20261010_cpu_verification.md](20261010_cpu_verification.md) |
| 2026-10-02 구현 기록 | [20261002_implementation.md](20261002_implementation.md) |
| Colab T4 Copy 초기 수렴 프로필 | [20261002_colab_t4_copy.md](20261002_colab_t4_copy.md) |
| GRT 구현 및 합성 메모리 벤치마크 명세 | [20260916_spec.md](20260916_spec.md) |
| Gated Register Transformer (GRT) — Technical Specification | [20260608_spec.md](20260608_spec.md) |
| 출력은 기본적으로 configs.base.yaml에 설정된 rtla.output_dir (기본: traces/) 아래에 저장됩니다. | [20260608_run_tests.md](20260608_run_tests.md) |
| Gated Register Transformer with Global Attention Pooling and Overwrite-driven Decay | [20260608_proposal.md](20260608_proposal.md) |
| install editable package and misc deps | [20260608_experiment1.md](20260608_experiment1.md) |
