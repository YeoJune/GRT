Experiment 1 — First run & debugging (5060 Ti)

Goal

- Run the toy passkey experiment, verify training stability, collect RTLA traces and tune early hyperparameters.

Assumptions

- GPU: NVIDIA 5060 Ti (CUDA available)
- Python 3.10+ and dependencies installed via `pip install -e .`
- `configs/experiment/toy_passkey.yaml` is the active config for this run

Step 0 — Quick checks (on your machine)

- Confirm CUDA device and name:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu')"
nvidia-smi
```

Step 1 — Prepare environment

- Create venv, install project editable, install torch matching CUDA version (example commands):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# install editable package and misc deps
pip install -e .
# if torch not installed or needs a specific CUDA build, install separately, e.g.:
# pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Step 2 — Sanity check (Phase 1)

```powershell
python scripts/sanity_check.py
```

Expect: all checks PASS.

Step 3 — Short training run (first experiment)

- Run a short run (we set `max_steps` small in the CLI or config). Example: 200 steps first.

```powershell
python scripts/train.py configs/experiment/toy_passkey.yaml --max_steps 200
```

Recommended experiment standard

- `data.tokenizer: gpt2`
- `model.vocab_size: 50257`
- `training.batch_size: 2`
- `training.mixed_precision: fp16`
- `wandb.enabled: true`
- `rtla.enabled: true` with `trace_every_n_steps: 100`
  What to watch while training

- GPU memory usage: `nvidia-smi -l 2`
- Errors in logs (OOM, NaN loss)
- If loss is NaN or diverging quickly, stop and proceed to tuning steps below

Step 4 — Collect RTLA trace

- If `rtla.enabled` is true in config, traces will be saved automatically at configured steps.
- Otherwise run `run_rtla.py` on a saved checkpoint:

```powershell
python scripts/run_rtla.py configs/experiment/toy_passkey.yaml --checkpoint checkpoints/step_000200.pt --input_text "Testing RTLA"
```

Step 5 — Inspect RTLA panels

- Panels are saved to the `rtla.output_dir` (default: `traces/`). Open the PNG(s) and check:
  - Panel A: write-gate heatmap — ensure not all red (write collapse) and not all zero
  - Panel D: prediction error peaks occur at meaningful timesteps

Step 6 — Early tuning checklist (if problems observed)

- If write gate collapse (mean W > 0.85): reduce `model.register.write_gate_bias_init` (more negative), e.g. -2.5 -> -3.0
- If write gates are too low (never write): raise bias towards -1.0
- If model concentrates in few registers: increase `model.register.dropout_prob` (e.g. 0.10 -> 0.2) or reduce ALU capacity
- If OOM: reduce `training.batch_size`, reduce `model.alu.d_ff`, or use smaller `d_model`
- If unstable (NaNs): reduce learning rate (`training.lr` by factor 2-5), enable grad clipping (already present), or switch to fp32 temporarily

Step 7 — Iterate

- Apply one change at a time, rerun for 50-200 steps, inspect RTLA + loss curves, repeat until stable.

Useful commands recap

- Run one quick train:

```powershell
python scripts/train.py configs/experiment/toy_passkey.yaml --max_steps 200
```

- Evaluate a checkpoint:

```powershell
python scripts/evaluate.py configs/experiment/toy_passkey.yaml --checkpoint checkpoints/step_000000.pt
```

- Create/save RTLA panels from a checkpoint:

```powershell
python scripts/run_rtla.py configs/experiment/toy_passkey.yaml --checkpoint checkpoints/step_000000.pt --input_text "The quick brown fox"
```

Notes

- We tuned `configs/experiment/toy_passkey.yaml` to use the standard GPT-2 tokenizer setup (`data.tokenizer: gpt2`, `model.vocab_size: 50257`) plus `mixed_precision: fp16` and `batch_size: 2` for a 5060 Ti. If your GPU supports `bf16` you can try that instead.
- When making hyperparameter changes, track them (W&B or a simple notes file) so results are reproducible.
