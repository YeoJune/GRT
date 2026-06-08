"""GRT Phase 1 Sanity Check."""

from __future__ import annotations

import os
import sys
import traceback

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from grt.config import GRTConfig
from grt.model.grt import GRTModel


PASS = "✅ PASS"
FAIL = "❌ FAIL"


def run(name: str, fn) -> bool:
    try:
        fn()
        print(f"{PASS}  {name}")
        return True
    except Exception:
        print(f"{FAIL}  {name}")
        traceback.print_exc()
        return False


cfg = GRTConfig()
B = 2
N = cfg.model.segment_len
M = cfg.model.num_registers
V = cfg.model.vocab_size
L = N * 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GRTModel(cfg.model).to(device)
model.eval()

results: list[bool] = []


def check_param_count() -> None:
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 0, "모델 파라미터가 없음"
    print(f"      파라미터 수: {n_params:,}")


results.append(run("파라미터 수 > 0", check_param_count))


def check_forward_shape() -> None:
    ids = torch.randint(0, V, (B, L), device=device)
    with torch.no_grad():
        out = model(ids)
    assert out.logits.shape == (B, L, V), f"logits shape 오류: expected {(B, L, V)}, got {out.logits.shape}"


results.append(run("Forward pass output shape [B, L, V]", check_forward_shape))


def check_alu_fixed_len() -> None:
    observed: list[int] = []

    def hook(module, inp, out):
        observed.append(inp[0].shape[1])

    handle = model.alu.encoder.layers[0].register_forward_hook(hook)
    ids = torch.randint(0, V, (B, L), device=device)
    with torch.no_grad():
        model(ids)
    handle.remove()
    expected = N + M
    for seq_len in observed:
        assert seq_len == expected, f"ALU seq_len 오류: expected {expected}, got {seq_len}"


results.append(run(f"ALU 입력 길이 항상 N+M={N+M}", check_alu_fixed_len))


def check_gate_range() -> None:
    ids = torch.randint(0, V, (B, L), device=device)
    with torch.no_grad():
        out = model(ids, return_trace=True)
    trace = out.trace
    assert trace is not None
    for step, (r_gate, w_gate) in enumerate(zip(trace.r_gates, trace.w_gates)):
        assert r_gate.min() >= 0 and r_gate.max() <= 1, f"R_gate out of [0,1] at t={step}"
        assert w_gate.min() >= 0 and w_gate.max() <= 1, f"W_gate out of [0,1] at t={step}"


results.append(run("게이트 값 범위 [0, 1]", check_gate_range))


def check_register_updates() -> None:
    ids = torch.randint(0, V, (B, L), device=device)
    with torch.no_grad():
        out = model(ids, return_trace=True)
    trace = out.trace
    assert trace is not None
    norms = [s.cpu() for s in trace.s_norms]
    changes = sum(1 for i in range(1, len(norms)) if not torch.allclose(norms[i], norms[i - 1], atol=1e-6))
    assert changes > 0, "레지스터 상태가 전혀 변하지 않음 (write gate 고장 의심)"


results.append(run("레지스터 상태가 세그먼트 간 변경됨", check_register_updates))


def check_write_gate_init() -> None:
    ids = torch.randint(0, V, (B, L), device=device)
    with torch.no_grad():
        out = model(ids, return_trace=True)
    trace = out.trace
    assert trace is not None
    w_mean = torch.stack(trace.w_gates).mean().item()
    assert 0.05 < w_mean < 0.30, f"write gate 초기 평균={w_mean:.3f}, bias 초기화 확인 필요"
    print(f"      write gate 초기 평균: {w_mean:.4f}")


results.append(run("Write gate 초기 평균 ≈ 0.12 (0.05~0.30)", check_write_gate_init))


def check_backward() -> None:
    model.train()
    ids = torch.randint(0, V, (B, L), device=device)
    out = model(ids)
    labels = ids.clone()
    loss = F.cross_entropy(out.logits.view(-1, V), labels.view(-1))
    loss.backward()
    if cfg.model.register.s0_learnable:
        assert model.s0.grad is not None, "S_0 gradient가 None"
    model.eval()


results.append(run("Backward pass 오류 없음 + S_0 gradient 전파", check_backward))


def check_vram() -> None:
    if not torch.cuda.is_available():
        print("      GPU 없음, VRAM 체크 스킵")
        return
    torch.cuda.reset_peak_memory_stats(device)
    ids = torch.randint(0, V, (B, L), device=device)
    with torch.no_grad():
        model(ids)
    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
    assert peak_gb < 4.0, f"VRAM {peak_gb:.2f} GB > 4 GB 초과 (단일 forward, B=2)"
    print(f"      Peak VRAM: {peak_gb:.3f} GB")


results.append(run("VRAM 사용량 < 4 GB (B=2, L=512)", check_vram))


print("\n" + "─" * 50)
passed = sum(results)
total = len(results)
print(f"결과: {passed}/{total} PASS")
if passed == total:
    print("🎉 모든 체크 통과 — Phase 2로 진행하세요.")
else:
    print("⚠️  실패 항목을 수정한 후 다시 실행하세요.")
    sys.exit(1)
