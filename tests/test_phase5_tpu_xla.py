# Basic-Lite TRACE
# Copyright © 2026 Miguel19x
# Todos los derechos reservados.
# SPDX-License-Identifier: LicenseRef-Basic-Lite-TRACE-Source-Available
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

"""Tests unitarios para soporte TPU v5e-8 (BF16 + torch_xla) en Fase 5."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn

from scripts.train_phase5_sft_grpo import (
    pad_batch_to_static_length,
    resolve_device_and_dtype,
    run_sft_loop,
)
from trace.reasoning.trainer import (
    gather_token_logprobs,
    grpo_policy_loss,
    sft_loss_on_completion,
)

NOTEBOOK_PATH = (
    Path(__file__).resolve().parent.parent / "notebooks" / "fase5_bitnet_2b_think_tir_tpu_v5e8_kaggle.ipynb"
)


class _FakeXLAModule:
    """Frontera mínima de `torch_xla.core.xla_model` para tests deterministas en CPU."""

    def xla_device(self) -> torch.device:
        return torch.device("cpu")

    def mark_step(self) -> None:
        return None

    def optimizer_step(self, optimizer: torch.optim.Optimizer, *, barrier: bool = True) -> None:
        del barrier
        optimizer.step()

    def save(self, data: Any, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, p)


@pytest.mark.parametrize(
    ("requested", "xla_avail", "cuda_avail", "cuda_bf16", "expected_backend", "expected_dtype"),
    [
        ("xla", True, False, False, "xla", torch.bfloat16),
        ("auto", True, True, True, "xla", torch.bfloat16),
        ("auto", False, True, False, "cuda", torch.float32),
        ("auto", False, True, True, "cuda", torch.bfloat16),
        ("auto", False, False, False, "cpu", torch.float32),
        ("cpu", True, True, True, "cpu", torch.float32),
    ],
)
def test_resolve_device_dtype_matrix(
    requested: str,
    xla_avail: bool,
    cuda_avail: bool,
    cuda_bf16: bool,
    expected_backend: str,
    expected_dtype: torch.dtype,
) -> None:
    fake_xm = _FakeXLAModule()
    device, backend, dtype = resolve_device_and_dtype(
        requested,
        xla_available=xla_avail,
        cuda_available=cuda_avail,
        cuda_bf16_supported=cuda_bf16,
        xla_device_fn=fake_xm.xla_device,
    )
    assert backend == expected_backend
    assert dtype == expected_dtype
    if expected_backend == "xla":
        assert device == torch.device("cpu")


def test_resolve_device_raises_when_xla_unavailable() -> None:
    with pytest.raises(RuntimeError, match="torch_xla"):
        resolve_device_and_dtype("xla", xla_available=False)


def test_pad_batch_to_static_length_pads_short_sequences() -> None:
    ids, mask = pad_batch_to_static_length(
        [[10, 20, 30], [11, 22]],
        [[0, 1, 1], [0, 1]],
        max_length=6,
        pad_id=99,
    )
    assert ids.shape == (2, 6)
    assert mask.shape == (2, 6)
    assert ids[0].tolist() == [10, 20, 30, 99, 99, 99]
    assert ids[1].tolist() == [11, 22, 99, 99, 99, 99]
    assert mask[0].tolist() == [0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert mask[1].tolist() == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]


def test_pad_batch_to_static_length_truncates_long_sequences() -> None:
    ids, mask = pad_batch_to_static_length(
        [[1, 2, 3, 4, 5, 6]],
        [[0, 0, 1, 1, 1, 1]],
        max_length=4,
        pad_id=0,
    )
    assert ids.shape == (1, 4)
    assert mask.shape == (1, 4)
    assert ids[0].tolist() == [1, 2, 3, 4]
    assert mask[0].tolist() == [0.0, 0.0, 1.0, 1.0]


@pytest.mark.parametrize("bad_len", [0, 1, -4])
def test_pad_batch_to_static_length_rejects_invalid_max_length(bad_len: int) -> None:
    with pytest.raises(ValueError, match="max_length"):
        pad_batch_to_static_length([[1, 2]], max_length=bad_len, pad_id=0)


def test_sft_loss_on_completion_matches_padded_static_batch() -> None:
    torch.manual_seed(17)
    vocab = 16
    raw_seq = [1, 2, 3, 4, 5]
    raw_mask = [0, 0, 1, 1, 1]
    ids_pad, mask_pad = pad_batch_to_static_length([raw_seq], [raw_mask], max_length=8, pad_id=0)
    logits_pad = torch.randn(1, 8, vocab, requires_grad=True)
    logits_raw = logits_pad[:, :5, :].detach().clone().requires_grad_(True)

    loss_pad = sft_loss_on_completion(
        logits_pad,
        ids_pad,
        torch.tensor([2], dtype=torch.long),
        pad_id=0,
        loss_mask=mask_pad,
    )
    loss_raw = sft_loss_on_completion(
        logits_raw,
        torch.tensor([raw_seq], dtype=torch.long),
        torch.tensor([2], dtype=torch.long),
        pad_id=0,
        loss_mask=torch.tensor([raw_mask], dtype=torch.float32),
    )
    assert torch.allclose(loss_pad, loss_raw, atol=1e-6)
    loss_pad.backward()
    assert logits_pad.grad is not None
    assert torch.all(logits_pad.grad[:, 4:, :] == 0.0)


def test_sft_loss_on_completion_all_masked_returns_finite_zero() -> None:
    torch.manual_seed(19)
    ids_pad, mask_pad = pad_batch_to_static_length([[1, 2, 3, 4]], [[0, 0, 0, 0]], max_length=6, pad_id=0)
    logits = torch.randn(1, 6, 12, requires_grad=True)
    loss = sft_loss_on_completion(
        logits,
        ids_pad,
        torch.tensor([6], dtype=torch.long),
        pad_id=0,
        loss_mask=mask_pad,
    )
    assert torch.isfinite(loss)
    assert float(loss.detach()) == 0.0
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.all(logits.grad == 0.0)


def test_sft_loss_on_completion_has_no_item_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(23)
    logits = torch.randn(1, 6, 12, requires_grad=True)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    pl = torch.tensor([3], dtype=torch.long)

    def _forbid_item(self: torch.Tensor) -> Any:
        raise AssertionError("sft_loss_on_completion no debe invocar .item() (rompe grafo XLA)")

    monkeypatch.setattr(torch.Tensor, "item", _forbid_item)
    loss = sft_loss_on_completion(logits, ids, pl, pad_id=0)
    loss.backward()
    assert logits.grad is not None


def test_grpo_policy_loss_defer_metrics_avoids_item_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(29)
    logits_new = torch.randn(2, 6, 12, requires_grad=True)
    logits_old = logits_new.detach().clone()
    logits_ref = torch.randn(2, 6, 12)
    ids = torch.randint(1, 12, (2, 6))
    mask = torch.ones(2, 5)
    adv = torch.tensor([1.0, -1.0])

    log_new = gather_token_logprobs(logits_new, ids)
    log_old = gather_token_logprobs(logits_old, ids)
    log_ref = gather_token_logprobs(logits_ref, ids)

    ref_loss, ref_metrics = grpo_policy_loss(log_new.detach(), log_old, log_ref, adv, mask, kl_beta=0.05)

    def _forbid_item(self: torch.Tensor) -> Any:
        raise AssertionError("grpo_policy_loss(defer_metrics=True) no debe invocar .item()")

    monkeypatch.setattr(torch.Tensor, "item", _forbid_item)
    loss, deferred = grpo_policy_loss(
        log_new,
        log_old,
        log_ref,
        adv,
        mask,
        kl_beta=0.05,
        defer_metrics=True,
    )
    loss.backward()
    monkeypatch.undo()

    assert torch.allclose(loss.detach(), ref_loss.detach(), atol=1e-6)
    for key in ("loss", "kl_mean", "ratio_mean"):
        assert isinstance(deferred[key], torch.Tensor)
        assert float(deferred[key].item()) == pytest.approx(ref_metrics[key], abs=1e-6)


def test_sft_loop_xla_requires_xm_module() -> None:
    model = nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    with pytest.raises(ValueError, match="xm"):
        run_sft_loop(
            model=model,
            optimizer=optimizer,
            trainable_params=list(model.parameters()),
            step_loss_fn=lambda _s: (model(torch.ones(1, 4)) ** 2).mean(),
            sft_steps=2,
            grad_accum=1,
            lr=0.01,
            backend="xla",
            xm=None,
        )


def test_sft_loop_xla_defers_item_sync_to_log_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(0)
    model = nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    fake_xm = _FakeXLAModule()

    orig_item = torch.Tensor.item
    item_calls = [0]

    def _counting_item(self: torch.Tensor) -> Any:
        item_calls[0] += 1
        return orig_item(self)

    monkeypatch.setattr(torch.Tensor, "item", _counting_item)

    result = run_sft_loop(
        model=model,
        optimizer=optimizer,
        trainable_params=list(model.parameters()),
        step_loss_fn=lambda _step: (model(torch.ones(1, 4)) ** 2).mean(),
        sft_steps=5,
        grad_accum=2,
        lr=0.01,
        backend="xla",
        xm=fake_xm,
        log_every=4,
        max_wall_hours=8.0,
    )
    # En 5 pasos con log_every=4 solo se materializa `.item()` en paso 4 y paso final 5
    assert item_calls[0] == 2
    assert result["steps_completed"] == 5
    assert result["stopped_early_wall_budget"] is False
    assert len(result["losses"]) == 2


def test_sft_loop_xla_reduces_loss_with_weight_updates() -> None:
    torch.manual_seed(1)
    model = nn.Linear(4, 4)
    initial_weight = model.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    fake_xm = _FakeXLAModule()

    result = run_sft_loop(
        model=model,
        optimizer=optimizer,
        trainable_params=list(model.parameters()),
        step_loss_fn=lambda _step: (model(torch.ones(1, 4)) ** 2).mean(),
        sft_steps=8,
        grad_accum=2,
        lr=0.1,
        backend="xla",
        xm=fake_xm,
        log_every=4,
        max_wall_hours=8.0,
    )
    assert result["steps_completed"] == 8
    assert not torch.allclose(model.weight.detach(), initial_weight)
    assert result["losses"][-1] < result["losses"][0]


def test_sft_loop_xla_early_guardrail_aborts_on_flat_loss() -> None:
    torch.manual_seed(3)
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    fake_xm = _FakeXLAModule()

    def flat_loss(_step: int) -> torch.Tensor:
        return (model.weight.sum() * 0.0) + 2.0

    with pytest.raises(RuntimeError, match="ABORTADO TEMPRANO EN PASO 160"):
        run_sft_loop(
            model=model,
            optimizer=optimizer,
            trainable_params=list(model.parameters()),
            step_loss_fn=flat_loss,
            sft_steps=240,
            grad_accum=8,
            lr=0.01,
            backend="xla",
            xm=fake_xm,
            log_every=80,
            max_wall_hours=8.0,
            early_guardrail=True,
        )


def test_sft_loop_xla_early_guardrail_uses_40_step_windows() -> None:
    torch.manual_seed(4)
    model = nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    fake_xm = _FakeXLAModule()

    def dropping_with_dip_at_step_80(step: int) -> torch.Tensor:
        # Ventana [0..39] = 2.0, dip puntual en paso 80 (step=79) = 0.9, ventana [120..159] = 1.0.
        # Con ventanas de 40 pasos (1.0 <= 2.0 * 0.85) NO debe abortar; comparar solo paso 80 vs 160 abortaría.
        if step < 40:
            val = 2.0
        elif step == 79:
            val = 0.9
        else:
            val = 1.0
        return (model.weight.sum() * 0.0) + val

    result = run_sft_loop(
        model=model,
        optimizer=optimizer,
        trainable_params=list(model.parameters()),
        step_loss_fn=dropping_with_dip_at_step_80,
        sft_steps=240,
        grad_accum=8,
        lr=0.01,
        backend="xla",
        xm=fake_xm,
        log_every=80,
        max_wall_hours=8.0,
        early_guardrail=True,
    )
    assert result["steps_completed"] == 240
    assert result["stopped_early_wall_budget"] is False


def test_sft_loop_stops_cleanly_on_wall_budget(tmp_path: Path) -> None:
    torch.manual_seed(2)
    model = nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    fake_xm = _FakeXLAModule()

    # Reloj determinista con start_sec propagado: start=0.0, paso 1 -> 1800s (0.5h), paso 2 -> 3700s (>1.0h)
    ticks = iter([1800.0, 3700.0, 4000.0])

    def fake_clock() -> float:
        return next(ticks)

    ckpt_path = tmp_path / "wall_guard_ckpt.pt"
    result = run_sft_loop(
        model=model,
        optimizer=optimizer,
        trainable_params=list(model.parameters()),
        step_loss_fn=lambda _step: (model(torch.ones(1, 4)) ** 2).mean(),
        sft_steps=10,
        grad_accum=1,
        lr=0.01,
        backend="xla",
        xm=fake_xm,
        log_every=5,
        max_wall_hours=1.0,
        clock_fn=fake_clock,
        start_sec=0.0,
        checkpoint_path=ckpt_path,
    )
    assert result["stopped_early_wall_budget"] is True
    assert result["steps_completed"] == 2
    assert ckpt_path.exists()
    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert set(loaded.keys()) == set(model.state_dict().keys())


def test_tpu_kaggle_notebook_valid_nbformat_v4() -> None:
    assert NOTEBOOK_PATH.exists(), f"Falta notebook TPU: {NOTEBOOK_PATH}"
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    assert isinstance(nb["cells"], list) and len(nb["cells"]) >= 4
    all_code = "\n".join(
        "".join(cell["source"]) for cell in nb["cells"] if cell.get("cell_type") == "code"
    )
    assert "PJRT_DEVICE" in all_code
    assert "--device xla" in all_code
    assert "--max-wall-hours 8.0" in all_code


def test_tpu_kaggle_notebook_avoids_kernel_tpu_lock() -> None:
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    for idx, cell in enumerate(nb["cells"]):
        if cell.get("cell_type") != "code":
            continue
        py_lines: list[str] = []
        in_magic_cont = False
        for line in "".join(cell["source"]).splitlines():
            stripped = line.lstrip()
            if in_magic_cont:
                in_magic_cont = line.rstrip().endswith("\\")
                continue
            if stripped.startswith(("!", "%")):
                indent = line[: len(line) - len(stripped)]
                py_lines.append(f"{indent}pass")
                in_magic_cont = line.rstrip().endswith("\\")
            else:
                py_lines.append(line)
        tree = ast.parse("\n".join(py_lines))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "xla_device":
                    pytest.fail(
                        f"Celda {idx} invoca xla_device() en el kernel Jupyter (provoca libtpu_lockfile crash)"
                    )
                if isinstance(func, ast.Name) and func.id == "xla_device":
                    pytest.fail(
                        f"Celda {idx} invoca xla_device() en el kernel Jupyter (provoca libtpu_lockfile crash)"
                    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gather_token_logprobs_matches_log_softmax_oracle(dtype: torch.dtype) -> None:
    torch.manual_seed(31)
    logits = torch.randn(2, 7, 32, dtype=dtype, requires_grad=True)
    target_ids = torch.randint(0, 32, (2, 7), dtype=torch.long)

    got = gather_token_logprobs(logits, target_ids)
    expected = (
        torch.nn.functional.log_softmax(logits[:, :-1, :].float(), dim=-1)
        .gather(dim=-1, index=target_ids[:, 1:].unsqueeze(-1))
        .squeeze(-1)
    )

    assert got.shape == (2, 6)
    assert torch.allclose(got, expected, atol=1e-5, rtol=1e-5)
    got.sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_sft_loop_xla_updates_lr_tensor_in_place_with_adamw() -> None:
    torch.manual_seed(37)
    model = nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.01)

    recorded_ids: list[int] = []
    recorded_lrs: list[float] = []

    class _RecordingXLAModule(_FakeXLAModule):
        def optimizer_step(self, opt: torch.optim.Optimizer, *, barrier: bool = True) -> None:
            lr_obj = opt.param_groups[0]["lr"]
            assert isinstance(lr_obj, torch.Tensor), "En XLA pg['lr'] debe ser un tensor 0-D"
            assert lr_obj.ndim == 0
            recorded_ids.append(id(lr_obj))
            recorded_lrs.append(float(lr_obj.detach().cpu().item()))
            super().optimizer_step(opt, barrier=barrier)

    result = run_sft_loop(
        model=model,
        optimizer=optimizer,
        trainable_params=list(model.parameters()),
        step_loss_fn=lambda _step: (model(torch.ones(1, 4)) ** 2).mean(),
        sft_steps=6,
        grad_accum=2,
        lr=1e-2,
        backend="xla",
        xm=_RecordingXLAModule(),
        log_every=6,
        max_wall_hours=8.0,
    )
    assert result["steps_completed"] == 6
    assert len(recorded_ids) == 3
    assert len(set(recorded_ids)) == 1, "pg['lr'] debe actualizarse in-place con fill_() sin reasignar objeto"
    assert recorded_lrs[0] > recorded_lrs[1] > recorded_lrs[2] > 0.0


def test_sft_loss_on_completion_avoids_boolean_setitem(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(41)
    logits = torch.randn(2, 6, 16, requires_grad=True)
    ids = torch.tensor([[1, 2, 3, 4, 5, 0], [1, 2, 3, 4, 0, 0]], dtype=torch.long)
    pl = torch.tensor([2, 3], dtype=torch.long)
    mask = torch.tensor([[0, 0, 1, 1, 1, 0], [0, 0, 0, 1, 0, 0]], dtype=torch.float32)

    orig_setitem = torch.Tensor.__setitem__

    def _forbid_bool_setitem(self: torch.Tensor, key: Any, value: Any) -> Any:
        if isinstance(key, torch.Tensor) and key.dtype == torch.bool:
            raise AssertionError("sft_loss_on_completion no debe usar indexado booleano in-place (usar masked_fill)")
        return orig_setitem(self, key, value)

    monkeypatch.setattr(torch.Tensor, "__setitem__", _forbid_bool_setitem)
    loss = sft_loss_on_completion(logits, ids, pl, pad_id=0, loss_mask=mask)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad is not None


@pytest.mark.parametrize("use_loss_mask", [False, True])
def test_sft_loss_on_completion_matches_oracle(use_loss_mask: bool) -> None:
    torch.manual_seed(43)
    logits = torch.randn(2, 5, 10, requires_grad=True)
    ids = torch.tensor([[1, 2, 3, 4, 0], [1, 2, 3, 5, 6]], dtype=torch.long)
    pl = torch.tensor([2, 3], dtype=torch.long)
    loss_mask = (
        torch.tensor([[0, 0, 1, 0, 0], [0, 0, 0, 1, 1]], dtype=torch.float32) if use_loss_mask else None
    )

    got = sft_loss_on_completion(logits, ids, pl, pad_id=0, loss_mask=loss_mask)

    # Oráculo explícito posición a posición
    ce_vals: list[torch.Tensor] = []
    for b in range(2):
        for t in range(4):
            target_pos = t + 1
            if target_pos < int(pl[b]):
                continue
            if loss_mask is not None:
                if loss_mask[b, target_pos] <= 0:
                    continue
            elif int(ids[b, target_pos]) == 0:
                continue
            ce_vals.append(torch.nn.functional.cross_entropy(logits[b, t], ids[b, target_pos]))
    expected = torch.stack(ce_vals).mean()
    assert torch.allclose(got, expected, atol=1e-6)


def test_tpu_kaggle_notebook_pins_torch_constraint() -> None:
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_cells = ["".join(cell["source"]) for cell in nb["cells"] if cell.get("cell_type") == "code"]
    assert code_cells, "El notebook TPU debe tener celdas de código"
    cell1 = code_cells[0]
    assert "/tmp/torch_constraint.txt" in cell1
    assert "-c /tmp/torch_constraint.txt" in cell1
    assert "pip" in cell1

