"""PyTest suite for P0 verification: LoRA integration, horizons, and save/load."""

from pathlib import Path
import torch
import pytest

from paxg_lab.constants import (
    MODEL_REPO,
    MODEL_REVISION,
    TIMEFRAME_HORIZONS,
    QUANTILES,
    MEDIAN_QUANTILE_INDEX,
)
from paxg_lab.model.lora import (
    build_lora_timesfm3,
    verify_lora_parameters,
    save_lora_adapter,
    load_lora_adapter,
)
from paxg_lab.model.loss import combined_forecast_loss
from timesfm3 import TimesFM3Torch


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="function")
def base_model(device):
    model = TimesFM3Torch.from_pretrained(MODEL_REPO, revision=MODEL_REVISION)
    model.to(device)
    model.eval()
    return model


def test_horizons_constant():
    assert TIMEFRAME_HORIZONS["1h"] == 24
    assert TIMEFRAME_HORIZONS["4h"] == 6


def test_base_inference_horizons(base_model, device):
    b, v, ctx_len = 1, 1, 128
    ctx = torch.randn(b, v, ctx_len, device=device)

    # 1h forecast
    out_1h = base_model.decode(target=ctx, horizon=TIMEFRAME_HORIZONS["1h"])
    assert out_1h.shape == (b, v, 24, 9)
    assert torch.isfinite(out_1h).all().item()

    # 4h forecast
    out_4h = base_model.decode(target=ctx, horizon=TIMEFRAME_HORIZONS["4h"])
    assert out_4h.shape == (b, v, 6, 9)
    assert torch.isfinite(out_4h).all().item()


def test_lora_initialization_parity(base_model, device):
    lora_model = build_lora_timesfm3(base_model, lora_r=4, lora_alpha=8)
    lora_model.to(device)
    lora_model.eval()

    info = verify_lora_parameters(lora_model)
    assert info["is_base_frozen"]
    assert info["trainable_params"] > 0

    ctx = torch.randn(1, 1, 128, device=device)
    with torch.no_grad():
        base_pred = base_model.decode(target=ctx, horizon=24)
        lora_pred = lora_model.base_model.model.decode(target=ctx, horizon=24)

    max_diff = torch.max(torch.abs(base_pred - lora_pred)).item()
    assert max_diff < 1e-5


def test_lora_train_step_and_base_invariance(base_model, device):
    lora_model = build_lora_timesfm3(base_model, lora_r=4, lora_alpha=8)
    lora_model.to(device)
    lora_model.train()

    base_weights_snapshot = {
        name: param.clone().detach()
        for name, param in lora_model.named_parameters()
        if "lora" not in name.lower()
    }

    ctx = 100.0 + torch.randn(1, 1, 128, device=device)
    tgt = 100.0 + torch.randn(1, 1, 24, device=device)

    opt = torch.optim.AdamW(lora_model.parameters(), lr=1e-4)
    opt.zero_grad()

    pred = lora_model.base_model.model.forward_decode(target=ctx, horizon=24)
    loss, _ = combined_forecast_loss(pred, tgt, last_context_price=ctx[:, :, -1:])
    loss.backward()

    # Assert LoRA has grad, base does not
    for name, param in lora_model.named_parameters():
        if "lora" in name.lower() and param.requires_grad:
            assert param.grad is not None
            assert torch.isfinite(param.grad).all().item()
        else:
            assert param.grad is None

    opt.step()

    # Assert base weights 100% invariant
    for name, param in lora_model.named_parameters():
        if "lora" not in name.lower():
            assert torch.equal(param.data, base_weights_snapshot[name])


def test_save_and_reload_adapter(base_model, device, tmp_path):
    lora_model = build_lora_timesfm3(base_model, lora_r=4, lora_alpha=8)
    lora_model.to(device)
    lora_model.eval()

    adapter_dir = tmp_path / "adapter_test"
    save_lora_adapter(lora_model, adapter_dir, metadata={"timeframe": "1h", "horizon": 24})

    assert (adapter_dir / "adapter_model.safetensors").exists()
    assert (adapter_dir / "adapter_config.json").exists()

    clean_base = TimesFM3Torch.from_pretrained(MODEL_REPO, revision=MODEL_REVISION).to(device)
    clean_base.eval()
    reloaded = load_lora_adapter(clean_base, adapter_dir).to(device)
    reloaded.eval()

    ctx = torch.randn(1, 1, 128, device=device)
    with torch.no_grad():
        out_orig = lora_model.base_model.model.decode(target=ctx, horizon=24)
        out_reloaded = reloaded.base_model.model.decode(target=ctx, horizon=24)

    assert torch.allclose(out_orig, out_reloaded, atol=1e-5, rtol=1e-5)
