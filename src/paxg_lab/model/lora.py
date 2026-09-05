"""LoRA integration for TimesFM 3.0 via Hugging Face PEFT."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model

from ..constants import DEFAULT_LORA_TARGETS, EXTENDED_LORA_TARGETS


def build_lora_timesfm3(
    model: nn.Module,
    lora_r: int = 4,
    lora_alpha: int = 8,
    lora_dropout: float = 0.1,
    extended_targets: bool = False,
    custom_targets: list[str] | None = None,
) -> PeftModel:
    """Wraps TimesFM 3.0 model with PEFT LoRA adapter.

    Args:
        model: Base TimesFM3Torch instance.
        lora_r: LoRA rank (default 4).
        lora_alpha: LoRA scaling factor (default 8, typically 2 * r).
        lora_dropout: LoRA dropout probability (default 0.1).
        extended_targets: If True, attaches to query_proj, value_proj, key_proj, out_proj.
                          If False, attaches to query_proj and value_proj.
        custom_targets: Custom list of module names to target.

    Returns:
        PeftModel wrapping TimesFM3Torch with adapter weights active.
    """
    if custom_targets is not None:
        target_modules = list(custom_targets)
    elif extended_targets:
        target_modules = list(EXTENDED_LORA_TARGETS)
    else:
        target_modules = list(DEFAULT_LORA_TARGETS)

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
    )

    peft_model = get_peft_model(model, peft_config)
    return peft_model


def verify_lora_parameters(peft_model: nn.Module) -> dict[str, Any]:
    """Verifies that only LoRA parameters have gradients enabled.

    Raises RuntimeError if any base model parameter has requires_grad=True.
    """
    total_params = 0
    trainable_params = 0
    base_trainable_params = []

    for name, param in peft_model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            # Ensure parameter belongs to LoRA adapter
            if "lora" not in name.lower():
                base_trainable_params.append(name)

    if base_trainable_params:
        raise RuntimeError(
            f"Base model parameters have requires_grad=True: {base_trainable_params[:5]}..."
        )

    if trainable_params == 0:
        raise RuntimeError("No trainable LoRA parameters found.")

    trainable_pct = 100.0 * trainable_params / total_params

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_percentage": trainable_pct,
        "is_base_frozen": len(base_trainable_params) == 0,
    }


def save_lora_adapter(
    peft_model: PeftModel,
    save_dir: str | Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Saves LoRA adapter weights and optional metadata manifest.

    Args:
        peft_model: PeftModel instance.
        save_dir: Destination directory.
        metadata: Optional dictionary with training details, horizon, timeframe, etc.

    Returns:
        Path to save directory.
    """
    out_path = Path(save_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save adapter weights (safetensors) and adapter_config.json
    peft_model.save_pretrained(str(out_path))

    # Save additional paxg manifest if provided
    if metadata is not None:
        manifest_path = out_path / "paxg_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    return out_path


def load_lora_adapter(
    base_model: nn.Module,
    adapter_dir: str | Path,
    is_trainable: bool = False,
) -> PeftModel:
    """Loads a saved LoRA adapter onto a TimesFM 3.0 base model.

    Args:
        base_model: Fresh TimesFM3Torch instance.
        adapter_dir: Directory containing adapter_model.safetensors and adapter_config.json.
        is_trainable: Whether adapter should be loaded in trainable mode.

    Returns:
        PeftModel with restored weights.
    """
    path_str = str(adapter_dir)
    peft_model = PeftModel.from_pretrained(base_model, path_str, is_trainable=is_trainable)
    return peft_model
