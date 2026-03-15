#!/usr/bin/env python3
"""
Converts all torch.nn.Linear layers in a HuggingFace model to
INT8 symmetric per-output-channel quantisation.

Usage: python -m vkflame.tools.quantise <model_path> <output_path>
"""
import sys
import json
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM


def quantise_model(model_path: str, output_path: str) -> None:
    print(f"[quantise] loading {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    n_quantised = 0
    original_bytes = 0
    quantised_bytes = 0

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue

        w: torch.Tensor = module.weight.detach().float()
        original_bytes += w.numel() * 2  # fp16 original

        # Symmetric per-output-channel: scale = max(|w|) / 127
        scale: torch.Tensor = w.abs().max(dim=1).values / 127.0
        scale = scale.clamp(min=1e-8)

        w_int8 = (w / scale.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)

        module.register_buffer("weight_int8",  w_int8)
        module.register_buffer("weight_scale", scale.half())
        # Clear the original weight to free memory
        module.weight = None  # type: ignore[assignment]

        quantised_bytes += w_int8.numel()
        n_quantised += 1

        if n_quantised % 20 == 0:
            print(f"  [{n_quantised}] {name}")

    saved_gb = (original_bytes - quantised_bytes) / 1e9
    print(f"[quantise] {n_quantised} layers quantised, {saved_gb:.1f} GB saved")

    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)

    (out / "vkflame_quantised.json").write_text(
        json.dumps({"quantised": True, "n_layers": n_quantised}, indent=2)
    )
    print(f"[quantise] saved to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m vkflame.tools.quantise <model_path> <output_path>")
        sys.exit(1)
    quantise_model(sys.argv[1], sys.argv[2])
