from __future__ import annotations

import shlex
from pathlib import Path
from typing import Mapping

import torch


LORA_SUFFIXES = (".lora_A.weight", ".lora_B.weight")
PROJECTOR_PREFIX = "base_model.model.model.visual.merger."


LANGUAGE_QUANT_TYPES = {
    "Q8_0": "q8_0",
    "Q4_K_M": "q4_k_m",
}


def language_quantized_filename(quant_type: str) -> str:
    """Return the canonical filename for a language-only GGUF quantization."""
    try:
        suffix = LANGUAGE_QUANT_TYPES[quant_type]
    except KeyError as error:
        supported = ", ".join(LANGUAGE_QUANT_TYPES)
        raise ValueError(
            f"unsupported language quantization {quant_type!r}; choose {supported}"
        ) from error
    return f"language-base-{suffix}.gguf"


def language_quantize_command(
    quantizer: Path, source_bf16: Path, output: Path, quant_type: str
):
    """Build an argument vector for direct BF16 language quantization."""
    language_quantized_filename(quant_type)
    return [str(quantizer), str(source_bf16), str(output), quant_type]


def partition_adapter_tensors(tensors: Mapping[str, torch.Tensor]):
    """Split the AeroDPO adapter into language LoRA and visual merger tensors."""
    lora = {}
    projector = {}
    for name, tensor in tensors.items():
        if name.endswith(LORA_SUFFIXES):
            lora[name] = tensor
            continue
        if name.startswith(PROJECTOR_PREFIX):
            projector[name.removeprefix("base_model.model.model.")] = tensor
            continue
        raise ValueError(f"unexpected adapter tensor: {name}")

    if not lora:
        raise ValueError("adapter contains no language LoRA tensors")
    if not projector:
        raise ValueError("adapter contains no visual.merger tensors")
    return lora, projector

def projector_overrides_for_base(projector: Mapping[str, torch.Tensor]):
    """Map canonical AeroDPO projector names to Qwen3-VL checkpoint names."""
    return {f"model.{name}": tensor for name, tensor in projector.items()}


def language_only_adapter_config(config: Mapping):
    """Return a PEFT config which contains only the language LoRA adapter."""
    result = dict(config)
    result["modules_to_save"] = []
    return result


def conversion_commands(
    llama_cpp_dir: Path,
    base_dir: Path,
    language_adapter_dir: Path,
    staged_dir: Path,
    output_dir: Path,
):
    """Return the reproducible llama.cpp conversion commands for AeroDPO."""
    def quote(path: Path) -> str:
        return shlex.quote(str(path))

    converter = llama_cpp_dir / "convert_hf_to_gguf.py"
    lora_converter = llama_cpp_dir / "convert_lora_to_gguf.py"
    quantizer = llama_cpp_dir / "build/bin/llama-quantize"
    text_bf16 = output_dir / "language-base-bf16.gguf"
    text_q8 = output_dir / "language-base-q8_0.gguf"
    lora_f16 = output_dir / "aerodpo-language-lora-f16.gguf"
    mmproj_bf16 = output_dir / "aerodpo-mmproj-bf16.gguf"
    return {
        "text_bf16": (
            f"python {quote(converter)} {quote(base_dir)} --outfile {quote(text_bf16)} "
            "--outtype bf16"
        ),
        "text_q8": f"{quote(quantizer)} {quote(text_bf16)} {quote(text_q8)} Q8_0",
        "lora_f16": (
            f"python {quote(lora_converter)} --base {quote(base_dir)} --outfile {quote(lora_f16)} "
            f"--outtype f16 {quote(language_adapter_dir)}"
        ),
        "mmproj_bf16": (
            f"python {quote(converter)} {quote(staged_dir)} --mmproj "
            f"--outfile {quote(mmproj_bf16)} --outtype bf16"
        ),
    }
