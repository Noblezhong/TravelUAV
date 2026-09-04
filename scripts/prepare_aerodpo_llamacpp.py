#!/usr/bin/env python3
"""Prepare a Q8_0 language-only llama.cpp deployment of AeroDPO.

The source Qwen3-VL base and AeroDPO adapter are never edited.  The adapter is
split into a PEFT language-only LoRA folder and a patched, temporary HF model
used exclusively to export the BF16 multimodal projector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForImageTextToText, AutoProcessor

from src.model_wrapper.aerodpo_llamacpp_artifacts import (
    conversion_commands,
    language_only_adapter_config,
    partition_adapter_tensors,
    projector_overrides_for_base,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-model-dir",
        type=Path,
        default=Path("/HDD1/code/AeroDPO/model_zoo/AeroDPO/AeroVLA-2B-384-Merged"),
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=Path("/HDD1/code/AeroDPO/model_zoo/AeroDPO/AeroDPO-2B-384-LoRA"),
    )
    parser.add_argument(
        "--llama-cpp-dir", type=Path, default=Path("/HDD1/code/llama.cpp")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/HDD1/code/AeroDPO/model_zoo/AeroDPO-llamacpp-q8_0"),
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=Path("/HDD1/code/AeroDPO/model_zoo/AeroDPO-llamacpp-mmproj-staging"),
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="write the split adapter and staged projector checkpoint without GGUF conversion",
    )
    return parser.parse_args()


def require_empty_or_missing(path: Path):
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty path: {path}")


def write_language_adapter(adapter_dir: Path, output_dir: Path, base_model_dir: Path):
    source_weights = adapter_dir / "adapter_model.safetensors"
    source_config = adapter_dir / "adapter_config.json"
    tensors = load_file(str(source_weights), device="cpu")
    lora, projector = partition_adapter_tensors(tensors)
    if len(lora) != 392 or len(projector) != 6:
        raise ValueError(
            f"unexpected AeroDPO adapter layout: {len(lora)} LoRA, {len(projector)} projector tensors"
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    save_file(lora, str(output_dir / "adapter_model.safetensors"))
    config = language_only_adapter_config(json.loads(source_config.read_text()))
    config["base_model_name_or_path"] = str(base_model_dir)
    (output_dir / "adapter_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return projector


def stage_projector_checkpoint(base_model_dir: Path, projector, staging_dir: Path):
    require_empty_or_missing(staging_dir)
    model = AutoModelForImageTextToText.from_pretrained(
        base_model_dir,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    missing, unexpected = model.load_state_dict(
        projector_overrides_for_base(projector), strict=False
    )
    if unexpected:
        raise RuntimeError(f"projector keys rejected by Qwen3-VL: {unexpected}")
    if len(missing) <= len(projector):
        raise RuntimeError("projector update did not load into the base model")
    model.save_pretrained(staging_dir, safe_serialization=True, max_shard_size="10GB")
    AutoProcessor.from_pretrained(base_model_dir, trust_remote_code=True).save_pretrained(staging_dir)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run(command: str):
    print(f"+ {command}", flush=True)
    subprocess.run(command, shell=True, check=True)


def main():
    args = parse_args()
    for required in (
        args.base_model_dir / "config.json",
        args.base_model_dir / "model.safetensors",
        args.adapter_dir / "adapter_config.json",
        args.adapter_dir / "adapter_model.safetensors",
        args.llama_cpp_dir / "convert_hf_to_gguf.py",
        args.llama_cpp_dir / "convert_lora_to_gguf.py",
        args.llama_cpp_dir / "build/bin/llama-quantize",
    ):
        if not required.exists():
            raise FileNotFoundError(required)
    require_empty_or_missing(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    language_adapter_dir = args.output_dir / "language_adapter"
    projector = write_language_adapter(
        args.adapter_dir, language_adapter_dir, args.base_model_dir
    )
    stage_projector_checkpoint(args.base_model_dir, projector, args.staging_dir)
    commands = conversion_commands(
        args.llama_cpp_dir,
        args.base_model_dir,
        language_adapter_dir,
        args.staging_dir,
        args.output_dir,
    )
    manifest = {
        "base_model_dir": str(args.base_model_dir),
        "adapter_dir": str(args.adapter_dir),
        "llama_cpp_dir": str(args.llama_cpp_dir),
        "base_model_sha256": sha256(args.base_model_dir / "model.safetensors"),
        "adapter_sha256": sha256(args.adapter_dir / "adapter_model.safetensors"),
        "lora_tensor_count": 392,
        "projector_tensor_count": 6,
        "commands": commands,
    }
    (args.output_dir / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.prepare_only:
        return
    for name in ("text_bf16", "text_q8", "lora_f16", "mmproj_bf16"):
        run(commands[name])


if __name__ == "__main__":
    main()
