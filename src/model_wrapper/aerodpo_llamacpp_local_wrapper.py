import ctypes
import os
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from src.model_wrapper.base_model import BaseModelWrapper
from src.vlnce_src.aerodpo_eval_contract import parse_aerodpo_action_text


class _NativeProfile(ctypes.Structure):
    _fields_ = [
        ("vision_encode_ms", ctypes.c_double),
        ("prefill_ms", ctypes.c_double),
        ("decode_ms", ctypes.c_double),
        ("total_ms", ctypes.c_double),
        ("generated_tokens", ctypes.c_uint32),
    ]


class AeroDPOLlamaCppLocalBridge:
    """ctypes binding for the process-local llama.cpp multimodal bridge."""

    def __init__(self, text_model, lora_model, mmproj_model, library_path, context=2048):
        library_path = Path(library_path)
        if not library_path.is_file():
            raise FileNotFoundError(f"AeroDPO local bridge not found: {library_path}")
        for label, path in {
            "text GGUF": text_model,
            "LoRA GGUF": lora_model,
            "mmproj GGUF": mmproj_model,
        }.items():
            if not Path(path).is_file():
                raise FileNotFoundError(f"AeroDPO {label} not found: {path}")
        self.lib = ctypes.CDLL(str(library_path))
        self.lib.aerodpo_llamacpp_local_create.argtypes = [
            ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32,
            ctypes.c_char_p, ctypes.c_size_t,
        ]
        self.lib.aerodpo_llamacpp_local_create.restype = ctypes.c_void_p
        self.lib.aerodpo_llamacpp_local_generate.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint32,
            ctypes.c_uint32, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_char_p,
            ctypes.c_size_t, ctypes.POINTER(_NativeProfile), ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self.lib.aerodpo_llamacpp_local_generate.restype = ctypes.c_int
        self.lib.aerodpo_llamacpp_local_free.argtypes = [ctypes.c_void_p]
        self.lib.aerodpo_llamacpp_local_free.restype = None
        error = ctypes.create_string_buffer(4096)
        self.handle = self.lib.aerodpo_llamacpp_local_create(
            os.fsencode(text_model), os.fsencode(lora_model), os.fsencode(mmproj_model),
            int(context), error, ctypes.sizeof(error),
        )
        if not self.handle:
            raise RuntimeError(error.value.decode("utf-8", errors="replace"))

    def close(self):
        if getattr(self, "handle", None):
            self.lib.aerodpo_llamacpp_local_free(self.handle)
            self.handle = None

    def __del__(self):
        self.close()

    def generate(self, rgb, prompt, max_tokens=20):
        rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        if rgb.shape != (768, 384, 3):
            raise ValueError(f"expected RGB [768, 384, 3], got {rgb.shape}")
        output = ctypes.create_string_buffer(256)
        error = ctypes.create_string_buffer(4096)
        profile = _NativeProfile()
        status = self.lib.aerodpo_llamacpp_local_generate(
            self.handle,
            rgb.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            384, 768, prompt.encode("utf-8"), int(max_tokens), output,
            ctypes.sizeof(output), ctypes.byref(profile), error, ctypes.sizeof(error),
        )
        if status != 0:
            raise RuntimeError(error.value.decode("utf-8", errors="replace"))
        return output.value.decode("utf-8", errors="replace").strip(), {
            "native_bridge_latency_ms": float(profile.total_ms),
            "native_vision_encode_latency_ms": float(profile.vision_encode_ms),
            "native_prefill_latency_ms": float(profile.prefill_ms),
            "native_decode_latency_ms": float(profile.decode_ms),
            "native_generated_tokens": int(profile.generated_tokens),
        }


class AeroDPOLlamaCppLocalEvalModelWrapper(BaseModelWrapper):
    resolution = 384

    def __init__(self, model_args):
        super().__init__()
        required = ("llama_text_model_path", "model_path", "llama_mmproj_path", "llama_local_bridge_path")
        missing = [name for name in required if not getattr(model_args, name, None)]
        if missing:
            raise ValueError(f"llamacpp_local requires: {', '.join(missing)}")
        self.bridge = AeroDPOLlamaCppLocalBridge(
            model_args.llama_text_model_path,
            model_args.model_path,
            model_args.llama_mmproj_path,
            model_args.llama_local_bridge_path,
            getattr(model_args, "llama_local_context", 2048),
        )

    def eval(self):
        return None

    @staticmethod
    def _semantic_direction(current_state, target_position):
        position = np.asarray(current_state["position"], dtype=np.float64)
        orientation = current_state["orientation"]
        quaternion = [orientation.get(key, default) for key, default in (("x", 0.0), ("y", 0.0), ("z", 0.0), ("w", 1.0))] if isinstance(orientation, dict) else orientation
        world_vector = np.asarray(target_position, dtype=np.float64) - position
        if float(np.linalg.norm(world_vector[:2])) < 0.01:
            return ""
        body_vector = Rotation.from_quat(quaternion).inv().apply(world_vector)
        angle = float(np.degrees(np.arctan2(body_vector[1], body_vector[0])))
        if -15 <= angle <= 15:
            return "straight ahead "
        if 15 < angle <= 60:
            return "forward-right "
        if 60 < angle <= 120:
            return "to your right "
        if 120 < angle <= 180:
            return "to your right rear "
        if -60 <= angle < -15:
            return "forward-left "
        if -120 <= angle < -60:
            return "to your left "
        return "to your left rear "

    @staticmethod
    def _object_description(instruction):
        try:
            return instruction.split("degrees from you.", 1)[1].split(" Please control", 1)[0].strip()
        except (AttributeError, IndexError):
            return str(instruction).strip()

    def _make_mosaic(self, latest):
        front = Image.fromarray(cv2.cvtColor(latest["rgb"][0], cv2.COLOR_BGR2RGB))
        down = Image.fromarray(cv2.cvtColor(latest["rgb"][1], cv2.COLOR_BGR2RGB))
        target = (self.resolution, self.resolution)
        mosaic = Image.new("RGB", (self.resolution, self.resolution * 2), (0, 0, 0))
        mosaic.paste(front.resize(target, Image.Resampling.BICUBIC), (0, 0))
        mosaic.paste(down.resize(target, Image.Resampling.BICUBIC), (0, self.resolution))
        return mosaic

    def prepare_inputs(self, episodes, target_positions, instructions):
        prepared, prompts = [], []
        for index, episode in enumerate(episodes):
            latest = episode[-1]
            prompt = f"Fly {self._semantic_direction(latest['sensors']['state'], target_positions[index])}and find the target. {self._object_description(instructions[index])}\nAction: "
            prepared.append({"image": self._make_mosaic(latest), "prompt": prompt})
            prompts.append(prompt)
        return prepared, prompts

    def run_profiled(self, inputs):
        if len(inputs) != 1:
            raise ValueError(f"AeroDPO local backend requires batch size 1, got {len(inputs)}")
        item = inputs[0]
        start = time.perf_counter()
        rgb = np.ascontiguousarray(np.asarray(item["image"], dtype=np.uint8))
        text, native = self.bridge.generate(rgb, item["prompt"], max_tokens=20)
        action, should_stop = parse_aerodpo_action_text(text)
        total_ms = (time.perf_counter() - start) * 1000.0
        return [action], [should_stop], {
            "model_inference_latency_ms": float(total_ms),
            "image_text_to_action_latency_ms": float(total_ms),
            "model_output_text": [text],
            **native,
        }
