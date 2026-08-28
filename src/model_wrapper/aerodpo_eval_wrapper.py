import time

import cv2
import numpy as np
import torch
from peft import PeftModel
from PIL import Image
from qwen_vl_utils import process_vision_info
from scipy.spatial.transform import Rotation
from transformers import AutoModelForImageTextToText, AutoProcessor

from src.vlnce_src.aerodpo_eval_contract import parse_aerodpo_action_text
from src.model_wrapper.base_model import BaseModelWrapper


class AeroDPOEvalModelWrapper(BaseModelWrapper):
    """AeroDPO inference wrapper with its original observation and prompt path."""

    resolution = 384

    def __init__(self, model_args):
        super().__init__()
        if not model_args.base_model_path:
            raise ValueError("--base_model_path is required for AeroDPO evaluation")
        if not model_args.model_path:
            raise ValueError("--model_path must point to the AeroDPO LoRA adapter")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(
            model_args.base_model_path, trust_remote_code=True
        )
        self.processor.image_processor.max_pixels = self.resolution * self.resolution * 2
        self.processor.tokenizer.padding_side = "left"
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id

        base_model = AutoModelForImageTextToText.from_pretrained(
            model_args.base_model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        self.model = PeftModel.from_pretrained(base_model, model_args.model_path)
        self.model = self.model.merge_and_unload()
        self.model.to(self.device)
        self.model.eval()

    def _sync_cuda(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def eval(self):
        self.model.eval()

    @staticmethod
    def _semantic_direction(current_state, target_position):
        position = np.asarray(current_state["position"], dtype=np.float64)
        orientation = current_state["orientation"]
        if isinstance(orientation, dict):
            quaternion = [
                orientation.get("x", 0.0),
                orientation.get("y", 0.0),
                orientation.get("z", 0.0),
                orientation.get("w", 1.0),
            ]
        else:
            quaternion = orientation
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
            return instruction.split("degrees from you.", 1)[1].split(
                " Please control", 1
            )[0].strip()
        except (AttributeError, IndexError):
            return str(instruction).strip()

    def prepare_inputs(self, episodes, target_positions, instructions):
        messages_batch = []
        prompts = []
        for index, episode in enumerate(episodes):
            latest = episode[-1]
            front = Image.fromarray(cv2.cvtColor(latest["rgb"][0], cv2.COLOR_BGR2RGB))
            down = Image.fromarray(cv2.cvtColor(latest["rgb"][1], cv2.COLOR_BGR2RGB))
            target_size = (self.resolution, self.resolution)
            front = front.resize(target_size, resample=Image.Resampling.BICUBIC)
            down = down.resize(target_size, resample=Image.Resampling.BICUBIC)
            mosaic = Image.new("RGB", (self.resolution, self.resolution * 2), (0, 0, 0))
            mosaic.paste(front, (0, 0))
            mosaic.paste(down, (0, self.resolution))

            direction = self._semantic_direction(
                latest["sensors"]["state"], target_positions[index]
            )
            prompt = (
                f"Fly {direction}and find the target. "
                f"{self._object_description(instructions[index])}\nAction: "
            )
            prompts.append(prompt)
            messages_batch.append([{
                "role": "user",
                "content": [
                    {"type": "image", "image": mosaic},
                    {"type": "text", "text": prompt},
                ],
            }])

        texts = [
            self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for messages in messages_batch
        ]
        image_inputs, video_inputs = process_vision_info(messages_batch)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding="longest",
            return_tensors="pt",
        )
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
        return {key: value.to(self.device) for key, value in inputs.items()}, prompts

    def run_profiled(self, inputs):
        with torch.inference_mode():
            self._sync_cuda()
            start = time.perf_counter()
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                use_cache=True,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )
            self._sync_cuda()
            model_inference_latency_ms = (time.perf_counter() - start) * 1000.0

        input_length = inputs["input_ids"].shape[1]
        actions = []
        stops = []
        texts = []
        for output_ids in generated_ids:
            text = self.processor.decode(
                output_ids[input_length:], skip_special_tokens=True
            ).strip()
            action, should_stop = parse_aerodpo_action_text(text)
            actions.append(action)
            stops.append(should_stop)
            texts.append(text)
        return actions, stops, {
            "model_inference_latency_ms": model_inference_latency_ms,
            "model_output_text": texts,
        }
