import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.model_wrapper.utils.travel_util import (
    inputs_to_batch,
    load_model,
    prepare_data_to_inputs,
)


class EdgeVLMWrapper:
    def __init__(self, model_args, data_args):
        self.tokenizer, self.model, self.image_processor = load_model(model_args)
        self.model.to(torch.bfloat16)
        self.model_args = model_args
        self.data_args = data_args

    def eval(self):
        self.model.eval()

    def _sync_cuda(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def prepare_inputs(
        self,
        episodes: List[List[Dict[str, Any]]],
        target_positions,
        assist_notices: Optional[List[Optional[str]]] = None,
    ):
        inputs = []
        rot_to_targets = []
        for i in range(len(episodes)):
            input_item, rot_to_target = prepare_data_to_inputs(
                episodes=episodes[i],
                tokenizer=self.tokenizer,
                image_processor=self.image_processor,
                data_args=self.data_args,
                target_point=target_positions[i],
                assist_notice=assist_notices[i] if assist_notices is not None else None,
            )
            inputs.append(input_item)
            rot_to_targets.append(rot_to_target)

        batch = inputs_to_batch(tokenizer=self.tokenizer, instances=inputs)
        inputs_device = {
            k: v.to(self.model.device)
            for k, v in batch.items()
            if "prompts" not in k and "images" not in k and "historys" not in k
        }
        inputs_device["prompts"] = [item for item in batch["prompts"]]
        inputs_device["images"] = [item.to(self.model.device) for item in batch["images"]]
        inputs_device["historys"] = [
            item.to(device=self.model.device, dtype=self.model.dtype) for item in batch["historys"]
        ]
        inputs_device["orientations"] = inputs_device["orientations"].to(dtype=self.model.dtype)
        inputs_device["return_waypoints"] = True
        inputs_device["use_cache"] = False
        return inputs_device, rot_to_targets

    def run_llm_model(self, inputs):
        with torch.inference_mode():
            self._sync_cuda()
            start = time.perf_counter()
            waypoints_llm = self.model(**inputs).detach().cpu().to(dtype=torch.float32).numpy()
            self._sync_cuda()
            latency_ms = (time.perf_counter() - start) * 1000.0

        waypoints_llm_new = []
        for waypoint in waypoints_llm:
            waypoint_new = waypoint[:3] / (1e-6 + np.linalg.norm(waypoint[:3])) * waypoint[3]
            waypoints_llm_new.append(waypoint_new)
        return np.array(waypoints_llm_new), latency_ms

    def run_coarse(self, episodes, target_positions, assist_notices=None):
        inputs, _ = self.prepare_inputs(episodes, target_positions, assist_notices)
        coarse_local, llm_latency_ms = self.run_llm_model(inputs)
        return coarse_local, llm_latency_ms
