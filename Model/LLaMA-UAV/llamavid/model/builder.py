#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llamavid.model import *
from llamavid.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from peft import PeftModel


LLAMAVID_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _resolve_config_path(path_value, extra_roots=None):
    if not isinstance(path_value, str) or not path_value:
        return path_value
    if os.path.isabs(path_value) and os.path.exists(path_value):
        return path_value

    candidate_roots = [LLAMAVID_ROOT]
    if extra_roots:
        candidate_roots.extend(root for root in extra_roots if root)

    for root in candidate_roots:
        candidate = os.path.abspath(os.path.join(root, path_value))
        if os.path.exists(candidate):
            return candidate
    return path_value


def _normalize_multimodal_config_paths(config, model_path=None, model_base=None):
    extra_roots = []
    if model_path:
        extra_roots.append(model_path)
        extra_roots.append(os.path.dirname(model_path))
    if model_base:
        extra_roots.append(model_base)
        extra_roots.append(os.path.dirname(model_base))

    for attr_name in ("mm_vision_tower", "vision_tower", "image_processor", "pretrain_qformer"):
        if hasattr(config, attr_name):
            setattr(config, attr_name, _resolve_config_path(getattr(config, attr_name), extra_roots=extra_roots))


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda"):
    kwargs = {"device_map": device_map}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if 'vid' or 'uav' in model_name.lower():
        # Load LLaMA-VID model
        if model_base is not None:
            # this may be mm projector only
            from peft import PeftModel
            print('Loading LLaVA from base model...')
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            cfg_pretrained = AutoConfig.from_pretrained(model_path)
            _normalize_multimodal_config_paths(cfg_pretrained, model_path=model_path, model_base=model_base)
            model = LlavaLlamaAttForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            model.to(torch.bfloat16)
            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            model = LlavaLlamaAttForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            _normalize_multimodal_config_paths(model.config, model_path=model_path, model_base=model_base)

    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if 'vid' or 'uav' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower.to(device=device, dtype=torch.bfloat16)
        image_processor = vision_tower.image_processor
        model.config.model_path = model_path
        model.get_model().initialize_attention_modules(model.config, for_eval=True)

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
