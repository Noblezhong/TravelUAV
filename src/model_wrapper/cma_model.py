"""
Import CMA model from AerialVLN codebase (the exact architecture used for training).
"""
import sys
import os
from pathlib import Path

# Add AerialVLN to Python path (before any AerialVLN imports)
_AIRVLN_PATH = "/HDD1/code/AirVLN_ws/AirVLN"
if _AIRVLN_PATH not in sys.path:
    sys.path.insert(0, _AIRVLN_PATH)
    sys.path.insert(0, str(Path(_AIRVLN_PATH).parent))  # project_prefix

# Save and clear sys.argv to prevent AerialVLN param.py from parsing eval args
_saved_argv = sys.argv
sys.argv = [sys.argv[0], "--run_type", "train", "--policy_type", "cma"]

import torch
from Model.cma_policy import CMAPolicy
from src.common.param import args as airvln_args

# Restore sys.argv
sys.argv = _saved_argv

# Action constants (matching AerialVLN)
FORWARD_STEP = 5.0
UP_DOWN_STEP = 2.0
LEFT_RIGHT_STEP = 5.0
TURN_ANGLE = 15.0


def load_cma_checkpoint(ckpt_path, vocab_size, device="cuda"):
    """Load CMA model from AerialVLN training checkpoint."""
    # Fix project_prefix so depth encoder can find the pretrained weights
    airvln_args.project_prefix = str(Path("/HDD1/code/AirVLN_ws"))
    airvln_args.vocab_size = vocab_size
    airvln_args.policy_type = "cma"
    airvln_args.PROGRESS_MONITOR_use = False
    airvln_args.PROGRESS_MONITOR_alpha = 1.0
    airvln_args.tokenizer_use_bert = False
    airvln_args.rgb_encoder_use_place365 = False
    airvln_args.maxInput = 300
    airvln_args.ablate_rgb = False
    airvln_args.ablate_depth = False
    airvln_args.ablate_instruction = False
    airvln_args.featdropout = 0.4
    airvln_args.action_feature = 32
    airvln_args.vlnbert = "prevalent"
    airvln_args.SEQ2SEQ_use_prev_action = False
    airvln_args.Image_Height_RGB = 224
    airvln_args.Image_Width_RGB = 224
    airvln_args.Image_Height_DEPTH = 256
    airvln_args.Image_Width_DEPTH = 256
    airvln_args.DistributedDataParallel = False
    airvln_args.trainer_gpu_device = 0

    from gym import spaces
    import numpy as np

    observation_space = spaces.Dict({
        "rgb": spaces.Box(low=0, high=255, shape=(224, 224, 3), dtype=np.uint8),
        "depth": spaces.Box(low=0, high=1, shape=(256, 256, 1), dtype=np.float32),
        "instruction": spaces.Discrete(0),
        "progress": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        "teacher_action": spaces.Box(low=0, high=100, shape=(1,)),
    })
    action_space = spaces.Discrete(8)

    model = CMAPolicy.from_config(
        observation_space=observation_space,
        action_space=action_space,
        device=torch.device(device),
    )
    model.to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)

    new_state = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state[k] = v

    model.load_state_dict(new_state, strict=True)
    model.eval()
    return model
