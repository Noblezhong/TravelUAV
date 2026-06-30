"""
CMA model wrapper for TravelUAV closed-loop eval.
Implements the same interface as ProfileTravelModelWrapper.
"""
import time
import re
import string
import os
import numpy as np
from typing import Any, Dict, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from scipy.spatial.transform import Rotation as R

from src.model_wrapper.base_model import BaseModelWrapper
from src.model_wrapper.cma_model import (
    CMAPolicy,
    load_cma_checkpoint,
    FORWARD_STEP,
    UP_DOWN_STEP,
    LEFT_RIGHT_STEP,
    TURN_ANGLE,
)

# ============================================================
# Tokenizer (matching AerialVLN)
# ============================================================
SENTENCE_SPLIT_REGEX = re.compile(r"(\W+)")


def _split_sentence(sentence):
    toks = []
    for word in [s.strip().lower() for s in SENTENCE_SPLIT_REGEX.split(sentence.strip()) if len(s.strip()) > 0]:
        if all(c in string.punctuation for c in word) and not all(c in "." for c in word):
            toks += list(word)
        else:
            toks.append(word)
    return toks


def _load_vocab(vocab_path):
    """Load vocab file (one word per line)."""
    word2idx = defaultdict(lambda: 1)  # <UNK> = 1
    idx2word = {}
    with open(vocab_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            word = line.strip()
            word2idx[word] = i
            idx2word[i] = word
    return word2idx, idx2word


def tokenize(text, word2idx, max_length=300):
    encoding = [word2idx[w] for w in _split_sentence(text)]
    if not encoding:
        encoding = [0]
    if len(encoding) < max_length:
        encoding += [0] * (max_length - len(encoding))
    return np.array(encoding[:max_length], dtype=np.int64)


# ============================================================
# Image pre-processing
# ============================================================
rgb_resize = T.Compose([T.ToPILImage(), T.Resize(224), T.CenterCrop(224)])
depth_resize = T.Compose([T.ToPILImage(), T.Resize((256, 256))])

# Feature extractor matching convert_traveluav.py's RGBFeatureExtractor:
# ResNet50 with ImageNet Normalize + AdaptiveAvgPool2d(4,4) + forward hook
# Training used pre-extracted rgb_features [2048, 4, 4], not raw images.
_rgb_feat_extractor = None


def _get_rgb_feature_extractor(device):
    """Lazy init ResNet50 feature extractor matching training pipeline."""
    global _rgb_feat_extractor
    if _rgb_feat_extractor is None:
        cnn = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        cnn.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        cnn.fc = nn.Sequential()
        cnn.eval()
        cnn.to(device)
        for param in cnn.parameters():
            param.requires_grad = False
        _rgb_feat_extractor = cnn
    return _rgb_feat_extractor


_rgb_feat_hook_output = None


def _rgb_feat_hook(module, inp, out):
    global _rgb_feat_hook_output
    _rgb_feat_hook_output = out


# ImageNet normalization matching convert_traveluav.py
_rgb_normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
_rgb_to_tensor = T.ToTensor()


@torch.no_grad()
def extract_rgb_features(rgb_np, device):
    """Extract ResNet50 [2048, 4, 4] features with ImageNet Normalize.
    Matches convert_traveluav.py's RGBFeatureExtractor pipeline exactly."""
    global _rgb_feat_hook_output
    cnn = _get_rgb_feature_extractor(device)
    # Convert uint8 [H,W,3] → tensor [3,H,W] float [0,1] → Normalize
    img_tensor = _rgb_normalize(_rgb_to_tensor(rgb_np)).unsqueeze(0).to(device)
    _rgb_feat_hook_output = None
    hook_handle = cnn.avgpool.register_forward_hook(_rgb_feat_hook)
    cnn(img_tensor)
    hook_handle.remove()
    return _rgb_feat_hook_output  # [1, 2048, 4, 4]


# ============================================================
# Action naming
# ============================================================
ACTION_NAMES = {
    0: 'STOP', 1: 'FORWARD', 2: 'TURN_LEFT', 3: 'TURN_RIGHT',
    4: 'GO_UP', 5: 'GO_DOWN', 6: 'MOVE_LEFT', 7: 'MOVE_RIGHT',
}


# ============================================================
# Action → Waypoint conversion
# ============================================================
def action_to_waypoint_offset(action: int):
    """
    Convert discrete AerialVLN action to relative position offset (x, y, z)
    and yaw change (degrees), in the drone's local frame (X=forward, Y=right, Z=down/NED).
    """
    if action == 0:   # STOP
        return (0.0, 0.0, 0.0, 0.0)
    elif action == 1: # FORWARD
        return (FORWARD_STEP, 0.0, 0.0, 0.0)
    elif action == 2: # TURN_LEFT
        return (0.0, 0.0, 0.0, TURN_ANGLE)
    elif action == 3: # TURN_RIGHT
        return (0.0, 0.0, 0.0, -TURN_ANGLE)
    elif action == 4: # GO_UP
        return (0.0, 0.0, -UP_DOWN_STEP, 0.0)  # NED: down = negative Z = going up
    elif action == 5: # GO_DOWN
        return (0.0, 0.0, UP_DOWN_STEP, 0.0)
    elif action == 6: # MOVE_LEFT
        return (0.0, -LEFT_RIGHT_STEP, 0.0, 0.0)
    elif action == 7: # MOVE_RIGHT
        return (0.0, LEFT_RIGHT_STEP, 0.0, 0.0)
    return (0.0, 0.0, 0.0, 0.0)


def apply_action_to_pose(position, orientation_quat, dx, dy, dz, dyaw_deg):
    """
    Apply a relative action (in local NED frame) to a global pose.
    Returns new position and orientation.
    """
    # Convert quaternion to rotation matrix (w,x,y,z → scalar-last for scipy)
    q = orientation_quat  # [x, y, z, w] (Unreal)
    r = R.from_quat([q[0], q[1], q[2], q[3]])  # scipy uses [x, y, z, w]

    # Local displacement in NED: X=forward, Y=right, Z=down
    local_disp = np.array([dx, dy, dz])
    global_disp = r.apply(local_disp)

    new_pos = np.array(position) + global_disp

    # Apply yaw rotation (around Z axis)
    dyaw_rad = np.deg2rad(dyaw_deg)
    yaw_rot = R.from_euler("z", dyaw_rad)
    new_r = r * yaw_rot
    new_q = new_r.as_quat()  # [x, y, z, w]

    return new_pos.tolist(), new_q.tolist()


# ============================================================
# CMA Model Wrapper
# ============================================================
class CMAModelWrapper(BaseModelWrapper):
    def __init__(self, ckpt_path, vocab_path, device="cuda"):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load vocab
        self.word2idx, self.idx2word = _load_vocab(vocab_path)
        vocab_size = len(self.word2idx)
        print(f"CMA wrapper: loaded vocab ({vocab_size} words)")

        # Load model
        self.model = load_cma_checkpoint(ckpt_path, vocab_size, device=str(self.device))
        print(f"CMA wrapper: loaded checkpoint from {ckpt_path}")

        # Per-episode state
        self.rnn_states = None
        self.prev_actions = None
        self.not_done_masks = None
        self.instruction_tokens = None
        self.active_episode_id = None
        self.last_action = None

    def eval(self):
        self.model.eval()

    def _get_current_obs(self, ep_history):
        """Extract latest observation from episode history."""
        latest = ep_history[-1]
        # latest is a single dict with 'rgb', 'depth', 'sensors' keys
        if isinstance(latest, dict) and 'rgb' in latest:
            return latest
        # Fallback: latest might be a list of dicts
        if isinstance(latest, list):
            for item in reversed(latest):
                if isinstance(item, dict) and 'rgb' in item:
                    return item
        return None

    def _preprocess_frame(self, obs_dict):
        """Preprocess depth frame for AerialVLN CMA encoder.
        RGB is handled separately via extract_rgb_features (matching training pipeline)."""
        depth_img = obs_dict['depth'][0]  # numpy array [H, W] uint8

        depth_resized = np.array(depth_resize(depth_img))    # [256, 256] uint8
        depth_resized = depth_resized.astype(np.float32) / 255.0  # normalize to [0,1]
        depth_resized = depth_resized[:, :, np.newaxis]     # [256, 256, 1] float32

        return depth_resized

    @torch.no_grad()
    def prepare_inputs(self, episodes, target_positions, assist_notices=None):
        """Extract observations from episodes; also handle episode transitions."""
        current_ep_id = None
        if len(episodes) > 0 and len(episodes[0]) > 0:
            obs = self._get_current_obs(episodes[0])
            if obs is not None:
                current_ep_id = id(episodes[0])  # Use list id as proxy

        # Reset RNN state on new episode
        if current_ep_id != self.active_episode_id:
            self.rnn_states = None
            self.prev_actions = None
            self.not_done_masks = None
            self.instruction_tokens = None
            self.last_action = None
            self._step_counter = 0
            self.active_episode_id = current_ep_id

        inputs = {
            'episodes': episodes,
            'target_positions': target_positions,
        }
        return inputs, None

    @torch.no_grad()
    def run_profiled(self, inputs, episodes, rot_to_targets):
        """Run CMA inference for one step and return waypoints."""
        batch_size = len(episodes)
        waypoints_list = []

        llm_start = time.perf_counter()

        for i in range(batch_size):
            obs_dict = self._get_current_obs(episodes[i])
            if obs_dict is None:
                import logging
                logging.getLogger(__name__).warning(f"obs_dict is None for episode {i}")
                waypoints_list.append([np.zeros(3)])
                continue

            rgb_np = np.array(rgb_resize(obs_dict['rgb'][0]))   # [224, 224, 3] uint8
            depth_np = self._preprocess_frame(obs_dict)
            rgb_features = extract_rgb_features(rgb_np, self.device)  # [1, 2048, 4, 4]

            # Get current pose
            sensors = obs_dict.get('sensors', {})
            state = sensors.get('state', {})
            cur_pos = state.get('position', [0, 0, 0])
            cur_ori = sensors.get('imu', {}).get('orientation', [0, 0, 0, 1])

            # Tokenize instruction on first call
            if self.instruction_tokens is None:
                instr_text = obs_dict.get('instruction', '')
                if isinstance(instr_text, np.ndarray):
                    instr_text = str(instr_text)
                self.instruction_tokens = tokenize(instr_text, self.word2idx)
                self.instruction_tokens = torch.from_numpy(
                    self.instruction_tokens
                ).unsqueeze(0).to(self.device)

            # Build observation dict for AerialVLN CMA (matching training pipeline)
            cma_obs = {
                'rgb_features': rgb_features,                                     # [1, 2048, 4, 4]
                'depth': torch.from_numpy(depth_np).unsqueeze(0).to(self.device), # [1, 256, 256, 1] float32
                'instruction': self.instruction_tokens,                            # [1, 300] int64
            }

            # Initialize RNN state
            if self.rnn_states is None:
                self.rnn_states = torch.zeros(
                    1, self.model.net.num_recurrent_layers, self.model.net._hidden_size,
                    device=self.device,
                )
                self.prev_actions = torch.zeros(1, 1, dtype=torch.long, device=self.device)
                self.not_done_masks = torch.zeros(1, 1, dtype=torch.uint8, device=self.device)

            # Run CMA forward
            actions, new_rnn = self.model.act(
                cma_obs, self.rnn_states, self.prev_actions, self.not_done_masks,
                deterministic=True,
            )

            # Convert action → waypoint offset → global waypoint
            action = actions[0].item()
            self.last_action = action  # track for predict_done
            dx, dy, dz, dyaw = action_to_waypoint_offset(action)

            new_pos, new_ori = apply_action_to_pose(cur_pos, cur_ori, dx, dy, dz, dyaw)

            # Extract yaw from new orientation quaternion [x, y, z, w]
            r = R.from_quat([new_ori[0], new_ori[1], new_ori[2], new_ori[3]])
            yaw_deg = float(np.rad2deg(r.as_euler('xyz')[2]))

            # Build waypoint [x, y, z, yaw_deg] — yaw used by move_to_position for turns
            waypoint = [float(new_pos[0]), float(new_pos[1]), float(new_pos[2]), yaw_deg]
            if not hasattr(self, '_step_counter'):
                self._step_counter = 0
            self._step_counter += 1

            # Per-step log (suppressed during DAgger collect)
            if not os.environ.get('CMA_QUIET'):
                aname = ACTION_NAMES.get(action, str(action))
                print(f"  [CMA step {self._step_counter}] pos=({cur_pos[0]:.1f},{cur_pos[1]:.1f},{cur_pos[2]:.1f}) → {aname} → wp=({new_pos[0]:.1f},{new_pos[1]:.1f},{new_pos[2]:.1f}) yaw={yaw_deg:.1f}°")
            waypoints_list.append([waypoint])

            # Update RNN state for next step
            self.rnn_states = new_rnn
            self.prev_actions = actions.unsqueeze(-1)
            self.not_done_masks = torch.ones(1, 1, dtype=torch.uint8, device=self.device)

        llm_latency_ms = (time.perf_counter() - llm_start) * 1000.0

        profile_info = {
            "llm_latency_ms": llm_latency_ms,
            "traj_latency_ms": 0.0,
            "llm_output": waypoints_list,
        }
        return waypoints_list, profile_info

    def predict_done(self, episodes, object_infos):
        """CMA uses STOP action (0) as termination signal."""
        if hasattr(self, 'last_action') and self.last_action == 0:
            return [True] * len(episodes)
        return [False] * len(episodes)
