"""
CMA model wrapper for TravelUAV closed-loop eval.
Implements the same interface as ProfileTravelModelWrapper.
"""
import time
import re
import string
import numpy as np
from typing import Any, Dict, Tuple
from collections import defaultdict

import torch
import torch.nn.functional as F
import torchvision.transforms as T
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
rgb_transform = T.Compose([
    T.ToPILImage(),
    T.Resize(224),
    T.CenterCrop(224),
    T.ToTensor(),
])

depth_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((256, 256)),
    T.ToTensor(),
])


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

    def eval(self):
        self.model.eval()

    def _get_current_obs(self, ep_history):
        """Extract latest observation from episode history."""
        # ep_history is a list of observations (each observation is a list of dicts)
        # Latest entry with 'rgb' key contains the current sensor data
        latest = ep_history[-1]
        # Find the dict with 'rgb' in the list
        obs_dict = None
        for item in reversed(latest):
            if isinstance(item, dict) and 'rgb' in item:
                obs_dict = item
                break
        if obs_dict is None:
            for item in latest:
                if isinstance(item, dict) and 'rgb' in item:
                    obs_dict = item
                    break
        return obs_dict

    def _preprocess_frame(self, obs_dict):
        """Preprocess a single observation frame for CMA."""
        # Front camera is first in the multi-view list
        rgb_img = obs_dict['rgb'][0]   # numpy array [H, W, 3]
        depth_img = obs_dict['depth'][0]  # numpy array [H, W]

        rgb_tensor = rgb_transform(rgb_img)      # [3, 224, 224]
        depth_tensor = depth_transform(depth_img) # [1, 256, 256]

        return rgb_tensor, depth_tensor

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
                waypoints_list.append([np.zeros(7)])
                continue

            rgb_tensor, depth_tensor = self._preprocess_frame(obs_dict)

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

            # Build observation dict for CMA
            rgb_batch = rgb_tensor.unsqueeze(0).to(self.device)       # [1, 3, 224, 224]
            depth_batch = depth_tensor.unsqueeze(0).to(self.device)    # [1, 1, 256, 256]
            instr_batch = self.instruction_tokens                     # [1, 300]

            cma_obs = {
                'rgb_tensor': rgb_batch,  # Special key for our encoder
                'rgb': depth_tensor.unsqueeze(0).unsqueeze(-1).to(self.device),  # Dummy - not used
                'depth': depth_tensor.permute(1, 2, 0).unsqueeze(0).to(self.device),  # [1, 256, 256, 1]
                'instruction': instr_batch,
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
            dx, dy, dz, dyaw = action_to_waypoint_offset(action)

            new_pos, new_ori = apply_action_to_pose(cur_pos, cur_ori, dx, dy, dz, dyaw)

            # Build waypoint list (in TravelUAV format: [x, y, z, qx, qy, qz, qw])
            waypoint = [new_pos[0], new_pos[1], new_pos[2],
                        new_ori[0], new_ori[1], new_ori[2], new_ori[3]]
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
        """CMA doesn't have explicit done prediction. Return False."""
        return [False] * len(episodes)
