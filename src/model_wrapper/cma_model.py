"""
Self-contained CMA model definition (from AerialVLN) for TravelUAV eval.
Supports loading AirVLN CMA checkpoint and running closed-loop inference.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ============================================================
# Action space (matching AerialVLN)
# ============================================================
# STOP=0, FORWARD=1, TURN_LEFT=2, TURN_RIGHT=3,
# GO_UP=4, GO_DOWN=5, MOVE_LEFT=6, MOVE_RIGHT=7

FORWARD_STEP = 5.0       # meters
UP_DOWN_STEP = 2.0       # meters
LEFT_RIGHT_STEP = 5.0    # meters
TURN_ANGLE = 15.0        # degrees


# ============================================================
# Instruction Encoder
# ============================================================
class InstructionEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_size=50, hidden_size=128, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.encoder_rnn = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
        )
        self._output_size = hidden_size * 2 if bidirectional else hidden_size
        self.config = type('Config', (), {
            'final_state_only': False,
            'rnn_type': 'LSTM',
        })()

    @property
    def output_size(self):
        return self._output_size

    def forward(self, observations):
        instr = observations["instruction"].long()
        # instr shape: [B, L]
        embedded = self.embedding(instr)
        embedded = embedded.permute(1, 0, 2)  # [L, B, E] for LSTM
        output, _ = self.encoder_rnn(embedded)
        output = output.permute(1, 0, 2)  # [B, L, H]
        return output


# ============================================================
# RGB Feature Extractor
# ============================================================
class RGBFeatureExtractor(nn.Module):
    """TorchVision ResNet50, spatial output [2048, 4, 4] (matching CMA config)."""

    def __init__(self, device="cuda"):
        super().__init__()
        cnn = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        cnn.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        cnn.fc = nn.Sequential()
        for p in cnn.parameters():
            p.requires_grad = False
        self.cnn = cnn
        self._features = None
        self._hook = cnn.avgpool.register_forward_hook(self._hook_fn)

        # ImageNet normalization
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        self.output_shape = (2048, 4, 4)
        self.output_size = 256

    def _hook_fn(self, m, i, o):
        self._features = o  # [B, 2048, 4, 4]

    def forward(self, rgb_tensor):
        """rgb_tensor: [B, 3, H, W] in [0, 255]"""
        x = rgb_tensor.float() / 255.0
        x = (x - self.mean) / self.std
        self.cnn(x)
        feat = self._features.clone()
        return feat  # [B, 2048, 4, 4]


# ============================================================
# Depth Encoder (simplified, same architecture as VlnResnetDepthEncoder)
# ============================================================
class DepthEncoder(nn.Module):
    """Simple CNN depth encoder with spatial output (matching CMA)."""

    def __init__(self):
        super().__init__()
        # Lightweight CNN to match expected output shape
        layers = []
        in_ch = 1
        for out_ch in [32, 64, 128]:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ])
            in_ch = out_ch
        self.backbone = nn.Sequential(*layers)
        self.spatial_embeddings = nn.Embedding(8 * 8, 64)
        self.output_size = 128
        self.output_shape = (128 + 64, 8, 8)

    def forward(self, observations):
        if "depth_features" in observations:
            x = observations["depth_features"]
        else:
            d = observations["depth"]  # [B, H, W, 1]
            d = d.permute(0, 3, 1, 2)  # [B, 1, H, W]
            x = self.backbone(d)       # [B, 128, H', W']

        b, c, h, w = x.size()
        emb = self.spatial_embeddings(
            torch.arange(0, self.spatial_embeddings.num_embeddings, device=x.device)
        ).view(1, 64, h, w).expand(b, 64, h, w)
        return torch.cat([x, emb], dim=1)  # [B, 192, H', W']


# ============================================================
# GRU State Encoder
# ============================================================
class GRUStateEncoder(nn.Module):
    def __init__(self, input_size, hidden_size=512):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1)
        self.hidden_size = hidden_size

    @property
    def num_recurrent_layers(self):
        return 1

    def forward(self, x, h, masks):
        """
        x: [B, input_size]
        h: [B, 1, hidden_size]
        masks: [B, 1]  (0 = reset, 1 = continue)
        """
        if h.dim() == 2:
            h = h.unsqueeze(0)  # [1, B, H]
        else:
            h = h.permute(1, 0, 2)  # [L, B, H] → ensure proper dim

        # Reset hidden state where mask is 0
        h = h * masks.float().unsqueeze(0).permute(2, 1, 0)

        x = x.unsqueeze(0)  # [1, B, input_size]
        output, new_h = self.gru(x, h.contiguous())
        return output.squeeze(0), new_h


# ============================================================
# CMANet (from AerialVLN cma_policy.py)
# ============================================================
class CMANet(nn.Module):
    def __init__(self, vocab_size, num_actions=8, hidden_size=512, device="cuda"):
        super().__init__()
        self.device = device

        # Encoders
        self.instruction_encoder = InstructionEncoder(vocab_size)
        self.rgb_encoder = RGBFeatureExtractor(device)
        self.depth_encoder = DepthEncoder()

        # Previous action embedding
        self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)

        # Linear projections
        self.rgb_linear = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(2048, self.rgb_encoder.output_size), nn.ReLU(True),
        )
        self.depth_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(self.depth_encoder.output_shape), self.depth_encoder.output_size),
            nn.ReLU(True),
        )

        # First GRU
        rnn_input_size = self.depth_encoder.output_size + self.rgb_encoder.output_size + 32
        self.state_encoder = GRUStateEncoder(rnn_input_size, hidden_size)
        self._hidden_size = hidden_size

        # Cross-modal attention projections
        self.rgb_kv = nn.Conv1d(
            self.rgb_encoder.output_shape[0],
            hidden_size // 2 + self.rgb_encoder.output_size, 1,
        )
        self.depth_kv = nn.Conv1d(
            self.depth_encoder.output_shape[0],
            hidden_size // 2 + self.depth_encoder.output_size, 1,
        )
        self.state_q = nn.Linear(hidden_size, hidden_size // 2)
        self.text_k = nn.Conv1d(self.instruction_encoder.output_size, hidden_size // 2, 1)
        self.text_q = nn.Linear(self.instruction_encoder.output_size, hidden_size // 2)
        self.register_buffer("_scale", torch.tensor(1.0 / ((hidden_size // 2) ** 0.5)))

        # Second GRU
        self._output_size_before_compress = (
            hidden_size + self.rgb_encoder.output_size +
            self.depth_encoder.output_size + self.instruction_encoder.output_size
        )
        self.second_state_compress = nn.Sequential(
            nn.Linear(self._output_size_before_compress + 32, hidden_size), nn.ReLU(True),
        )
        self.second_state_encoder = GRUStateEncoder(hidden_size, hidden_size)

        self.num_recurrent_layers = 2

    @property
    def output_size(self):
        return self._hidden_size

    def _attn(self, q, k, v, mask=None):
        logits = torch.einsum("nc,nci->ni", q, k)
        if mask is not None:
            logits = logits - mask.float() * 1e8
        attn = F.softmax(logits * self._scale, dim=1)
        return torch.einsum("ni,nci->nc", attn, v)

    def forward(self, observations, rnn_states, prev_actions, masks):
        B = prev_actions.size(0)

        # Embed previous actions
        prev_act_emb = self.prev_action_embedding(
            ((prev_actions.float() + 1) * masks.float()).long().view(-1)
        )

        # Instruction embedding
        instr_emb = self.instruction_encoder(observations)

        # Depth embedding
        depth_emb = self.depth_encoder(observations)
        depth_emb = torch.flatten(depth_emb, 2)

        # RGB embedding
        rgb_emb = self.rgb_encoder(observations["rgb_tensor"])
        rgb_emb = torch.flatten(rgb_emb, 2)

        # Linear projections
        rgb_in = self.rgb_linear(rgb_emb)
        depth_in = self.depth_linear(depth_emb)

        # First GRU
        state_in = torch.cat([rgb_in, depth_in, prev_act_emb], dim=1)
        rnn_out = rnn_states.clone()
        state, rnn_out[:, 0:1, :] = self.state_encoder(
            state_in, rnn_states[:, 0:1, :], masks,
        )

        # Text attention: state → instruction
        text_state_q = self.state_q(state)
        text_state_k = self.text_k(instr_emb)
        text_mask = (instr_emb.sum(dim=2) == 0.0)
        text_emb = self._attn(text_state_q, text_state_k, instr_emb, text_mask)

        # Visual attention: text → RGB / depth
        rgb_k, rgb_v = torch.split(self.rgb_kv(rgb_emb), self._hidden_size // 2, dim=1)
        depth_k, depth_v = torch.split(self.depth_kv(depth_emb), self._hidden_size // 2, dim=1)
        text_q = self.text_q(text_emb)
        rgb_attn = self._attn(text_q, rgb_k, rgb_v)
        depth_attn = self._attn(text_q, depth_k, depth_v)

        # Second GRU
        x = torch.cat([state, text_emb, rgb_attn, depth_attn, prev_act_emb], dim=1)
        x = self.second_state_compress(x)
        x, rnn_out[:, 1:2, :] = self.second_state_encoder(x, rnn_states[:, 1:2, :], masks)

        return x, rnn_out


# ============================================================
# CMA Policy (action head on top of CMANet)
# ============================================================
class CMAPolicy(nn.Module):
    def __init__(self, vocab_size, num_actions=8, hidden_size=512, device="cuda"):
        super().__init__()
        self.net = CMANet(vocab_size, num_actions, hidden_size, device)
        self.action_head = nn.Linear(hidden_size, num_actions)

    def forward(self, observations, rnn_states, prev_actions, masks):
        features, new_rnn = self.net(observations, rnn_states, prev_actions, masks)
        logits = self.action_head(features)
        return logits, new_rnn

    def act(self, observations, rnn_states, prev_actions, masks, deterministic=True):
        logits, new_rnn = self.forward(observations, rnn_states, prev_actions, masks)
        if deterministic:
            actions = logits.argmax(dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)
            actions = probs.multinomial(1).squeeze(-1)
        return actions, new_rnn


# ============================================================
# Load checkpoint
# ============================================================
def load_cma_checkpoint(ckpt_path, vocab_size, device="cuda"):
    """Load CMA model from AerialVLN training checkpoint."""
    model = CMAPolicy(vocab_size=vocab_size, device=device)
    model.to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)

    # Handle potential DistributedDataParallel prefix
    new_state = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state[k] = v

    model.load_state_dict(new_state, strict=False)
    model.eval()
    return model
