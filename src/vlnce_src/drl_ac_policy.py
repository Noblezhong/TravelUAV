"""Split Actor-Critic policy for DRL scheduler.

Observation is 7-dim: [buf, wp_dist, bw, inflight, drift, td, NE].
Actor sees dims 0-5 with dim 6 (NE) zeroed — so it effectively uses 6 features.
Critic sees all 7 dims including NE, which helps the value function distinguish
"close to target" from "far".

During evaluation NE is computed from the sim (same as during training)
so there is no train/eval distribution shift.  The Actor never learns to
depend on NE because that input is always zero during forward / backward.
"""

from typing import Tuple

import torch
from stable_baselines3.common.policies import ActorCriticPolicy


class SplitACPolicy(ActorCriticPolicy):
    """MlpPolicy where the Actor input masks out the NE dimension."""

    def _actor_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Return 7-dim tensor with NE (index 6) zeroed for the Actor."""
        # Pilot-A: request_age (index 7) unmasked --- the Actor needs
        # request-schedule memory to learn preventive requesting and the
        # CONTINUE_REQUEST mode. NE (index 6) stays visible: that is the
        # load-bearing convergence wall, do NOT mask it.
        return obs

    def _critic_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Return the full 7-dim tensor including NE for the Critic."""
        return obs

    def forward(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.extract_features(obs)
        actor_feat = self.extract_features(self._actor_obs(obs))
        pi_latent, _ = self.mlp_extractor(actor_feat)
        _, vf_latent = self.mlp_extractor(features)
        values = self.value_net(vf_latent)
        distribution = self._get_action_dist_from_latent(pi_latent)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.extract_features(obs)
        actor_feat = self.extract_features(self._actor_obs(obs))
        pi_latent, _ = self.mlp_extractor(actor_feat)
        _, vf_latent = self.mlp_extractor(features)
        values = self.value_net(vf_latent)
        distribution = self._get_action_dist_from_latent(pi_latent)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values, log_prob, entropy
