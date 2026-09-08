"""Compatibility shim for PPO.load deserialization of pre-pilot7 checkpoints.

Old DRL-scheduler checkpoints were saved with policy_class=SplitACPolicy;
sb3's PPO.load reconstructs the policy from the saved policy_class, so the
class name must stay importable (drl_scheduler_train.py / drl_scheduler_eval.py
/ match_eval.py all import it). Since pilot7 the observation is 5-dim and the
Actor and Critic share one observation, so the class does nothing beyond stock
sb3 MlpPolicy — new training uses ``"MlpPolicy"`` directly.
"""

from stable_baselines3.common.policies import ActorCriticPolicy


class SplitACPolicy(ActorCriticPolicy):
    pass
