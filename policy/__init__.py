from brax.envs import env
from ml_collections import FrozenConfigDict

from .ebm_policy import EBMPolicy

__all__ = []
__all__ += ["EBMPolicy"]


def make_policy(cfg: FrozenConfigDict, env: env.Env, target_action_size: int):
    return globals()[cfg.POLICY.CLASS](cfg, env, target_action_size)
