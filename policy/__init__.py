from brax.envs import env
from ml_collections import FrozenConfigDict

from .mlp_policy import MLPPolicy
from .ebm_policy import EBMPolicy
from .ebm_amp_policy import EBMAMPPolicy

__all__ = []
__all__ += ["MLPPolicy"]
__all__ += ["EBMPolicy"]
__all__ += ["EBMAMPPolicy"]


def make_policy(cfg: FrozenConfigDict, env: env.Env, target_action_size: int):
    return globals()[cfg.POLICY.CLASS](cfg, env, target_action_size)
