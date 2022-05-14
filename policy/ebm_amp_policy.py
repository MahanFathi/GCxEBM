import jax
from jax import numpy as jnp
from brax.envs import env
from ml_collections import FrozenConfigDict

from policy.ebm import EBM
from policy.base_policy import BasePolicy
from util import net
from util.types import *


class EBMAMPPolicy(BasePolicy):
    """Energy-based Model + AMP of the policy"""

    def __init__(self, cfg: FrozenConfigDict, env: env.Env, target_action_size: int):
        super(EBMAMPPolicy, self).__init__(cfg, env, target_action_size)
        self.ebm = EBM(cfg, env, env.goal_size, target_action_size) # options = goals for now
        self.amp = net.make_mlp(
            list(cfg.EBM.AMP_LAYERS) + [target_action_size],
            target_action_size,
        )


    def init(self, rng: PRNGKey):
        key_ebm, key_amp = jax.random.split(rng)
        return {
            'ebm': self.ebm.init(key_ebm),
            'amp': self.amp.init(key_amp),
        }


    def apply(self, params: Params, observation: jnp.ndarray, goal: jnp.ndarray, key: PRNGKey, train_mode: bool = None) -> jnp.ndarray:

        params_ebm, params_amp = params['ebm'], params['amp']

        batch_size, observation_size = observation.shape

        key_init_a, key_infer_a = jax.random.split(key)
        a_init = jax.random.normal(key_init_a, (batch_size, self.target_action_size))

        a = self.ebm.infer_a(params_ebm, observation, goal, a_init, key_infer_a, train_mode)

        return self.amp.apply(params_amp, a)
