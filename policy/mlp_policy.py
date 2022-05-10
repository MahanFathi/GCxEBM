import jax
from jax import numpy as jnp
from brax.envs import env
from ml_collections import FrozenConfigDict

from policy.ebm import EBM
from policy.base_policy import BasePolicy
from util import net
from util.types import *


class MLPPolicy(BasePolicy):
    """Multi-layer Perceptron model of the policy"""

    def __init__(self, cfg: FrozenConfigDict, env: env.Env, target_action_size: int):
        super(MLPPolicy, self).__init__(cfg, env, target_action_size)
        self.mlp_policy = net.make_mlp(
            list(cfg.POLICY.MLP_LAYERS) + [target_action_size],
            env.observation_size + env.goal_size)


    def init(self, rng: PRNGKey):
        return self.mlp_policy.init(rng)


    def apply(self, params: Params, observation: jnp.ndarray, goal: jnp.ndarray, key: PRNGKey, train_mode: bool = True) -> jnp.ndarray:
        return self.mlp_policy.apply(params, jnp.concatenate([observation, goal], axis=-1))
