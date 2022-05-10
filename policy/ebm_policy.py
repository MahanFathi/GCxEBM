import jax
from jax import numpy as jnp
from brax.envs import env
from ml_collections import FrozenConfigDict

from policy.ebm import EBM
from policy.base_policy import BasePolicy
from util.types import *


class EBMPolicy(BasePolicy):

    def __init__(self, cfg: FrozenConfigDict, env: env.Env, target_action_size: int):
        super(EBMPolicy, self).__init__(cfg, env)
        self.ebm = EBM(cfg, env, env.goal_size, target_action_size) # options = goals for now


    def init(self, rng: PRNGKey):
        return self.ebm.init(rng)


    def apply(self, params: Params, observation: jnp.ndarray, goal: jnp.ndarray, key: PRNGKey, train_mode: bool = True) -> jnp.ndarray:

        batch_size, observation_size = observation.shape

        key_init_a, key_infer_a = jax.random.split(key)
        a_init = jax.random.normal(key_init_a, (batch_size, self.env.action_size))

        a = self.ebm.infer_a(params, observation, goal, a_init, key_infer_a, train_mode)

        return a
