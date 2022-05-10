from abc import ABC, abstractmethod

import jax
from jax import numpy as jnp
from brax import jumpy as jp
from brax.envs import env
from ml_collections import FrozenConfigDict

from util.types import *


class BasePolicy(ABC):
    """Base policy"""

    def __init__(self, cfg: FrozenConfigDict, env: env.Env, target_action_size: int):
        self.cfg = cfg
        self.env = env
        self.target_action_size = target_action_size
        self._apply_sequence = jax.vmap(self.apply, in_axes=(None, 0, 0, 0, None))


    def init(self, rng: PRNGKey):
        """should return the parameters of policy in form of python tree
        """
        pass


    def apply(self, params: Params, observation: jnp.ndarray, goal: jnp.ndarray, key: PRNGKey, train_mode: bool = True) -> jnp.ndarray:
        pass


    def apply_sequence(self, params: Params, observation: jnp.ndarray, goal: jnp.ndarray, key: PRNGKey, train_mode: bool = True) -> jnp.ndarray:
        sequence_length, batch_size, observation_size = observation.shape # goal also admits the same shape
        key = jp.random_split(key, sequence_length)
        return self._apply_sequence(params, observation, goal, key, train_mode)
