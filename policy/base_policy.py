from abc import ABC, abstractmethod

from jax import numpy as jnp
from brax.envs import env
from ml_collections import FrozenConfigDict

from util.types import *


class BasePolicy(ABC):

    def __init__(self, cfg: FrozenConfigDict, env: env.Env, target_action_size: int):
        self.cfg = cfg
        self.env = env
        self.target_action_size = target_action_size


    def init(self, rng: PRNGKey):
        """should return the parameters of policy in form of python tree
        """
        pass


    def apply(self, params: Params, observation: jnp.ndarray, goal: jnp.ndarray, key: PRNGKey, train_mode: bool = True) -> jnp.ndarray:
        pass
