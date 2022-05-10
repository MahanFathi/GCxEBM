from typing import Dict

import flax
import optax
import brax
from brax import envs
from typing import Any
import jax.numpy as jnp

Params = Any
PRNGKey = jnp.ndarray


@flax.struct.dataclass
class StepData:
    """Contains data for one environment step."""
    obs: jnp.ndarray
    goal: jnp.ndarray
    qp: brax.physics.base.QP
    rewards: jnp.ndarray
    dones: jnp.ndarray
    truncation: jnp.ndarray
    actions: jnp.ndarray
    logits: jnp.ndarray


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""
    optimizer_state: optax.OptState
    params: Params
    key: PRNGKey
    normalizer_params: Params


@flax.struct.dataclass
class EvalEnvState:
    """Env state + eval metrics."""
    core: envs.State
    qp: brax.physics.base.QP
    current_episode_metrics: Dict[str, jnp.ndarray]
    completed_episodes_metrics: Dict[str, jnp.ndarray]
    completed_episodes: jnp.ndarray
    completed_episodes_steps: jnp.ndarray
