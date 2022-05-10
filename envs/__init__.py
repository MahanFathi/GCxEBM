from typing import Callable, Optional

from brax.envs.env import Env

from .envs import ant
from . import wrappers

_envs = {
    'ant': ant.Ant,
}


def get_environment(env_name, **kwargs):
  return _envs[env_name](**kwargs)


def create(env_name: str,
           episode_length: int = 1000,
           action_repeat: int = 1,
           auto_reset: bool = True,
           batch_size: Optional[int] = None,
           eval_metrics: bool = False,
           **kwargs) -> Env:
  """Creates an Env with a specified brax system."""
  env = _envs[env_name](**kwargs)
  if episode_length is not None:
    env = wrappers.EpisodeWrapper(env, episode_length, action_repeat)
  if batch_size:
    env = wrappers.VectorWrapper(env, batch_size)
  if auto_reset:
    env = wrappers.AutoResetWrapper(env)
  if eval_metrics:
    env = wrappers.EvalWrapper(env)

  return env  # type: ignore


def create_fn(env_name: str, **kwargs) -> Callable[..., Env]:
  """Returns a function that when called, creates an Env."""
  return functools.partial(create, env_name, **kwargs)
