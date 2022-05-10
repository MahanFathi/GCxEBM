# Copyright 2022 The Brax Authors (+Mahan).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wrappers for goal-cusomized Brax env."""

from typing import Dict

from brax import jumpy as jp
from brax.envs import env as brax_env
import flax
import jax


class VectorWrapper(brax_env.Wrapper):
    """DEPRECATED Vectorizes Brax env. Use VmapWrapper instead."""

    def __init__(self, env: brax_env.Env, batch_size: int):
        super().__init__(env)
        self.batch_size = batch_size

    def reset(self, rng: jp.ndarray) -> brax_env.State:
        rng = jp.random_split(rng, self.batch_size)
        return jp.vmap(self.env.reset)(rng)

    def sample_goal(self, rng: jp.ndarray) -> jp.ndarray:
        rng = jp.random_split(rng, self.batch_size)
        return jp.vmap(self.env.sample_goal)(rng)

    def step(self, state: brax_env.State, action: jp.ndarray, goal: jp.ndarray) -> brax_env.State:
        return jp.vmap(self.env.step)(state, action, goal)


class VmapWrapper(brax_env.Wrapper):
    """Vectorizes Brax env."""

    def reset(self, rng: jp.ndarray) -> brax_env.State:
        return jp.vmap(self.env.reset)(rng)

    def sample_goal(self, rng: jp.ndarray) -> jp.ndarray:
        rng = jp.random_split(rng, self.batch_size)
        return jp.vmap(self.env.sample_goal)(rng)

    def step(self, state: brax_env.State, action: jp.ndarray, goal:jp.ndarray) -> brax_env.State:
        return jp.vmap(self.env.step)(state, action, goal)


class EpisodeWrapper(brax_env.Wrapper):
    """Maintains episode step count and sets done at episode end."""

    def __init__(self, env: brax_env.Env, episode_length: int,
                 action_repeat: int):
        super().__init__(env)
        if hasattr(self.unwrapped, 'sys'):
            self.unwrapped.sys.config.dt *= action_repeat
            self.unwrapped.sys.config.substeps *= action_repeat
        self.episode_length = episode_length
        self.action_repeat = action_repeat

    def reset(self, rng: jp.ndarray) -> brax_env.State:
        state = self.env.reset(rng)
        state.info['steps'] = jp.zeros(())
        state.info['truncation'] = jp.zeros(())
        return state

    def step(self, state: brax_env.State, action: jp.ndarray, goal: jp.ndarray) -> brax_env.State:
        state = self.env.step(state, action, goal)
        steps = state.info['steps'] + self.action_repeat
        one = jp.ones_like(state.done)
        zero = jp.zeros_like(state.done)
        done = jp.where(steps >= self.episode_length, one, state.done)
        state.info['truncation'] = jp.where(steps >= self.episode_length,
                                            1 - state.done, zero)
        state.info['steps'] = steps
        return state.replace(done=done)


class AutoResetWrapper(brax_env.Wrapper):
    """Automatically resets Brax envs that are done."""

    def reset(self, rng: jp.ndarray) -> brax_env.State:
        state = self.env.reset(rng)
        state.info['first_qp'] = state.qp
        state.info['first_obs'] = state.obs
        return state

    def step(self, state: brax_env.State, action: jp.ndarray, goal: jp.ndarray) -> brax_env.State:
        if 'steps' in state.info:
            steps = state.info['steps']
            steps = jp.where(state.done, jp.zeros_like(steps), steps)
            state.info.update(steps=steps)
        state = state.replace(done=jp.zeros_like(state.done))
        state = self.env.step(state, action, goal)

        def where_done(x, y):
            done = state.done
            if done.shape:
                done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
            return jp.where(done, x, y)

        qp = jp.tree_map(where_done, state.info['first_qp'], state.qp)
        obs = where_done(state.info['first_obs'], state.obs)
        return state.replace(qp=qp, obs=obs)


@flax.struct.dataclass
class EvalMetrics:
    current_episode_metrics: Dict[str, jp.ndarray]
    completed_episodes_metrics: Dict[str, jp.ndarray]
    completed_episodes: jp.ndarray
    completed_episodes_steps: jp.ndarray


class EvalWrapper(brax_env.Wrapper):
    """Brax env with eval metrics."""

    def reset(self, rng: jp.ndarray) -> brax_env.State:
        reset_state = self.env.reset(rng)
        reset_state.metrics['reward'] = reset_state.reward
        eval_metrics = EvalMetrics(
            current_episode_metrics=jax.tree_map(jp.zeros_like,
                                                 reset_state.metrics),
            completed_episodes_metrics=jax.tree_map(
                lambda x: jp.zeros_like(jp.sum(x)), reset_state.metrics),
            completed_episodes=jp.zeros(()),
            completed_episodes_steps=jp.zeros(()))
        reset_state.info['eval_metrics'] = eval_metrics
        return reset_state

    def step(self, state: brax_env.State, action: jp.ndarray, goal: jp.ndarray) -> brax_env.State:
        state_metrics = state.info['eval_metrics']
        if not isinstance(state_metrics, EvalMetrics):
            raise ValueError(
                f'Incorrect type for state_metrics: {type(state_metrics)}')
        del state.info['eval_metrics']
        nstate = self.env.step(state, action, goal)
        nstate.metrics['reward'] = nstate.reward
        # steps stores the highest step reached when done = True, and then
        # the next steps becomes action_repeat
        completed_episodes_steps = state_metrics.completed_episodes_steps + jp.sum(
            nstate.info['steps'] * nstate.done)
        current_episode_metrics = jax.tree_multimap(
            lambda a, b: a + b, state_metrics.current_episode_metrics,
            nstate.metrics)
        completed_episodes = state_metrics.completed_episodes + jp.sum(nstate.done)
        completed_episodes_metrics = jax.tree_multimap(
            lambda a, b: a + jp.sum(b * nstate.done),
            state_metrics.completed_episodes_metrics, current_episode_metrics)
        current_episode_metrics = jax.tree_multimap(
            lambda a, b: a * (1 - nstate.done) + b * nstate.done,
            current_episode_metrics, nstate.metrics)

        eval_metrics = EvalMetrics(
            current_episode_metrics=current_episode_metrics,
            completed_episodes_metrics=completed_episodes_metrics,
            completed_episodes=completed_episodes,
            completed_episodes_steps=completed_episodes_steps)
        nstate.info['eval_metrics'] = eval_metrics
        return nstate
