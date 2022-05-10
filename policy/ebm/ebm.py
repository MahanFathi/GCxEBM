from functools import partial

import jax
from jax import numpy as jnp
from ml_collections import FrozenConfigDict

from envs.base_env import BaseEnv
from util.net import build_ebm_net
from util.types import *

# from absl import logging
# from jax.experimental import host_callback as jhcb


class EBM(object):

    def __init__(
            self,
            cfg: FrozenConfigDict,
            env: BaseEnv,
            option_size: int,
            target_action_size: int = None
    ):

        self.cfg = cfg
        self.env = env

        self.state_size = env.observation_size
        self.action_size = env.action_size
        self.option_size = option_size

        if target_action_size:
            self.target_action_size = target_action_size
        else:
            self.target_action_size = 2 * self.action_size # assuming outputs to be parameters of a normal dist, mu + sigma


        # build net
        self._ebm_net = build_ebm_net(cfg, self.state_size, self.target_action_size)

        # define derivatives
        self._dedz = jax.jit(jax.vmap(jax.grad(self.apply, 2), in_axes=(None, 0, 0, 0)))
        self._deda = jax.jit(jax.vmap(jax.grad(self.apply, 3), in_axes=(None, 0, 0, 0)))

        if cfg.EBM.GRAD_CLIP:
            self.dedz = lambda params, s, z, a: jnp.clip(
                self._dedz(params, s, z, a), -cfg.EBM.GRAD_CLIP, cfg.EBM.GRAD_CLIP)
            self.deda = lambda params, s, z, a: jnp.clip(
                self._deda(params, s, z, a), -cfg.EBM.GRAD_CLIP, cfg.EBM.GRAD_CLIP)
        else:
            self.dedz = self._dedz
            self.deda = self._deda


    def init(self, key: PRNGKey):
        return self._ebm_net.init(key)


    @partial(jax.jit, static_argnums=(0,))
    def apply(self, params: Params, s: jnp.ndarray, z: jnp.ndarray, a: jnp.ndarray):
        return self._ebm_net.apply(params, s, z, a).squeeze() ** 2 # (batch_size, 1).squeeze()


    @partial(jax.jit, static_argnums=(0, ))
    def _step_z_grad_descent(self, carry, unused_t):
        params: Params
        s: jnp.ndarray
        z: jnp.ndarray
        a: jnp.ndarray
        key: PRNGKey
        langevin_gd: bool
        params, s, z, a, key, langevin_gd = carry

        alpha = self.cfg.EBM.ALPHA

        key, langevin_key = jax.random.split(key)
        omega = jax.random.normal(langevin_key, z.shape) * jnp.sqrt(alpha)
        omega *= langevin_gd # TODO: dirty way around jax compiling scan functions
        # TODO: do not add noise at final step?

        z += -alpha / 2. * self.dedz(params, s, z, a) + omega

        return (params, s, z, a, key, langevin_gd), ()


    @partial(jax.jit, static_argnums=(0, ))
    def _step_a_grad_descent(self, carry, unused_t):
        params: Params
        s: jnp.ndarray
        z: jnp.ndarray
        a: jnp.ndarray
        key: PRNGKey
        langevin_gd: bool
        params, s, z, a, key, langevin_gd = carry

        alpha = self.cfg.EBM.ALPHA

        key, langevin_key = jax.random.split(key)
        omega = jax.random.normal(langevin_key, a.shape) * jnp.sqrt(alpha)
        omega *= langevin_gd # TODO: dirty way around jax compiling scan functions
        # TODO: do not add noise at final step?

        a += -alpha / 2. * self.deda(params, s, z, a) + omega

        return (params, s, z, a, key, langevin_gd), ()


    @partial(jax.jit, static_argnums=(0, ))
    def _step_z_and_a_grad_descent(self, carry, unused_t):
        params: Params
        s: jnp.ndarray
        z: jnp.ndarray
        a: jnp.ndarray
        key: PRNGKey
        langevin_gd: bool
        params, s, z, a, key, langevin_gd = carry

        alpha = self.cfg.EBM.ALPHA # NOTE: here we assume we have used same alpha for z and a during training

        key, langevin_key_z, langevin_key_a = jax.random.split(key, 3)
        omega_z = jax.random.normal(langevin_key_z, z.shape) * jnp.sqrt(alpha)
        omega_a = jax.random.normal(langevin_key_a, a.shape) * jnp.sqrt(alpha)
        omega_z *= langevin_gd # TODO: dirty way around jax compiling scan functions
        omega_a *= langevin_gd # TODO: dirty way around jax compiling scan functions
        # TODO: do not add noise at final step?

        dedz = self.dedz(params, s, z, a)
        deda = self.deda(params, s, z, a)
        z += -alpha / 2. * dedz + omega_z
        a += -alpha / 2. * deda + omega_a

        return (params, s, z, a, key, langevin_gd), ()


    def infer_z(self, params: Params, s: jnp.ndarray, z: jnp.ndarray, a: jnp.ndarray, key: PRNGKey, langevin_gd: bool = None):

        if langevin_gd is None:
            langevin_gd = self.cfg.EBM.LANGEVIN_GD

        (_, _, z, _, _, _), _ = jax.lax.scan(
            self._step_z_grad_descent,
            (params, s, z, a, key, langevin_gd), (), self.cfg.EBM.K)

        return z


    def infer_a(self, params: Params, s: jnp.ndarray, z: jnp.ndarray, a: jnp.ndarray, key: PRNGKey, langevin_gd: bool = None):

        if langevin_gd is None:
            langevin_gd = self.cfg.EBM.LANGEVIN_GD

        (_, _, _, a, _, _), _ = jax.lax.scan(
            self._step_a_grad_descent,
            (params, s, z, a, key, langevin_gd), (), self.cfg.EBM.K)

        return a


    def infer_z_and_a(self, params: Params, s: jnp.ndarray, z: jnp.ndarray, a: jnp.ndarray, key: PRNGKey, langevin_gd: bool = None):

        if langevin_gd is None:
            langevin_gd = self.cfg.EBM.LANGEVIN_GD

        (_, _, z, a, _, _), _ = jax.lax.scan(
            self._step_z_and_a_grad_descent,
            (params, s, z, a, key, langevin_gd), (), self.cfg.EBM.K)

        return z, a
