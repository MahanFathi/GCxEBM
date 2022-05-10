from typing import Any, Callable, Dict, Optional, Tuple
import functools
import time

from absl import logging
from brax import envs
from brax.training import distribution
from brax.training import normalization
from brax.training import pmap
import flax
import jax
import jax.numpy as jnp
import optax
from ml_collections import FrozenConfigDict

from policy import make_policy
from rl import get_rl_loss
from util.eval import EvalEnvState, wrap_for_eval
from util import net, logger
from util.types import *


def train(
        cfg: FrozenConfigDict,
        environment_fn: Callable[..., envs.Env],
        progress_fn: Optional[Callable[[int, Dict[str, Any]], None]] = None,
        seed: int = 0,
):
    """Training pipeline. Stolen from BRAX for the MOST part."""

    # CONFIG EXTRACTION
    num_timesteps = cfg.TRAIN.NUM_TIMESTEPS
    episode_length = cfg.TRAIN.EPISODE_LENGTH
    num_update_epochs = cfg.TRAIN.NUM_UPDATE_EPOCHS
    action_repeat = cfg.TRAIN.ACTION_REPEAT
    num_envs = cfg.TRAIN.NUM_ENVS
    max_devices_per_host = cfg.TRAIN.MAX_DEVICES_PER_HOST
    learning_rate = cfg.TRAIN.LEARNING_RATE
    unroll_length = cfg.TRAIN.UNROLL_LENGTH
    batch_size = cfg.TRAIN.BATCH_SIZE
    num_minibatches = cfg.TRAIN.NUM_MINIBATCHES
    normalize_observations = cfg.TRAIN.NORMALIZE_OBSERVATIONS
    num_eval_envs = cfg.TRAIN.NUM_EVAL_ENVS
    log_frequency = cfg.LOG.FREQUENCY
    log_to_file = cfg.LOG.TO_FILE
    log_save_params = cfg.LOG.SAVE_PARAMS and cfg.LOG.TO_FILE

    assert batch_size * num_minibatches % num_envs == 0
    xt = time.time()

    # PROCESS BOOKKEEPING
    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count
    if max_devices_per_host:
        local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
    logging.info(
        'Device count: %d, process count: %d (id %d), local device count: %d, '
        'devices to be used count: %d',
        jax.device_count(), process_count, process_id, local_device_count,
        local_devices_to_use)

    # MAKE THEM FILES
    if log_to_file and process_id == 0:
        logger.get_logdir_path(cfg)

    # KEY MANAGEMENT
    key = jax.random.PRNGKey(seed)
    key, key_models, key_env, key_eval = jax.random.split(key, 4)
    # Make sure every process gets a different random key, otherwise they will be
    # doing identical work.
    key_env = jax.random.split(key_env, process_count)[process_id]
    key = jax.random.split(key, process_count)[process_id]
    # key_models should be the same, so that models are initialized the same way
    # for different processes.
    # key_eval is also used in one process so no need to split.

    # ENV SETTINGS
    core_env = environment_fn(
        action_repeat=action_repeat,
        batch_size=num_envs // local_devices_to_use // process_count,
        episode_length=episode_length,
    )
    key_envs = jax.random.split(key_env, local_devices_to_use)
    step_fn = core_env.step
    first_state = jax.tree_multimap(lambda *args: jnp.stack(args),
                                    *[core_env.reset(key) for key in key_envs])

    core_eval_env = environment_fn(
        action_repeat=action_repeat,
        batch_size=num_eval_envs,
        episode_length=episode_length)
    eval_first_state, eval_step_fn = wrap_for_eval(core_eval_env, key_eval)

    # NETWORKS
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=core_env.action_size)

    policy_model = make_policy(
        cfg,
        core_env,
        parametric_action_distribution.param_size,
    )
    value_model = net.make_model(
        cfg.VALUE_NET.FEATURES + [1],
        core_env.observation_size + core_env.goal_size, # we also need to train a universal vf
    )

    key_policy, key_value = jax.random.split(key_models)
    policy_params = policy_model.init(key_policy)
    value_params = value_model.init(key_value)
    init_params = {
        'policy': policy_params,
        'value': value_params,
    }
    optimizer = optax.adam(learning_rate=learning_rate)
    optimizer_state = optimizer.init(init_params)
    optimizer_state, init_params = pmap.bcast_local_devices(
        (optimizer_state, init_params), local_devices_to_use)

    normalizer_params, obs_normalizer_update_fn, obs_normalizer_apply_fn = (
        normalization.create_observation_normalizer(
            core_env.observation_size, normalize_observations,
            num_leading_batch_dims=2, pmap_to_devices=local_devices_to_use))

    key_debug = jax.random.PRNGKey(seed + 666)

    # LOSS AND GRAD
    loss_fn = functools.partial(
        get_rl_loss(cfg.TRAIN.LOSS_FN),
        cfg=cfg,
        parametric_action_distribution=parametric_action_distribution,
        policy_apply=policy_model.apply,
        value_apply=value_model.apply,
    )
    grad_loss = jax.grad(loss_fn, has_aux=True)

    def do_one_step(carry, unused_target_t):
        state, goal, normalizer_params, policy_params, key = carry
        key, key_sample, key_action_logits = jax.random.split(key, 3)
        normalized_obs = obs_normalizer_apply_fn(normalizer_params, state.obs)
        logits = policy_model.apply(policy_params, normalized_obs, goal, key_action_logits)
        actions = parametric_action_distribution.sample_no_postprocessing(
            logits, key_sample)
        postprocessed_actions = parametric_action_distribution.postprocess(actions)
        nstate = step_fn(state, postprocessed_actions, goal)
        return (nstate, goal, normalizer_params, policy_params, key), StepData(
            obs=state.obs, # what state you're at
            goal=goal, # what goal you wanna acheive
            qp=state.qp, # your state specifics
            rewards=nstate.reward, # the reward you get from taking action `a` at state `s` towards acheiving the `goal`
            dones=nstate.done, # are you gonna be done after taking this action at this state?
            truncation=nstate.info['truncation'], # same
            actions=actions, # the action you took
            logits=logits, # same
        )

    def do_one_step_eval(carry, unused_target_t):
        state, policy_params, normalizer_params, key = carry
        key, key_sample = jax.random.split(key)
        # TODO: Make this nicer ([0] comes from pmapping).
        obs = obs_normalizer_apply_fn(
            jax.tree_map(lambda x: x[0], normalizer_params), state.obs)
        logits = policy_model.apply(policy_params, obs)
        actions = parametric_action_distribution.sample(logits, key_sample)
        nstate = eval_step_fn(state, actions)
        return (nstate, policy_params, normalizer_params, key), ()

    def generate_unroll(carry, unused_target_t):
        state, normalizer_params, policy_params, key = carry
        key, key_goal = jax.random.split(key)
        goal = core_env.sample_goal(key_goal)
        (state, _, _, key), data = jax.lax.scan(
            do_one_step, (state, goal, normalizer_params, policy_params, key), (),
            length=unroll_length)
        # data: (unroll_length, batch_size, [obs_size])
        # add the last datapoint
        data = data.replace(
            obs=jnp.concatenate(
                [data.obs, jnp.expand_dims(state.obs, axis=0)]),
            # rewards=jnp.concatenate( # NOTE: commented out since the rewards at each state should represent the reward given to taking that action from that state
            #     [data.rewards, jnp.expand_dims(state.reward, axis=0)]),
            goal=jnp.concatenate(
                [data.goal, jnp.expand_dims(goal, axis=0)]), # the goal you had when reaching this (last) state (later needed for value function)
            qp=data.qp.replace( # TODO: use tree_map
                pos=jnp.concatenate(
                    [data.qp.pos, jnp.expand_dims(state.qp.pos, axis=0)]),
                rot=jnp.concatenate(
                    [data.qp.rot, jnp.expand_dims(state.qp.rot, axis=0)]),
                vel=jnp.concatenate(
                    [data.qp.vel, jnp.expand_dims(state.qp.vel, axis=0)]),
                ang=jnp.concatenate(
                    [data.qp.ang, jnp.expand_dims(state.qp.ang, axis=0)])),
        )
        return (state, normalizer_params, policy_params, key), data

    @jax.jit
    def run_eval(state, key, policy_params,
                 normalizer_params) -> Tuple[EvalEnvState, PRNGKey]:
        normalizer_params = jax.tree_map(lambda x: x[0], normalizer_params)
        policy_params = jax.tree_map(lambda x: x[0], policy_params)
        (state, _, _, key), _ = jax.lax.scan(
            generate_dmp_unroll_eval, (state, normalizer_params, policy_params, key), (),
            length=episode_length // action_repeat // dmp_unroll_length)
        return state, key

    def update_model(carry, data):
        optimizer_state, params, key = carry
        key, key_loss = jax.random.split(key)
        loss_grad, metrics = grad_loss(params, data, key_loss)
        loss_grad = jax.lax.pmean(loss_grad, axis_name='i')
        params_update, optimizer_state = optimizer.update(
            loss_grad,
            optimizer_state,
        )
        params = optax.apply_updates(params, params_update)
        return (optimizer_state, params, key), metrics

    def minimize_epoch(carry, unused_t):
        optimizer_state, params, data, key = carry
        key, key_perm, key_grad = jax.random.split(key, 3)
        permutation = jax.random.permutation(key_perm, data.obs.shape[1])

        def convert_data(data, permutation):
            data = jnp.take(data, permutation, axis=1, mode='clip')
            data = jnp.reshape(
                data, [data.shape[0], num_minibatches, -1] + list(data.shape[2:]))
            data = jnp.swapaxes(data, 0, 1)
            return data

        ndata = jax.tree_map(lambda x: convert_data(x, permutation), data)
        (optimizer_state, params, _), metrics = jax.lax.scan(
            update_model, (optimizer_state, params, key_grad),
            ndata,
            length=num_minibatches)
        return (optimizer_state, params, data, key), metrics

    def run_epoch(carry: Tuple[TrainingState, envs.State], unused_t):
        training_state, state = carry
        key_minimize, key_generate_unroll, new_key = jax.random.split(
            training_state.key, 3)
        (state, _, _, _), data = jax.lax.scan(
            generate_unroll, (state, training_state.normalizer_params,
                              training_state.params['policy'],
                              key_generate_unroll), (),
            length=batch_size * num_minibatches // num_envs)
        # make unroll first
        data = jax.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
        # data: (batch_size, unroll_length + 1, [obs_size])
        data = jax.tree_map(
            lambda x: jnp.reshape(x, [x.shape[0], -1] + list(x.shape[3:])), data)
        # data: (unroll_length + 1, batch_size, [obs_size])

        # Update normalization params and normalize observations.
        normalizer_params = obs_normalizer_update_fn(
            training_state.normalizer_params, data.obs[:-1])
        data = data.replace(
            obs=obs_normalizer_apply_fn(normalizer_params, data.obs))

        (optimizer_state, params, _, _), metrics = jax.lax.scan(
            minimize_epoch, (training_state.optimizer_state, training_state.params,
                             data, key_minimize), (),
            length=num_update_epochs)

        new_training_state = TrainingState(
            optimizer_state=optimizer_state,
            params=params,
            normalizer_params=normalizer_params,
            key=new_key)
        return (new_training_state, state), metrics

    num_epochs = num_timesteps // (
            batch_size * unroll_length * num_minibatches * action_repeat)

    def _minimize_loop(training_state, state):
        synchro = pmap.is_replicated(
            (training_state.optimizer_state, training_state.params,
             training_state.normalizer_params),
            axis_name='i')
        (training_state, state), losses = jax.lax.scan(
            run_epoch, (training_state, state), (),
            length=num_epochs // log_frequency)
        losses = jax.tree_map(jnp.mean, losses)
        return (training_state, state), losses, synchro

    minimize_loop = jax.pmap(_minimize_loop, axis_name='i')

    training_state = TrainingState(
        optimizer_state=optimizer_state,
        params=init_params,
        key=jnp.stack(jax.random.split(key, local_devices_to_use)),
        normalizer_params=normalizer_params)
    training_walltime = 0
    eval_walltime = 0
    sps = 0
    eval_sps = 0
    losses = {}
    state = first_state
    metrics = {}

    for it in range(log_frequency + 1):
        logging.info('starting iteration %s %s', it, time.time() - xt)
        t = time.time()

        if process_id == 0:
            eval_state, key_debug = (
                run_eval(eval_first_state, key_debug,
                         training_state.params['policy'],
                         training_state.normalizer_params))
            eval_state.completed_episodes.block_until_ready()
            eval_walltime += time.time() - t
            eval_sps = (episode_length * eval_first_state.core.reward.shape[0] /
                        (time.time() - t))
            avg_episode_length = (
                    eval_state.completed_episodes_steps / eval_state.completed_episodes)
            metrics = dict(
                dict({
                    f'eval/episode_{name}': value / eval_state.completed_episodes
                    for name, value in eval_state.completed_episodes_metrics.items()
                }),
                **dict({
                    f'losses/{name}': jnp.mean(value)
                    for name, value in losses.items()
                }),
                **dict({
                    'eval/completed_episodes': eval_state.completed_episodes,
                    'eval/avg_episode_length': avg_episode_length,
                    'speed/sps': sps,
                    'speed/eval_sps': eval_sps,
                    'speed/training_walltime': training_walltime,
                    'speed/eval_walltime': eval_walltime,
                    'speed/timestamp': training_walltime,
                }))
            logging.info(metrics)
            if log_save_params:
                save_params(
                    (normalizer_params, policy_params),
                    logger.get_logdir_path(cfg),
                    str(int(training_state.normalizer_params[0][0])),
                )
            if progress_fn:
                progress_fn(int(training_state.normalizer_params[0][0]) * action_repeat,
                            metrics)

        if it == log_frequency:
            break

        t = time.time()
        previous_step = training_state.normalizer_params[0][0]
        # optimization
        (training_state, state), losses, synchro = minimize_loop(training_state, state)
        assert synchro[0], (it, training_state)
        jax.tree_map(lambda x: x.block_until_ready(), losses)
        sps = ((training_state.normalizer_params[0][0] - previous_step) /
               (time.time() - t)) * action_repeat
        training_walltime += time.time() - t

    # To undo the pmap.
    normalizer_params = jax.tree_map(lambda x: x[0],
                                     training_state.normalizer_params)
    policy_params = jax.tree_map(lambda x: x[0],
                                 training_state.params['policy'])

    logging.info('total steps: %s', normalizer_params[0] * action_repeat)

    inference_fn = make_inference_fn(
        cfg,
        core_env,
        core_env.sys.config.dt,
        normalize_observations,
    )

    params = normalizer_params, policy_params

    pmap.synchronize_hosts()

    return (inference_fn, params, metrics, (obs_normalizer_apply_fn, policy_model, parametric_action_distribution))


def make_inference_fn(cfg, core_env, dt, normalize_observations):
    """Creates params and inference function for the PPO w/ NDP agent."""
    _, obs_normalizer_apply_fn = normalization.make_data_and_apply_fn(
        core_env.observation_size, normalize_observations)
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=core_env.action_size)

    policy_model = NDP(
        cfg,
        core_env,
        parametric_action_distribution.param_size,
        dt,
    )

    def inference_fn(params, state, obs, key):
        normalizer_params, policy_params = params
        obs = obs_normalizer_apply_fn(normalizer_params, obs)
        action = parametric_action_distribution.sample(
            policy_model.apply(policy_params, state.qp, obs), key)
        return action

    return inference_fn


def save_params(params: Params, logdir: str, name: str):
    params_dir = logdir.joinpath("params")
    params_dir.mkdir(exist_ok=True)
    params_file = params_dir.joinpath("{}.flax".format(name))

    param_bytes = flax.serialization.to_bytes(params)

    with open(params_file, "wb") as f:
        f.write(param_bytes)
