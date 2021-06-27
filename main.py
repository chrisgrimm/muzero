import os
import time

import gym
import jax.random as jrng
import numpy as np
import optax
import ray

import common
import eval
from actors import parallel_actor
from environments.vec_env import atari_wrappers
from networks import muzero_functions, jitted_muzero_functions
from networks import muzero_def
from replay_buffers import trajectory_replay_buffer as trb
from replay_buffers.trajectory_replay_buffer import TrajectoryReplayBuffer, ReplaySpec


def muzero_wrap_atari(env_id, eval=False):
    env = gym.make(env_id)
    env = atari_wrappers.NoopResetEnv(env, noop_max=30)
    if not eval:
        env = atari_wrappers.EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = atari_wrappers.FireResetEnv(env)
    env = atari_wrappers.WarpFrame(env, height=96, width=96, grayscale=False)
    return env


def get_temperature(
        ts: int
) -> float:
    if ts < 500_000:
        return 1.0
    elif ts < 750_000:
        return 0.5
    else:
        return 0.25
    

def main():



    config = {
        'gamma': 0.997,
        'num_stack': 32,
        'obs_shape': (96, 96, 3),
        'embedding_shape': (6, 6, 256),
        'num_actions': 18, #?
        'num_simulations': 5,
        'model_rollout_length': 5,
        'env_rollout_length': 10,
        'update_actor_params_every': 1000,
        'update_temperature_every': 1000,
        'train_agent_every': 1,
        'batch_size': 32,
        'buffer_capacity': 1_000_000,
        'seed': 1234,
        'num_cat': 601,
        'cat_min': -300,
        'cat_max': 300,
        'learning_rate': 0.00025, # TODO this needs to be a schedule
        'num_actors': 32,
        'num_training_steps': 1_000_000,
        'min_buffer_length': 1_000,
        'env_name': 'PongNoFrameskip-v4',
        'adam_eps': 0.01 / 32,
        'eval_every': 100_000,
    }

    forward_frames = config['env_rollout_length'] + config['model_rollout_length'] + 1
    backward_frames = config['num_stack'] - 1

    # construct buffer.
    buffer_spec = [
        ReplaySpec('obs', config['obs_shape'], dtype=np.uint8, on_reset=True, oob_handling=trb.oob_identity),
        ReplaySpec('search_v', (), dtype=np.float32, on_reset=True, oob_handling=trb.oob_identity),
        ReplaySpec('search_pi', (config['num_actions'],), dtype=np.float32, on_reset=True,
                   oob_handling=trb.oob_identity),
        ReplaySpec('a', (), dtype=np.uint8, on_reset=False,
                   oob_handling=trb.make_oob_random_action(config['num_actions'])),
        ReplaySpec('r', (), dtype=np.float32, on_reset=False, oob_handling=trb.oob_zero),
        ReplaySpec('done', (), dtype=np.bool, on_reset=False, oob_handling=trb.oob_only_reset_at_end_of_traj),
    ]

    buffer = TrajectoryReplayBuffer(config['buffer_capacity'], buffer_spec, use_priority=True)

    key = jrng.PRNGKey(config['seed'])
    key, muzero_init_key, runner_init_key, eval_key = jrng.split(key, 4)

    muzero_params, muzero_comps = muzero_def.init_muzero(
        key=muzero_init_key,
        embed=muzero_functions.embed,
        reward=muzero_functions.reward,
        value=muzero_functions.value,
        policy=muzero_functions.policy,
        dynamics=muzero_functions.dynamics,
        config=config
    )

    optimizer = optax.adam(config['learning_rate'], eps=config['adam_eps'])
    opt_state = optimizer.init(muzero_params)

    muzero_train_fn = jitted_muzero_functions.make_train_function(muzero_comps, optimizer, config)


    pa_handle = parallel_actor.init_runner(
        config['num_actors'],
        lambda: muzero_wrap_atari(config['env_name'], eval=False),
        muzero_params,
        lambda: jitted_muzero_functions.make_actor(muzero_comps, config),
        get_temperature(0),
        runner_init_key,
        config
    )

    eval_env = muzero_wrap_atari(config['env_name'], eval=True)
    eval_actor = jitted_muzero_functions.make_actor(muzero_comps, config)

    ts = 1
    while ts < config['num_training_steps'] + 1:
        trajectories, pa_handle = parallel_actor.get_if_ready(pa_handle)
        if len(trajectories) > 0:
            traj_return = np.mean([eval.get_return(traj) for traj in trajectories])
            print(ts, 'traj_return!', traj_return)

        parallel_actor.feed_buffer(trajectories, buffer, config)

        if len(buffer) < config['min_buffer_length']:
            # give the buffer some more time.
            time.sleep(1)
            continue

        # update muzero params for actors
        if ts % config['update_actor_params_every'] == 0:
            pa_handle = parallel_actor.update_muzero_params(pa_handle, muzero_params)

        # update action selection temperature
        if ts % config['update_temperature_every'] == 0:
            temp = get_temperature(ts)
            pa_handle = parallel_actor.update_temperature(pa_handle, temp)

        if ts % config['train_agent_every'] == 0:
            samples = buffer.sample_traj(config['batch_size'], (-backward_frames, forward_frames))
            loss, priorities, muzero_params, opt_state = muzero_train_fn(
                muzero_params, opt_state,
                samples['obs'], samples['a'], samples['r'], samples['search_pi'],
                samples['search_v'], samples['importance_weights'])
            buffer.update_priorities(samples['indices'], priorities)
            print(ts, 'loss!', loss)

        if ts % config['eval_every'] == 0:
            eval_key, new_eval_key = jrng.split(eval_key)
            avg_return = eval.evaluate_agent(eval_env, eval_actor, new_eval_key, muzero_params, config)
            print(ts, 'eval!', avg_return)
        ts += 1






if __name__ == '__main__':

    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['RAY_BACKEND_LOG_LEVEL'] = 'error'

    ray.init()
    try:
        main()
    finally:
        ray.shutdown()

