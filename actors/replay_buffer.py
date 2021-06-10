import numpy as np
import ray

@ray.remote
class ReplayBuffer:
    def __init__(self, replay_buffer_size):
        self.replay_buffer_size = replay_buffer_size
        self.buffer = {}
        self.game_id = 0

    def add(self, trajectory):
        self.buffer[self.game_id] = trajectory
        self.game_id = (self.game_id + 1) % self.replay_buffer_size
    
    def sample(self, batch_size, K, n):
        obs_batch, act_batch, reward_batch, next_obs_batch = [], [], [], []
        selected_games = np.random.choice(len(self.buffer), batch_size)

        for game_id in selected_games:
            obs_batch_per_id, act_batch_per_id, reward_batch_per_id, next_obs_batch_per_id = [], [], [], []
            start_idx = np.random.choice(len(self.buffer[game_id]))
            episode_per_game_id = self.buffer[game_id]
            len_episode_per_game_id = len(episode_per_game_id)
            
            for current_idx in range(start_idx, start_idx + K + n):
                if current_idx <= len_episode_per_game_id:
                    obs_batch_per_id.append(episode_per_game_id[current_idx][0])
                    act_batch_per_id.append(episode_per_game_id[current_idx][1])
                    reward_batch_per_id.append(episode_per_game_id[current_idx][2])
                    next_obs_batch_per_id.append(episode_per_game_id[current_idx][3])
                else:  
                    # if the current_idx is beyond episode's terminationn, then return obs, act, reward such that it appears to be an absorbing state
                    obs_batch_per_id.append(episode_per_game_id[len_episode_per_game_id - 1][0])
                    act_batch_per_id.append()
                    reward_batch_per_id.append(0.)
                    next_obs_batch_per_id.append(episode_per_game_id[len_episode_per_game_id - 1][3])

            obs_batch.append(np.array(obs_batch_per_id))
            act_batch.append(np.array(act_batch_per_id))
            reward_batch.append(np.array(reward_batch_per_id))
            next_obs_batch.append(np.array(next_obs_batch_per_id))
        
        return obs_batch, act_batch, reward_batch, next_obs_batch
