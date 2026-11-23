from stable_baselines3.common.save_util import load_from_pkl
import os
import sys
import numpy as np
import pandas as pd
import glob
import gzip
import pickle

def sum_rewards_from_start_indices(episode_start_indices, rewards):
    """
    Sum step rewards between episode start indices.

    Args:
        episode_start_indices: list of ints (sorted), indices where episodes start
        rewards: list or 1D array of rewards per step

    Returns:
        List of total rewards per episode
    """
    n = len(rewards)
    episode_start_indices = list(episode_start_indices)

    if not episode_start_indices:
        return []
    if episode_start_indices[0] < 0 or episode_start_indices[-1] >= n:
        raise ValueError("Episode start indices out of bounds")

    episode_rewards = []

    for i, start in enumerate(episode_start_indices):
        end = episode_start_indices[i + 1] if i + 1 < len(episode_start_indices) else n
        episode_rewards.append(sum(rewards[start:end-1]))

    return episode_rewards

RAW_RESULTS_DIR=sys.argv[1] 
PROCESSED_RESULTS_DIR=sys.argv[2] 

envs = ["Acrobot", "Mujoco", "Atari"]
venv_manager = ["guix", "guix_with_masked_AVX", "pip_frozen", "pip_unfrozen", "conda_frozen", "conda_unfrozen"]
 
for env in envs:
    for venv in venv_manager:
    # Get all result folders for the experiment
        log_dir = [f for f in glob.glob(f"{RAW_RESULTS_DIR}/{env}/{env}_{venv}/*")][0]
        os.makedirs(f"{PROCESSED_RESULTS_DIR}/{env}/", exist_ok=True)
        cpu_folders = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d))]
        print(f"{env}_{venv}: processing experiments for {len(cpu_folders)} hostnames")
        all_data = []
        for cpu in cpu_folders:
            if env == 'Atari':
                try:
                    data = pd.read_csv(f"{log_dir}/{cpu}/0.monitor.csv", header=1)
                except FileNotFoundError:
                    continue
                all_data.append(pd.DataFrame({
                    'computer_name': [cpu] * len(data['l']),
                    'episode_step': np.cumsum(data['l']),
                    'episode_rewards': data['r']
                }))
            else:
                rewards = []
                episode_starts = []
                for seed_data in glob.glob(f"{log_dir}/{cpu}/rollouts_rollout_buffer_*_steps.pkl"):
                    try:
                        with gzip.open(seed_data, "rb") as filename:
                            rollout_data = pickle.load(filename)
                        rewards += list(rollout_data.rewards.flatten('F'))
                        episode_starts += list(rollout_data.episode_starts.flatten('F'))
                    except FileNotFoundError:
                        continue

                # Compute episode rewards and steps
                episode_steps = np.argwhere(episode_starts).squeeze()  # Start from second episode
                episode_rewards = sum_rewards_from_start_indices(episode_steps, rewards)
                # Prepare data for this CPU
                all_data.append(pd.DataFrame({
                    'computer_name': [cpu] * len(episode_rewards[:-1]),
                    'episode_step': episode_steps[:-1],
                    'episode_rewards': episode_rewards[:-1]
                }))

        # Concatenate all data into a single DataFrame
        df = pd.concat(all_data, ignore_index=True)
        df.to_csv(f'{PROCESSED_RESULTS_DIR}/{env}/{env}_{venv}.csv', index=False)



cpu_folders = [d for d in os.listdir(f"{RAW_RESULTS_DIR}/Acrobot_GPU/Acrobot-v1") if os.path.isdir(os.path.join(f"{RAW_RESULTS_DIR}/Acrobot_GPU/Acrobot-v1", d))]
print(f"Acrobot_GPU/: processing experiments for {len(cpu_folders)} hostnames")
all_data = []

for cpu in cpu_folders:
    rewards = []
    episode_starts = []
    for seed_data in np.sort(glob.glob(f"{RAW_RESULTS_DIR}/Acrobot_GPU/Acrobot-v1/{cpu}/rollouts_rollout_buffer_*_steps.pkl")):
        try:
            with gzip.open(seed_data, "rb") as filename:
                rollout_data = pickle.load(filename)
            rewards += list(rollout_data.rewards.flatten('F'))
            episode_starts += list(rollout_data.episode_starts.flatten('F'))
        except FileNotFoundError:
            continue

    # Compute episode rewards and steps
    episode_steps = np.argwhere(episode_starts).squeeze()  # Start from second episode
    episode_rewards = sum_rewards_from_start_indices(episode_steps, rewards)
    # Prepare data for this CPU
    all_data.append(pd.DataFrame({
                    'computer_name': [cpu] * len(episode_rewards[:-1]),
                    'episode_step': episode_steps[:-1],
                    'episode_rewards': episode_rewards[:-1]
                }))
df = pd.concat(all_data, ignore_index=True)
df.to_csv(f'../results/Acrobot_GPU.csv', index=False)
