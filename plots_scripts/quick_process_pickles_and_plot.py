import numpy as np
import pandas as pd
import glob
import gzip
import pickle
import matplotlib.pyplot as plt

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

logdir = '../Atari/Atari_pip_frozen/results/BreakoutNoFrameskip-v4/sic-install-ubuntu.lille.inria.fr'
all_data = []
for cpu in [logdir]:
    rewards = []
    episode_starts = []
    try:
        data = pd.read_csv(logdir + "/0.monitor.csv", header=1)
    except FileNotFoundError:
        continue
    print(data)
    # Prepare data for this CPU
    all_data.append(pd.DataFrame({
        'computer_name': [cpu] * len(data['l']),
        'episode_step': np.cumsum(data['l']),
        'episode_rewards': data['r']
    }))

# Concatenate all data into a single DataFrame
df = pd.concat(all_data, ignore_index=True)
df.to_csv(f'test_process_pickles.csv', index=False)
plt.plot(df['episode_rewards'])
plt.savefig('test_process_pickles')
