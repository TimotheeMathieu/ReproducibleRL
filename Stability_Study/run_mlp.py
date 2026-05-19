import gymnasium as gym
from mlp import play
import numpy as np
import pandas as pd
import argparse
import os 
from tqdm import tqdm

parser = argparse.ArgumentParser(
    prog='run_ppo',
    description='Run ppo on Acrobot and get computation time')

parser.add_argument('output', default="results") 
parser.add_argument('--n-seeds', help='number of seeds', type=int,  default=1000) 
args = parser.parse_args()

def get_reward(perturbation=1e-6,  seed=None):
    # Create environment
    env = gym.make('Hopper-v5', render_mode="rgb_array")

    # Reset environment
    observation, info = env.reset(seed=seed)
    done = False
    total_reward = 0
    iteration_n = 0
    rewards = []

    # Run the policy
    while not done:
        iteration_n += 1
        action = play(observation)
        action = np.array(action)*(1-perturbation)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

    env.close()
    return total_reward

seeds = range(args.n_seeds)
perturbations = np.logspace(-6, -1, num=20)
df = pd.DataFrame()
for s in tqdm(seeds):
    for p in perturbations:
        df = pd.concat([df, pd.DataFrame({"seed":[s],
                                          "perturbation":[p],
                                          "base_reward":[get_reward(0,s)],
                                          "perturbed_reward":[get_reward(p,s)]
                                          })],ignore_index=True)

df.to_csv(os.path.join(args.output, "results_sensibility_mujoco.csv"))
