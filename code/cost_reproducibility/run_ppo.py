import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.save_util import load_from_pkl
from stable_baselines3.common.utils import set_random_seed
import torch as th
import socket
import os, sys
import time

COMMON_SEED = 42

import argparse

parser = argparse.ArgumentParser(
    prog='run_ppo',
    description='Run ppo on Acrobot and get computation time')

parser.add_argument('workflow') 
parser.add_argument('output_folder', default="../results") 
parser.add_argument('--deterministic', action='store_true')
parser.add_argument('--n-timesteps', help='number of timesteps', type=int, default=None) 
args = parser.parse_args()

n_timesteps = args.n_timesteps
output_folder = args.output_folder
hostname = socket.gethostname()
output_file = output_folder + f"/results_cost_repro_{hostname}.txt"

# Set the random seed for stable-baselines3 and pytorch
set_random_seed(COMMON_SEED)

if args.deterministic:
    th.use_deterministic_algorithms(True)

for env in ["Acrobot-v1", "Hopper-v5"]:
    print("Environment: ", env)
    if n_timesteps is None:
        n_timesteps = 50_000 if env == "Acrobot-v1" else 1_000_000

    beg_time = time.time()
    # Train the model on a few steps
    model = PPO("MlpPolicy", env, verbose=0, seed=COMMON_SEED, device="cpu")
    model.learn(total_timesteps=n_timesteps)
    s = model.env.reset()
    done = False
    episode_reward = 0
    while not done:
        action = model.predict(s, deterministic=True)[0]
        s, r, done, _ = model.env.step(action)
        episode_reward += r[0]
    end_time = time.time()

    with open(output_file, "a") as myfile:
        myfile.write(args.workflow+", "+str(end_time-beg_time)+","+env+","+str(episode_reward)+"\n")
