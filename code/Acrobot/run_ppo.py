import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.save_util import load_from_pkl
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
import torch as th
import socket
import os, sys
import argparse

COMMON_SEED = 42

th.use_deterministic_algorithms(True)
th.set_default_tensor_type(th.FloatTensor)

hostname = socket.gethostname()

# Set the random seed for stable-baselines3 and pytorch
set_random_seed(COMMON_SEED)

# Set number of timesteps and outputs from arguments or from defaults
parser = argparse.ArgumentParser(
    prog='run_ppo',
    description='Run ppo on Acrobot and get computation time')

parser.add_argument('output', type=str, default="results") 
parser.add_argument('--n-timesteps', help='number of timesteps', type=int,  default=50000) 
args = parser.parse_args()

output_name = args.output
n_timesteps = args.n_timesteps


log_dir = f"{output_name}/Acrobot-v1/{hostname}"
env = make_vec_env("Acrobot-v1", monitor_dir=log_dir, n_envs=1, seed=COMMON_SEED)

model = PPO("MlpPolicy", env, verbose=0, seed=COMMON_SEED, device="cpu")
model.learn(total_timesteps=n_timesteps)
