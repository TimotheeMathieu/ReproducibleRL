import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
import torch as th
import argparse
import socket
import os, sys
from stable_baselines3.common.env_util import make_vec_env

COMMON_SEED = 42

th.use_deterministic_algorithms(True)
th.set_default_tensor_type(th.FloatTensor)

hostname = socket.gethostname()

# Set the random seed for stable-baselines3 and pytorch
set_random_seed(COMMON_SEED)

# Set number of timesteps and outputs from arguments or from defaults
parser = argparse.ArgumentParser(
    prog="run_ppo", description="Run ppo on Acrobot and get computation time"
)

parser.add_argument("output", type=str, default="results")
parser.add_argument("--n-timesteps", help="number of timesteps", type=int, default=None)
args = parser.parse_args()

n_timesteps = args.n_timesteps
output_name = args.output

# Save the rollouts data observed during training
# From https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#checkpointcallback
log_dir = f"{output_name}/Hopper-v5/{hostname}"
env = make_vec_env("Hopper-v5", n_envs=1, seed=COMMON_SEED, wrapper_class=TimeLimit, wrapper_kwargs={'max_episode_steps':1000}, monitor_dir=log_dir)
model = PPO.load(
    "../pretrain/Hopper-v5.zip",
    verbose=1,
    seed=COMMON_SEED,
    device="cpu",
    env=env,
)

if n_timesteps is None:
    n_timesteps = 20000

model.learn(total_timesteps=n_timesteps, reset_num_timesteps=True)
