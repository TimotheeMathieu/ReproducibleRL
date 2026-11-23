import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
import torch as th
import socket
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import argparse
import ale_py

COMMON_SEED = 42

gym.register_envs(ale_py)
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
parser.add_argument('--n-timesteps', help='number of timesteps', type=int, default=None) 
args = parser.parse_args()

n_timesteps = args.n_timesteps
output_name = args.output

# Save the rollouts data observed during training
# Inspired from https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#checkpointcallback
log_dir = f"{output_name}/BreakoutNoFrameskip-v4/{hostname}"

env = make_atari_env('BreakoutNoFrameskip-v4', n_envs=1, seed=COMMON_SEED, monitor_dir=log_dir)
# Stack 4 frames (to represent the dynamics)
env = VecFrameStack(env, n_stack=4)
# Train the model on a few steps
model = PPO.load("../pretrain_model/ppo/BreakoutNoFrameskip-v4_1/BreakoutNoFrameskip-v4.zip",
                 verbose=1, device="cpu", env=env,
                 seed=COMMON_SEED)

if n_timesteps is None:
    n_timesteps =  20000

model.learn(total_timesteps=n_timesteps) 
