import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
import numpy as np
from stable_baselines3.common.utils import set_random_seed
import torch as th
import argparse
import socket
import os, sys
from stable_baselines3.common.env_util import make_vec_env
import minatar
from minatar.gym import register_envs
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
register_envs()

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
log_dir = f"{output_name}/MinatarBreakout_MLP/{hostname}"
def make_env():
    env = Monitor(gym.make("MinAtar/Breakout-v1"), filename = log_dir+"/0.monitor.csv")
    return env
env = DummyVecEnv([make_env])
model = PPO.load(
    "pretrain/MinatarBreakout_mlp.zip",
    verbose=1,
    seed=COMMON_SEED,
    device="cpu",
    env=env,
)

if n_timesteps is None:
    n_timesteps = 5000

model.learn(total_timesteps=n_timesteps, reset_num_timesteps=True)

class MinAtarCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            sample = th.as_tensor(
                observation_space.sample()[None]
            ).float()
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        return self.linear(self.cnn(observations))

class MinAtarImageWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        h, w, c = env.observation_space.shape

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(h, w, c),
            dtype=np.uint8,
        )

    def observation(self, obs):
        return obs.astype(np.uint8) * 255

log_dir = f"{output_name}/MinatarBreakout_CNN/{hostname}"

def make_env():
    env = Monitor(gym.make("MinAtar/Breakout-v1"), filename = log_dir+"/0.monitor.csv")
    env = MinAtarImageWrapper(env)
    return env
env = DummyVecEnv([make_env])

policy_kwargs = dict(
    features_extractor_class=MinAtarCNN,
    features_extractor_kwargs=dict(features_dim=64),
)

# Convert from HWC -> CHW
env = VecTransposeImage(env)
model = PPO.load(
    "pretrain/MinatarBreakout_cnn.zip",
    verbose=1,
    seed=COMMON_SEED,
    device="cpu",
    env=env,
)

model.learn(total_timesteps=n_timesteps, reset_num_timesteps=True)
