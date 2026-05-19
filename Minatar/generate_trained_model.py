import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
import torch as th
import torch.nn as nn
import os
from minatar.gym import register_envs
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

register_envs()

seed = 1234
# Set the random seed for stable-baselines3 and pytorch
set_random_seed(seed)

os.makedirs("pretrain", exist_ok=True)

# Train the model on a few steps
model = PPO("MlpPolicy", "MinAtar/Breakout-v1", verbose=1, seed=seed, device='cpu')

model.learn(total_timesteps=500_000)
model.save("pretrain/MinatarBreakout_mlp")

# Train the model on a few steps


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

def make_env():
    env = gym.make("MinAtar/Breakout-v1")
    env = MinAtarImageWrapper(env)
    return env
env = DummyVecEnv([make_env])

policy_kwargs = dict(
    features_extractor_class=MinAtarCNN,
    features_extractor_kwargs=dict(features_dim=64),
)

# Convert from HWC -> CHW
env = VecTransposeImage(env)

model = PPO("CnnPolicy", env, verbose=1, seed=seed, device='cpu', policy_kwargs=policy_kwargs,)

model.learn(total_timesteps=200_000)
model.save("pretrain/MinatarBreakout_cnn")
