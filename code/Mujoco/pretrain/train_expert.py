import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
import torch as th
from stable_baselines3.common.env_util import make_vec_env

th.use_deterministic_algorithms(True)
th.set_default_tensor_type(th.FloatTensor)
# Set the random seed for stable-baselines3 and pytorch
set_random_seed(42)
env = make_vec_env("Hopper-v5", n_envs=1, seed=42, wrapper_class=TimeLimit, wrapper_kwargs={'max_episode_steps':1000})
model = PPO('MlpPolicy', env, verbose=1, seed=1)
model.learn(1e5)
model.save('Hopper-v5')
