# Test script for reproducibility 
# test there is no randomness in Atari


import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import ale_py
import numpy as np

COMMON_SEED = 42

gym.register_envs(ale_py)

for j in range(100):
    env = make_atari_env('BreakoutNoFrameskip-v4', n_envs=1)
    # Stack 4 frames (to represent the dynamics)
    env = VecFrameStack(env, n_stack=4)
    model = PPO.load("pretrain_model/ppo/BreakoutNoFrameskip-v4_1/BreakoutNoFrameskip-v4.zip",
                    verbose=1, device="cpu", env=env,
                    seed=COMMON_SEED)
    s = model.get_env().reset()
    # env.action_space.seed(42)
    done = False
    list_s = [s]
    while not done:
        action = model.predict(s, deterministic=True)[0]
        s, _, done, _ = model.get_env().step(action)
        list_s.append(s)
    list_s1 = np.array(list_s)

    gym.register_envs(ale_py)
    env = make_atari_env('BreakoutNoFrameskip-v4', n_envs=1)
    # Stack 4 frames (to represent the dynamics)
    env = VecFrameStack(env, n_stack=4)
    model = PPO.load("pretrain_model/ppo/BreakoutNoFrameskip-v4_1/BreakoutNoFrameskip-v4.zip",
                    verbose=1, device="cpu", env=env,
                    seed=COMMON_SEED)
    s = model.get_env().reset()
    done = False
    list_s = [s]
    while not done:
        action = model.predict(s, deterministic=True)[0]
        s, _, done, _ = model.get_env().step(action)
        list_s.append(s)

    list_s2 = np.array(list_s)
    print(len(list_s2), len(list_s1))
    for i, s in enumerate(list_s1):
        assert np.array_equal(list_s2[i], s)