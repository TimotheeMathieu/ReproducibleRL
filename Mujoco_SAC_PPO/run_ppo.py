import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
import torch as th
import socket
import argparse

th.use_deterministic_algorithms(True)
th.set_default_tensor_type(th.FloatTensor)

hostname = socket.gethostname()

parser = argparse.ArgumentParser(
    prog='run_ppo',
    description='Run ppo on Mujoco')

parser.add_argument('output', default="results") 
parser.add_argument('--environment', default="Hopper-v5") 
parser.add_argument('--n-seeds', help='number of timesteps', type=int,  default=50) 
parser.add_argument('--pretrained', help='Use pretrained model', action='store_true')

args = parser.parse_args()
output_name = args.output
n_seeds = args.n_seeds
env = args.environment

def run_seed(seed):
    # Set the random seed for stable-baselines3 and pytorch
    set_random_seed(seed)

    # Save the rollouts data observed during training
    # Inspired from https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#checkpointcallback
    if args.pretrained:
        log_dir = f"{output_name}/ppo_logs/{env}_pretrained/{hostname}"
    else:
        log_dir = f"{output_name}/ppo_logs/{env}/{hostname}"
    # Train the model on a few steps
    model = PPO("MlpPolicy", env, verbose=1, seed=seed, device='cpu', tensorboard_log=log_dir)
    if args.pretrained:
        n_steps = 50_000
        model.load("../Mujoco/pretrain_model/ppo-Hopper-v3.zip")
    else:
        n_steps = 1_000_000
    model.learn(total_timesteps=n_steps, log_interval=10, tb_log_name='seed_'+str(seed))
    model.save(log_dir+f'/final_seed_{seed}')
    episode_rewards = []
    for ep in range(100):
        model.env._seeds[0] = ep
        s = model.env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = model.predict(s, deterministic=True)[0]
            s, r, done, _ = model.env.step(action)
            episode_reward += r[0]
        episode_rewards.append(episode_reward)
    np.save(log_dir+f"/final_rewards_seed_{seed}", episode_rewards)

for seed in range(n_seeds):
    run_seed(seed)
