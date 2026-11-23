from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_util import make_vec_env
import torch as th
import socket
import os
import argparse

th.use_deterministic_algorithms(True)
th.set_default_tensor_type(th.FloatTensor)

hostname = socket.gethostname()

# Set number of timesteps and outputs from arguments or from defaults
parser = argparse.ArgumentParser(
    prog='run_ppo',
    description='Run ppo on Acrobot and get computation time')

parser.add_argument('output', type=str, default="results") 
args = parser.parse_args()

output_name = args.output

for seeds in range(50):
    for gae_lambda in [0.5, 0.9, 0.99]:
        for n_steps in [512, 1024, 2048]:
            for n_epochs in [16, 64, 256]:
                # Set the random seed for stable-baselines3 and pytorch
                set_random_seed(seeds)

                log_dir = f"{output_name}-PPO-seed-{seeds}-gs-{n_epochs}-ns-{n_steps}-gae-{gae_lambda}/Acrobot-v1/{hostname}"
                print("Computing", log_dir)
                if not os.path.exists(log_dir):
                    env = make_vec_env("Acrobot-v1", monitor_dir=log_dir, n_envs=1, seed=seeds)

                    model = PPO("MlpPolicy", env, verbose=0, n_steps=n_steps, gae_lambda=gae_lambda, n_epochs=n_epochs, device='cpu')
                    model.save(log_dir + '-step-0')
                    for i in range(1, 51):
                        model.learn(total_timesteps=2048, reset_num_timesteps=False)
                        model.save(log_dir + f'after-{int(i * 2048)}-new-spls')


