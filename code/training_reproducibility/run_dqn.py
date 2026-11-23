from stable_baselines3 import DQN
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
    prog='run_pdqn',
    description='Run dqn on Acrobot and get computation time')

parser.add_argument('output', type=str, default="results") 
args = parser.parse_args()

output_name = args.output

for seeds in range(50):
    for gradient_steps in [1, 10, 100]:
        for train_freq in [10, 100, 1000]:
            for target_update in [1, 10, 100]:
                # Set the random seed for stable-baselines3 and pytorch
                set_random_seed(seeds)

                log_dir = f"{output_name}-DQN-seed-{seeds}-gs-{gradient_steps}-tf-{train_freq}-tu-{target_update}/Acrobot-v1/{hostname}"
                print("Computing", log_dir)
                if not os.path.exists(log_dir):
                    env = make_vec_env("Acrobot-v1", monitor_dir=log_dir, n_envs=1, seed=seeds)

                    model = DQN("MlpPolicy", env, verbose=0, train_freq=train_freq, gradient_steps=gradient_steps, target_update_interval=int(target_update * train_freq), seed=seeds, device="cpu", learning_starts=100_000, exploration_fraction=1, exploration_final_eps=0.3, exploration_initial_eps=0.3, buffer_size=100_000)
                    model.learn(total_timesteps=100_000, reset_num_timesteps=True)
                    model.save(log_dir + '-step-0')

                    periods = 100_000 // train_freq
                    for i in range(1, 101):
                        model.learn(total_timesteps=1000, reset_num_timesteps=False)
                        model.save(log_dir + f'after-{int(i * 1000)}-new-spls')


