import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import glob

envs = ["Acrobot", "Mujoco", "Atari"]
venv_manager = ["guix", "guix_with_masked_AVX", "pip_frozen",  "conda_frozen", "pip_unfrozen", "conda_unfrozen"]

def read_data_from_dir(results_dir):
    df = pd.DataFrame()
    for env in envs:
        for venv in venv_manager:
            print(env, venv)
            machine_dir = list(glob.glob(f"{results_dir}/{env}/{env}_{venv}/*"))
            machine_already_seen = []
            for machine in machine_dir:
                machine_name = machine.split("/")[-1]
                if machine_name not in machine_already_seen:
                    data = pd.read_csv(machine+"/0.monitor.csv", header=0)
                    data["env"] = env
                    data["venv"] = venv
                    data["computer_name"] = machine_name
                    data["episode_step"] = np.arange(len(data))
                    df = pd.concat([df,data ], ignore_index = True)
                    machine_already_seen.append(machine_name)
                else:
                    print("Double ", machine_name)
    return df



def read_unfrozen_from_dir(results_dir):
    df = pd.DataFrame()
    for env in envs:
        for venv in ["pip_unfrozen", "conda_unfrozen"]:
            machine_dir = list(glob.glob(f"{results_dir}/{env}/{env}_{venv}/*"))
            machine_already_seen = []
            for machine in machine_dir:
                machine_name = machine.split("/")[-1]
                if machine_name not in machine_already_seen:
                    data = pd.read_csv(machine+"/0.monitor.csv", header=0,skiprows=[0])
                    data["env"] = env
                    data["venv"] = venv
                    data["computer_name"] = machine_name
                    data["episode_step"] = np.arange(len(data))
                    df = pd.concat([df,data ], ignore_index = True)
                    machine_already_seen.append(machine_name)
                else:
                    print("Double ", machine_name)
    return df


def read_minatar_from_dir(results_dir):
    df = pd.DataFrame()
    for nn in ["MLP", "CNN"]:
        machine_dir = list(glob.glob(f"{results_dir}/Minatar/MinatarBreakout_{nn}/*"))
        machine_already_seen = []
        for machine in machine_dir:
            machine_name = machine.split("/")[-1].split("-")[0]
            if machine_name not in machine_already_seen:
                data = pd.read_csv(machine+"/0.monitor.csv", header=0,skiprows=[0])
                data["env"] = "Minatar"
                data["nn"] = nn
                data["computer_name"] = machine_name
                data["episode_step"] = np.arange(len(data))
                df = pd.concat([df,data ], ignore_index = True)
                machine_already_seen.append(machine_name)
            else:
                print("Double ", machine_name)
    return df

def read_acrobot_gpu_from_dir(results_dir):
    df = pd.DataFrame()
    machine_dir = list(glob.glob(f"{results_dir}/Acrobot_GPU/Acrobot-v1/*"))
    machine_already_seen = []
    for machine in machine_dir:
        machine_name = machine.split("/")[-1].split("-")[0]
        if machine_name not in machine_already_seen:
            data = pd.read_csv(machine+"/0.monitor.csv", header=0,skiprows=[0])
            data["env"] = "Acrobot GPU"
            data["computer_name"] = machine_name
            data["episode_step"] = np.arange(len(data))
            df = pd.concat([df,data ], ignore_index = True)
            machine_already_seen.append(machine_name)
        else:
            print("Double ", machine_name)
    return df
