import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from read_data import read_data_from_dir, read_minatar_from_dir
import os

"""
Create figures to show the variations of the rewards obtained on different environments, for different workflows, and for different computers.
"""
print("--- starting plot_g5k.py : generating figures for the variations of the rewards obtained on different environments, for different workflows, and for different computers ---")

df_g5k = pd.concat([pd.read_csv("data_g5k_hardware/grid5000_default_queue_resources.csv"),
                    pd.read_csv("data_g5k_hardware/grid5000_abaca_queue_resources.csv")], ignore_index = True)


envs = ["Acrobot", "Mujoco", "Atari"]
venv_manager = ["guix", "guix_with_masked_AVX", "pip_frozen",  "conda_frozen", "pip_unfrozen", "conda_unfrozen"]

def make_reward_fig(dfenvenv, ax):
    df = dfenvenv
    ax.set_title(venv)

    for name in np.sort(df["computer_name"].unique()):
        ax.plot(
            np.cumsum(df.loc[df["computer_name"]==name, "l"]),
            df.loc[df["computer_name"]==name, "r"],
            alpha = 0.6
        )
        ax.set_xlabel("Samples")
        ax.set_ylabel("Online episodic returns")

df = read_data_from_dir(sys.argv[2])

df["r"] = df["r"].astype(np.float64)

for env in envs:
    for venv in venv_manager:
        dfs = []
        fig, ax = plt.subplots(figsize=(5,3))
        dfenvenv = df.loc[(df["env"] == env) & (df["venv"] == venv)]
        make_reward_fig(dfenvenv, ax)
        output_path = os.path.join(sys.argv[1], f"rewards_{env}_{venv}.pdf")
        fig.savefig(output_path, bbox_inches="tight")
        print("saved figure to : ", output_path)

