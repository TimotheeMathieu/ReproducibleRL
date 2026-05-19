import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys


fig,axes = plt.subplots(1,4, figsize=(12,2.5),sharex=True, sharey=True)
axes = axes.reshape([2,2])
for i, perturbation in enumerate(["weight", "action"]):
    for j, initialization in enumerate(["fixed", "random"]):
        df = pd.read_csv(f"{sys.argv[1]}/results_sensibility_{perturbation}_{initialization}_mujoco.csv", index_col=0)
        df["relative discrepancy"] = np.abs(df["base_reward"] - df["perturbed_reward"])/ df["base_reward"]

        ax = axes[i,j]
        sns.lineplot(df, x = "perturbation", y="relative discrepancy", errorbar = ("pi", 100), label="min-max", ax = ax)
        sns.lineplot(df, x = "perturbation", y="relative discrepancy", errorbar = ("pi", 90), label="90% pi", ax = ax)
        ax.set_xscale("log")
        ax.set_title(f"perturbation on {perturbation},\n init {initialization}", fontsize=11)
        ax.set_ylim(0,1)
fig.savefig(f"{sys.argv[2]}/sensibility_mujoco_pi.pdf", bbox_inches="tight")
