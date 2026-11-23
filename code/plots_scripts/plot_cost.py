import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns
import sys
import os

"""
Create figures to show the time taken for training on different environments, for different workflows.
data is stored in results_cost_repro_archlinux.txt, with columns workflow, time, env, reward.
"""

print("--- starting plot_cost.py : generating figures for the time taken for training on different environments, for different workflows ---")

df = pd.read_csv(f"{sys.argv[2]}/results_cost_repro_archlinux.txt", names=["workflow", "time", "env", "reward"])
df=  df.sort_values(by="workflow")
df.loc[df["workflow"]=="guix-avx2", "workflow"] = "guix_no_avx2"
fig, axes = plt.subplots(1,2, figsize=(8,4))
plt.subplots_adjust(wspace=0.3)
sns.boxplot(df.loc[df["env"]=="Acrobot-v1"], x="workflow", y="time", ax=axes[0])
axes[0].set_title("Time for training on Acrobot-v1")
sns.boxplot(df.loc[df["env"]=="Hopper-v5"], x="workflow", y="time", ax=axes[1])
axes[1].set_title("Time for training on Hopper-v5")

n_uniq = len(df[df["workflow"]=="guix"])
fig.suptitle(f"Time in seconds, aggregated over {n_uniq} repetitions")

output_path = os.path.join(sys.argv[1], "cost_reproducibility.pdf")
fig.savefig(output_path)
print("saved figure to : ", output_path)