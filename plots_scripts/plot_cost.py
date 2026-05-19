import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns
import sys

df = pd.read_csv(f"{sys.argv[2]}/results_cost_repro_archlinux.txt", names=["workflow", "time", "env", "reward"])
df=  df.sort_values(by="workflow")

fig, axes = plt.subplots(1,2, figsize=(8,4))
plt.subplots_adjust(wspace=0.3)
sns.boxplot(df.loc[df["env"]=="Acrobot-v1"], x="workflow", y="time", ax=axes[0])
axes[0].set_title("Time for training on Acrobot-v1")
sns.boxplot(df.loc[df["env"]=="Hopper-v5"], x="workflow", y="time", ax=axes[1])
axes[1].set_title("Time for training on Hopper-v5")

n_uniq = len(df[df["workflow"]=="guix"])
fig.suptitle(f"Time in seconds, aggregated over {n_uniq} repetitions")
fig.savefig(f"{sys.argv[1]}/cost_reproducibility.pdf")
