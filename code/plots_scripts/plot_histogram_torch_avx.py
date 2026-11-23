import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys

df_avx = pd.read_csv(f"{sys.argv[2]}/results_torch_avx_archlinux.csv")
df_noavx = pd.read_csv(f"{sys.argv[2]}/results_torch_noavx_archlinux.csv")

df_diff = df_avx[["seed", "size","relu"]]
df_diff["diff"] = df_avx["value"] - df_noavx["value"]

fig, axes = plt.subplots(1,2, figsize=(9,3), sharey=True)

sns.histplot(df_diff.loc[df_diff["relu"]==True], x="diff", hue="size", ax = axes[0], bins=20, palette="tab10",stat="probability")
sns.histplot(df_diff.loc[df_diff["relu"]==False], x="diff", hue="size", ax = axes[1], bins=20, palette="tab10",stat="probability")

axes[0].set_title("With ReLU")
axes[1].set_title("Without ReLU")

fig.savefig(f"{sys.argv[1]}/torch_histogram_avx.pdf", bbox_inches="tight")
