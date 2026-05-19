import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys

df_avx = pd.read_csv(f"{sys.argv[2]}/results_torch_avx_archlinux.csv")
df_noavx = pd.read_csv(f"{sys.argv[2]}/results_torch_noavx_archlinux.csv")

df_diff = df_avx[["seed", "size"]]
df_diff["diff"] = df_avx["value"] - df_noavx["value"]

fig, ax = plt.subplots()

sns.histplot(df_diff, x="diff", hue="size", ax = ax, palette="tab10")

fig.savefig(f"{sys.argv[1]}/torch_histogram_avx.pdf")
