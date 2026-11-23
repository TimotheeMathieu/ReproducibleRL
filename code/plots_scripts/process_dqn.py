import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import sys
N_POINTS = 500  # resolution of the common x-grid
N_BOOT = 1000  # bootstrap resamples for the 95% CI

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 13,
    "axes.labelsize": 24,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "lines.linewidth": 3,
    "axes.linewidth": 1.5,
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
})

computers = ["burns", "flanders"]

# rows = []
# for computer in computers:
#     for gradient_steps in [1, 10, 100]:
#         for train_freq in [10, 100, 1000]:
#             for target_update in [1, 10, 100]:
#                 curves = []

#                 for seed in range(50):
#                     log_dir = (
#                         f"{computer}_logs/ReproRL/DQN-seed-{seed}-gs-{gradient_steps}"
#                         f"-tf-{train_freq}-tu-{target_update}"
#                         f"/Acrobot-v1/{computer}/0.monitor.csv"
#                     )
#                     try:
#                         log_df = pd.read_csv(log_dir, delimiter=",", skiprows=1)
#                     except FileNotFoundError:
#                         continue
#                     x = log_df['l'].cumsum().to_numpy()
#                     y = log_df['r'].to_numpy()
#                     curves.append((seed, x, y))

#                 if not curves:
#                     continue

#                 # common x-grid from 0 to the minimum of all seeds' last timestep
#                 # (use min so every curve covers the full range)
#                 x_end = min(x[-1] for _, x, _ in curves)
#                 x_common = np.linspace(0, x_end, N_POINTS)

#                 for seed, x, y in curves:
#                     reward = np.interp(x_common, x, y)
#                     for xc, r in zip(x_common, reward):
#                         rows.append({
#                             "computer": computer,
#                             "gradient_steps": gradient_steps,
#                             "train_freq": train_freq,
#                             "target_update": target_update,
#                             "seed": seed,
#                             "x": xc,
#                             "reward": r,
#                         })

# df = pd.DataFrame(rows)
# df.to_csv("training.csv", index=False)
df = pd.read_csv("data_feedback_loop/training.csv")
fig, axes = plt.subplots(3, 9, figsize=(36, 12), sharey=True)

for computer in computers:
    for gi, gradient_steps in enumerate([1, 10, 100]):
        for ti, train_freq in enumerate([10, 100, 1000]):
            for ui, target_update in enumerate([1, 10, 100]):
                sub = df[
                    (df["computer"] == computer)
                    & (df["gradient_steps"] == gradient_steps)
                    & (df["train_freq"] == train_freq)
                    & (df["target_update"] == target_update)
                ]
                if sub.empty:
                    continue

                # seeds x grid-points matrix, then bootstrap over seeds
                pivot = sub.pivot_table(index="seed", columns="x", values="reward").sort_index(axis=1)
                data = pivot.to_numpy()
                x_common = pivot.columns.to_numpy()
                mean = data.mean(axis=0)

                res = stats.bootstrap(
                    (data,), np.mean, axis=0, n_resamples=N_BOOT,
                    confidence_level=0.95, rng=0,
                )
                ci_low = res.confidence_interval.low
                ci_high = res.confidence_interval.high

                col = ti * 3 + ui
                ax = axes[gi, col]
                ax.axvspan(0, 100_000, color="grey", alpha=0.15, zorder=0)
                ax.text(50_000, 0.97, "Fill buffer\nwith random\nsamples", color="grey",
                        fontsize=14, ha="center", va="top", transform=ax.get_xaxis_transform())
                ax.plot(x_common, mean, label=computer)
                ax.fill_between(x_common, ci_low, ci_high, alpha=0.3)

                ax.axvline(100_000 + train_freq, color="red", linestyle=":", linewidth=2.5)
                ax.text(97_000, 0.08, "First Q update", color="red", fontsize=14,
                        ha="right", va="bottom", rotation=90, transform=ax.get_xaxis_transform())
                ax.set_title(f"Grad. steps/Q update={gradient_steps}\nNew samples/Q update={train_freq}\nNew samples/Qtarget update={train_freq * target_update}")
                ax.set_xticks([0, 100_000, 200_000])
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k"))
                # first panel whose upper right corner is free of curves
                if col == 3 and gi == 0:
                    ax.legend(loc='upper right')

fig.supylabel("Online smoothed episodic\n" + r"reward of $\epsilon$-greedy policy", fontsize=24, x=0.002)
fig.supxlabel("Samples", fontsize=24)
plt.tight_layout()
plt.savefig(sys.argv[1]+'/training.pdf', dpi=300)