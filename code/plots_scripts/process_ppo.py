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
    "axes.titlesize": 15,
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
# # each computer's PPO logs live under a different ReproRL-style subfolder
# repro_dirs = {"burns": "ReproRL", "flanders": "ReproducibleRL_fitted_Q"}

# rows = []
# for computer in computers:
#     repro_dir = repro_dirs[computer]
#     for gradient_steps in [16, 64, 256]:
#         for n_steps in [512, 1042, 2048]:
#             for gae_lambda in [0.5, 0.9, 0.99]:
#                 curves = []

#                 for seed in range(50):
#                     log_dir = (
#                         f"logs-ppo/ppo_logs_{computer}/{repro_dir}/-PPO-seed-{seed}-gs-{gradient_steps}"
#                         f"-ns-{n_steps}-gae-{gae_lambda}"
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
#                             "n_steps": n_steps,
#                             "gae_lambda": gae_lambda,
#                             "seed": seed,
#                             "x": xc,
#                             "reward": r,
#                         })

# df = pd.DataFrame(rows)
# df.to_csv("training_ppo.csv", index=False)
df = pd.read_csv("data_feedback_loop/training_ppo.csv")
fig, axes = plt.subplots(3, 9, figsize=(36, 12), sharey=True)

for computer in computers:
    for gi, gradient_steps in enumerate([16, 64, 256]):
        for ni, n_steps in enumerate([512, 1042, 2048]):
            for gaei, gae_lambda in enumerate([0.5, 0.9, 0.99]):
                sub = df[
                    (df["computer"] == computer)
                    & (df["gradient_steps"] == gradient_steps)
                    & (df["n_steps"] == n_steps)
                    & (df["gae_lambda"] == gae_lambda)
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

                col = ni * 3 + gaei
                ax = axes[gi, col]
                ax.plot(x_common, mean, label=computer)
                ax.fill_between(x_common, ci_low, ci_high, alpha=0.3)

                ax.set_title(f"Grad. steps/update={gradient_steps}\nn_steps={n_steps} gae_lambda={gae_lambda}")
                ax.set_xticks([0, 50_000, 100_000])
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k"))

                if col == 0 and gi == 0:
                    ax.legend(loc='lower right')

fig.supylabel("Online smoothed episodic\n" + r"reward of $\epsilon$-greedy policy", fontsize=24, x=0.002)
fig.supxlabel("Samples", fontsize=24)
plt.tight_layout()
plt.savefig(sys.argv[1]+'/training_ppo.pdf', dpi=300)
