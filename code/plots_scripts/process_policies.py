import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.stats import bootstrap
import sys
# each computer's PPO logs live under a different ReproRL-style subfolder
repro_dirs = {"burns": "ReproRL", "flanders": "ReproducibleRL_fitted_Q"}

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


def bootstrap_ci(values, confidence_level=0.95, n_resamples=1000):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size < 2 or np.all(values == values[0]):
        mean = float(values[0]) if values.size else float("nan")
        return mean, mean, mean
    res = bootstrap(
        (values,), np.mean, confidence_level=confidence_level,
        n_resamples=n_resamples, rng=0,
    )
    return float(values.mean()), float(res.confidence_interval.low), float(res.confidence_interval.high)


# def compute_seed_diffs(seed, gradient_steps, n_steps, gae_lambda, checkpoints):
#     print(seed)
#     burns_path = (
#         f"logs-ppo/ppo_logs_burns/{repro_dirs['burns']}/-PPO-seed-{seed}-gs-{gradient_steps}"
#         f"-ns-{n_steps}-gae-{gae_lambda}"
#         f"/Acrobot-v1/burns-step-0.zip"
#     )
#     flanders_path = (
#         f"logs-ppo/ppo_logs_flanders/{repro_dirs['flanders']}/-PPO-seed-{seed}-gs-{gradient_steps}"
#         f"-ns-{n_steps}-gae-{gae_lambda}"
#         f"/Acrobot-v1/flanders-step-0.zip"
#     )
#     try:
#         burns_params = PPO.load(burns_path).policy.parameters_to_vector()
#         flanders_params = PPO.load(flanders_path).policy.parameters_to_vector()
#         b_norm, f_norm = np.linalg.norm(burns_params), np.linalg.norm(flanders_params)
#         diffs = [float(np.linalg.norm(burns_params - flanders_params) / (0.5 * (b_norm + f_norm)))]
#         cosines = [float(np.dot(burns_params, flanders_params) / (b_norm * f_norm))]
#     except FileNotFoundError:
#         diffs = [float("nan")]
#         cosines = [float("nan")]
#     for k in checkpoints[1:]:
#         burns_path = (
#             f"logs-ppo/ppo_logs_burns/{repro_dirs['burns']}/-PPO-seed-{seed}-gs-{gradient_steps}"
#             f"-ns-{n_steps}-gae-{gae_lambda}"
#             f"/Acrobot-v1/burnsafter-{k}-new-spls.zip"
#         )
#         flanders_path = (
#             f"logs-ppo/ppo_logs_flanders/{repro_dirs['flanders']}/-PPO-seed-{seed}-gs-{gradient_steps}"
#             f"-ns-{n_steps}-gae-{gae_lambda}"
#             f"/Acrobot-v1/flandersafter-{k}-new-spls.zip"
#         )
#         try:
#             burns_params = PPO.load(burns_path).policy.parameters_to_vector()
#             flanders_params = PPO.load(flanders_path).policy.parameters_to_vector()
#             b_norm, f_norm = np.linalg.norm(burns_params), np.linalg.norm(flanders_params)
#             diff = float(np.linalg.norm(burns_params - flanders_params) / (0.5 * (b_norm + f_norm)))
#             cosine = float(np.dot(burns_params, flanders_params) / (b_norm * f_norm))
#         except FileNotFoundError:
#             diff = float("nan")
#             cosine = float("nan")
#         diffs.append(diff)
#         cosines.append(cosine)
#     return np.array(diffs), np.array(cosines)


checkpoints = list(range(0, 102_401, 2048))
x_axis = np.array(checkpoints)

# rows = []
# for gradient_steps in [16, 64, 256]:
#     for n_steps in [512, 1042, 2048]:
#         for gae_lambda in [0.5, 0.9, 0.99]:
#             results = Parallel(n_jobs=15)(
#                 delayed(compute_seed_diffs)(seed, gradient_steps, n_steps, gae_lambda, checkpoints)
#                 for seed in range(30)
#             )
#             for seed, (diffs, cosines) in enumerate(results):
#                 for checkpoint, diff, cosine in zip(checkpoints, diffs, cosines):
#                     rows.append({
#                         "gradient_steps": gradient_steps,
#                         "n_steps": n_steps,
#                         "gae_lambda": gae_lambda,
#                         "seed": seed,
#                         "checkpoint": checkpoint,
#                         "diff": diff,
#                         "cosine": cosine,
#                     })

# df = pd.DataFrame(rows)
# df.to_csv("ppo_diff.csv", index=False)
df = pd.read_csv("data_feedback_loop/old_data/ppo_diff.csv")
fig, axes = plt.subplots(3, 9, figsize=(36, 12))

for gi, gradient_steps in enumerate([16, 64, 256]):
    for ni, n_steps in enumerate([512, 1042, 2048]):
        for gaei, gae_lambda in enumerate([0.5, 0.9, 0.99]):
            sub = df[
                (df["gradient_steps"] == gradient_steps)
                & (df["n_steps"] == n_steps)
                & (df["gae_lambda"] == gae_lambda)
            ]
            if sub.empty:
                continue

            # drop seeds whose curve is entirely NaN, matching previous behavior
            valid_seeds = sub.groupby("seed")["diff"].apply(lambda s: not s.isna().all())
            sub = sub[sub["seed"].isin(valid_seeds[valid_seeds].index)]
            if sub.empty:
                continue

            stats = sub.groupby("checkpoint").agg(
                diff_stats=("diff", bootstrap_ci),
                cosine_stats=("cosine", bootstrap_ci),
            ).reindex(checkpoints)

            mean, lo, hi = map(np.array, zip(*stats["diff_stats"]))
            cos_mean, cos_lo, cos_hi = map(np.array, zip(*stats["cosine_stats"]))

            col = ni * 3 + gaei
            ax = axes[gi, col]
            ax.plot(x_axis, mean)
            ax.fill_between(x_axis, lo, hi, alpha=0.3)
            ax.set_title(f"Grad. steps/update={gradient_steps}\nn_steps={n_steps} gae_lambda={gae_lambda}")
            ax.set_xticks([0, 50_000, 100_000])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k"))

            ax2 = ax.twinx()
            ax2.plot(x_axis, cos_mean, color="green")
            ax2.fill_between(x_axis, cos_lo, cos_hi, alpha=0.15, color="green")
            ax2.tick_params(axis="y", labelcolor="green")

fig.supylabel(r"$\frac{\ell_2(\hat{\pi}_{\text{flanders}} - \hat{\pi}_{\text{burns}})}{\frac{1}{2}(\ell_2(\hat{\pi}_{\text{flanders}}) + \ell_2(\hat{\pi}_{\text{burns}}))}$", fontsize=34, x=0.002)
fig.supxlabel("Samples", fontsize=24)
fig.text(0.998, 0.5, r"$\cos(\hat{\pi}_{\text{flanders}},\, \hat{\pi}_{\text{burns}})$", color="green", fontsize=24,
         rotation=90, ha="right", va="center")
plt.tight_layout(rect=(0, 0, 0.985, 1))
plt.savefig(sys.argv[1]+'/ppo_diff.pdf', dpi=300)
print("Saved ppo_diff.pdf")
