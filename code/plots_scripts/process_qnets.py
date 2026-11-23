import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.stats import bootstrap
import sys

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


# def compute_seed_diffs(seed, gradient_steps, train_freq, target_update, checkpoints):
#     print(seed)
#     burns_path = (
#         f"burns_logs/ReproRL/DQN-seed-{seed}-gs-{gradient_steps}"
#         f"-tf-{train_freq}-tu-{target_update}"
#         f"/Acrobot-v1/burns-step-0.zip"
#     )
#     flanders_path = (
#         f"flanders_logs/ReproRL/DQN-seed-{seed}-gs-{gradient_steps}"
#         f"-tf-{train_freq}-tu-{target_update}"
#         f"/Acrobot-v1/flanders-step-0.zip"
#     )
#     burns_params = DQN.load(burns_path).policy.q_net.parameters_to_vector()
#     flanders_params = DQN.load(flanders_path).policy.q_net.parameters_to_vector()
#     b_norm, f_norm = np.linalg.norm(burns_params), np.linalg.norm(flanders_params)
#     diffs = [float(np.linalg.norm(burns_params - flanders_params) / (0.5 * (b_norm + f_norm)))]
#     cosines = [float(np.dot(burns_params, flanders_params) / (b_norm * f_norm))]
#     for k in checkpoints[1:]:
#         burns_path = (
#             f"burns_logs/ReproRL/DQN-seed-{seed}-gs-{gradient_steps}"
#             f"-tf-{train_freq}-tu-{target_update}"
#             f"/Acrobot-v1/burnsafter-{k}-new-spls.zip"
#         )
#         flanders_path = (
#             f"flanders_logs/ReproRL/DQN-seed-{seed}-gs-{gradient_steps}"
#             f"-tf-{train_freq}-tu-{target_update}"
#             f"/Acrobot-v1/flandersafter-{k}-new-spls.zip"
#         )
#         try:
#             burns_params = DQN.load(burns_path).policy.q_net.parameters_to_vector()
#             flanders_params = DQN.load(flanders_path).policy.q_net.parameters_to_vector()
#             b_norm, f_norm = np.linalg.norm(burns_params), np.linalg.norm(flanders_params)
#             diff = float(np.linalg.norm(burns_params - flanders_params) / (0.5 * (b_norm + f_norm)))
#             cosine = float(np.dot(burns_params, flanders_params) / (b_norm * f_norm))
#         except FileNotFoundError:
#             diff = float("nan")
#             cosine = float("nan")
#         diffs.append(diff)
#         cosines.append(cosine)
#     return np.array(diffs), np.array(cosines)


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

checkpoints = list(range(0, 100_001, 1000))
# checkpoints count the samples collected after the 100k random ones filling the buffer
x_axis = 100_000 + np.array(checkpoints)

# rows = []
# for gradient_steps in [1, 10, 100]:
#     for train_freq in [10, 100, 1000]:
#         for target_update in [1, 10, 100]:
#             results = Parallel(n_jobs=15)(
#                 delayed(compute_seed_diffs)(seed, gradient_steps, train_freq, target_update, checkpoints)
#                 for seed in range(30)
#             )
#             for seed, (diffs, cosines) in enumerate(results):
#                 for checkpoint, diff, cosine in zip(checkpoints, diffs, cosines):
#                     rows.append({
#                         "gradient_steps": gradient_steps,
#                         "train_freq": train_freq,
#                         "target_update": target_update,
#                         "seed": seed,
#                         "checkpoint": checkpoint,
#                         "diff": diff,
#                         "cosine": cosine,
#                     })

# df = pd.DataFrame(rows)
# df.to_csv("qnets_diff.csv", index=False)
df = pd.read_csv("data_feedback_loop/old_data/qnets_diff.csv")
fig, axes = plt.subplots(3, 9, figsize=(36, 12))

for gi, gradient_steps in enumerate([1, 10, 100]):
    for ti, train_freq in enumerate([10, 100, 1000]):
        for ui, target_update in enumerate([1, 10, 100]):
            sub = df[
                (df["gradient_steps"] == gradient_steps)
                & (df["train_freq"] == train_freq)
                & (df["target_update"] == target_update)
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

            col = ti * 3 + ui
            ax = axes[gi, col]
            ax.plot(x_axis, mean)
            ax.fill_between(x_axis, lo, hi, alpha=0.3)
            ax.axvline(100_000 + train_freq, color="red", linestyle=":", linewidth=2.5)
            ax.text(100_000 + train_freq, 0.7, "First Q update", color="red", fontsize=14,
                    ha="left", va="top", rotation=90, transform=ax.get_xaxis_transform())
            ax.set_title(f"Grad. steps/Q update={gradient_steps}\nNew samples/Q update={train_freq}\nNew samples/Qtarget update={train_freq * target_update}", pad=22)
            ax.set_xticks([100_000, 150_000, 200_000])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k"))
            ax.yaxis.get_offset_text().set_fontsize(13)

            ax2 = ax.twinx()
            ax2.plot(x_axis, cos_mean, color="green")
            ax2.fill_between(x_axis, cos_lo, cos_hi, alpha=0.15, color="green")
            ax2.tick_params(axis="y", labelcolor="green")
            ax2.yaxis.get_offset_text().set_fontsize(13)

fig.supylabel(r"$\frac{\ell_2(\hat{Q}_{\text{flanders}} - \hat{Q}_{\text{burns}})}{\frac{1}{2}(\ell_2(\hat{Q}_{\text{flanders}}) + \ell_2(\hat{Q}_{\text{burns}}))}$", fontsize=34, x=0.002)
fig.supxlabel("Samples", fontsize=24)
fig.text(0.998, 0.5, r"$\cos(\hat{Q}_{\text{flanders}},\, \hat{Q}_{\text{burns}})$", color="green", fontsize=24,
         rotation=90, ha="right", va="center")
plt.tight_layout(rect=(0, 0, 0.985, 1))
plt.savefig(sys.argv[1]+'/qnets_diff.pdf', dpi=300)
print("Saved qnets_diff.pdf")