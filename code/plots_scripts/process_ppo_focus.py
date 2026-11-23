import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import sys
N_POINTS = 500  # resolution of the common x-grid
N_BOOT = 1000  # bootstrap resamples for the 95% CI
# df = pd.DataFrame(rows)
# df.to_csv("training_ppo.csv", index=False)
df = pd.read_csv("data_feedback_loop/training_ppo.csv")

for computer in ['burns', 'flanders']:
    sub = df[
        (df["computer"] == computer)
        & (df["gradient_steps"] == 16)
        & (df["n_steps"] == 2048)
        & (df["gae_lambda"] == 0.99)
    ]

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

    plt.plot(x_common, mean, label=computer)
    plt.fill_between(x_common, ci_low, ci_high, alpha=0.3)
plt.ylabel("Online smoothed\nepisodic returns", fontsize=18)
plt.xlabel("Samples", fontsize=18)
plt.legend(loc='lower right', fontsize=18)
plt.tight_layout()
plt.savefig(sys.argv[1]+'/training_ppo_focus.pdf', dpi=300)
