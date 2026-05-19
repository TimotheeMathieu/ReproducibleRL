import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

df_g5k = pd.concat([pd.read_csv("data_g5k_hardware/grid5000_default_queue_resources.csv"),
                    pd.read_csv("data_g5k_hardware/grid5000_abaca_queue_resources.csv")], ignore_index = True)
def make_reward_fig(df, ax):
    for name in np.sort(df["computer_name"].unique()):
        hardware = df_g5k.loc[df_g5k["Cluster"]==name.split("-")[0],["CPU.1", "Access Condition"]].values.astype(str)
        if len(hardware)>0:
            hardware = "<br> ".join(list(hardware[0]))
        else:
            hardware = "Unknown"
        ax.plot(
            df.loc[df["computer_name"]==name, "episode_step"],
            df.loc[df["computer_name"]==name, "episode_rewards"],
        )
        ax.set_xlabel("episode step")
        ax.set_ylabel("episode reward")


frameworks = {"pip": "../results/Acrobot/Acrobot_Pip_Unfrozen.csv",
              "pip frozen": "../results/Acrobot/Acrobot_Pip_Frozen.csv",
              "guix": "../results/Acrobot/Acrobot_Guix.csv",
              "guix no-avx": "../results/Acrobot/Acrobot_Guix_with_masked_AVX.csv",
              "conda": "../results/Acrobot/Acrobot_Conda_UnFrozen.csv",
              "conda frozen": "../results/Acrobot/Acrobot_Conda_Frozen.csv"
              }

for framework in frameworks:
    df = pd.read_csv(frameworks[framework])
    n_uniq = len(np.sort(df["computer_name"].unique()))
    print(f"Framework {framework} has {n_uniq} computers")

for id_framework in frameworks:
    fig, ax = plt.subplots(figsize=(8,4))
    df = pd.read_csv(frameworks[id_framework])
    make_reward_fig(df, ax)
    fig.savefig(f"{sys.argv[1]}/acrobot_rewards_{id_framework}.pdf")

def isin_interval(x, interval):
    if (x >= interval[0]) and (x <= interval[1]):
        return True
    else:
        return False

def get_num_curves(X, epsilon):
    # https://math.stackexchange.com/questions/4139118/how-do-we-find-the-smallest-number-of-intervals-covering-the-subset-of-0-1
    sorted_values = np.sort(np.unique(X))
    intervals = []
    while len(sorted_values) > 0:
        if len(sorted_values)==1:
            intervals = intervals + [[sorted_values[0]-epsilon/2, sorted_values[0]+epsilon/2]]
        elif sorted_values[-1] - sorted_values[0] > epsilon:
            intervals = intervals +  [[sorted_values[0], sorted_values[0] + epsilon],
                                      [sorted_values[-1]-epsilon, sorted_values[-1]]]
        else:
            intervals = intervals + [[sorted_values[0], sorted_values[-1]]]
        isnotin = [not np.any([isin_interval(x, interval) for interval in intervals]) for x in sorted_values]
        sorted_values = sorted_values[isnotin]
    intervals_ids = []
    for idx in range(len(X)):
        for j, interval in enumerate(intervals):
            if isin_interval(X[idx], interval):
                intervals_ids.append(j)
                break

    return len(intervals), intervals_ids

def make_fig_epsilon(df, ax):
    tot_max_time = 40000
    names = []
    for name in np.unique(df["computer_name"]):
        max_time = np.max(df.loc[df["computer_name"]==name, "episode_step"])
        if max_time >= tot_max_time:
            names.append(name)
    print(f"Working with {len(names)} cpus.")

    times = np.linspace(0,tot_max_time, num=200)
    groups = [np.unique(names)]

    df_interp =  pd.DataFrame()
    for name  in np.unique(names):
        df_name = df.loc[df["computer_name"] == name]
        df_interp = pd.concat([df_interp,
                               pd.DataFrame({
                                   "name":[name]*len(times),
                                   "time": times,
                                   "reward": np.interp(times, df_name["episode_step"], df_name["episode_rewards"])
                               })], ignore_index = True)
    epsilons = [0, 1e-2, 1e-1, 1, 10][::-1]
    for epsilon in epsilons:
        res = []
        for t in times:
            data_at_time = df_interp.loc[df_interp["time"] == t]
            num, ids = get_num_curves(np.array(data_at_time["reward"]), epsilon)
            res.append(num)
        ax.plot(times,res,label=str(epsilon))

    ax.set_xlabel("episode step")
    ax.set_ylabel("number of groups")


for id_framework in frameworks:
    fig, ax = plt.subplots(figsize=(8,4))
    df = pd.read_csv(frameworks[id_framework])
    make_fig_epsilon(df, ax)
    plt.legend(title="$\\varepsilon$")
    fig.savefig(f"{sys.argv[1]}/acrobot_epsicurve_{id_framework}.pdf")
