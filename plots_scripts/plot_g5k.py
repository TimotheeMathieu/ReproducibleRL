import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from read_data import read_data_from_dir, read_minatar_from_dir

df_g5k = pd.concat([pd.read_csv("data_g5k_hardware/grid5000_default_queue_resources.csv"),
                    pd.read_csv("data_g5k_hardware/grid5000_abaca_queue_resources.csv")], ignore_index = True)


envs = ["Acrobot", "Mujoco", "Atari"]
venv_manager = ["guix", "guix_with_masked_AVX", "pip_frozen",  "conda_frozen", "pip_unfrozen", "conda_unfrozen"]


def make_reward_fig(dfenvenv, ax):
    df = dfenvenv
    ax.set_title(venv)

    for name in np.sort(df["computer_name"].unique()):
        ax.plot(
            df.loc[(df["computer_name"]==name) & (df["computer_name"]==name), "episode_step"],
            df.loc[df["computer_name"]==name, "r"],
            alpha = 0.6
        )
        ax.set_xlabel("episode step")
        ax.set_ylabel("episode reward")

df = read_data_from_dir(sys.argv[2])

df["r"] = df["r"].astype(np.float64)

for env in envs:
    for venv in venv_manager:
        dfs = []
        fig, ax = plt.subplots(figsize=(5,3))
        dfenvenv = df.loc[(df["env"] == env) & (df["venv"] == venv)]
        make_reward_fig(dfenvenv, ax)
        fig.savefig(f"{sys.argv[1]}/rewards_{env}_{venv}.pdf", bbox_inches="tight")

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
    names = np.unique(df["computer_name"])
    max_time = 0
    for name in np.unique(df["computer_name"]):
        max_time = max(max_time, np.max(df.loc[df["computer_name"]==name, "episode_step"]))
    max_time = int(max_time)

    times = np.linspace(0,max_time, num=200)
    groups = [np.unique(names)]

    df_interp =  pd.DataFrame()
    for name  in np.unique(names):
        df_name = df.loc[df["computer_name"] == name]

        df_interp = pd.concat([df_interp,
                               pd.DataFrame({
                                   "name":[name]*len(times),
                                   "time": times,
                                   "reward": np.interp(times, df_name["episode_step"], df_name["r"])
                               })], ignore_index = True)
    epsilons = [0, 1e-2,1][::-1]
    for epsilon in epsilons:
        res = []
        for t in times:
            data_at_time = df_interp.loc[df_interp["time"] == t]
            num, ids = get_num_curves(np.array(data_at_time["reward"]), epsilon)
            res.append(num)
        ax.plot(times,res,label=str(epsilon))

    ax.set_xlabel("episode step")
    ax.set_ylabel("number of groups")


def count_groups(df, epsilon=0):
    names = np.unique(df["computer_name"])
    max_time = 0
    for name in np.unique(df["computer_name"]):
        max_time = max(max_time, np.max(df.loc[df["computer_name"]==name, "episode_step"]))
    max_time = int(max_time)

    times = np.linspace(0,max_time, num=200)
    groups = [np.unique(names)]
    df_interp =  pd.DataFrame()
    for name  in np.unique(names):
        df_name = df.loc[df["computer_name"] == name]
        
        df_interp = pd.concat([df_interp,
                               pd.DataFrame({
                                   "name":[name]*len(times),
                                   "time": times,
                                   "reward": np.interp(times, df_name["episode_step"], df_name["r"])
                               })], ignore_index = True)
    res = []
    if len(df_interp) >0:
        for t in times:
            data_at_time = df_interp.loc[df_interp["time"] == t]
            num, ids = get_num_curves(np.array(data_at_time["reward"]), epsilon)
            res.append(num)
        return np.max(res)
    else:
        return np.nan


for env in envs:
    for venv in venv_manager:
        fig, ax = plt.subplots()
        df_env = df.loc[(df["env"]==env) & (df["venv"]==venv)]
        make_fig_epsilon(df_env, ax)
        plt.legend(title="$\\varepsilon$")
        fig.savefig(f"{sys.argv[1]}/epsicurve_{venv}_{env}.pdf")

print("Number of distinct curves")
print("=========================")
for env in envs:
    dfs = []
    for venv in venv_manager:
        df_env = df.loc[(df["env"]==env) & (df["venv"]==venv)]
        print(f"{env}_{venv}", count_groups(df_env), "over", len(np.unique(df_env["computer_name"])))


# data_loc = f"../results/Acrobot_GPU.csv"
# df = pd.read_csv(data_loc)
# print(f"Acrobot_GPU", count_groups(df), "over", len(np.unique(df["computer_name"])))


print("Number of distinct curves over intersection ")
print("===========================================")

def intersect(names1, names2):
    return list(set(names1).intersection(set(names2)))

all_computers =np.unique(df["computer_name"])

names = None
for env in envs:
    dfs = []
    fig, axes = plt.subplots(3,2,figsize=(15,5))
    for venv in venv_manager:
        if "AVX" not in venv:
            df_env = df.loc[(df["env"]==env) & (df["venv"]==venv)]
            if names is None:
                names = list(np.unique(df_env["computer_name"]))
            else:
                names = intersect(names, np.unique(df_env["computer_name"]))
    #### missing computers
    print("==========Missing computer (no avx)=========", env)
    print([ computer for computer in all_computers if computer not in names])

print(names)
print("Number of distinct curves")
print("=========================")

def isin(names1, names2):
    return [n in list(names2) for n in list(names1)]

for env in envs:
    dfs = []
    fig, axes = plt.subplots(3,2,figsize=(15,5))
    for venv in venv_manager:
        df_env = df.loc[(df["env"]==env) & (df["venv"]==venv)]
        df_env = df_env.loc[isin(df_env["computer_name"], names)]
        print(f"{env}_{venv}", count_groups(df_env), "over", len(np.unique(df_env["computer_name"])))


fig, axes = plt.subplots(1,2,figsize=(10,7))
df = read_minatar_from_dir(sys.argv[2])

for j, nn in enumerate(["MLP", "CNN"]):
    dfnn = df.loc[(df["nn"] == nn)]
    ax = axes[j]
    ax.set_title(nn)
    for name in np.sort(dfnn["computer_name"].unique()):
        ax.plot(
            dfnn.loc[(dfnn["computer_name"]==name) & (dfnn["computer_name"]==name), "episode_step"],
            dfnn.loc[dfnn["computer_name"]==name, "r"],
            alpha = 0.6
        )
    ax.set_xlabel("episode step")
    ax.set_ylabel("episode reward")
    print(f"Minatar {nn}", count_groups(dfnn), "over", len(np.unique(dfnn["computer_name"])))
fig.savefig(f"{sys.argv[1]}/rewards_Minatar.pdf")



