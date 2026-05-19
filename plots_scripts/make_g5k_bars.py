import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import seaborn as sns
import distinctipy
from distinctipy import examples

from read_data import read_data_from_dir, read_minatar_from_dir, read_acrobot_gpu_from_dir


df_g5k = pd.concat([pd.read_csv("data_g5k_hardware/grid5000_default_queue_resources.csv"),
                    pd.read_csv("data_g5k_hardware/grid5000_abaca_queue_resources.csv")], ignore_index = True)

def name_to_cpu(name):
    return df_g5k.loc[df_g5k["Cluster"]==name, "CPU.1"].item()

envs = ["Acrobot", "Mujoco", "Atari"]
venv_manager = ["guix_with_masked_AVX", "guix",  "pip_frozen","conda_frozen", "pip_unfrozen", "conda_unfrozen"]
xlabels = ["guix\nw/o AVX2","guix",  "pip frz", "conda frz", "pip", "conda"]*3

df = read_data_from_dir(sys.argv[2])
df["r"] = df["r"].astype(np.float64)

all_the_names = np.unique(df["computer_name"])
intel_names = [name for name in all_the_names if not ("AMD" in name_to_cpu(name))]
amd_names = [name for name in all_the_names if "AMD" in name_to_cpu(name)]

def isin_interval(x, interval):
    if (x >= interval[0]) and (x <= interval[1]):
        return True
    else:
        return False

# get the differents intervals
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

# create cluster and match computer in it
def count_groups(df):
    times = np.unique(df["episode_step"])
    n_times = len(times)
    if n_times < 1:
        return np.nan, [np.nan], None, None
    times = times[np.arange(0,n_times,max(1,n_times//10)).astype(int)]

    res = []
    max_counts = []
    groups= None
    for t in times:
        data_at_time = df.loc[df["episode_step"] == t]
        num, ids = get_num_curves(np.array(data_at_time["r"]), 0)
        unique, indices, inverse, counts = np.unique(ids, return_index=True, return_inverse=True,return_counts=True)

        idsort = np.flip(np.argsort(counts))
        counts = counts[idsort]
        unique = unique[idsort]
        inverse = np.hstack([ unique[j]*np.ones(c) for j,c in enumerate(counts[idsort])])
        
        if len(counts) > len(max_counts):
            max_counts = counts
            max_inverse = inverse
            
            groups = {j: data_at_time.loc[ids == j, "computer_name"] for j in unique}
        res.append(num)
    print(groups)
    return np.max(res), max_counts, max_inverse, groups

print("Number of distinct curves over intersection -- Intel ")
print("=====================================================")

def intersect(names1, names2):
    return list(set(names1).intersection(set(names2)))

names = intel_names
for env in envs:
    dfs = []
    fig, axes = plt.subplots(3,2,figsize=(15,5))
    for venv in venv_manager:
        df_env = df.loc[(df["env"]==env) & (df["venv"]==venv)]
        names = intersect(names, np.unique(df_env["computer_name"]))

def isin(names1, names2):
    return np.array([n in list(names2) for n in list(names1)])

print("*********************************")

res = {}
for env in envs:
    dfs = []
    fig, axes = plt.subplots(3,2,figsize=(15,5))
    for venv in venv_manager:
        df_env = df.loc[(df["env"]==env) & (df["venv"]==venv)]
        df_env = df_env.loc[isin(df_env["computer_name"], names)]
        res[(env,venv)] = count_groups(df_env)
        print(env, venv)
        print(count_groups(df_env)[1])

print("*********************************")

fig, ax = plt.subplots(figsize=(16,8))

cm = distinctipy.get_colors(50, pastel_factor=0.75,rng=3)

current_height = {(env, venv):0 for env in envs for venv in venv_manager}

dfs_toplot = []
step1 = 9
step2 = 1.35
xs = {}
for i, env in enumerate(envs):
    for j,venv in enumerate(venv_manager):
        xs[(env,venv)] = i * step1 + j* step2 

nums = [ len(res[(env, venv)][1]) for env in envs for venv in venv_manager]


def get_color(computer_id, res):
    return res[2][computer_id]


def make_bars(nums,xs, names, width, res, ax,  id_computer=False,n_workflow=6,n_extra=1):
    mydf = pd.DataFrame()

    for computer_id in np.arange(len(names)):
        dfcount = pd.DataFrame({"x": [xs[(env,venv)]  for env,venv in list(res.keys())],
                                "color": [get_color(computer_id,res[(env, venv)]) for env,venv in list(res.keys())],
                                "env": [env for env,venv in list(res.keys())],
                                "computer_id" : [computer_id]*len(list(res.keys()))
                                })
        mydf = pd.concat([mydf, dfcount], ignore_index = True)
    
    thexs = np.unique(mydf["x"])
    if id_computer is False:
        myxss = [thexs[(n_workflow*idenv) : n_workflow*(idenv+1)] for idenv in range(3)]
    else:
        myxss = [thexs[:(2*n_workflow+n_extra)]]
    
    for myxs in myxss:
        mydfx=mydf.loc[mydf["x"]==myxs[0]]
        mydfx = mydfx.sort_values(by=["color"])
        order = mydfx['computer_id']
        order = order.values.copy()
        for x in myxs:
            for color in np.unique(mydfx["color"]):
                idgroup = [ f for f in (mydfx["color"]==color)]
                group = order[idgroup]
                mydfx2 = mydf.loc[(mydf["x"]==x) & isin(mydf["computer_id"], group)].sort_values(by=["color"])
                order[idgroup] = mydfx2["computer_id"].values
            mydfx = mydf.loc[mydf["x"]==x].iloc[order]
        if id_computer is False:
            toplot = myxs
        else:
            toplot = thexs
        for x in toplot:
            mydfx = mydf.loc[mydf["x"]==x]
            mydfx = mydfx.sort_values(by=["color"])
        
            id_color = 0
            colors = []
            for j,computer_id in enumerate(order):
                data = mydfx.loc[mydfx["computer_id"]==computer_id]
                mycolor = data["color"].item()
                if mycolor in colors:
                    color = cm[np.where(np.array(colors)==mycolor)[0][0]]
                else:
                    color = cm[len(colors)]
                    colors.append(mycolor)
                ax.bar([x], [1], width,bottom = j, color = color)
    

def make_bars_coherent(nums,xs, names, envs, venv_manager, width, res, ax,  id_computer=False):
    current_height = {(env, venv):0 for env in envs for venv in venv_manager}
    nums = [ len(res[(env, venv)]) for env in envs for venv in venv_manager]
    bottom = np.zeros(len(res.keys()))
    mycounts = {(env, venv):res[(env,venv)][1] for env in envs for venv in venv_manager}
    idcol = 0
    while len(mycounts) > 0:
        dfcount = pd.DataFrame({"x":[xs[(env,venv)]  for env,venv in list(mycounts.keys())],
                           "Counts":[mycounts[(env, venv)][0] for env,venv in list(mycounts.keys())],
                           "env": [env for env,venv in list(mycounts.keys())],
                           })
        p = ax.bar(dfcount["x"], dfcount["Counts"], width, 
                   bottom=bottom, color=cm[idcol], alpha=0.6)
        # ax.bar_label(p, label_type='center')
        idcol += 1

        bottom = bottom + dfcount["Counts"]

        dfs_toplot.append(dfcount)
        reskeys = list(mycounts.keys())
        todel = []
        for j, k in enumerate(reskeys):
            if len(mycounts[k]) == 1:
                del mycounts[k]
                todel.append(j)
            else:
                mycounts[k] = mycounts[k][1:]
        bottom = np.array([bottom[i] for  i in range(len(bottom)) if i not in todel])


####################### Do plot ###########################"

fig, ax = plt.subplots(figsize=(16,8))
xticksx = [xs[(env,venv)]  for env, venv in xs.keys() ]
ax.set_xticks(xticksx)
ax.text(xticksx[3]-0.5, -4, "Acrobot", ha='center', va='top')
ax.text(xticksx[9]-0.5, -4, "Hopper", ha='center', va='top')
ax.text(xticksx[15]-0.5 , -4, "Atari", ha='center', va='top')

for j in range(len(nums)):
    ax.text(xticksx[j]-0.1, len(names)+len(names)/10, nums[j])

ax.text(-2.25, len(names)+len(names)/10, "Nb groups")
ax.set_xticklabels(xlabels)

ax.set_ylabel("Nb Computer")
width = 1.1

make_bars_coherent(nums,xs, names, envs, venv_manager, width, res, ax)

fig.savefig(sys.argv[1]+"/bars_g5k_intel.pdf", bbox_inches="tight")

print("Number of distinct curves over intersection -- AMD ")
print("====================================================")


envs = ["Acrobot", "Mujoco", "Atari"]
venv_manager = ["guix", "pip_frozen","conda_frozen", "pip_unfrozen", "conda_unfrozen"]
xlabels = ["guix",  "pip frz", "conda frz", "pip", "conda"]*3


names = amd_names
for env in envs:
    dfs = []
    fig, axes = plt.subplots(3,2,figsize=(15,5))
    for venv in venv_manager:
        df_env = df.loc[(df["env"]==env) & (df["venv"]==venv)]
        names = intersect(names, np.unique(df_env["computer_name"]))

res = {}
for env in envs:
    dfs = []
    fig, axes = plt.subplots(3,2,figsize=(15,5))
    for venv in venv_manager:
        df_env = df.loc[(df["env"]==env) & (df["venv"]==venv)]
        df_env = df_env.loc[isin(df_env["computer_name"], names)]
        res[(env,venv)] = count_groups(df_env)

fig, ax = plt.subplots(figsize=(16,8))

# cm = plt.get_cmap("tab20")
current_height = {(env, venv):0 for env in envs for venv in venv_manager}

dfs_toplot = []
step1 = 8
step2 = 1.35
xs = {}
for i, env in enumerate(envs):
    for j,venv in enumerate(venv_manager):
        xs[(env,venv)] = i * step1 + j* step2 

nums = [ len(res[(env, venv)][1]) for env in envs for venv in venv_manager]

fig, ax = plt.subplots(figsize=(16,8))
xticksx = [xs[(env,venv)]  for env, venv in xs.keys() ]
ax.set_xticks(xticksx)
ax.text(xticksx[2], -1, "Acrobot", ha='center', va='top')
ax.text(xticksx[7], -1, "Hopper", ha='center', va='top')
ax.text(xticksx[12] , -1, "Atari", ha='center', va='top')

for j in range(len(nums)):
    ax.text(xticksx[j]-0.1, len(names)+1, nums[j])

ax.text(-2.25, len(names)+1, "Nb groups")
ax.set_xticklabels(xlabels)

ax.set_ylabel("Nb Computer")

width = 1.1

make_bars_coherent(nums,xs, names, envs, venv_manager, width, res, ax)

fig.savefig(sys.argv[1]+"/bars_g5k_amd.pdf", bbox_inches="tight")

#################################### ALL sauf avx


print("Number of distinct curves over intersection -- No avx ")
print("====================================================")

names = np.unique(df["computer_name"])
for env in envs:
    dfs = []
    fig, axes = plt.subplots(3,2,figsize=(15,5))
    for venv in venv_manager:
        if "AVX" not in venv:
            df_env = df.loc[(df["env"]==env) & (df["venv"]==venv)]
            names = intersect(names, np.unique(df_env["computer_name"]))

res = {}
for env in envs:
    dfs = []
    fig, axes = plt.subplots(3,2,figsize=(15,5))
    for venv in venv_manager:
        df_env = df.loc[(df["env"]==env) & (df["venv"]==venv)]
        df_env = df_env.loc[isin(df_env["computer_name"], names)]
        res[(env,venv)] = count_groups(df_env)

# fig, ax = plt.subplots(figsize=(16,8))

step1 = 9
step2 = 1.35
xs = {}
for i, env in enumerate(envs):
    for j,venv in enumerate(venv_manager):
        xs[(env,venv)] = i * step1 + j* step2 

nums = [ len(res[(env, venv)][1]) for env in envs for venv in venv_manager]


fig, ax = plt.subplots(figsize=(16,8))
xticksx = [xs[(env,venv)]  for env, venv in xs.keys() ]
ax.set_xticks(xticksx)

ax.text(xticksx[2], -4, "Acrobot", ha='center', va='top')
ax.text(xticksx[7], -4, "Hopper", ha='center', va='top')
ax.text(xticksx[12] , -4, "Atari", ha='center', va='top')

for j in range(len(nums)):
    ax.text(xticksx[j]-0.1, len(names)*1.07, nums[j])

ax.text(-2.5, len(names)*1.07, "Nb groups")
ax.set_xticklabels(xlabels)

# ax.set_xlabel("Environment",labelpad=15)
ax.set_ylabel("Nb Computer")


width = 1.1

make_bars_coherent(nums,xs, names, envs, venv_manager, width, res, ax)

fig.savefig(sys.argv[1]+"/bars_g5k_noavx.pdf", bbox_inches="tight")


############# Minatar ###################
print("============= Minatar ==============")

df = read_minatar_from_dir(sys.argv[2])
for j, nn in enumerate(["MLP", "CNN"]):
    dfnn = df.loc[(df["nn"] == nn)]
    N = len(np.unique(dfnn["computer_name"]))
    print(f"Minatar {nn}", count_groups(dfnn), "over", len(np.unique(dfnn["computer_name"])))

print(np.unique(dfnn["computer_name"]))
fig, ax = plt.subplots(figsize=(10,4))
current_height = {"MLP":0, "CNN":0}

res = {}

envs = ["Minatar"]
venv_manager = ["MLP", "CNN"]

for nn in ["MLP", "CNN"]:
    dfnn = df.loc[(df["nn"] == nn)]
    res[("Minatar", nn)] = count_groups(dfnn)


step1 = 9
step2 = 1
xs = {}
for i, env in enumerate(envs):
    for j,venv in enumerate(venv_manager):
        xs[(env,venv)] = i * step1 + j* step2 

nums = [ len(res[(env, venv)][1]) for env in envs for venv in venv_manager]

xticks = {"MLP":0, "CNN":1}

ax.text(xticks["MLP"], N+5, nums[0])
ax.text(xticks["CNN"], N+5, nums[1])

ax.set_xticks([0,1])
ax.set_xticklabels(["MLP", "CNN"])
ax.text(-0.75, N+5.1, "Nb groups")
ax.set_ylabel("Nb Computer")


width = 0.9

make_bars_coherent(nums,xs, names, envs, venv_manager, width, res, ax)

fig.savefig(sys.argv[1]+"/bars_g5k_minatar.pdf", bbox_inches="tight")

############# Acrobot_GPU ###################

df = read_acrobot_gpu_from_dir(sys.argv[2])
N = len(np.unique(df["computer_name"]))
print("============= Acrobot GPU==============")
print(f"Acrobot_gpu", count_groups(df), "over", len(np.unique(df["computer_name"])))

fig, ax = plt.subplots(figsize=(6,6))
current_height = 0

envs = ["Acrobot"]
venv_manager = ["GPU"]

res =  {("Acrobot", "GPU"):count_groups(df)}
nums = len(res)
bottom = 0
width = 0.9
idcol = 0

step1 = 9
step2 = 1.35
xs = {}
for i, env in enumerate(envs):
    for j,venv in enumerate(venv_manager):
        xs[(env,venv)] = i * step1 + j* step2 
print(xs)
nums = [ len(res[(env, venv)][1]) for env in envs for venv in venv_manager]

xticks = {"Acrobot GPU": 0}

ax.text(-0.02, N*(1.1), nums[0])

ax.set_xticks([0])
ax.set_xticklabels(["Acrobot GPU"])
ax.text(-0.6, N*(1.1), "Nb groups")
ax.set_ylabel("Nb Computer")


width = 0.9

make_bars_coherent(nums,xs, names, envs, venv_manager, width, res, ax)

fig.savefig(sys.argv[1]+"/bars_g5k_acrobot_GPU.pdf", bbox_inches="tight")
