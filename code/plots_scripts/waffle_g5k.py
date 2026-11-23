import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import seaborn as sns
import distinctipy
from distinctipy import examples
from pywaffle import Waffle
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

def count_groups(df):
    times = np.unique(df["episode_step"])
    n_times = len(times)
    times = times[np.arange(0,n_times,n_times//10).astype(int)]

    res = []
    max_counts = []
    groups= None
    for t in times:
        data_at_time = df.loc[df["episode_step"] == t]
        num, ids = get_num_curves(np.array(data_at_time["r"]), 0)
        unique, indices, inverse, counts = np.unique(ids, return_index=True, return_inverse=True,return_counts=True)
        if len(counts) > len(max_counts):
            max_counts = counts
            max_inverse = inverse
            groups = {j: list(data_at_time.loc[ids == j, "computer_name"]) for j in np.unique(ids)}
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

cm = distinctipy.get_colors(40, pastel_factor=0.75,rng=3)

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

def name_to_id(name, names):
    return np.where(np.array(names) == name)[0][0]

def make_waffles(nums,xs, names, width, res):
    toplot = {}
    for env, venv in list(res.keys()):
        mydf = pd.DataFrame()
        n, max_counts, max_inverse, groups = res[(env, venv)]
        for idgroup in list(groups.keys()):
            group = groups[idgroup]
            dfcount = pd.DataFrame({"id_group": [idgroup]*len(group),
                                    "color": [cm[idgroup]]*len(group),
                                    "env": [env]*len(group),
                                    "venv": [venv]*len(group),
                                    "name_id": [name_to_id(name, names) for name in group],
                                    "cpu":[ name_to_cpu(name) for name in group]
                                    })
            mydf = pd.concat([mydf, dfcount], ignore_index = True)
        toplot[(env, venv)] = mydf
    print("Plotting Waffle")
    nenvs = len(envs)
    nvenvs = len(venv_manager)
    fig, axes = plt.subplots(nvenvs,nenvs,figsize=(12, 12),facecolor='#DDDDDD')
    for i, env in enumerate(envs):
        for k, venv in enumerate(venv_manager):
            mydf = toplot[(env, venv)]
            plot={
                  "values":np.ones(len(mydf)),
                  "colors":list(mydf["color"]),
                  # "characters":[str(mid) for mid in mydf["name_id"]],
                  "font_size":11,
                  "icon_style":"regular",
                  'title': {'label': env + "_"+venv, 'loc': 'left', 'fontsize': 12}
                  }
            Waffle.make_waffle(axes[k,i],
                               **plot,
                               alpha = 0.5,
                               rows=6,
                               block_arranging_style='snake',
                               cmap_name="Accent")

    for i, env in enumerate(envs):
        for k, venv in enumerate(venv_manager):
            mydf = toplot[(env, venv)]
            plot={
                  "values":np.ones(len(mydf)),
                  "colors": ["k"]*len(mydf),
                  "characters":[str(mid) for mid in mydf["name_id"]],
                  "font_size":9,
                  # "icon_style":"regular",
                  'title': {'label': env + "_"+venv, 'loc': 'left', 'fontsize': 12}
                  }
            Waffle.make_waffle(axes[k,i],
                               **plot,
                               rows=6,
                               block_arranging_style='snake',
                               cmap_name="Accent")


    # fig = plt.figure(
    #     FigureClass=Waffle,
    #     plots = plots,
    #     rows=6,
        
    #     alpha=0.5,
    #     ,
    # )

    # import pdb; breakpoint()

    return fig
        
    
####################### Do plot ###########################"

# fig, ax = plt.subplots(figsize=(16,8))
fig = make_waffles(nums,xs, names,1.1, res)
fig.savefig("test_waffle.pdf")


fig, ax = plt.subplots(figsize=(12,10))

# Hide axes
ax.axis('off')

# Create table
table = ax.table(
    cellText=[[name_to_id(name, names),name, name_to_cpu(name)] for name in names],
    colLabels=["Id", "name", "cpu"],
    loc='center'
)

# Adjust style
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(3)))
fig.savefig("table_ids.pdf")

# xticksx = [xs[(env,venv)]  for env, venv in xs.keys() ]
# ax.set_xticks(xticksx)
# ax.text(xticksx[3]-0.5, -4, "Acrobot", ha='center', va='top')
# ax.text(xticksx[9]-0.5, -4, "Mujoco", ha='center', va='top')
# ax.text(xticksx[15]-0.5 , -4, "Atari", ha='center', va='top')

# for j in range(len(nums)):
#     ax.text(xticksx[j]-0.1, len(names)+3, nums[j])

# ax.text(-2.25, len(names)+3, "Nb groups")
# ax.set_xticklabels(xlabels)

# ax.set_ylabel("Computer number")

# make_bars(nums,xs, names,1.1, res, ax, id_computer=False)

# fig.savefig(sys.argv[1]+"/bars_g5k_intel_v1.pdf", bbox_inches="tight")

# fig, ax = plt.subplots(figsize=(16,8))
# xticksx = [xs[(env,venv)]  for env, venv in xs.keys() ]
# ax.set_xticks(xticksx)
# ax.text(xticksx[3]-0.5, -4, "Acrobot", ha='center', va='top')
# ax.text(xticksx[9]-0.5, -4, "Mujoco", ha='center', va='top')
# ax.text(xticksx[15]-0.5 , -4, "Atari", ha='center', va='top')

# for j in range(len(nums)):
#     ax.text(xticksx[j]-0.1, len(names)+3, nums[j])

# ax.text(-2.25, len(names)+3, "Nb groups")
# ax.set_xticklabels(xlabels)

# ax.set_ylabel("Computer id")

# make_bars(nums,xs, names,1.1, res, ax, id_computer=True)

# fig.savefig(sys.argv[1]+"/bars_g5k_intel_v2.pdf", bbox_inches="tight")

# print("Number of distinct curves over intersection -- AMD ")
# print("====================================================")


# envs = ["Acrobot", "Mujoco", "Atari"]
# venv_manager = ["guix", "pip_frozen","conda_frozen", "pip_unfrozen", "conda_unfrozen"]
# xlabels = ["guix",  "pip frz", "conda frz", "pip", "conda"]*3


# names = amd_names
# for env in envs:
#     dfs = []
#     fig, axes = plt.subplots(3,2,figsize=(15,5))
#     for venv in venv_manager:
#         df_env = df.loc[(df["env"]==env) & (df["venv"]==venv)]
#         names = intersect(names, np.unique(df_env["computer_name"]))

# res = {}
# for env in envs:
#     dfs = []
#     fig, axes = plt.subplots(3,2,figsize=(15,5))
#     for venv in venv_manager:
#         df_env = df.loc[(df["env"]==env) & (df["venv"]==venv)]
#         df_env = df_env.loc[isin(df_env["computer_name"], names)]
#         res[(env,venv)] = count_groups(df_env)

# fig, ax = plt.subplots(figsize=(16,8))

# # cm = plt.get_cmap("tab20")
# current_height = {(env, venv):0 for env in envs for venv in venv_manager}

# dfs_toplot = []
# step1 = 8
# step2 = 1.35
# xs = {}
# for i, env in enumerate(envs):
#     for j,venv in enumerate(venv_manager):
#         xs[(env,venv)] = i * step1 + j* step2 

# nums = [ len(res[(env, venv)][1]) for env in envs for venv in venv_manager]

# fig, ax = plt.subplots(figsize=(16,8))
# xticksx = [xs[(env,venv)]  for env, venv in xs.keys() ]
# ax.set_xticks(xticksx)
# ax.text(xticksx[2]-0.5, -1, "Acrobot", ha='center', va='top')
# ax.text(xticksx[7]-0.5, -1, "Mujoco", ha='center', va='top')
# ax.text(xticksx[12]-0.5 , -1, "Atari", ha='center', va='top')

# for j in range(len(nums)):
#     ax.text(xticksx[j]-0.1, len(names)+1, nums[j])

# ax.text(-2.25, len(names)+1, "Nb groups")
# ax.set_xticklabels(xlabels)

# ax.set_ylabel("Computer number")

# make_bars(nums,xs, names,1.1, res, ax, id_computer=False,n_workflow=5)

# fig.savefig(sys.argv[1]+"/bars_g5k_amd_v1.pdf", bbox_inches="tight")


# fig, ax = plt.subplots(figsize=(16,8))
# xticksx = [xs[(env,venv)]  for env, venv in xs.keys() ]
# ax.set_xticks(xticksx)
# ax.text(xticksx[2]-0.5, -1, "Acrobot", ha='center', va='top')
# ax.text(xticksx[7]-0.5, -1, "Mujoco", ha='center', va='top')
# ax.text(xticksx[12]-0.5 , -1, "Atari", ha='center', va='top')

# for j in range(len(nums)):
#     ax.text(xticksx[j]-0.1, len(names)+1, nums[j])

# ax.text(-2.25, len(names)+1, "Nb groups")
# ax.set_xticklabels(xlabels)

# ax.set_ylabel("Computer id")

# make_bars(nums,xs, names,1.1, res, ax, id_computer=True,n_workflow=5)

# fig.savefig(sys.argv[1]+"/bars_g5k_amd_v2.pdf", bbox_inches="tight")


# #################################### ALL sauf avx


# print("Number of distinct curves over intersection -- No avx ")
# print("====================================================")

# names = np.unique(df["computer_name"])
# for env in envs:
#     dfs = []
#     fig, axes = plt.subplots(3,2,figsize=(15,5))
#     for venv in venv_manager:
#         if "AVX" not in venv:
#             df_env = df.loc[(df["env"]==env) & (df["venv"]==venv)]
#             names = intersect(names, np.unique(df_env["computer_name"]))

# res = {}
# for env in envs:
#     dfs = []
#     fig, axes = plt.subplots(3,2,figsize=(15,5))
#     for venv in venv_manager:
#         df_env = df.loc[(df["env"]==env) & (df["venv"]==venv)]
#         df_env = df_env.loc[isin(df_env["computer_name"], names)]
#         res[(env,venv)] = count_groups(df_env)

# # fig, ax = plt.subplots(figsize=(16,8))

# step1 = 9
# step2 = 1.35
# xs = {}
# for i, env in enumerate(envs):
#     for j,venv in enumerate(venv_manager):
#         xs[(env,venv)] = i * step1 + j* step2 

# nums = [ len(res[(env, venv)][1]) for env in envs for venv in venv_manager]


# fig, ax = plt.subplots(figsize=(16,8))
# xticksx = [xs[(env,venv)]  for env, venv in xs.keys() ]
# ax.set_xticks(xticksx)
# ax.text(len(xlabels) / 8, -4, "Acrobot", ha='center', va='top')
# ax.text(len(xlabels)* (1/ 2+1/32), -4, "Mujoco", ha='center', va='top')
# ax.text(len(xlabels)*(31/32) , -4, "Atari", ha='center', va='top')

# for j in range(len(nums)):
#     ax.text(xticksx[j]-0.1, len(names)+3.5, nums[j])

# ax.text(-2.25, len(names)+3.5, "Nb groups")
# ax.set_xticklabels(xlabels)

# # ax.set_xlabel("Environment",labelpad=15)
# ax.set_ylabel("Computer number")

# make_bars(nums,xs, names,1.1, res, ax, id_computer=False,n_workflow=5)

# fig.savefig(sys.argv[1]+"/bars_g5k_noavx_v1.pdf", bbox_inches="tight")



# fig, ax = plt.subplots(figsize=(16,8))
# xticksx = [xs[(env,venv)]  for env, venv in xs.keys() ]
# ax.set_xticks(xticksx)
# ax.text(len(xlabels) / 8, -4, "Acrobot", ha='center', va='top')
# ax.text(len(xlabels)* (1/ 2+1/32), -4, "Mujoco", ha='center', va='top')
# ax.text(len(xlabels)*(31/32) , -4, "Atari", ha='center', va='top')

# for j in range(len(nums)):
#     ax.text(xticksx[j]-0.1, len(names)+3.5, nums[j])

# ax.text(-2.25, len(names)+3.5, "Nb groups")
# ax.set_xticklabels(xlabels)

# # ax.set_xlabel("Environment",labelpad=15)
# ax.set_ylabel("Computer id")

# make_bars(nums,xs, names,1.1, res, ax, id_computer=True,n_workflow=5, n_extra=0)

# fig.savefig(sys.argv[1]+"/bars_g5k_noavx_v2.pdf", bbox_inches="tight")




# ############# Minatar ###################
# print("============= Minatar ==============")

# df = read_minatar_from_dir(sys.argv[2])
# for j, nn in enumerate(["MLP", "CNN"]):
#     dfnn = df.loc[(df["nn"] == nn)]
#     N = len(np.unique(dfnn["computer_name"]))
#     print(f"Minatar {nn}", count_groups(dfnn), "over", len(np.unique(dfnn["computer_name"])))

# print(np.unique(dfnn["computer_name"]))
# fig, ax = plt.subplots(figsize=(10,4))
# current_height = {"MLP":0, "CNN":0}

# res = {}
# for nn in ["MLP", "CNN"]:
#     dfnn = df.loc[(df["nn"] == nn)]
#     res[nn] = count_groups(dfnn)[1]
# nums = [ len(res[nn]) for nn in ["MLP", "CNN"]]
# bottom = np.zeros(2)
# width = 0.9
# idcol = 0

# xticks = {"MLP":0, "CNN":1}

# while len(res) > 0:
#     df = pd.DataFrame({
#         "x":[ xticks[nn] for nn in res.keys()],
#         "Counts":[res[nn][0] for nn in res.keys()],
#         "env": [env for nn in res.keys()],
#                        })


#     p = ax.bar(df["x"], df["Counts"], width, bottom=bottom, color=cm[idcol], alpha=0.6)
#     # ax.bar_label(p, label_type='center')
#     idcol += 1

#     bottom = bottom + df["Counts"]

#     reskeys = list(res.keys())
#     todel = []
#     for j, k in enumerate(reskeys):
#         if len(res[k]) == 1:
#             del res[k]
#             todel.append(j)
#         else:
#             res[k] = res[k][1:]
#     bottom = np.array([bottom[i] for  i in range(len(bottom)) if i not in todel ])

# ax.text(xticks["MLP"], N+5, nums[0])
# ax.text(xticks["CNN"], N+5, nums[1])

# ax.set_xticks([0,1])
# ax.set_xticklabels(["MLP", "CNN"])
# ax.text(-0.75, N+5.1, "Nb groups")
# ax.set_ylabel("Computer id")

# fig.savefig(sys.argv[1]+"/bars_g5k_minatar.pdf", bbox_inches="tight")

# ############# Acrobot_GPU ###################

# df = read_acrobot_gpu_from_dir(sys.argv[2])
# N = len(np.unique(df["computer_name"]))
# print("============= Acrobot GPU==============")
# print(f"Acrobot_gpu", count_groups(df), "over", len(np.unique(df["computer_name"])))

# fig, ax = plt.subplots(figsize=(6,6))
# current_height = 0
# res =  count_groups(df)[1]
# nums = len(res)
# bottom = 0
# width = 0.9
# idcol = 0

# xticks = {"Acrobot GPU":0}

# while len(res) > 0:
#     df = pd.DataFrame({
#         "x":[0],
#         "Counts":[res[0]],
#         "env": ["Acrobot GPU"]})

#     p = ax.bar(df["x"], df["Counts"], width, bottom=bottom, color=cm[idcol], alpha=0.6)
#     # ax.bar_label(p, label_type='center')
#     idcol += 1

#     bottom = bottom + df["Counts"]

#     if len(res) == 1:
#         break
#     else:
#         res = res[1:]

# ax.text(0, N+3.3, nums)

# ax.set_xticks([0])
# ax.set_xticklabels(["Acrobot GPU"])
# ax.text(-0.6, N+3.3, "Nb groups")
# ax.set_ylabel("Computer id")

# fig.savefig(sys.argv[1]+"/bars_g5k_acrobot_GPU.pdf", bbox_inches="tight")
