import plotly.graph_objects as go
import pandas as pd
from plotly_tools import plotly_figs
import numpy as np
from tqdm import tqdm
my_figs = plotly_figs()

df_g5k = pd.concat([pd.read_csv("data_g5k_hardware/grid5000_default_queue_resources.csv"),
                    pd.read_csv("data_g5k_hardware/grid5000_abaca_queue_resources.csv")], ignore_index = True)
def make_fig(df):
    fig = go.Figure()
    for name in np.sort(df["computer_name"].unique()):
        hardware = df_g5k.loc[df_g5k["Cluster"]==name.split("-")[0],["CPU.1", "Access Condition"]].values.astype(str)
        if len(hardware)>0:
            hardware = "<br> ".join(list(hardware[0]))
        else:
            hardware = "Unknown"
        fig.add_trace(go.Scatter(
            x = df.loc[df["computer_name"]==name, "episode_step"],
            y = df.loc[df["computer_name"]==name, "episode_rewards"],
            hovertemplate =
            '<i>Reward</i>: %{y:.2f}<br>'+
            '<b>Computer:'+name+'</b><br>'+
            "<i>Specs:</i>"+hardware, name=name,
        ))
    fig.update_layout(hoverdistance=100,legend={'traceorder':'normal', "title": 
                                                f"Computer name ({len(df["computer_name"].unique())} unique PC)"}, 
                      xaxis_title="Environment step", yaxis_title="Reward",
                      width=1000, height=600,)
    return fig


frameworks = {"pip": "../results/Acrobot/Acrobot_Pip_Unfrozen.csv",
              "pip frozen": "../results/Acrobot/Acrobot_Pip_Frozen.csv",
              "guix": "../results/Acrobot/Acrobot_Guix.csv",
              "guix no-avx": "../results/Acrobot/Acrobot_Guix_with_masked_AVX.csv",
              "conda": "../results/Acrobot/Acrobot_Conda_UnFrozen.csv",
              "conda frozen": "../results/Acrobot/Acrobot_Conda_Frozen.csv"
              }

for framework in frameworks:
    df = pd.read_csv(frameworks[framework])
    print(f"Framework {framework} has {len(np.sort(df["computer_name"].unique()))} computers")

figs = {}
for id_framework in frameworks:
    df = pd.read_csv(frameworks[id_framework])
    figs[id_framework] = make_fig(df)

my_figs.add_menu_figs("Framework","Regret curves", figs)



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

def make_fig_epsilon(df):
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
    fig = go.Figure()
    epsilons = [0, 1e-2, 1e-1, 1, 10][::-1]
    for epsilon in tqdm(epsilons):
        res = []
        for t in times:
            data_at_time = df_interp.loc[df_interp["time"] == t]
            num, ids = get_num_curves(np.array(data_at_time["reward"]), epsilon)
            res.append(num)
        fig.add_trace(go.Scatter(
            x = times,
            y = res,
            name=str(epsilon)))
    fig.update_layout(hoverdistance=100,legend={'traceorder':'normal', "title":f"Epsilon ({len(df["computer_name"].unique())} unique PC)"},
                      xaxis_title="Environment step", yaxis_title="Number of group of curves", 
                      width=1000, height= 600)
    return fig


figs = {}
for id_framework in frameworks:
    df = pd.read_csv(frameworks[id_framework])
    figs[id_framework] = make_fig_epsilon(df)

my_figs.add_menu_figs("Framework","Epsilon curves", figs)


def get_hardware(name):
    hardware = df_g5k.loc[df_g5k["Cluster"]==name.split("-")[0],["CPU.1", "Access Condition"]].values.astype(str)
    if len(hardware) >0:
        return (", ".join(list(hardware[0]))).replace(u'\xa0', u' ')
    else:
        return "Unknown"

def make_table_epsilon(df):
    tot_max_time = 40001
    names = []
    for name in np.unique(df["computer_name"]):
        max_time = np.max(df.loc[df["computer_name"]==name, "episode_step"])
        if max_time >= tot_max_time:
            names.append(name)
    print(f"Working with {len(names)} cpus.")
    times = np.linspace(0,tot_max_time, num=2000)
    groups = [np.unique(df["computer_name"])]

    df_interp =  pd.DataFrame()
    for name  in np.unique(names):
        df_name = df.loc[df["computer_name"] == name]
        df_interp = pd.concat([df_interp,
                               pd.DataFrame({
                                   "name":[name]*len(times),
                                   "time": times,
                                   "reward": np.interp(times, df_name["episode_step"], df_name["episode_rewards"])
                               })], ignore_index = True)
    fig = go.Figure()
    data_at_time = df_interp.loc[df_interp["time"] == tot_max_time]
    num, ids = get_num_curves(np.array(data_at_time["reward"]), 0)

    groups = np.array(ids)
    names = np.array(names)
    id_sort = np.argsort(groups)
    cpus = np.array([get_hardware(name) for name in names])
    
    print(pd.DataFrame({"name":names[id_sort],
                        "cpu" : cpus[id_sort],
                        "group":groups[id_sort]
                        }).to_markdown())

    fig = go.Figure(data=[go.Table(header=dict(values=['Name', "cpu", 'Group']),
                 cells=dict(values=[names[id_sort], 
                                    cpus[id_sort],
                                    groups[id_sort]
                                    ]))
                     ])

    return fig

figs = {}
for id_framework in frameworks:
    print(id_framework)
    df = pd.read_csv(frameworks[id_framework])
    figs[id_framework] = make_table_epsilon(df)

my_figs.add_menu_figs("Framework","Groups at max time, slack 0", figs)

my_figs.save_to_file("plots.html")
