import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import seaborn as sns
import distinctipy
from distinctipy import examples
from pywaffle import Waffle
from read_data import read_data_from_dir, read_minatar_from_dir, read_acrobot_gpu_from_dir
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly_tools import plotly_figs


######## Loading Data #############

# Load informations about computing nodes
# df_g5k = pd.concat([pd.read_csv("data_g5k_hardware/grid5000_default_queue_resources.csv"),
#                     pd.read_csv("data_g5k_hardware/grid5000_abaca_queue_resources.csv")], ignore_index = True)
df_g5k = pd.concat([pd.read_csv("./data_g5k_hardware/grid5000_default_queue_resources.csv"),
                    pd.read_csv("./data_g5k_hardware/grid5000_abaca_queue_resources.csv")], ignore_index = True)

# Load results
df = read_data_from_dir(sys.argv[2])
# df = read_data_from_dir("./code/results/results_g5k")
df["r"] = df["r"].astype(np.float64)


########## Tools ###########

# Get the CPU typre from Name of the machine
def name_to_cpu(name):
    return df_g5k.loc[df_g5k["Cluster"]==name, "CPU.1"].item()

def name_to_gpu(name):
    return df_g5k.loc[df_g5k["Cluster"]==name, "Accelerators"].item()

def name_to_ram(name):
    return df_g5k.loc[df_g5k["Cluster"]==name, "Memory"].item()

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
    times = times[np.arange(0,n_times,n_times//10).astype(int)]

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

# return the list of items present in the two lists passed as parameters
def intersect(names1, names2):
    return list(set(names1).intersection(set(names2)))

# Return a boolean mask of 'names1' if the value of the element is (or no) in the names2
def isin(names1, names2):
    return np.array([n in list(names2) for n in list(names1)])

def name_to_id(name, names):
    return np.where(np.array(names) == name)[0][0]

########### MAIN FUNCTION ##############

# create Waffle graph 
def make_waffles(names, res, envs, venv_managers):

    def _best_shape(n, target_ratio=2/3):

        best_shape=None        
        for nb_rows in range(3,n+1):    #start at 3 to skip the case with prime number
            nb_cols=int(np.ceil(n/nb_rows))

            if nb_cols < nb_rows : continue
            
            total=nb_rows*nb_cols
            nb_empty_cell=total-n 
            ratio_diff=abs((nb_rows/nb_cols)-target_ratio)  #how close from the ratio?
            
            current_score=(nb_empty_cell,ratio_diff)  #use tuple to compare with first then second criteria
            
            if best_shape is None or current_score < best_shape[0]:     #if "less empty cell, or closer to expected ratio"
                best_shape = (current_score, nb_rows, nb_cols)
    
        return best_shape[1], best_shape[2] #nb_rows,nb_cols

    def _reshape_ratio(vec, target_ratio=2/3):
        """
        convert a vector to a matrix with a ratio target (complete with NaN)
        """
        n = len(vec)
        rows, cols = _best_shape(n, target_ratio)
        
        # create a NaN matrix, then fill it up in "zigzag"
        mat = np.full((rows, cols),"",dtype=object)
        idx = 0
        for j in range(cols):
            if j % 2 == 0:
                # column down
                for i in reversed(range(rows)):
                    if idx < n:
                        mat[i, j] = vec[idx]
                        idx += 1
            else:
                # column up 
                for i in range(rows):
                    if idx < n:
                        mat[i, j] = vec[idx]
                        idx += 1
        
        return mat

    def _get_converted_colors(base_colors):
        my_colorsc=[]
        max_colors = len(base_colors)
        for color_id in range(0,max_colors):
            (r, g, b) = cm[color_id]
            r_i = int(r * 255)
            g_i = int(g * 255)
            b_i = int(b * 255)        
            new_color = f'rgb({r_i},{g_i},{b_i})'        
            my_colorsc.append([color_id/max_colors, new_color])
            my_colorsc.append([(color_id+1)/max_colors, new_color])
        return my_colorsc

    def _extract_info_to_plot(results):
        toplot={}
        for env, venv in results.keys():
            mydf = pd.DataFrame()
            n, max_counts, max_inverse, groups = results[(env, venv)]

            color_id_group = 0
            for idgroup in groups.keys():
                group = groups[idgroup]
                dfcount = pd.DataFrame({"color_id_group": [color_id_group]*len(group),
                                        "env": [env]*len(group),
                                        "venv": [venv]*len(group),
                                        "name_id": [name_to_id(name, names) for name in group],
                                        "name": [name for name in group],
                                        })
                dfcount = dfcount.sort_values(["name_id"])
                mydf = pd.concat([mydf, dfcount], ignore_index = True)
                color_id_group = color_id_group +1
            toplot[(env, venv)] = mydf        
        return toplot

    #Setup colors
    max_colors = max((v[0] for v in res.values()), default=None)
    cm = distinctipy.get_colors(max_colors, pastel_factor=0.75,rng=3)
    my_colorsc=_get_converted_colors(cm)

    toplot=_extract_info_to_plot(res)
    
    print("*** Plotting Waffle ***")

    nenvs = len(envs)
    nvenvs = len(venv_managers)

    # setup titles for subplots
    subplot_titles=[]
    for venv in venv_managers:
        for env in envs:
            number_of_cluster = max(toplot[(env, venv)]["color_id_group"])+1
            subplot_titles.append(f"{env}_{venv} : {number_of_cluster} cluster(s)")
    fig = make_subplots(nvenvs,nenvs, subplot_titles=subplot_titles, 
                        vertical_spacing=0.05, horizontal_spacing=0.03)

    # add the waffle graph to each subplot (1 by xp)
    for i, env in enumerate(envs):
        for k, venv in enumerate(venv_managers):
            mydf = toplot[(env, venv)]

            color_id_group_reorganized = _reshape_ratio(mydf["color_id_group"])
            text_reorganized = _reshape_ratio([str(mid) for mid in mydf["name_id"]])
            info_reorganized = _reshape_ratio([f"Hostname:{name},<br>CPU:{name_to_cpu(name)},<br>Accelerator:{name_to_gpu(name)},<br>Mem:{name_to_ram(name)}" for name in mydf["name"]])

            fig.add_trace(
                    go.Heatmap(
                    z=color_id_group_reorganized,
                    text=text_reorganized,
                    texttemplate="%{text}",
                    textfont={"size":19},
                    xgap=1,
                    ygap=1,
                    coloraxis="coloraxis",
                    name= env+"_"+venv,
                    customdata=info_reorganized,
                    hovertemplate=
                        "<b>%{fullData.name}</b><br>" +
                        "Machine_ID: %{text}<extra></extra><br>" +
                        "Cluster_number: %{z}<br>" +
                        "%{customdata}<br>"
                    ),
                row=k+1,
                col=i+1,)

    fig.update_xaxes(showticklabels=False, ticks="")
    fig.update_yaxes(showticklabels=False, ticks="")
    fig.update_layout(
        coloraxis={
            "colorscale":my_colorsc,
            "cmin":0,
            "cmax":max_colors
            },
        plot_bgcolor="#FFFFFF",
        height=max(190*len(venv_managers),650), 
        hoverlabel_font_size=15,
    )
    return fig

###################### End main function ######################

###################### start MAIN process ######################



all_the_names = np.unique(df["computer_name"])
intel_names = [name for name in all_the_names if "AMD" not in name_to_cpu(name)]
amd_names = [name for name in all_the_names if "AMD" in name_to_cpu(name)]

proc_cluster = [("Intel + AMD (w/o masked AVX)",all_the_names),("Intel",intel_names),("AMD (w/o masked AVX)",amd_names)]


myfigs = plotly_figs(offline=True)



for proc_name,names in proc_cluster:

    envs = ["Acrobot", "Mujoco", "Atari"]
    venv_managers= ["guix_with_masked_AVX", "guix", "pip_frozen", "pip_unfrozen","conda_frozen",  "conda_unfrozen"]

    if proc_name!="Intel" and venv_managers[0]=="guix_with_masked_AVX":
        del venv_managers[0]

    if proc_name=="Intel" and venv_managers[0]!="guix_with_masked_AVX":
        venv_managers.insert(0,"guix_with_masked_AVX")

    print("Number of distinct curves over intersection -- " + proc_name)
    print("=====================================================")

    # get the names of computers that have results ON EACH XP
    for env in envs:
        for venv in venv_managers:
            df_env = df.loc[(df["env"]==env) & (df["venv"]==venv)]
            names = intersect(names, np.unique(df_env["computer_name"]))

    names.sort()

    print("*********************************")

    # compute and get the clusters results for each env / venv
    res = {}
    for env in envs:
        for venv in venv_managers:
            df_env = df.loc[(df["env"]==env) & (df["venv"]==venv)]
            df_env = df_env.loc[isin(df_env["computer_name"], names)]
            res[(env,venv)] = count_groups(df_env)

    print("*********************************")

    ####################### Do plot ###########################"

    fig = make_waffles(names, res,  envs, venv_managers)    
    fig.update_layout(
        title={
            "text": proc_name,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size":30},
            "automargin":True,
            "y":0.975
        },
    )

    myfigs.add_fig(fig, proc_name)


##################### Minatar ###################

df = read_minatar_from_dir(sys.argv[2])
for j, nn in enumerate(["MLP", "CNN"]):
    dfnn = df.loc[(df["nn"] == nn)]
    N = len(np.unique(dfnn["computer_name"]))

names = np.unique(dfnn["computer_name"])

envs = ["Breakout Minatar"]
venv_managers= ["MLP", "CNN"]

res = {}
for nn in ["MLP", "CNN"]:
    dfnn = df.loc[(df["nn"] == nn)]
    res[("Breakout Minatar", nn)] = count_groups(dfnn)
 
fig = make_waffles(names, res,  envs, venv_managers)        
fig.update_layout(
    title={
        "text": "Minatar CNN VS MLP",
        "x": 0.5,
        "xanchor": "center",
        "yanchor": "top",
        "font": {"size":30},
        "automargin":True,
        "y":0.975
    },
)


myfigs.add_fig(fig, "Minatar")


####### Acrobot GPU #########



df = read_acrobot_gpu_from_dir(sys.argv[2])
N = len(np.unique(df["computer_name"]))
names = np.unique(df["computer_name"])

envs = ["Acrobot"]
venv_managers= ["GPU"]

print("============= Acrobot GPU==============")
print(f"Acrobot_gpu", count_groups(df), "over", len(np.unique(df["computer_name"])))

res = {("Acrobot", "GPU"): count_groups(df)}
 
fig = make_waffles(names, res,  envs, venv_managers)     
   
fig.update_layout(
    title={
        "text": "Acrobot GPU",
        "x": 0.5,
        "xanchor": "center",
        "yanchor": "top",
        "font": {"size":30},
        "automargin":True,
        "y":0.975
    },
)

myfigs.add_fig(fig, "Acrobot GPU")


myfigs.save_to_file(sys.argv[1]+"/g5k_website.html")
