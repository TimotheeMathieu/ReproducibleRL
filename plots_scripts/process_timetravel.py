import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns
from tqdm import tqdm
import glob
import sys

df = pd.DataFrame()

def commit_to_date(commit):
    if commit == "720d2b57eb":
        return "20/10/2025"
    elif commit == "801d1108b5":
        return "08/08/2025"
    elif commit == "81d309de87":
        return "08/06/2025"
    elif commit == "a0a9044d35":
        return "22/01/2025"
    elif commit == "b44b2e346c":
        return "04/08/2024"
    elif commit == "7b62d614e7":
        return "23/03/2024"
    else:
        return None

for commit in list(glob.glob(f"{sys.argv[1]}/*")):
    data = pd.read_csv(commit+"/Acrobot-v1/archlinux/0.monitor.csv", header=0,skiprows=[0])
    data["date"] = commit_to_date(commit.split("/")[-1])
    data["episode_step"] = np.arange(len(data))
    df = pd.concat([df, data], ignore_index=True)

print(df)

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
        counts = np.unique_counts(ids).counts
        if len(counts) > len(max_counts):
            max_counts = counts
            groups = {j: list(data_at_time.loc[ids == j, "date"]) for j in np.unique(ids)}
        res.append(num)
    print(groups)
    return np.max(res), np.sort(max_counts)

print(count_groups(df))
