import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import glob
import statsmodels.api as sm
import seaborn as sns
from stats_tools import compare_workflows, plot_ci
import sys
import os

"""
Create figures to show the variations of the results for the Mujoco environments, for SAC and PPO, based on the machine, the workflow used, and the seed.
    - TODO detailler les différentes figures générées
"""


print("--- starting plot_stats_mujoco.py : generating figures for the variations of the results for the Mujoco environments, for SAC and PPO, based on the machine, the workflow used, and the seed ---")

for env in ["HalfCheetah-v5", "Hopper-v5"]:
    df = pd.DataFrame()
    for fname in glob.glob(f"../results/mujoco_sac_vs_ppo/*_{env}_*.csv"):
        data = np.loadtxt(fname)
        workflow = " ".join(fname.split("/")[-1].split("_")[:-3])
        df = pd.concat([df, 
                        pd.DataFrame({
                            "workflow": [workflow] * len(data),
                            "reward": data,
                            "algo" : [workflow.split(" ")[0]]*len(data),
                            "machine": [" ".join(workflow.split(" ")[2:])]*len(data),
                            "seed" : np.arange(len(data))
                        })], ignore_index=True)

    # Make our analysis on differences
    dfdiff = pd.DataFrame()
    for machine in df["machine"].unique():
        dfmachine = df.loc[df["machine"]==machine]
        sac_reward = np.array(dfmachine.sort_values(by=["seed"]).loc[dfmachine["algo"]=="sac", "reward"])
        ppo_reward = np.array(dfmachine.sort_values(by=["seed"]).loc[dfmachine["algo"]=="ppo", "reward"])
        n_min = min(len(sac_reward), len(ppo_reward))
        dfdiff = pd.concat([dfdiff, 
                            pd.DataFrame({
                            "workflow": [workflow] *n_min,
                            "diff_reward":  sac_reward[:n_min]-ppo_reward[:n_min] ,
                            "machine": [machine]*n_min
                        })], ignore_index=True)

    
    for n_seeds in [10,30]:
        fig, axes = plt.subplots(2, 1, figsize=(8,7), sharex=True)
        ax = axes[1]
        machines_names = dfdiff["machine"].unique()
        for j, machine in enumerate(machines_names):
            values = np.array(dfdiff.loc[(dfdiff["machine"]==machine), "diff_reward"])[:n_seeds]
            ci = stats.bootstrap((values,), np.mean).confidence_interval
            plot_ci(ci[0], ci[1], np.mean(dfdiff.loc[(dfdiff["machine"]==machine), "diff_reward"]), j, ax = ax )
        ax.set_yticks(np.arange(len(machines_names)), machines_names)

        ax = axes[0]
        for j, machine in enumerate(machines_names):
            sns.histplot(np.array(dfdiff.loc[dfdiff["machine"]==machine, "diff_reward"])[:n_seeds],bins=15,  ax = ax)

        output_path = os.path.join(sys.argv[1], f"mujoco_{env}_{n_seeds}_CI_bootstrap.pdf")
        fig.savefig(f"{sys.argv[1]}/mujoco_{env}_{n_seeds}_CI_bootstrap.pdf", bbox_inches='tight')
        print("saved figure to : ", output_path)

        # Plot of individual, grouped, not differences
        for algo in ["ppo", "sac"]:
            fig, ax = plt.subplots(1, 1, figsize=(8,3.5), sharex=True)
            dfalgo = df.loc[df["algo"]==algo]
            machines_names = dfalgo["machine"].unique()
            minx = np.min(dfalgo["reward"])
            maxx = np.max(dfalgo["reward"])
            for j, machine in enumerate(machines_names):
                values = np.array(dfalgo.loc[(dfalgo["machine"]==machine), "reward"])[:n_seeds]
                if len(set(values))>1:
                    ci = stats.bootstrap((values,), np.mean).confidence_interval
                else:
                    ci = [values[0],values[0]]
                plot_ci(ci[0], ci[1], np.mean(dfalgo.loc[(dfalgo["machine"]==machine), "reward"]), j, ax = ax )

            ax.set_yticks(np.arange(len(machines_names)), machines_names)
            output_path = os.path.join(sys.argv[1], f"mujoco_individual_{algo}_{env}_{n_seeds}_CI_bootstrap.pdf")
            fig.savefig(output_path, bbox_inches='tight')
            print("saved figure to : ", output_path)

        # Plot of both, grouped, not differences
        fig, ax = plt.subplots(1, 1, figsize=(8,3.5), sharex=True)
        id_alg_machine = 0
        names = []
        machines_names = dfalgo["machine"].unique()
        minx = np.min(df["reward"])
        maxx = np.max(df["reward"])
        for algo in ["ppo", "sac"]:
            dfalgo = df.loc[df["algo"]==algo]
            for j, machine in enumerate(machines_names):
                values = np.array(dfalgo.loc[(dfalgo["machine"]==machine), "reward"])[:n_seeds]
                if len(set(values))>1:
                    ci = stats.bootstrap((values,), np.mean).confidence_interval
                else:
                    ci = [values[0],values[0]]
                plot_ci(ci[0], ci[1], np.mean(dfalgo.loc[(dfalgo["machine"]==machine), "reward"]), 
                        id_alg_machine, ax = ax)
                names.append((id_alg_machine,f"{'w/ avx512' if  machine.split(' ')[0] == 'flanders' else 'w/o avx 512'}: {algo}"))
                id_alg_machine += 1
        ax.set_yticks([i for i, a in names], [a for i,a  in names])
        output_path = os.path.join(sys.argv[1], f"mujoco_all_algo_{env}_{n_seeds}_CI_bootstrap.pdf")
        fig.savefig(output_path, bbox_inches='tight')
        print("saved figure to : ", output_path)
