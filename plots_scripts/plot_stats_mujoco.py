import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import glob
import statsmodels.api as sm
import seaborn as sns
from stats_tools import compare_workflows, plot_ci
import sys

for env in ["HalfCheetah-v5", "Hopper-v5"]:
    df = pd.DataFrame()
    for fname in glob.glob(f"../results/mujoco_sac_vs_ppo/*_{env}_*.csv"):
        data = np.loadtxt(fname)
        workflow = " ".join(fname.split("/")[-1].split("_")[:-3])
        print(workflow, "n data is ", len(data))
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

    ## Some basic stats
    # fig, axes = plt.subplots(1,5,figsize=(12,5), sharey=True, sharex=True)
    # axes = axes.ravel()
    # for i, machine in enumerate(dfdiff["machine"].unique()):
    #     sns.histplot(dfdiff.loc[dfdiff["machine"]==machine, "diff_reward"],bins=15,  ax = axes[i])
    #     axes[i].set_title(machine)

    # fig.savefig(f"{sys.argv[1]}/mujoco_{env}_histograms.pdf")

    fig, axes = plt.subplots(1,5,figsize=(12,5), sharey=True, sharex=True)
    axes = axes.ravel()
    for i, machine in enumerate(dfdiff["machine"].unique()):
        fig = sm.qqplot(dfdiff.loc[dfdiff["machine"]==machine, "diff_reward"], line = "s", ax = axes[i])
        axes[i].set_title(machine)
    fig.savefig(f"{sys.argv[1]}/mujoco_{env}_qqplot.pdf")

    # fig, ax = plt.subplots(figsize=(8,5))
    # sns.boxplot(dfdiff, x="machine", y="diff_reward", ax = ax)

    # fig.savefig(f"{sys.argv[1]}/mujoco_{env}_boxplots.pdf")
    # Seems not too far from Gaussian

    ## If we beleive that Gaussian nonetheless, can do Anova

    print("For environment ", env, "Anova results are ",stats.f_oneway(
        *[dfdiff.loc[dfdiff["machine"]==machine, "diff_reward"] for machine in dfdiff["machine"].unique()]
    ))

    ## Anova stats says that all equal

    for n_seeds in [5, 10,30, 50]:
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

        fig.savefig(f"{sys.argv[1]}/mujoco_{env}_{n_seeds}_CI_bootstrap.pdf", bbox_inches='tight')

        # Plot of individual, grouped, not differences
        for algo in ["ppo", "sac"]:
            fig, ax = plt.subplots(1, 1, figsize=(8,3.5), sharex=True)
            dfalgo = df.loc[df["algo"]==algo]
            machines_names = dfalgo["machine"].unique()
            minx = np.min(dfalgo["reward"])
            maxx = np.max(dfalgo["reward"])
            for j, machine in enumerate(machines_names):
                values = np.array(dfalgo.loc[(dfalgo["machine"]==machine), "reward"])[:n_seeds]
                ci = stats.bootstrap((values,), np.mean).confidence_interval
                plot_ci(ci[0], ci[1], np.mean(dfalgo.loc[(dfalgo["machine"]==machine), "reward"]), j, ax = ax )
                # sns.histplot(np.array(dfalgo.loc[dfalgo["machine"]==machine, "reward"])[:n_seeds],bins=np.linspace(minx, maxx,num=20),  ax = axes[0])

            ax.set_yticks(np.arange(len(machines_names)), machines_names)
            fig.savefig(f"{sys.argv[1]}/mujoco_individual_{algo}_{env}_{n_seeds}_CI_bootstrap.pdf", bbox_inches='tight')

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
                ci = stats.bootstrap((values,), np.mean).confidence_interval
                plot_ci(ci[0], ci[1], np.mean(dfalgo.loc[(dfalgo["machine"]==machine), "reward"]), 
                        id_alg_machine, ax = ax)
                names.append((id_alg_machine,f"{'w/ avx512' if  machine.split(' ')[0] == 'flanders' else 'w/o avx 512'}: {algo}"))
                id_alg_machine += 1
        ax.set_yticks([i for i, a in names], [a for i,a  in names])
        fig.savefig(f"{sys.argv[1]}/mujoco_all_algo_{env}_{n_seeds}_CI_bootstrap.pdf", bbox_inches='tight')
