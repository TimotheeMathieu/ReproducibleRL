import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.special import binom

def compare_workflows(
    df,
    alpha=0.05,
    B=10_000,
    seed=None,
):
    """
    Permutation test with step-down method, see Testing Statistical Hypotheses by E. L. Lehmann, Joseph P. Romano (Section 15.4.4), https://doi.org/10.1007/0-387-27605-X, Springer

    """


    workflows_names = df["workflow"].unique()
    data = np.array(
        [np.array(df.loc[df["workflow"] == workflow, "reward"]) for workflow in workflows_names]
    )

    n_workflows = len(workflows_names)
    vs = [
        (workflows_names[i], workflows_names[j])
        for i in range(n_workflows)
        for j in range(n_workflows)
        if i < j
    ]

    mean_workflow1 = [
        np.mean(df.loc[df["workflow"] == vs[i][0], "reward"]) for i in range(len(vs))
    ]

    mean_workflow2 = [
        np.mean(df.loc[df["workflow"] == vs[i][1], "reward"]) for i in range(len(vs))
    ]
    mean_diff = [
        np.mean(
            np.array(df.loc[df["workflow"] == vs[i][0], "reward"])
            - np.array(df.loc[df["workflow"] == vs[i][1], "reward"])
        )
        for i in range(len(vs))
    ]
    std_diff = [
        np.std(
            np.array(df.loc[df["workflow"] == vs[i][0], "reward"])
            - np.array(df.loc[df["workflow"] == vs[i][1], "reward"])
        )
        for i in range(len(vs))
    ]


    results_perm = _permutation_test(data, B, alpha, seed) == 1


    decisions = [
        "accept" if results_perm[i][j] else "reject"
        for i in range(n_workflows)
        for j in range(n_workflows)
        if i < j
    ]
    results = pd.DataFrame(
        {
            "Workflow1 vs Workflow2": [
                "{0} vs {1}".format(vs[i][0], vs[i][1]) for i in range(len(vs))
            ],
            "mean Workflow1": mean_workflow1,
            "mean Workflow2": mean_workflow2,
            "mean diff": mean_diff,
            "std diff": std_diff,
            "decisions": decisions,
        }
    )

    return results



def get_shaffer(k):
    if k in [0,1]:
        return [0]
    else:
        results = []
        for j in range(1, k+1):
            results = results + [int(binom(j, 2) + x) for x in get_shaffer(k-j)]
        return np.unique(results)


def _permutation_test(data, B, alpha, seed):
    """
    Permutation test with Step-Down method
    """
    n_fit = len(data[0])
    n_workflows = len(data)

    # We do all the pairwise comparisons.
    comparisons = np.array(
        [(i, j) for i in range(n_workflows) for j in range(n_workflows) if i < j]
    )
    shaffer_nbrs = get_shaffer(n_workflows)

    decisions = np.array(["accept" for i in range(len(comparisons))])
    comparisons_alive = np.arange(len(comparisons))
    seeder = np.random.RandomState(seed)

    print("Beginning permutation test")
    while True:
        current_comparisons = comparisons[comparisons_alive]
        print(f"Still {len(current_comparisons)} comparisons to test")

        # make a generator of permutations
        if B is None:
            permutations = combinations(2 * n_fit, n_fit)
        else:
            permutations = (seeder.permutation(2 * n_fit) for _ in range(B))

        # Test statistics
        T0_max = 0
        for id_comp, (i, j) in enumerate(current_comparisons):
            Z = np.hstack([data[i], data[j]])
            T = np.abs(np.mean(Z[:n_fit]) - np.mean(Z[n_fit : (2 * n_fit)]))
            if T > T0_max:
                T0_max = T
                id_comp_max = comparisons_alive[id_comp]

        # Permutation distribution of Tmax
        Tmax_values = []
        for perm in permutations:
            Tmax = 0
            for id_comp, (i, j) in enumerate(current_comparisons):
                Z = np.hstack([data[i], data[j]])
                Z = Z[perm]
                T = np.abs(np.mean(Z[:n_fit]) - np.mean(Z[n_fit : (2 * n_fit)]))
                if T > Tmax:
                    Tmax = T
            Tmax_values.append(Tmax)

        Tmax_values = np.sort(Tmax_values)
        icumulative_probas = (
            np.arange(len(Tmax_values))[::-1] / B
        )  # This corresponds to 1 - F(t) = P(T > t)
        eff_n_h0 = np.max(shaffer_nbrs[shaffer_nbrs <= len(current_comparisons)])

        admissible_values = Tmax_values[
            icumulative_probas <= alpha / eff_n_h0
        ]  # acceptance region
        if len(admissible_values) > 0:
            threshold = np.min(admissible_values)
        else:
            raise ValueError(
                f"There is not enough fits, the comparisons cannot be done with the precision {alpha}"
            )

        if T0_max > threshold:
            assert decisions[id_comp_max] == "accept"
            decisions[id_comp_max] = "reject"
            comparisons_alive = np.arange(len(comparisons))[decisions == "accept"]
        else:
            break
        if len(comparisons_alive) == 0:
            break

    # make a result array with 1 if accept and 0 if reject.
    results = np.zeros([n_workflows, n_workflows])
    for id_comp in range(len(comparisons)):
        if decisions[id_comp] == "reject":
            i, j = comparisons[id_comp]
            results[i, j] = 0
        else:
            i, j = comparisons[id_comp]
            results[i, j] = 1

    results = results + results.T + np.eye(n_workflows)
    return results


def sig_p_value(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p <= 0.05:
        return "*"
    else:
        return ""


## Confidence intervals

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

def plot_ci(a,b, mean, i, ax, width = 1):
    cmap =  plt.get_cmap("tab10")
    color = cmap(i)
    end_width = width/2
    lw = 1.5
    # ax.plot([i, i], [a,b], color ="black", linewidth=lw, alpha=0.8)
    ax.plot( [a,a], [i-end_width/2, i+end_width/2],color ="black", linewidth=lw, alpha=0.85)
    ax.plot( [b,b],[i-end_width/2, i+end_width/2],color ="black", linewidth=lw, alpha=0.85)
    ax.plot( [a,a+(b-a)/10],[i-end_width/2, i-end_width/2],color ="black", linewidth=lw, alpha=0.85)
    ax.plot( [a,a+(b-a)/10], [i+end_width/2, i+end_width/2],color ="black", linewidth=lw, alpha=0.85)
    ax.plot( [b-(b-a)/10,b],[i+end_width/2, i+end_width/2],color ="black", linewidth=lw, alpha=0.85)
    ax.plot( [b-(b-a)/10,b],[i-end_width/2, i-end_width/2],color ="black", linewidth=lw, alpha=0.85)

    rec = Rectangle((a, i-end_width/2), b-a, end_width)
    pc = PatchCollection([rec], facecolor=color, alpha=0.7,
                         edgecolor="black")
    ax.add_collection(pc)
    ax.scatter([mean],[i], color="black", marker="*", s= 100)

