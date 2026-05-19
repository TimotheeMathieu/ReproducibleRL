import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys

on_props = [0, 0.2, 0.4, 0.6, 0.8, 1]
pcs = ['burns', 'flanders']
colors = {'burns': '#d62728', 'flanders': '#2ca02c'}
n_seeds = 20

# Load all data: data[pc][on_prop] -> array of shape (n_seeds, n_timesteps)
data = {}
for pc in pcs:
    data[pc] = {}
    for on_prop in on_props:
        seeds = []
        for seed in range(n_seeds):
            seeds.append(np.loadtxt(f'{sys.argv[2]}/{pc}_fitted_q_rl_scores_{on_prop}_{seed}.txt'))
        data[pc][on_prop] = np.array(seeds)

n_steps = data['burns'][0].shape[1]
x = np.arange(n_steps)

fig = plt.figure(figsize=(18, 8))
gs = gridspec.GridSpec(2, len(on_props), hspace=0.55, wspace=0.25)

# --- Top row: training curves per PC ---
axes_top = [fig.add_subplot(gs[0, 0])]
for i in range(1, len(on_props)):
    axes_top.append(fig.add_subplot(gs[0, i], sharey=axes_top[0]))

for i, on_prop in enumerate(on_props):
    ax = axes_top[i]
    for pc in pcs:
        arr = data[pc][on_prop]          # (n_seeds, n_steps)
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        ax.plot(x, mean, color=colors[pc], label=pc, linewidth=1.8)
        ax.fill_between(x, mean - std, mean + std, color=colors[pc], alpha=0.2)
    ax.set_title(f'online_prop = {on_prop}', fontsize=9)
    ax.set_xlabel('Iteration', fontsize=8)
    if i == 0:
        ax.set_ylabel('Score', fontsize=8)
        ax.legend(fontsize=7, loc='lower right')
    ax.tick_params(labelsize=7)

# --- Bottom row: mean ± std of |burns - flanders| over iterations ---
axes_bot = [fig.add_subplot(gs[1, 0])]
for i in range(1, len(on_props)):
    axes_bot.append(fig.add_subplot(gs[1, i], sharey=axes_bot[0]))

for i, on_prop in enumerate(on_props):
    ax = axes_bot[i]
    # per-seed absolute difference: (n_seeds, n_steps)
    abs_diff = np.abs(data['burns'][on_prop] - data['flanders'][on_prop])
    mean_diff = abs_diff.mean(axis=0)
    std_diff  = abs_diff.std(axis=0)
    ax.plot(x, mean_diff, color='steelblue', linewidth=1.8)
    ax.fill_between(x, mean_diff - std_diff, mean_diff + std_diff, color='steelblue', alpha=0.2)
    ax.set_title(f'online_prop = {on_prop}', fontsize=9)
    ax.set_xlabel('Iteration', fontsize=8)
    if i == 0:
        ax.set_ylabel('|burns - flanders|', fontsize=8)
    ax.tick_params(labelsize=7)

fig.suptitle('Training curves: burns vs flanders across online_prop\n(mean ± 1 std over seeds)', fontsize=12)
plt.savefig(f'{sys.argv[1]}/fitted_q.pdf', dpi=300, bbox_inches='tight')
print("Saved test.pdf")
