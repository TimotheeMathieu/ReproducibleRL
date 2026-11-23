"""On-policy, symmetric divergence between burns and flanders PPO policies (Acrobot-v1).

For each (config, seed, checkpoint) both policies are rolled out for `n_states` env steps
(stochastic actions, episodes chained). The visited states are pooled, so they are drawn from
the mixture 1/2 d_burns + 1/2 d_flanders, and per-state divergences between the two action
distributions are averaged over that pool.

Both rollouts share the env reset seed and the uniforms used to sample actions (common random
numbers): identical policies give identical rollouts, and `split_step` is the first step at
which the two rollouts take different actions (censored at `n_states`).

Columns (b = burns policy, f = flanders policy):
    tv          E_s[ 1/2 |pi_b(.|s) - pi_f(.|s)|_1 ]
    js          E_s[ JS(pi_b(.|s), pi_f(.|s)) ]                      (nats, <= log 2)
    kl_bf       E_s[ KL(pi_b(.|s) || pi_f(.|s)) ]
    kl_fb       E_s[ KL(pi_f(.|s) || pi_b(.|s)) ]
    disagree    P_s[ argmax pi_b(.|s) != argmax pi_f(.|s) ]
    split_step  first step where the two common-random-number rollouts differ
    identical   actor weights are bitwise identical

The CSV lives in data_feedback_loop/policy_divergence.csv next to this script. `compute` needs the
training checkpoints (and torch, gymnasium, joblib); `plot` only reads the CSV, and is what
make_all_plots.sh runs.

Usage:
    python process_policy_divergence.py compute [--out CSV]
    python process_policy_divergence.py plot OUT_DIR [--csv CSV] [--metric tv kl_bf ...] [--logy] [--normalized]
"""
import argparse
import io
import zipfile
from pathlib import Path

import numpy as np

GRADIENT_STEPS = [16, 64, 256]  # n_epochs in run_ppo.py ("gs" in the run folder names)
N_STEPS = [512, 1042, 2048]
GAE_LAMBDAS = [0.5, 0.9, 0.99]
FOCUS = dict(gradient_steps=16, n_steps=2048, gae_lambda=0.99)
CHECKPOINTS = list(range(0, 102_401, 2048))
ROOTS = {"burns": "reprorl/ReproRL", "flanders": "ppo_logs_flanders/ReproducibleRL_fitted_Q"}
DATA_DIR = Path(__file__).resolve().parent / "data_feedback_loop"
DIVERGENCE_CSV = DATA_DIR / "policy_divergence.csv"

_S = r"s\sim\frac{1}{2}(d_{\mathrm{burns}}+d_{\mathrm{flanders}})"
_B, _F = r"\hat{\pi}_{\mathrm{burns}}(\cdot|s)", r"\hat{\pi}_{\mathrm{flanders}}(\cdot|s)"
METRIC_LABELS = {
    "tv": rf"total variation",
    "js": rf"$\mathbb{{E}}_{{{_S}}}\,\mathrm{{JS}}({_B},\,{_F})$",
    "kl_bf": rf"$\mathbb{{E}}_{{{_S}}}\,\mathrm{{KL}}({_B}\,\|\,{_F})$",
    "kl_fb": rf"$\mathbb{{E}}_{{{_S}}}\,\mathrm{{KL}}({_F}\,\|\,{_B})$",
    "disagree": rf"$\mathbb{{P}}_{{{_S}}}[\arg\max{_B}\neq\arg\max{_F}]$",
    "split_step": "Steps before rollouts split",
    "identical": "Fraction of bitwise-identical policies",
}


# ----------------------------------------------------------------------------- compute

def checkpoint_path(roots, host, seed, gs, ns, gae, ckpt):
    run = f"{roots[host]}/-PPO-seed-{seed}-gs-{gs}-ns-{ns}-gae-{gae}/Acrobot-v1"
    return f"{run}/{host}-step-0.zip" if ckpt == 0 else f"{run}/{host}after-{ckpt}-new-spls.zip"


def load_actor(path):
    """Actor weights of the SB3 default PPO MlpPolicy (6-64-64-3, tanh), read without PPO.load."""
    import torch

    with zipfile.ZipFile(path) as z:
        sd = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu", weights_only=True)
    layers = ["mlp_extractor.policy_net.0", "mlp_extractor.policy_net.2", "action_net"]
    return [(sd[f"{l}.weight"].double().numpy(), sd[f"{l}.bias"].double().numpy()) for l in layers]


def log_probs(actor, obs):
    (w0, b0), (w1, b1), (w2, b2) = actor
    h = np.tanh(obs @ w0.T + b0)
    h = np.tanh(h @ w1.T + b1)
    logits = h @ w2.T + b2
    logits = logits - logits.max(-1, keepdims=True)
    return logits - np.log(np.exp(logits).sum(-1, keepdims=True))


def sample_actions(logp, u):
    """Inverse-CDF sampling from uniforms u, so two policies can share random numbers."""
    cdf = np.cumsum(np.exp(logp), axis=-1)
    return np.minimum((cdf < np.asarray(u)[..., None]).sum(-1), logp.shape[-1] - 1)


def rollout(env, actor, env_seed, u):
    """States visited by `actor` over len(u) env steps (episodes chained) and its actions."""
    states = np.empty((len(u), env.observation_space.shape[0]))
    actions = np.empty(len(u), dtype=np.int64)
    obs, _ = env.reset(seed=env_seed)
    for t, u_t in enumerate(u):
        states[t] = obs
        actions[t] = sample_actions(log_probs(actor, obs), u_t)
        obs, _, terminated, truncated, _ = env.step(actions[t])
        if terminated or truncated:
            obs, _ = env.reset()
    return states, actions


def divergences(lp, lq):
    p, q = np.exp(lp), np.exp(lq)
    lm = np.logaddexp(lp, lq) - np.log(2)
    return {
        "tv": 0.5 * np.abs(p - q).sum(-1).mean(),
        "js": (0.5 * (p * (lp - lm)).sum(-1) + 0.5 * (q * (lq - lm)).sum(-1)).mean(),
        "kl_bf": (p * (lp - lq)).sum(-1).mean(),
        "kl_fb": (q * (lq - lp)).sum(-1).mean(),
        "disagree": (lp.argmax(-1) != lq.argmax(-1)).mean(),
    }


def compare(burns, flanders, env, env_seed, u):
    if all(np.array_equal(x, y) for lb, lf in zip(burns, flanders) for x, y in zip(lb, lf)):
        return dict(tv=0.0, js=0.0, kl_bf=0.0, kl_fb=0.0, disagree=0.0, split_step=len(u), identical=True)
    states_b, actions_b = rollout(env, burns, env_seed, u)
    states_f, _ = rollout(env, flanders, env_seed, u)
    states = np.concatenate([states_b, states_f])  # samples of 1/2 d_burns + 1/2 d_flanders
    out = divergences(log_probs(burns, states), log_probs(flanders, states))
    # Rollouts share env seed and uniforms, so they coincide until flanders, put in burns'
    # state, would sample a different action than burns did.
    split = np.flatnonzero(sample_actions(log_probs(flanders, states_b), u) != actions_b)
    out["split_step"] = int(split[0]) if split.size else len(u)
    out["identical"] = False
    return out


def run_seed(roots, gs, ns, gae, seed, n_states, eval_seed):
    import gymnasium as gym

    env = gym.make("Acrobot-v1")
    rows = []
    for i, ckpt in enumerate(CHECKPOINTS):
        row = dict(gradient_steps=gs, n_steps=ns, gae_lambda=gae, seed=seed, checkpoint=ckpt)
        rng = np.random.default_rng([eval_seed, seed, i])
        env_seed, u = int(rng.integers(2**31)), rng.random(n_states)
        try:
            burns, flanders = [load_actor(checkpoint_path(roots, h, seed, gs, ns, gae, ckpt)) for h in ROOTS]
        except FileNotFoundError:
            rows.append(row)
            continue
        rows.append(row | compare(burns, flanders, env, env_seed, u))
    return rows


def compute(args):
    import pandas as pd
    from joblib import Parallel, delayed

    roots = {"burns": args.burns_root, "flanders": args.flanders_root}
    results = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(run_seed)(roots, gs, ns, gae, seed, args.n_states, args.eval_seed)
        for gs in GRADIENT_STEPS for ns in N_STEPS for gae in GAE_LAMBDAS for seed in range(args.n_seeds)
    )
    df = pd.DataFrame([row for rows in results for row in rows])
    df.to_csv(args.out, index=False)
    print(f"Saved {args.out} ({len(df)} rows)")


# ----------------------------------------------------------------------------- plot

def bootstrap_ci(values, confidence_level=0.95, n_resamples=1000):
    from scipy.stats import bootstrap

    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size < 2 or np.all(values == values[0]):
        mean = float(values[0]) if values.size else float("nan")
        return mean, mean, mean
    res = bootstrap(
        (values,), np.mean, confidence_level=confidence_level,
        n_resamples=n_resamples, rng=0,
    )
    return float(values.mean()), float(res.confidence_interval.low), float(res.confidence_interval.high)


def config_mean_ci(df, metric, gradient_steps, n_steps, gae_lambda):
    """Mean and 95% bootstrap CI over seeds at each checkpoint, or None if the config has no data."""
    sub = df[
        (df["gradient_steps"] == gradient_steps)
        & (df["n_steps"] == n_steps)
        & (df["gae_lambda"] == gae_lambda)
    ]
    if sub[metric].isna().all():
        return None
    stats = sub.groupby("checkpoint")[metric].apply(bootstrap_ci).reindex(CHECKPOINTS)
    return map(np.array, zip(*[s if isinstance(s, tuple) else (np.nan,) * 3 for s in stats]))


GRID_RC = {
    "font.size": 16,
    "axes.titlesize": 15,
    "axes.labelsize": 24,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "lines.linewidth": 3,
    "axes.linewidth": 1.5,
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
}


SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#7a5cc4"]
NORMALIZED_YLABEL = "Policy discrepancy"


def min_max(mean, lo, hi):
    """Scale the mean curve to [0, 1]; the CI band gets the same affine map, so it may overshoot."""
    m0, m1 = np.nanmin(mean), np.nanmax(mean)
    scale = m1 - m0 if m1 > m0 else 1.0
    return [(x - m0) / scale for x in (mean, lo, hi)]


def draw_metrics(ax, df, metrics, config, normalize, color=None):
    """Mean and CI band over seeds of each metric for one config, each min-max scaled if `normalize`."""
    for i, metric in enumerate(metrics):
        stats = config_mean_ci(df, metric, **config)
        if stats is None:
            continue
        mean, lo, hi = min_max(*stats) if normalize else stats
        (line,) = ax.plot(CHECKPOINTS, mean, color=SERIES_COLORS[i] if normalize else color,
                          label=METRIC_LABELS[metric])
        ax.fill_between(CHECKPOINTS, lo, hi, alpha=0.3, color=line.get_color())


def plot_grid(df, metrics, out, logy=False, normalize=False):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    with plt.rc_context(GRID_RC):
        fig, axes = plt.subplots(3, 9, figsize=(36, 12))
        for gi, gradient_steps in enumerate(GRADIENT_STEPS):
            for ni, n_steps in enumerate(N_STEPS):
                for gaei, gae_lambda in enumerate(GAE_LAMBDAS):
                    ax = axes[gi, ni * 3 + gaei]
                    ax.set_title(f"Grad. steps/update={gradient_steps}\nn_steps={n_steps} gae_lambda={gae_lambda}")
                    ax.set_xticks([0, 50_000, 100_000])
                    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k"))
                    config = dict(gradient_steps=gradient_steps, n_steps=n_steps, gae_lambda=gae_lambda)
                    draw_metrics(ax, df, metrics, config, normalize)
                    if normalize:
                        ax.set_yticks([0, 0.5, 1])
                    elif logy:
                        ax.set_yscale("log")
        fig.supxlabel("Samples", fontsize=24)
        if normalize:
            fig.supylabel(NORMALIZED_YLABEL, fontsize=28, x=0.002)
            handles, labels = axes[-1, -1].get_legend_handles_labels()
            fig.legend(handles, labels, loc="upper center", ncol=len(metrics), frameon=False,
                       fontsize=24, bbox_to_anchor=(0.5, 1.0))
            plt.tight_layout(rect=(0, 0, 1, 0.94))
        else:
            fig.supylabel(METRIC_LABELS[metrics[0]], fontsize=28, x=0.002)
            plt.tight_layout()
        plt.savefig(out, dpi=300)
        plt.close(fig)
    print(f"Saved {out}")


def plot_focus(df, metrics, out, logy=False, normalize=False):
    import matplotlib.pyplot as plt

    if all(config_mean_ci(df, metric, **FOCUS) is None for metric in metrics):
        print(f"No data for {FOCUS}, skipping {out}")
        return
    fig, ax = plt.subplots()
    draw_metrics(ax, df, metrics, FOCUS, normalize, color="blue")
    if normalize:
        ax.set_yticks([0, 0.5, 1])
        ax.set_ylabel(NORMALIZED_YLABEL, fontsize=18)
        ax.legend(loc="lower right", fontsize=11, frameon=False)
    else:
        if logy:
            ax.set_yscale("log")
        ax.set_ylabel(f"Policy discrepency ({METRIC_LABELS[metrics[0]]})", fontsize=14)
    ax.set_xlabel("Samples", fontsize=18)
    ax.tick_params(labelsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


def plot(args):
    import pandas as pd

    df = pd.read_csv(args.csv)
    df["identical"] = df["identical"].astype(float)
    for metric in args.metric:
        plot_grid(df, [metric], f"{args.out_dir}/ppo_policy_{metric}.pdf", logy=args.logy)
        plot_focus(df, [metric], f"{args.out_dir}/ppo_policy_{metric}_focus.pdf", logy=args.logy)
    if args.normalized:
        plot_grid(df, args.metric, f"{args.out_dir}/ppo_policy_discrepancy.pdf", normalize=True)
        plot_focus(df, args.metric, f"{args.out_dir}/ppo_policy_discrepancy_focus.pdf", normalize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    c = sub.add_parser("compute", help="roll out policies and write per-checkpoint divergences to a CSV")
    c.add_argument("--out", default=DIVERGENCE_CSV)
    c.add_argument("--burns-root", default=ROOTS["burns"])
    c.add_argument("--flanders-root", default=ROOTS["flanders"])
    c.add_argument("--n-seeds", type=int, default=50)
    c.add_argument("--n-states", type=int, default=2000, help="env steps per policy rollout")
    c.add_argument("--eval-seed", type=int, default=12345, help="seeds env resets and action sampling")
    c.add_argument("--n-jobs", type=int, default=15)
    c.set_defaults(func=compute)

    p = sub.add_parser("plot", help="3x9 grid over all configs + the focus config, mean and 95%% bootstrap CI over seeds")
    p.add_argument("out_dir")
    p.add_argument("--csv", default=DIVERGENCE_CSV)
    p.add_argument("--metric", nargs="+", default=["tv"], choices=list(METRIC_LABELS),
                   help="one or more metrics, two figures each")
    p.add_argument("--logy", action="store_true")
    p.add_argument("--normalized", action="store_true",
                   help="also overlay all --metric curves, each min-max scaled to [0, 1], in ppo_policy_discrepancy*.pdf")
    p.set_defaults(func=plot)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
