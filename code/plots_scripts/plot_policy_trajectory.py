"""Qualitative view of the burns / flanders policy divergence on Acrobot-v1.

`compute` plays one episode with the burns policy and one with the flanders policy, both greedy
(argmax action) and starting from the same env reset seed, and saves the visited states to a CSV.
`plot` reads that CSV and draws the panel:
    - a filmstrip of the arm with both policies overlaid,
    - horizontal position and height of the tip over time (the episode ends when height > 1),
    - the torque each policy applies.
It also saves the filmstrip alone, as one wide row, in a `_focus` figure.

CSV columns (one row per visited state, terminal state included):
    host                    burns or flanders
    gradient_steps, n_steps, gae_lambda, seed, checkpoint
                            training run and checkpoint the policy comes from
    env_seed                env reset seed of the episode
    step                    0 for the reset state, t after t env steps
    theta1, theta2          joint angles (rad, 0 = hanging down), i.e. env.unwrapped.state
    dtheta1, dtheta2        joint angular velocities
    action                  action taken in this state (0, 1, 2 = torque -1, 0, +1); empty on the terminal state

The CSV lives in data_feedback_loop/policy_trajectory.csv next to this script. `compute` needs the
training checkpoints (and torch, gymnasium); `plot` only reads the CSV, and is what
make_all_plots.sh runs.

Usage:
    python plot_policy_trajectory.py compute [--out CSV] [--seed 0] [--env-seed 0]
        [--checkpoint 102400] [--gradient-steps 16] [--n-steps 2048] [--gae-lambda 0.99]
    python plot_policy_trajectory.py plot OUT_DIR [--csv CSV] [--frames 14] [--strip-frames 7] [--png]
"""
import argparse

import numpy as np

from process_policy_divergence import CHECKPOINTS, DATA_DIR, FOCUS, ROOTS, checkpoint_path, load_actor, log_probs

TRAJECTORY_CSV = DATA_DIR / "policy_trajectory.csv"

RUN = ["gradient_steps", "n_steps", "gae_lambda", "seed", "checkpoint", "env_seed"]
STATE = ["theta1", "theta2", "dtheta1", "dtheta2"]


# ----------------------------------------------------------------------------- compute

def episode(actor, env_seed):
    """States (T+1, 4) visited by the greedy policy over one episode, and its actions (T,)."""
    import gymnasium as gym

    env = gym.make("Acrobot-v1")
    obs, _ = env.reset(seed=env_seed)
    states, actions = [env.unwrapped.state.copy()], []
    while True:
        actions.append(int(log_probs(actor, obs).argmax()))
        obs, _, terminated, truncated, _ = env.step(actions[-1])
        states.append(env.unwrapped.state.copy())
        if terminated or truncated:
            return np.array(states), np.array(actions)


def compute(args):
    import pandas as pd

    roots = {"burns": args.burns_root, "flanders": args.flanders_root}
    run = dict(gradient_steps=args.gradient_steps, n_steps=args.n_steps, gae_lambda=args.gae_lambda,
               seed=args.seed, checkpoint=args.checkpoint, env_seed=args.env_seed)
    frames = []
    for host in ROOTS:
        path = checkpoint_path(roots, host, args.seed, args.gradient_steps, args.n_steps, args.gae_lambda,
                               args.checkpoint)
        states, actions = episode(load_actor(path), args.env_seed)
        df = pd.DataFrame(states, columns=STATE).assign(host=host, step=np.arange(len(states)), **run)
        df["action"] = pd.array([*actions, None], dtype="Int64")
        frames.append(df)
        print(f"{host:>8}: {len(actions)} steps, torques {''.join('-0+'[a] for a in actions)}")
    df = pd.concat(frames)[["host", *RUN, "step", *STATE, "action"]]
    df.to_csv(args.out, index=False)
    print(f"Saved {args.out} ({len(df)} rows)")


# ----------------------------------------------------------------------------- plot

COLORS = {"burns": "#2a78d6", "flanders": "#eb6834"}
LABELS = {"burns": "Burns", "flanders": "Flanders"}
INK, MUTED, HAIRLINE = "#0b0b0b", "#898781", "#c3c2b7"
GOAL = 1.0  # Acrobot-v1 terminates when the tip height exceeds this

RC = {
    "font.size": 13,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 14,
    "axes.edgecolor": HAIRLINE,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def arm(theta1, theta2):
    """Elbow and tip positions (pivot at the origin, y up), as drawn by the env's renderer."""
    t12 = theta1 + theta2
    elbow = np.stack([np.sin(theta1), -np.cos(theta1)], -1)
    return elbow, elbow + np.stack([np.sin(t12), -np.cos(t12)], -1)


def draw_frame(ax, t, traj, fontsize=12):
    """Both arms at step t; an arm whose episode already ended stays at its final pose, faded."""
    import matplotlib.patheffects as pe

    ax.axhline(GOAL, color=MUTED, lw=1, ls="--", zorder=0)
    ax.plot(0, 0, "o", color=INK, ms=4, zorder=5)
    for host, (elbow, tip) in traj.items():
        k = min(t, len(tip) - 1)
        c, alpha = COLORS[host], 0.35 if t >= len(tip) else 1.0
        ax.plot([0, elbow[k, 0], tip[k, 0]], [0, elbow[k, 1], tip[k, 1]], color=c, lw=3,
                alpha=alpha, solid_capstyle="round", zorder=3,
                path_effects=[pe.withStroke(linewidth=5, foreground="white")])
        ax.plot(*elbow[k], "o", color=c, ms=4, alpha=alpha, zorder=4)
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(HAIRLINE)
    ax.set_title(f"t = {t}", fontsize=fontsize, color=INK, pad=3)


def policy_handles(goal=False):
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=COLORS[h], lw=3, label=f"{LABELS[h]} policy") for h in ROOTS]
    if goal:
        handles.append(Line2D([], [], color=MUTED, lw=1, ls="--", label="Goal height"))
    return handles


def save(fig, out, png):
    import matplotlib.pyplot as plt

    fig.savefig(out, dpi=300)
    if png:
        fig.savefig(out[:-4] + ".png", dpi=110)
    plt.close(fig)
    print(f"Saved {out}")


def plot_strip(traj, frame_t, out, png):
    """Focus figure: the filmstrip alone, as one wide row."""
    import matplotlib.pyplot as plt

    n = len(frame_t)
    width, wspace, header = 14, 0.06, 0.75  # inches; header holds the legend and frame titles
    frame = width * 0.99 / (n + (n - 1) * wspace)
    height = (frame + header) / 0.99
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, n, figsize=(width, height))
        for ax, t in zip(axes, frame_t):
            draw_frame(ax, int(t), traj, fontsize=15)
        fig.legend(handles=policy_handles(goal=True), loc="upper center", ncol=3, frameon=False,
                   fontsize=16, bbox_to_anchor=(0.5, 1.0))
        fig.subplots_adjust(left=0.005, right=0.995, top=1 - header / height, bottom=0.01, wspace=wspace)
        save(fig, out, png)


def plot(args):
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(args.csv)
    runs = df[RUN].drop_duplicates()
    if len(runs) != 1:
        raise SystemExit(f"{args.csv} holds {len(runs)} runs, expected one")
    run = next(runs.itertuples(index=False))  # keeps per-column dtypes, unlike iloc
    eps = {h: df[df["host"] == h].sort_values("step") for h in ROOTS}
    traj = {h: arm(e["theta1"].to_numpy(), e["theta2"].to_numpy()) for h, e in eps.items()}
    actions = {h: e["action"].dropna().to_numpy(int) for h, e in eps.items()}

    a_b, a_f = actions["burns"], actions["flanders"]
    n = min(len(a_b), len(a_f))
    n_differ = int((a_b[:n] != a_f[:n]).sum())

    t_max = max(len(a) for a in actions.values())
    n_cols = args.frames // 2
    frame_t = np.linspace(0, t_max, args.frames).round().astype(int)
    out = f"{args.out_dir}/ppo_policy_trajectory_seed{run.seed}_env{run.env_seed}.pdf"

    with plt.rc_context(RC):
        fig = plt.figure(figsize=(14, 11))
        outer = fig.add_gridspec(2, 1, height_ratios=[4.2, 6.2], hspace=0.12)
        strip = outer[0].subgridspec(2, n_cols, wspace=0.08, hspace=0.25)
        for i, t in enumerate(frame_t):
            draw_frame(fig.add_subplot(strip[i // n_cols, i % n_cols]), int(t), traj)

        series = outer[1].subgridspec(4, 1, height_ratios=[3, 3, 1, 1], hspace=0.35)
        ax_x = fig.add_subplot(series[0])
        ax_y = fig.add_subplot(series[1], sharex=ax_x)
        ax_tau = {h: fig.add_subplot(series[2 + i], sharex=ax_x) for i, h in enumerate(ROOTS)}

        for h, (elbow, tip) in traj.items():
            steps = np.arange(len(tip))
            for ax, coord in ((ax_x, 0), (ax_y, 1)):
                ax.plot(steps, tip[:, coord], color=COLORS[h], lw=2)
                ax.plot(steps[-1], tip[-1, coord], "o", color=COLORS[h], ms=8,
                        markeredgecolor="white", markeredgewidth=2, zorder=5)
            torque = actions[h] - 1
            ax = ax_tau[h]
            ax.bar(np.arange(len(torque)) + 0.5, torque, width=0.8, color=COLORS[h])
            ax.axhline(0, color=HAIRLINE, lw=1)
            ax.set_ylim(-1.3, 1.3)
            ax.set_yticks([-1, 0, 1])
            ax.set_ylabel(f"{LABELS[h]}\ntorque", fontsize=12, rotation=0, ha="right", va="center")

        ax_x.set_ylabel("Tip horizontal\nposition")
        ax_y.set_ylabel("Tip height")
        ax_y.axhline(GOAL, color=MUTED, lw=1, ls="--", zorder=0)
        ax_y.annotate("goal", (t_max, GOAL), xytext=(4, 0), textcoords="offset points",
                      va="center", color=MUTED, fontsize=12)
        for ax in (ax_x, ax_y):
            ax.axhline(0, color=HAIRLINE, lw=1, zorder=0)
            ax.set_ylim(-2.2, 2.2)
            ax.grid(axis="y", color="#e1e0d9", lw=0.8)
            ax.set_axisbelow(True)
        for ax in (ax_x, ax_y, ax_tau["burns"]):
            ax.tick_params(labelbottom=False)
        ax_tau["flanders"].set_xlabel("Step")
        ax_x.set_xlim(0, t_max)

        fig.legend(handles=policy_handles(), loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
        fig.text(0.5, 0.955, f"Acrobot-v1, episode seed {run.env_seed} · policies after {run.checkpoint:,} "
                 f"samples (training seed {run.seed}, gs={run.gradient_steps}, n_steps={run.n_steps}, "
                 f"gae={run.gae_lambda:g}) · greedy actions · actions differ on {n_differ}/{n} steps",
                 ha="center", fontsize=12, color="#52514e")
        fig.subplots_adjust(left=0.1, right=0.95, top=0.915, bottom=0.05)
        save(fig, out, args.png)

    strip_t = np.linspace(0, t_max, args.strip_frames).round().astype(int)
    plot_strip(traj, strip_t, out[:-4] + "_focus.pdf", args.png)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    c = sub.add_parser("compute", help="play one greedy episode per policy and write the visited states to a CSV")
    c.add_argument("--out", default=TRAJECTORY_CSV)
    c.add_argument("--burns-root", default=ROOTS["burns"])
    c.add_argument("--flanders-root", default=ROOTS["flanders"])
    c.add_argument("--seed", type=int, default=0, help="training seed of the two runs")
    c.add_argument("--env-seed", type=int, default=0, help="env reset seed of the episode")
    c.add_argument("--checkpoint", type=int, default=CHECKPOINTS[-1], choices=CHECKPOINTS)
    c.add_argument("--gradient-steps", type=int, default=FOCUS["gradient_steps"])
    c.add_argument("--n-steps", type=int, default=FOCUS["n_steps"])
    c.add_argument("--gae-lambda", type=float, default=FOCUS["gae_lambda"])
    c.set_defaults(func=compute)

    p = sub.add_parser("plot", help="filmstrip, tip position and torques of the two episodes in the CSV")
    p.add_argument("out_dir")
    p.add_argument("--csv", default=TRAJECTORY_CSV)
    p.add_argument("--frames", type=int, default=14, help="filmstrip frames (two rows)")
    p.add_argument("--strip-frames", type=int, default=7, help="filmstrip frames of the one-row focus figure")
    p.add_argument("--png", action="store_true", help="also save a PNG preview")
    p.set_defaults(func=plot)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
