# running the experiments is building data that can be turned into stat display
# this file is only taking whatever is in the runs folders is plotting on that
# so you need to run both the coin and zombie envs as specified in run_all in order to be able to use the analysis
# normally, the models can be viewed in the watch.py, so if you dont care about the stats --you just stop there
# will write stats into results/

# each run trains the network, 100 test games per said network, mean to 1 number per network per seed -- so 5 numbers per condition

import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")                                                                       # same as in watch, needs to be defined before pyplot
import matplotlib.pyplot as plt                                                             # Agg draws, no open windows

RUNS, OUT = "runs", "results"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,              # global plotting settings for file
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False})

PALETTE = {"baseline_random": "#999999", "baseline_flee": "#555555", "A": "#1f77b4",    # same conditions should get the same style in each figure, easy to recall
           "B": "#2ca02c", "C": "#ff7f0e", "D": "#d62728", "proxy": "#d62728",
           "mixed": "#ff7f0e", "true": "#2ca02c"}


def color(cond):
    return PALETTE.get(cond, PALETTE.get(cond.split("_")[0], "#9467bd"))                        # and it falls back if specificy is missing


def ci95(values):                                                                           # mean & half-width of 95% confidence intervall
    x = np.asarray(values, dtype=float)
    if len(x) < 2:
        return float(x.mean()), 0.0
    return float(x.mean()), float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))     # stadard dev for small sample


def save(fig, name):
    fig.tight_layout()                                                                              # adjust spacing, labels dont overlap
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=300)                                    # saves in png and pdf, pdf vector base good to use for the paper and keep quality
    plt.close(fig)      # frees memory after
    print("saved", name)


# --- loading ---
def load_runs(exp):
    rows = []
    for path in Path(RUNS, exp).glob("*/seed*/eval_episodes.csv"):                                  # every csv in every seed folder, all into 1 table
        cond, seed = path.parts[-3], int(path.parts[-2].replace("seed", ""))                        # split conditions, strip seeds for rows
        df = pd.read_csv(path)                                                                      # read 100-game tables
        row = df.drop(columns="episode").mean().to_dict()                                           # drop index
        row.update(cond=cond, seed=seed)                                                            # tag with condition & seed
        rows.append(row)
    return pd.DataFrame(rows)                                                                       # 1 row per run


def load_curves(exp, cond):
    out = []
    for path in sorted(Path(RUNS, exp, cond).glob("seed*/train_log.csv")):
        out.append(pd.read_csv(path))
    return out


# --- plotting ---
def bar_panel(ax, df, conds, metric, title, ylabel=None):                                           # bars with error bars and dots
    for i, cond in enumerate(conds):
        vals = df[df.cond == cond][metric].values                                                   # take rows == condition, col as metric, //as plain list
        if len(vals) == 0:
            continue
        m, h = ci95(vals)
        ax.bar(i, m, yerr=h, capsize=2.5, color=color(cond), alpha=0.85, linewidth=0)               # draw bar at (i, m) & vertical error bar at +/- h
        ax.scatter(np.full(len(vals), i) + np.linspace(-0.15, 0.15, len(vals)), vals,               # 1 small black dot at the top of each run, on the bar
                   s=6, c="k", zorder=3)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([c.replace("baseline_", "") for c in conds], rotation=30, ha="right")
    ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)


def curve_panel(ax, exp, conds, metric, title, window=10):                                          # learning curves
    for cond in conds:
        runs = load_curves(exp, cond)
        if not runs:
            continue
        n = min(len(r) for r in runs)
        arr = np.stack([r[metric].iloc[:n].rolling(window, min_periods=1).mean().values for r in runs])
        #                                                           for each training log (for run), takes col for metric
        #                                                           smooth it w/ moving average over 10 iterations
        #                                                           stack all runs in 1 table with 1 row/run | trim all runs on same lenght
        x = runs[0]["env_steps"].iloc[:n].values
        m, s = arr.mean(0), arr.std(0)
        ax.plot(x, m, color=color(cond), label=cond, lw=1.2)
        ax.fill_between(x, m - s, m + s, color=color(cond), alpha=0.2, linewidth=0)
        #                                                           averages across runs at every point in training, computes spread
        #                                                           draws average as a line +/- 1 std
        #                                                           shows how behaviour appears over the learning
    ax.set_title(title)
    ax.set_xlabel("environment steps")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))


def present(df, conds):
    return [c for c in conds if c in set(df.cond)]


# --- experiment 1 ---
# creates a fig_coin = exit & coin rate for each baseline and reward mode
# exit rate vs reward on extra conditions in like comparison format
# learning curves for coin
def analyze_coin():
    df = load_runs("coin")
    if df.empty:
        print("no coin runs found")
        return df
    conds = present(df, ["baseline_random", "proxy", "mixed", "true"])
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.3))                                              # 1 figure, 2 drawing areas, should fit column for paper
    bar_panel(axes[0], df, conds, "exit_reached", "Episodes reaching the exit", "fraction")
    bar_panel(axes[1], df, conds, "coins", "Coins collected per episode", "coins")
    save(fig, "fig_coin")

    scan = present(df, ["mixed_exit2", "mixed_exit5", "mixed"])
    if len(scan) > 1:
        vals = {"mixed_exit2": 2, "mixed_exit5": 5, "mixed": 10}
        fig, ax = plt.subplots(figsize=(3.4, 2.3))
        xs, ms, hs = [], [], []
        for c in scan:
            m, h = ci95(df[df.cond == c]["exit_reached"])
            xs.append(vals[c]); ms.append(m); hs.append(h)
        ax.errorbar(xs, ms, yerr=hs, marker="o", capsize=2.5, color=color("mixed"))
        ax.set_xscale("log")
        ax.set_xlabel("exit reward (coin reward = 1)")
        ax.set_ylabel("exit rate")
        ax.set_title("Tipping point: loophole vs goal")
        save(fig, "fig_coin_tipping")

    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.3))
    curve_panel(axes[0], "coin", present(df, ["proxy", "mixed", "true"]), "exit_reached", "Exit rate during training")
    curve_panel(axes[1], "coin", present(df, ["proxy", "mixed", "true"]), "coins", "Coins during training")
    axes[0].legend()
    save(fig, "fig_coin_curves")
    return df


# --- experiment 2 ---
def latex_table(summary, conds, metrics, labels, path):                                                 # builds text of table as lines of screen, want to fit it in the paper nicely
    lines = ["\\begin{tabular}{l" + "c" * len(metrics) + "}", "\\hline",
             "Cond. & " + " & ".join(labels) + " \\\\", "\\hline"]
    for c in conds:
        cells = [f"{summary.loc[c, m + '_mean']:.2f} $\\pm$ {summary.loc[c, m + '_ci']:.2f}" for m in metrics]
        lines.append(c.replace("_", "\\_") + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\hline", "\\end{tabular}"]
    Path(path).write_text("\n".join(lines))


def analyze_zombie():
    df = load_runs("zombie")
    if df.empty:
        print("no zombie runs found")
        return df
    df.to_csv(os.path.join(OUT, "summary_per_run.csv"), index=False)                                    # save a per run table

    metrics = ["ff_kills", "ff_kills_no_chaser", "chaser_kills", "survivors", "survival_time",
               "win", "shots", "shots_hit_runner", "shots_hit_chaser", "deaths_by_chaser", "ep_len"]
    summary = {}
    for cond, g in df.groupby("cond"):                                                                  # and creates a summary per conditions
        row = {}
        for m in metrics:
            row[m + "_mean"], row[m + "_ci"] = ci95(g[m])
        row["n_seeds"] = len(g)
        summary[cond] = row
    summary = pd.DataFrame(summary).T
    summary.to_csv(os.path.join(OUT, "summary_by_condition.csv"))

    main = present(df, ["baseline_random", "baseline_flee", "A", "B", "C", "D"])
    show = ["ff_kills", "ff_kills_no_chaser", "chaser_kills", "survivors"]
    latex_table(summary, main, show,
                ["FF kills", "FF (no chaser)", "Chaser kills", "Survivors"],
                os.path.join(OUT, "table_zombie.tex"))

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.3))
    bar_panel(axes[0], df, main, "ff_kills", "Friendly-fire kills / episode")
    bar_panel(axes[1], df, main, "ff_kills_no_chaser", "FF kills, no chaser near")
    bar_panel(axes[2], df, main, "chaser_kills", "Chasers killed / episode")
    bar_panel(axes[3], df, main, "survivors", "Runners alive at end")
    save(fig, "fig_zombie_bars")

    learned = present(df, ["A", "B", "C", "D"])
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.3))
    curve_panel(axes[0], "zombie", learned, "ff_kills", "Friendly-fire kills / episode")
    curve_panel(axes[1], "zombie", learned, "chaser_kills", "Chasers killed / episode")
    curve_panel(axes[2], "zombie", learned, "survivors", "Runners alive at end")
    axes[0].legend()
    save(fig, "fig_zombie_curves")                                                                  # fig spanning on format

    extra = present(df, ["A", "C", "C_bonus2", "C_bonus10", "C_aware", "D", "D_aware"])
    if len(extra) > 3:
        fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.4))
        bar_panel(axes[0], df, extra, "ff_kills", "Friendly-fire kills / episode")
        bar_panel(axes[1], df, extra, "chaser_kills", "Chasers killed / episode")
        save(fig, "fig_zombie_extra")

    pairs = [("A", "C"), ("A", "B"), ("B", "C"), ("C", "D"), ("A", "D"), ("C", "C_aware")]          # vs like questions
    #                                                                           qestion was what causes things, so conditions here can be seen as that
    #                                                                           A vs C: does the bonus cause betrayal?
    lines = ["Mann-Whitney U tests on per-seed means (two-sided). Unit = training run (seed).",
             "With 5 seeds per condition the smallest possible p-value is about 0.008.", ""]
    for m in ["ff_kills", "ff_kills_no_chaser", "chaser_kills", "survivors"]:
        lines.append(f"[{m}]")
        for a, b in pairs:
            x, y = df[df.cond == a][m].values, df[df.cond == b][m].values
            if len(x) < 2 or len(y) < 2:
                continue
            if np.allclose(x, y) and np.allclose(x, x[0]):
                lines.append(f"  {a} vs {b}: identical values, no test")
                continue
            u, pval = stats.mannwhitneyu(x, y, alternative="two-sided")                             # point of comparision, if identical -- test&condition is useless
            lines.append(f"  {a} (mean {x.mean():.2f}) vs {b} (mean {y.mean():.2f}): U={u:.1f}, p={pval:.4f}")
        lines.append("")
    Path(OUT, "stats.txt").write_text("\n".join(lines))                                             # results written and printed
    print("\n".join(lines))
    return df


if __name__ == "__main__":
    analyze_coin()
    analyze_zombie()
    print("\nAll outputs are in", OUT)



# FML