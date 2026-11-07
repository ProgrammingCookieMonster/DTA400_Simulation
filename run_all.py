# run multiple train progreams on different cores parallelly

import argparse, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
import config

p = argparse.ArgumentParser()
p.add_argument("--exp", required=True, choices=["coin", "zombie"])
p.add_argument("--workers", type=int, default=5)
p.add_argument("--extra", action="store_true")
p.add_argument("--steps", type=int, default=None)
p.add_argument("--out", default="runs")
args = p.parse_args()

jobs = []                                                                                               # 1 jon =  1 run
for cond in config.all_conditions(args.exp, args.extra):                                                # we run 1 job per condition // per seed
    for seed in config.SEEDS:                                                                           # so every seed runs all possible conditions given
        jobs.append((cond, seed, ["--cond", cond]))
if args.exp == "zombie":                                                                                # e.g. here we got 4 conds x 5 seeds = 20 jobs/runs
                                                                                                        # run actually 5 seeds to get confidence figures and tests
    for b in config.BASELINES:
        for seed in config.SEEDS:
            jobs.append((f"baseline_{b}", seed, ["--baseline", b]))
else:
    for seed in config.SEEDS:
        jobs.append(("baseline_random", seed, ["--baseline", "random", "--cond", "true"]))


def run(job):
    name, seed, extra = job
    run_dir = os.path.join(args.out, args.exp, name, f"seed{seed}")
    if os.path.exists(os.path.join(run_dir, "eval_episodes.csv")):
        return f"skip  {name} seed{seed}"                                                                   # can break the run and start again, it will skip finished jobs and continue
    os.makedirs(run_dir, exist_ok=True)
    cmd = [sys.executable, "train.py", "--exp", args.exp, "--seed", str(seed), "--out", args.out] + extra   # run system exec with python inside active venv, cannot execute training file without running with right dependancies calling python from right place
    if args.steps:
        cmd += ["--steps", str(args.steps)]
    with open(os.path.join(run_dir, "stdout.txt"), "w") as log:                                             # starts training program separated and waits for it to finish
        code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)                                   # everything goes into specified file inside run folder
    return f"{'done ' if code == 0 else 'FAIL '} {name} seed{seed}"                                         # returns 0 if success, anything else means fail


print(f"{len(jobs)} jobs, {args.workers} in parallel")
with ThreadPoolExecutor(max_workers=args.workers) as pool:                                                  # thread pool of a crew = specified n.o workers, my case 5/max6 threads
    for msg in pool.map(run, jobs):
        print(msg, flush=True)

# can test it with:
#   python run_all.py --exp coin --steps 5000 --workers 2
            # should delete whatever folders it creates after testing here
            # dont want to skip by naming when running the real experiment later

# test whole:
    # python run_all.py --exp coin --workers 5
    # python run_all.py --exp zombie --workers 5
                                                # for 5 cpu cores here
    # It is necessary to run the experiments for both envs
    # That, if the file for analysis is to be used, as the statistics are based on whatever the this file is producing upon running the envs