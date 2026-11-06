# train the actual agents
# what does a complete run
# Trains agent for one condition, one seed, tests it --> saves in own folder

import argparse, json, os, platform
import numpy as np
import pandas as pd
import torch

import config
from baselines import make_flee_policy, make_random_policy
from ppo import PPOTrainer

torch.set_num_threads(1)        # single thread work

def evaluate(env, policy, n_episodes, base_seed):                                           # test
    rows = []
    for ep in range(n_episodes):                                                            # plays 100 episodes
        obs = env.reset(seed=base_seed + ep)
        done = False
        while not done:
            obs, _, _, done, info = env.step(policy(obs, env))                              # collects stats
        rows.append(dict(episode=ep, **info["episode_start"]))
    return pd.DataFrame(rows)                                                               # turns list into a DataFrame table

def main():
        # allow for passing settings on program start without editing code,
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True, choices=["coin", "zombie"])                      # can be used as args.exp=zombie
    p.add_argument("--cond", default=None)
    p.add_argument("--baseline", default=None, choices=[None, "random", "flee"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--out", default="runs")
    args = p.parse_args()

    exp = config.EXPERIMENTS[args.exp]                                                      # args from default code file if no script args is given
    name = f"baseline_{args.baseline}" if args.baseline else args.cond                      # name run & folder
    assert name, "give --cond or --baseline"
    run_dir = os.path.join(args.out, args.exp, name, f"seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)                                                     # creates folders accounting for any missing parent ones too

    cond_for_env = args.cond or list(exp["main"])[0]
    factory = config.make_env_factory(args.exp, cond_for_env)
    steps = args.steps or exp["steps"]

    # settings saved in json, so a certain output run can be accounted for in its settings
    meta = dict(
        exp=args.exp, cond=name, seed=args.seed, total_env_steps=steps, env_kwargs=config.condition_kwargs(args.exp, cond_for_env),
        ppo=config.PPO, python=platform.python_version(), torch=torch.__version__, numpy=np.__version__
    )
    json.dump(meta, open(os.path.join(run_dir, "meta.json"), "w"), indent=2, default=str)

    # policy when baseline was requested is just policy from baseline, there is no training when that happens
    # otherwise a trainer is created and train, tweaks saved in torch
    env = factory()
    if args.baseline:
        policy = (make_random_policy(env.n_actions, args.seed) if args.baseline == "random" else make_flee_policy(args.seed))
    else:
        trainer = PPOTrainer(factory, config.PPO, args.seed, os.path.join(run_dir, "train_log.csv"))
        model = trainer.train(steps, watch_key=exp["watch_key"])
        torch.save(model.state_dict(), os.path.join(run_dir, "model.pt"))                   # collection of network weights

        def policy(obs, env):                                                               # wrap network of trained agent around the same policy model as baselines
                                                                                            # asks network for actions and turns them into numbers
            with torch.no_grad():
                a, _, _ = model.act(torch.as_tensor(obs))
            return a.numpy()                                                                # so we mean measure actually learned behavior

    df = evaluate(env, policy, config.EVAL_EPISODES, config.EVAL_BASE_SEED)
    df.to_csv(os.path.join(run_dir, "eval_episodes.csv"), index=False)                      # save table as csv file, one row per game no column for index
    print(f"\nEVAL {args.exp}/{name}/seed{args.seed} (mean over {len(df)} episodes)")
    print(df.drop(columns="episodes").mean().round(3).to_string())                          # quick summary stat overview as average 



if __name__ == "__main__":
    main()                                                                                  # does absolutely not run main on import