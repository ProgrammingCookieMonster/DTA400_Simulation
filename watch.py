# loads saved network and plays the world
# can view it as LIVE or even save it as a gif, pretty fun thing to try for a first time

import argparse, os
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--exp", required=True, choices=["coin", "zombie"])
p.add_argument("--cond", default=None)
p.add_argument("--baseline", default=None, choices=[None, "random", "flee"])
p.add_argument("--seed", type=int, default=1)
p.add_argument("--episode_seed", type=int, default=0)
p.add_argument("--episodes", type=int, default=1)
p.add_argument("--fps", type=float, default=6)
p.add_argument("--save", default=None)
p.add_argument("--runs", default="runs")
args = p.parse_args()

import matplotlib                                                                           # import only after reading args
if args.save:
    matplotlib.use("Agg")                                                                   # draw in memory without opening any window, option=save as GIF, no need to view
import matplotlib.pyplot as plt                                                             # backend for matplot has to be chosen before pyplot is imported tho
from matplotlib.animation import PillowWriter
import torch
import config
from baselines import make_flee_policy, make_random_policy
from ppo import ActorCritic

cond = args.cond or list(config.EXPERIMENTS[args.exp]["main"])[0]
env = config.make_env_factory(args.exp, cond)()

if args.baseline:
    policy = (make_random_policy(env.n_actions, args.seed) if args.baseline == "random"
              else make_flee_policy(args.seed))
else:
    path = os.path.join(args.runs, args.exp, cond, f"seed{args.seed}", "model.pt")
    model = ActorCritic(env.obs_dim, env.n_actions, config.PPO["hidden"])                   # load saved agent
    model.load_state_dict(torch.load(path))                                                 # copy weights from trained ones, loaded agent here behaves like the trained one
    model.eval()

    def policy(obs, env):
        with torch.no_grad():
            a, _, _ = model.act(torch.as_tensor(obs))
        return a.numpy()

fig, ax = plt.subplots(figsize=(6.5, 6.5))
writer = PillowWriter(fps=args.fps) if args.save else None                                  # create writer for GIF if args.save as so
if writer:
    writer.setup(fig, args.save, dpi=80)                                                    # it collects frames of the running world
else:
    plt.ion()                                                                               # if no save arg, just draw live window


def frame():                                                                                # draw current state of canvas for each frame
    env.render_frame(ax)
    if writer:
        writer.grab_frame()                                                                 # grab said frame + add to animation
    else:
        plt.pause(1.0 / args.fps)                                                           # update window with pause = 1fps in live view


for ep in range(args.episodes):
    obs = env.reset(seed=args.episode_seed + ep)                                            # start obs on first image at reset
    frame()
    done = False
    while not done:
        obs, _, _, done, info = env.step(policy(obs, env))                                  # obs per step
        frame()
    print("episode result:", {k: round(v, 2) for k, v in info["episode_stats"].items()})    # stats

if writer:
    writer.finish()                                                                         # episode end, save last frame, close GIF file
    print("saved", args.save)
else:
    plt.ioff()
    plt.show()


# here we got now:
    # live window:
        # python watch.py --exp coin --cond proxy --seed 1
        # python watch.py --exp coin --cond true --seed 1

    # save as GIF:
        # python watch.py --exp zombie --cond C --seed 1 --save c.gif
    
    # no model, use baseline instead:
        # python watch.py --exp zombie --baseline flee
    
    # check some other random seed output:
        # python watch.py --exp zombie --cond C --seed 1 --episodes 3 --episode_seed 10

# seed generally == which run's model to load, they can behave differently, agents are spawned at start differently
# episode_seed == which map to load, as maps in zombie have 15 walls and they form randomly when creating maps, can use different run models to play on the same map and see how they behave