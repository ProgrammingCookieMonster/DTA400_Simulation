# test visualisation of coin environment and agent movement
# TO RUN THIS SCRIPT: run from the root folder where envs/ folder is visible

import numpy as np
import matplotlib.pyplot as plt                     # convenient for creating windows
from envs.coin_env import CoinEnv


env = CoinEnv(reward_mode="proxy")
iteration = env.m_steps
obs = env.reset(seed=0)                             # agent start tile always same on seed 0
print("observation shape: ", obs.shape)             # coin env has only 1 agent, should print (agent, observations it sees) = (1, 3xsizexsize)

rng = np.random.default_rng(0)                      # separated rng for this file, creates reproductability for the random actions the agent does for test run
fig, ax = plt.subplots(figsize=(5, 5))               # subplot returns tuple=(fix, ax); 
                # fig -- picture frame/canvas surface
                #  ax -- graph area that gets drawn on the canvas, axes refers to the entire coordinate system, not just x/y individual
plt.ion()                                           # interactive mode = True, windows is drawn right away, updates while code runs
for _ in range(iteration):
    action = rng.integers(0, 4, size=1)             # defines action as array -- returning randomly 0-3 as that was defined as action option in previous file
    obs, reward, agent_done, ep_done, info = env.step(action)       # each action is unpacked into the 5 values
    env.render_frame(ax)                            # call to render the frame as defined in the env
    plt.pause(0.15)                                 # motion speed
    if ep_done:
        print(info["episode_stats"])
        break
plt.ioff(); plt.show()                              # keep window open until user closes it
