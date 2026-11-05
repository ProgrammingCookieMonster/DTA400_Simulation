# test zombie environment
# draw and visualize map and actions in ticks
# simple view, use policy from baselines for test

import matplotlib.pyplot as plt
from envs.zombie_env import ZombieEnv
from baselines import make_random_policy, make_flee_policy

env = ZombieEnv()
policy = make_flee_policy(seed=0)
obs = env.reset(seed=3)

fig, ax = plt.subplots(figsize=(6, 6))
plt.ion()
done = False
while not done:
    obs, rewards, agent_done, done, info = env.step(policy(obs, env))
    env.render_frame(ax)
    plt.pause(0.15)

print(info["episode_stats"])
plt.ioff()
plt.show()