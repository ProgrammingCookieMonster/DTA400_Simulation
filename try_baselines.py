# try the baseline agent by playing a few episodes with the envs running on the baseline model
# nothing is printed for said episodes, only the their average of data
# this should be enough to rate the baseline model

import pandas
from envs.zombie_env import ZombieEnv
from baselines import make_random_policy, make_flee_policy

for name in ("random", "flee"):
    env = ZombieEnv()
    policy = make_random_policy(env.n_actions, 0) if name == "random" else make_flee_policy(0)
    rows = []
    for ep in range(40):
        obs = env.reset(seed=ep)
        done = False
        while not done:
            obs, _, _, done, info = env.step(policy(obs, anv))
        rows.append(info|["episode_stats"])
    table = pd.DataFrame(rows)
    print(name, table[["ep_len", "survivors", "deaths_by_chaser"]].mean().round(1).to_dict())