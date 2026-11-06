# The baseline is to check that the trained agents are actually learning at least up to a certain baseline
# it is rather a file used for relativity: -- the agent is performing in a certain way, is that good? good relative to what exactly?
# so the baseline here is hosting 2 behaviours: 1 random && 1 hand-written flee agent design by human biased strategy
# this way we can check if the agents trained in this simulation will perform better than the simple ones scripted in this baseline

import numpy as np

# policy: f(obs, env) --> actions

def make_random_policy(n_actions, seed):
    rng = np.random.default_rng(seed)
    return lambda obs, env: rng.integers(0, n_actions, size=env.n_agents)               # 1 random action per agent, complete rng, can be repeated by using same rng

def make_flee_policy(seed):
    rng = np.random.default_rng(seed)

    def policy(obs, env): 
        actions = np.full(env.n_agents, 4)
        alive_c = env.chaser_pos[env.chaser_alive]
        if len(alive_c) == 0:
            return actions
        for i in np.where(env.runner_alive)[0]:
            best, best_score = 4, -1
            options = [(a, env.runner_pos[i] + env.DIRS[a]) for a in range(4)]          # 5 things i can do
            options.append((4, env.runner_pos[i]))
            rng.shuffle(options)                                                        # shuffles so equally good options are not biased on anything
            for a, p in options:                                                        
                if not env._free(p):
                    continue                                                            # skip if tile is blocked
                score = np.abs(alive_c - p).sum(1).min()                                # manhattan distance from that tile to each alive zombie, computes best score of choice to stay away from zombies
                if score > best_score:                                                  # no guns involved at all, it is baseline performance
                    best, best_score = a, score
            actions[i] = best
        return actions
    return policy