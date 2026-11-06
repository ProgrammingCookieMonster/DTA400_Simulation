# configuration - all numbers in one place
# bring changes to all envs

from envs.coin_env import CoinEnv
from envs.zombie_env import ZombieEnv

PPO = dict(
    lr=3e-4, gamma=0.99, gae_lambda=0.95, clip=0.2, ent_coef=0.01, vf_coef=0.5, epochs=4, minibatches=4, 
    max_grad_norm=0.5, hidden=128, n_envs=16, rollout_len=128,
)                                   # just labeled list of values, change here -- change everywhere
SEEDS = [1, 2, 3, 4, 5]             # train every condition 5 times, reproducible starting points with seeds
EVAL_EPISODES = 100                 # each condition trained on the same 100 maps, comparition is fair
EVAL_BASE_SEED = 10_000

# --- Experiment 1 ---
COIN_ENV = dict(size=7, respawn_steps=5, max_steps=100, coin_reward=1.0, exit_rewards=10.0, step_penalty=0.01)
COIN_STEPS = 500_000
COIN_CONDITIONS = {
    "proxy": dict(reward_mode="proxy"),
    "true": dict(reward_mode="true"),
    "mixed": dict(reward_mode="mixed")
}

# optional follow up -- exit reward testing might lie below the standard reward, just for exploration
COIN_EXTRA = {
    "mixed_exit2": dict(reward_mode="mixed", exit_reward=2.0),
    "mixed_exit5": dict(reward_mode="mixed", exit_reward=5.0)
}

# --- Experimet 2 ---
ZOMBIE_ENV = dict(
    size=15, n_runners=10, n_chasers=3, n_walls=15, n_guns=6, ammo_per_gun=3, max_ammo=6, chaser_hp=4, shoot_range=4, max_steps=300, last_k=5,
    chaser_speed=0.4, view_radius=2, survival_reward=0.01, chaser_hit_reward=0.1, chaser_kill_reward=1.0, runner_kill_reward=0.0
)           # standard
ZOMBIE_STEPS = 2_000_000
ZOMBIE_CONDITIONS = {
    # 4 different conditions
    "A": dict(team_reward=False, last_k_bonus=0.0, friendly_fire_penalty=0.0),          # runners learning only on survival evading pay
    "B": dict(team_reward=True, last_k_bonus=0.0, friendly_fire_penalty=0.0),           # would a shared team reward make them co-op?
    "C": dict(team_reward=False, last_k_bonus=5.0, friendly_fire_penalty=0.0),          # bonus for last alive would make them betray each other?
    "D": dict(team_reward=False, last_k_bonus=5.0, friendly_fire_penalty=3.0),          # penalty stops betrayal?
}

ZOMBIE_EXTRA = {                        # extra testable configurations
    "C_bonus2": dict(last_k_bonus=2.0),
    "C_bonus10": dict(last_k_bonus=10.0),
    "C_aware": dict(last_k_bonus=5.0, aware=True),
    "D_aware": dict(last_k_bonus=5.0, friendly_fire_penalty=3.0, aware=True),
}

BASELINES = ["random", "flee"]



EXPERIMENTS = {                         # gathering all data about each in one place
    "coin": dict(cls=CoinEnv, env=COIN_ENV, steps=COIN_STEPS, main=COIN_CONDITIONS, extra=COIN_EXTRA, watch_key="coins"),
    "zombie": dict(cls=ZombieEnv, env=ZOMBIE_ENV, steps=ZOMBIE_STEPS, main=ZOMBIE_CONDITIONS, extra=ZOMBIE_EXTRA, watch_key="ff_kills"),
}

# swap argumens for extra
def condition_kwargs(exp, cond):
    e = EXPERIMENTS[exp]
    table = {**e["main"], **e["extra"]}
    kw = dict(e["env"])                                                                 # start from default settings
    if exp == "zombie" and cond in e["extra"] and cond.startswith(("C_", "D_")):        # apply default and only swap the extra arguments
        base = e["main"][cond[0]]
        kw.update(base)
    kw.update(table[cond])
    return kw                                                                           # applies matching keys

def make_env_factory(exp, cond):                                                        # factory for experiment worlds
    cls, kw = EXPERIMENTS[exp]["cls"], condition_kwargs(exp, cond)
    return lambda: cls(**kw)        # call on class with dict unpacked into named args

def all_conditions(exp, include_extra=False):
    e = EXPERIMENTS[exp]
    return list(e["main"]) + (list(e["extra"]) if include_extra else [])