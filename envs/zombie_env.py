# Zombie vs. Runners environment
    # --> it does use the same main concepts from the previous env, but the complexity is higher:
        # 1. Many agents instead of 1. Observations here become tables with 1 row per runner.
        # 2. Agents can die mid-game, yet the game has to continue.
        # 3. Interaction with environment, zombies, other agents. Runners should be able to shoot both zombies and other runners.
            # --> Order of events within a step matters
# Any difference in behaviour is cased by rewards only.
# 15x15 grid world, also places 15 random obstacles around the map
# 10 runners, 3 zombies/chasers, 6 guns placed on free tiles
# Runner and zombie on the same tile in the same time = runner agent dies
# Runner and gun on the same tile at the same tiem = picks up gun, has 3 bullets; agent max_ammo = 6
# Zombie HP = 4, Runner HP = 1
#       This is the Ethical Part of the project being tested
#       will the agents learn to co-op or will betray each other
#           --> friendly fire kills are recorded separatedly, if armed runner shoots another while no zombie in near proximity, that is intentional
#               --> cold-blooded malice, cool stuff :)
# Zombies use dumb algorithm, their sole purpuse is to move towards the nearest alive Runner
#       -->  Zombies do generally move slower, so Runners can outrun them
# Episode ends after 300 steps, or when there are n.o Runner <= 5 --> declared Survivors and are rewarded
#                                                           Runner wants to survive for as long as possible and achieve this state

import numpy as np

class ZombieEnv:
    n_actions = 6                                                   # 0=up, 1=down, 2=left, 3=right, 4=stay, 5=shoot
    DIRS = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)])

    STAT_KEYS = [                                                   # data needed saved
                    "return_mean", "ep_len", "survivors", "survival_time", "win", 
                    "ff_kills", "ff_kills_no_chaser", "chaser_kills", "deaths_by_chaser",
                    "shots", "shots_hit_runner", "shots_hit_chaser"
    ]
        # environment default initialization
    def __init__(self, size=15, n_runners=10, n_chasers=3, n_walls=15, n_guns=6,
                 ammo_per_gun=3, max_ammo=6, chaser_hp=4, shoot_range=4, max_steps=300,
                 last_k=5, chaser_speed=0.4, view_radius=2, near_chaser_dist=4,
                 survival_reward=0.01, chaser_hit_reward=0.1, chaser_kill_reward=1.0, 
                 runner_kill_reward=0.0, team_reward=False, last_k_bonus=0.0,
                 friendly_fire_penalty=0.0, aware=False):
        self.size, self.N, self.M = size, n_runners, n_chasers
        self.n_walls, self.n_gun = n_walls, n_guns
        self.ammo_per_gun, self.max_ammo = ammo_per_gun, max_ammo
        self.chaser_hp0, self.shoot_range = chaser_hp, shoot_range
        self.max_steps, self.last_k = max_steps, last_k
        self.chaser_speed, self.R = chaser_speed, view_radius                                           # default speed is calculated to be enough so nobody dies too fast or everybody is too safe
        self.near_chaser_dist = near_chaser_dist
        self.suvival_reward, self.chaser_hit_reward = survival_reward, chaser_hit_reward
        self.chaser_kill_reward, self.runner_kill_reward = chaser_kill_reward, runner_kill_reward       # ff_kills are not rewarded directly, agents should learn later what happens with ffk
        self.team_reward, self.last_k_bonus = team_reward, last_k_bonus
        self.friendly_fire_penalty, self.aware = friendly_fire_penalty, aware

        self.n_agents = n_runners
        self.win_cells = (2 * self.R + 1) ** 2
        self.obs_dim = 4 * self.win_cells + 5 + 3 # numbers runner receives, env tiles, ammo, direction, survivors left, time left, armed runners (last 3: aware=True)
        #       win_cells: 5x5 pictures that register walls, other runners, chasers, gun laying down
        #           network size fixed: 108 observations per agent.
        self.rng = np.random.default_rng()

        # helpers ------------------
        @property
        def alive_mask(self):
            return self.runner_alive.copy()                                                             # hands a copy, no outside should change thing

        def _free(self, p):
            return 0 <= p[0] < self.size and 0 <= p[1] < self.size and not self.walls[p[0], p[1]]       # check if tile p:free, so if entity can stand on it

        def _pick(self, cands, k):
            idx = self.rng.choice(len(cands), size=k, replace=False)
            return cands[idx]                                   # picks (choice) and returns k n.o candidates from given list

        # reset ------------------
        def reset(self, seed=None):
            if seed is not None:
                self.rng = np.random.default_rng(seed)

            S = self.size
            self.t = 0
            self.walls = np.zeros((S, S), dtype=bool)                                       # grid: SxS table filled with False everywhere, n_walls: sets True for picked random positions
            self.walls.flat[self.rng.choice(S * S, self.n_walls, replace=False)] = True     # .flat: treats grid as 1 long list -- pick n_walls random cells from it: True

            free = np.argwhere(~self.walls)                                                 # free is wherever there is no wall
            self.runner_pos = self._pick(free, self.N).copy()                               # 10 free positions are picked for runners first
            used = {tuple(p) for p in self.runner_pos}                                      # save set of tiles already taken by runners

            rest = np.array([p for p in free if tuple(p) not in used])                      # the rest of the unused tiles, a gun should never spawn on a runner
            self.gun_pos = self._pick(rest, self.n_guns).copy()                             # so they are spawned on any other free tile, zombies cannot pick anyways
            used |= {tuple(p) for p in self.gun_pos}                                        # update used positions after appending guns to the map

            # same principle for zombies, but cannot be too close to the tiles runners spawned on, or to allow them to be cornered by spawn
            rest = np.array([p for p in free if tuple(p) not in used])                      # computes last remaining free tiles
            # compute distance between each zombie and each runner
            # uses Manhattan distance to give a table tile x runner -- dont completely understand it
            # keep each zombie's spawn tile at a distance of at least 7 tiles away from any runner
            dmin = np.abs(rest[:, None, :] - self.runner_pos[None, :, :]).sum(2).min(1)
            far = rest[dmin >= 7]
            self.chaser_pos = self._pick(far if len(far) >= self.M else rest, self.M).copy()        # falls back on any free tile if there arent enough to meet requirements

            # set up initial states the rest of the observations and their type, for each entity at spawn
            self.runner_alive = np.ones(self.N, dtype=bool)                                         # np of 1s, everybody is alive at spawn
            self.ammo = np.zeros(self.N, dtype=int)                                                 # np of 0s, no ammot at all when spawned
            self.facing = self.rng.integers(0, 4, self.N)                                           # spawn facing random direction by rng
            self.chaser_hp = np.full(self.M, self.chaser_hp0)                                       # spawn zombies in initial hp0
            self.chaser_alive = np.ones(self.M, dtype=bool)                                         # all zombies alive at spawn
            self.gun_present = np.ones(self.n_guns, dtype=bool)                                     # all guns appear available for pick-up at spawn

            self.ep_return = np.zeros(self.N)
            self.alive_steps = np.zeros(self.N)
            self.ff_kills = self.ff_kills_no_chaser = self.chaser_kills = 0                         # all counters to 0 on reseet
            self.deaths_by_chaser = self.shots = 0
            self.shots_hit_runner = self.shots_hit_chaser = 0
            self.last_shots = []
            return self._obs()                                                                      # return first observation on reset, first look on the map

        # observation -----------------
        def _obs(self):
            pass