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
        #                       a runner sees 5x5 around itself as into 2 tiles every direction + itself, so 2*R+1 = 5
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
        #   observations seen by the runner, the plan is to build a full-map from 4 different map layers: walls, runners, chasers, guns
            S, R, N = self.size, self.R, self.N
            wall = self.walls.astype(np.float32)                                                    # wall table considint of 1s - wall=True, 0s - free tile
            runner = np.zeros((S, S), dtype=np.float32) 
            np.add.at(runner, (self.runner_pos[self.runner_alive, 0], self.runner_pos[self.runner_alive, 1]), 1)
        #       runner map: fill in with 0s everywhere except for tiles where a living runner position is
        #       add.at function also avoids double counting if 2 runners are on the same tile
            chaser = np.zeros((S, S), dtype=np.float32)                                             # same for runner, fill map with 0s fist
            np.add.at(chaser, (self.chaser_pos[self.chaser_alive, 0], self.chaser_pos[self.chaser_alive, 1]), 1)        # same idea, fill in positions
            gun = np.zeros((S, S), dtype=np.float32)
            gun[self.gun_pos[self.gun_present, 0], self.gun_pos[self.gun_present, 1], 1] = 1                            # just fill in position of guns

            padded = np.pad(np.stack([wall, runner, chaser, gun]), ((0, 0), (R, R), (R, R)))                            # all maps on full_map
            padded[0, :R, :] = 1; padded[0, -R:, :] = 1
            padded[0, :, :R] = 1; padded[0, :, -R:] = 1
                                    # padding the maps so that even a runner at the edge of the map can still get a full 5x5 view, map can be sensed outside map otherwise
            
            k = 4 * self.with_cells
            n_alive = int(self.runner_alive.sum())                                                      # define all alive runners, this is data all runner see
            armed = self.runner_alive & (self.ammo > 0)                                                 # no gun picked up means no ammo, all runners can see the other runners who are actually armed, they might need to learn to avoid them later on
            obs = np.zeros((N, self.obs_dim), dtype=np.float32)
            for i in np.where(self.runner_alive)[0]:
                # cutting out a 5x5 window view around each runner, including padding
                r, c = self.runner_pos[i]
                win = padded[:, r:r + 2 * R + 1, c:c + 2 * R + 1].copy()                            # center window on runner's tile
                win[1, R, R] = -1                   # subtract itself from the view, so that it does not see itself as another runner accounted for
                obs[i, :k] = win.ravel()            # flatten runner window into 1D array, flat observation
                obs[i, k] = self.ammo[i] / self.max_ammo    # ammot is scaled [0, 1], other runners know it can be a treat that a runner has ammot at all, flat value
                obs[i, k + 1 + self.facing[i]] = 1  # direction runner is facing is important, shooting has to be associated with facing direction
                #                       # facing direction is encoded as a one-hot vector: [0, 0, 1, 0] for a 1 = direction is actually facing
                # finally,  last properties based on awareness:
                if self.aware:
                    obs[i, k + 5] = n_alive / N                     # alive runners number
                    obs[i, k + 6] = 1 - self.t / self.map_steps     # aware of time left it game, could learn to try to stay alive to become survivors or appearl to ff_kills in case a lot of time is left until that
                    obs[i, k + 7] = (armed.sum() - armed[i]) / N    # aware of how many other runners than itself are armed, might learn to avoid them or to gather up and work together
                return obs

        # chasers -----------------
        def _chaser_move(self, k):
            # there is no actual AI or learning for the chaser, they just blindly move towards the nearest runner if they are alive
            # if nobody is alive, there is nothing to chase, zombie stays put
            alive = np.where(self.runner_alive)[0]
            if len(alive) == 0:
                return

            pos = self.chaser_pos[k].copy()                         # current position of chaser k
            delta = self.runner_pos[alive] - pos                    # chaser position relative to all alive runners, as row and column
            j = int(np.argmin(np.abs(delta).sum(1)))                # picks position to the closes to follow
            dr, dc = delta[j]   # how far horizontal/vertical
            vert, horiz = (int(np.sign(dr)), 0), (0, int(np.sign(dc)))      # np.sign ranges [-1, +1], it will tell the chaser which way to go
            
            for m in ([vert, horiz] if abs(dr) >= abs(dc) else [horiz, vert]):
                new = pos + np.array(m)
                if m != (0, 0) and self._free(new):
                    self.chaser_pos[k] = new
                    return
                                # chaser can move 1 step vertically or horizontally at a time, towards the closest runner
                                # if the tile it tries to move at it blocked by a wall, it tries another direction
                                # ends method in return as on the first move that is free, does not get hard-stucked
            valid = [pos + d for d in self.DIRS if self._free(pos + d)]
            if valid:
                self.chaser_pos[k] = valid[int(self.rng.integers(len(valid)))]
                                # zombie wanders in random direction if both positions are stuck when trying to move towards runner

        # shooting -----------------
        def _shoot(self, i, r):
            if self.ammo[i] <= 0:
                return                      # no ammo, no shooting

                # otherwise, what happens per shot
            self.ammo[i] -= 1               # deplet 1 bullet per shot
            self.shots += 1                 # keeps count of the shots fired in each episode, stats
            d = self.DIRS[self.facing[i]]   # d = direction of shot
            start = self.runner_pos[i].copy()  # starts from the tile of the runner shooting
            end, kind = start.copy(), None
            for dist in range(1, self.shoot_range + 1):
                # walking out 1 tile at a time based on distance
                q = start + d * dist        #  tiled being checked
                if not self._free(q):
                    break                   # bullet will stop if the tile is a wall or the edge of the map
                end = q                     # the end of the bullet path becomes then q
                ck = np.where(self.chaser_alive & (self.chaser_pos == q).all(1))[0]     # check for zombie, q bullet tile at current time matches alive one?
                                        # if zombie found, it will loose 1 hp, shooter gets rewarded, bullet stops
                                        # if zombie hp == 0, it dies, shooter gets big reward for kill
                if len(ck) > 0:
                    k = ck[0]
                    self.chaser_hp[k] -= 1
                    self.shots_hit_chaser += 1      # register shots in zombies
                    r[i] += self.chaser_hit_reward
                    if self.chaser_hp[k] <= 0:      # dies
                        self.chaser_alive[k] = False        # alive mask is off for entity, becomes irrelevant to everybody
                        self.chaser_kills += 1      # counter for kills
                        r[i] += self.chaser_kill_reward     # reward for zombie kill
                    kind = "chaser"
                    break                           # stop bullet on hit regardless
                # check for runner in bullet path, sharing same tile as bullet at the same time, exclude shooter itself, they dont die of own bullet, so self shooting
                # no reward given as standard
                # runner dies 1 shot
                # on certain experiment conditions, friendly fire has a penalty
                rk = np.where(self.runner_alive & (self.runner_pos == q).all(1))[0]     # again, check if runner and bullet share same tile at same time
                rk = rk[rk != 1]
                if len(rk):
                    self.runner_alive[rk[0]] = False
                    self.ff_kills += 1
                    self.shots_hit_runner += 1
                    r[i] += self.runner_kill_reward - self.friendly_fire_penalty
                    alive_c = self.chaser_pos[self.chaser_alive]                        # was there any zombie in the proximity, can it be an accidental shot?
                    near = len(alive_c) > 0 and np.abs(alive_c - q).sum(1).min() <= self.near_chaser_dist
                    if not near:
                        self.ff_kills_no_chaser += 1                                    # this could mean runners kill other runners on purpose
                    kind = "runner"
                    break                                                               # stop bullet
            self.last_shots.append((start, end, kind))                                  # save shot start, end, what it hit

        # step -----------------
        def step(self, actions):
            # one tick in the game
            S = self.size
            self.t += 1                                                 # for each tick
            self.last_shots = []
            r = np.zeros(self.N, dtype=np.float32)
            was_alive = self.runner_alive.copy()

            # order or who acts matters
            # having the same actors act in the same order all the time would introduce bias
            # therefore, the runners will act in random orders, nobody gets a bias advantage to get to a gun first
            for i in self.rng.permutation(self.N):
                if not self.runner_alive[i]:
                    continue                    # dead ones are skipped
                a = int(actions[i])             # agent chooses an action from its available list
                if a < 4:                       # action of movement
                    self.facing[i] = a
                    new = self.runner_pos[i] + self.DIRS[a]
                    if self._free(new):                         # same principle, if chosen location is free, move there
                        self.runner_pos[i] = new
                        g = np.where(self.gun_present & (self.gun_pos == new).all(1))[0]        # if it happens that new position is same tile as a gun
                        if len(g):
                            self.gun_present[g[0]] = False      # gun is no longeer available on that tile
                            self.ammo[i] = min(self.ammo[i] + self.ammo_per_gun, self.max_ammo)
                                                # gain +3 bullets on picked up gun, never carry more than max=6
                elif a == 5:
                    self._shoot(i, r)

            for k in range(self.M):
                if self.chaser_alive[k] and self.rng.random() < self.chaser_speed:
                    self._chaser_move(k)        # move zombie each step
                
            for i in np.where(self.runner_alive)[0]:
                if (self.chaser_alive & (self.chaser_pos == self.runner_pos[i]).all()).any():               # zombie + runner share same time = runner dies
                    self.runner_alive[i] = False
                    self.death_by_chaser += 1                                                               # doc stats

            r[self.runner_alive] += self.survival_reward                                                    # each step a runner is alive is a small reward
            self.alive_steps[self.runner_alive] += 1
            n_alive = int(self.runner_alive.sum())
            win = n_alive <= self.last_k                                                                    # specified last k are declared survivors
            ep_done = win or self.t >= self.max_steps
                            # condition is that a specific number of runners are left or the time has run out
                            # remaining runners there are declared survivors and get a big reward
                            # agents will want to stay alive to reach said reward
            if win and self.last_k_bonus > 0:
                r[self.runner_alive] += self.last_k_bonus
            if self.team_reward:                                    # certain mode allows for team reward, agents learn to gang up and kill zombies instead?
                r[:] = r.mean()
            self.ep_return += r

            agent_done = (was_alive & ~self.runner_alive) | ep_done         # agent is done if it died on current step or episode ended
            
            info = {}       # into  propagated per step
            if ep_done:
                info["episode_stats"] = dict(
                    return_mean=float(self.ep_return.mean()), ep_len=self.t, survivors=n_alive,
                    survival_time=float(self.alive_steps.mean()), win=float(win), ff_kills=self.ff_kills, 
                    ff_kills_no_chaser=self.ff_kills_no_chaser, chaser_kills=self.chaser_kills, deaths_by_chaser=self.deaths_by_chaser,
                    shots=self.shots, shots_hit_runner=self.shots_hit_runner, shots_hit_chaser=self.shots_hit_chaser
                )
            return self._obs(), r, agent_done, ep_done, info

        # graphics/reder -----------------
