# Environment file has the following jobs:
    # 1. Define the rules for: grid, coins, exit, rewards
    # 2. Describe the world to the learning agent as a list of numbers--observations
    # 3. Record statistics at the end of each episode, so the data is ready for the paper the course requires me to write 

import numpy as np


class CoinEnv:
    # general attribute rules before a run
    n_agents = 1                                            # number of agents that exist and take actions in the environment
    n_actions = 4                                           # for each direction: up, down, left, right || 0, 1, 2, 3
    m_steps = 100                                           # change for different length of episode
    MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]              # change in (row, column)
    # recorded data per finished episide (for logs & plots)
    STAT_KEYS = ["return_mean", "ep_len", "exit_reached", "coins"]          # saves total reward, how many steps the run lasted, if it reached the exit, n.o coins collected

    def __init__(self, reward_mode="proxy", size=7, respawn_steps=5, max_steps=m_steps, coin_reward=1.0, exit_reward=10.0, step_penalty=0.01):
        # recognize 3 gamemodes, agent will be tried in all of them, certain parameters might be changed based on the gamemode in use
        assert reward_mode in ("proxy", "true", "mixed")
        assert size >= 5                # minimum size of the grid is 5x5, otherwise fixed positions of coins and exit will overlap
        self.mode = reward_mode
        self.size = size
        self.respawn_steps = respawn_steps
        self.max_steps = max_steps
        self.coin_reward = coin_reward
        self.exit_reward = exit_reward
        self.step_penalty = step_penalty
        
        self.exit_pos = (size - 1, size - 1)            # grid exit is always in the bottom right corner
        self.coins = [(1, 1), (1, 3), (3, 1)]           # always same position for coins, (row, column), deliberately far from exit
        self.obs_dim = 3 * size * size                  # 3 channels/"maps" agent receives: own, coins, exit -- position | per each step // 5x5 grid = 3 x 5x5 = 75 numbers/step
        self.rng = np.random.default_rng()              # random number generator for respawn of coins

    @property                                           # property allows e.g. calls env.alive_mask called as a variable
    def alive_mask(self):                               # checks for alive playing agents. for the zombie run that is important to check which survivors are alive mid-simulation
        return np.array([True])


            # (internal helper function)
            # lerner for this simulation is a neural network which only accepts numbers as input
            # the situations are turned here into 3 grids of 0s, where 1s are marking the existance of something else than an empty grid cell
            # layer0: agents; layer1: coins; layer2: exit (e.g.)
    def _obs(self):
        o = np.zeros((3, self.size, self.size), dtype=np.float32)
        o[0][self.pos] = 1.0                                            # o[0]: layer0 -- agent's position is always recognized as an existing entity in the grid
        for coin, timer in zip(self.coins, self.timers):                # o[1]: layer1 -- walks through the states of coins & timers
            if timer == 0:                                              # timer > 0 = coin was picked up, is now invisible to the agent until respawn
                o[1][coin] = 1.0
        o[2][self.exit_pos] = 1.0                                       # o[2]: layer2 -- exit is always visible in the grid
        return o.reshape(1, -1)                                         # reshape flattens the data into a 1D array of numbers for the neural network

    # reset - starting a new episode
    # using seed for rng allows reproducibility of the start positions
    # seeded resets are used for evaluation of trained agents
    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)                       # set the random number generator to a specific seed for reproducibility, default produces new random episodes
        self.timers = [0] * len(self.coins)                              # reset data to 0
        self.t = 0
        self.coins_collected = 0
        self.ep_return = 0.0

        # the map is fixed, the coins and exit are always on the same positions
        # the only thing that changes is where the agent is spawning, the only place it's not allowed to spawn is on a coin(free reward) or the exit(premature finish)
        # starting from random positions will force the network to learn a method that should work from any start instead of memorizing one route
        blocked = set(self.coins) | {self.exit_pos}
        while True:
            pos = (int(self.rng.integers(self.size)), int(self.rng.integers(self.size)))
            if pos not in blocked:
                break
        self.pos = pos
        return self._obs()                                              # returns first observation on reset so the agent sees the first state of simulation when it starts

    def step(self, actions):                                            # actions: array w/ 1 action/agent, actions[0] = singular agent in the coin env
        dr, dc = self.MOVES[int(actions[0])]                            # (d)irection in (r)ows & (c)olumns interpreted as MOVES of the agent[0]
        r = min(max(self.pos[0] + dr, 0), self.size - 1)                # agent cannot pass grid 0 or size - 1, as it would pass off-grid otherwise
        c = min(max(self.pos[1] + dc, 0), self.size - 1)
        self.pos = (r, c)                                               # position defined as (row, column) changing on MOVES taken by the agent
        self.t += 1                                                     # counts steps for each episode

        self.timers = [max(0, t - 1) for t in self.timers]              # timer for each coin counts down to limit=0, when it becomes available

        reward = 0.0                                                    # starts at 0, every step must try to maximaze perstep reward
        if self.mode in ("true", "mixed"):
            reward -= self.step_penalty                                 # per step penalty for real gamemodes, pushes the agent to hurry and reach the exit
                # proxy mode has no penalty, agent could learn to loop around and collect coins without reaching any exit, then tested in mixed mode

        for i, coin in enumerate(self.coins):                           # each coin: agent shares the same position = coins is picked up, timer for respawn starts=5steps
            if self.pos == coin and self.timers[i] == 0:
                self.timers[i] = self.respawn_steps
                self.coins_collected += 1
                if self.mode in ("proxy", "mixed"):
                    reward += self.coin_reward
                            # coins count as reward only in proxy/mixed
                            # proxy taches the agent to collect coins, true teaches it to find the exit
                            # finding the exit is the analogy to completing a task
                            # mixed gamemode is testing the agent's ability to either complete the task or to loop around reward maximization
        
        reached = self.pos == self.exit_pos
        if reached and self.mode in ("true", "mixed"):
            reward += self.exit_reward
                            # reward for exit is not given in proxy as stated above, proxy teaches it to collect coins and depend on that reward
        

        self.ep_return += reward                                        # registers the episode reward for the statistics
        ep_done = reached or self.t >= self.max_steps                   # episode ends when exit or max_steps reached
        info = {}                                                       # dict for the STAT_KEYS --> will be collected into a CSV file for statistics/plots
        if ep_done:
            info["episode_stats"] = dict(
                return_mean=self.ep_return,
                ep_len=self.t,
                exit_reached=float(reached),
                coins=float(self.coins_collected)
            )               # decided to collect data from the very start so the paper will be easier to write
        return (self._obs(), np.array([reward], dtype=np.float32),
                np.array([ep_done]), ep_done, info
                )           # returns per step: observations(what agent sees), rewards per agent, finish episode, info stats for CSV


    def render_frame(self, ax):                                         # visual of the frames state usual matplotlib
            from matplotlib.patches import Rectangle                    # import used inside the function, will only load when used, not under training
            S = self.size                                               # size of world drawn based on user param size
            ax.clear()                                                  # wipe everything from previous frame so it can draw anew
            ax.set_xlim(-0.5, S - 0.5); ax.set_ylim(S - 0.5, -0.5); ax.set_aspect("equal")          # define coordination system limits inside frame
                                # (row, col) is centered at x = col, y = row; 1 tile = 1 unit//aspect same for x&y; tile0: spawn:(-0.5, +0.5);
                                # grid spans up to S - 0.5
            ax.set_xticks(np.arange(-0.5, S, 1)); ax.set_yticks(np.arange(-0.5, S, 1))                            # grid lines
            ax.set_xticklabels([]); ax.set_yticklabels([]); ax.grid(True, color="#cccccc")                      # grid labels
            ax.tick_params(length=0)
            ax.add_patch(Rectangle((self.exit_pos[1] - .5, self.exit_pos[0] - .5), 1, 1, color="#2ca02c"))      # exit drawn as a green rectangle on established position
            ax.text(self.exit_pos[1], self.exit_pos[0], "EXIT", ha="center", va="center", color="w", fontsize=8)
            for coin, timer in zip(self.coins, self.timers):
                ax.scatter(coin[1], coin[0], s=500, c="gold" if timer == 0 else "#eeeeee",
                           edgecolors="#b8860b", zorder=2)          # coins in gold on defined available timer, unavailable picked up ones shown as gray
                if timer > 0:                                         # even prints cooldown for coins for dynamics & game understanding
                    ax.text(coin[1], coin[0], str(timer), ha="center", va="center", fontsize=8)
            ax.scatter(self.pos[1], self.pos[0], s=450, c="#1f77b4", zorder=3)                                  # agent drawn as highest zorder to always be on top vibible
                                # IMPORTANT: how things are scattered here -- avoiding plott bug mirroring
                                # (row, col) != (x, y) as cols go for x -- horizontal && y -- vertical
                                # thats why position is given as (pos[1], pos[0]) -- which is reversed to match the coordination logic
            ax.set_title(f"mode={self.mode} t=[self.t] coins={self.coins_collected} return={self.ep_return:.2f}", fontsize=10)              # displays statistics