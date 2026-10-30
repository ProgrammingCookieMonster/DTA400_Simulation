"""
PPO trainer with Parameter Sharing, Reinforcement Learning.                 --- the learner ---
The zombie game has agents that die mid-game while many agents share the same brain trained by all, 
they share the same neural network.
Agents that go under the active mask mid-game cannot be trained with SB3/PettingZoo/SuperSuit as was intended initially

Should be able to run E environment for T steps each and collect data from them,
judge if and how every step computes an advantage, epoch step adjustment
to make the good actions more likely -- gradually -- && log as rows inside a CSV file

------------------------
Agent = function, numbers IN (observations from the world env), action OUT; function: weights;
Training -- repeating these steps in many episode runs: 
    1. Play many steps with current weights + record what happens
    2. Work out which actions are better and which are worse
    3. Adjust weights so the better actions play out more
After many iterations, the agent function should behave like it wants the reward.

----------------------
Neural network = stack of simple operations.
Linear layer + Tanh = grid of weights squashes the numbers and add offsets, Tanh ranges them (-1, 1)
    --> network can represent curved/non-linear relationships: e.g. zombie ahead + survivor has ammo ==> shoot, or something like that
    (the type of explanation coming from a student who deep-dives like this in the subject for the first time)
        --> 2 Networks: actor-critic design
        actor: takes observations, outputs preference scores for each; agent uses actions based on that score.
        critic: also takes observations, outputs its estimated opition of total future reward
            --- model compares if, after taken an action, the outcome is better/worse than what the critic has estimated
                Better/Worse decides which actions are to be done more and which less.

----------------------
Proximal Policy Optimization (PPO) measures the ration for how to nudge the weights
Drastic Policy changes can collapse the learning otherwise, PPO keeps it stable

----------------------
Parameters sharing -- zombie game: 10 runners use 1 network
Each runner inputs own observations, yet act differently, trains the same brain w/ inputs from all of them
"""

import csv                                                                              # for CSV logs
import time                                                                             # measurement for how long training takes
import numpy as np
import torch                                                                            # neural network library
import torch.nn as nn                                                                   # nn: layers for the network, tables of numbers =  tensors
from torch.distributions import Categorical


def layer_init(layer, std=np.sqrt(2), bias=0.0):                                        # weight initialization helper function, where weights start before training
    nn.init.orthogonal_(layer.weight, std)                                              # standard init to start with uniform policy and explore everything
    nn.init.constant_(layer.bias, bias)
    return layer

class ActorCritic(nn.Module):                                                           # neural network = always class from nn.Module
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()                                                              # and that module is initialized securely are requirement here
        self.actor = nn.Sequential(                                                     # sequential chain of layers, data flows in order through them
                    # hidden layer -- in between raw input and output layers, called hidden because they are in between
                    # they learn abstract internal representations of the world
                    # hidden = 128 -- n.o neurons of the layer, allows enough to track spation relationships on the grid, like coins, zombies, walls
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, n_actions), std=0.01)                          # action preferences run at 0.01, so all option are equally likely at first
                # ==> results in the action probabilities the agent takes, something like a form [2.5, -1.2, 0.4, 1.1] from the [up, down, left, right] options it had
                # -- these outputs are technically considered raw logits?
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),                          # Linear(a, b), input a numbers, outputs b, normalized by Tanh=(-1, 1) range
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),                               # so takes all observations and outputs by n.o neurons in hidden=128
            layer_init(nn.Linear(hidden, 1), std=1.0)                                   # expects 1 value: the final reward
        )

    def value(self, obs):                                                               # called for critic's forecast
        return self.critic(obs).squeeze(-1)
                        # --> outputs table of columns, squeezing it like this turns in into a plain list of forecasts

    def act(self, obs, deterministic=False):
                # False deterministic has a random chance to allow the agent to pick choices with a low change so it can still explore paths
                # True one is a greedy evaluation mode, it will only pick the option with the highest trained score
        dist = Categorical(logits=self.actor(obs))
                            # logits = raw score outputs generated by last linear layer before probability activation function is applied
                            # passed in a Caterogical, turns the raw logits into positive valid form e.g [0.65, 0.05, 0.10, .20] adding up to 1.0 = 100% chance distribution
        a = dist.probs.argmax(-1) if deterministic else dist.sample()                   # only runs highest change action if deterministic, has change to explore otherwise
        return a, dist.log_prob(a), self.value(obs)
                            # must return old probabilities too, so they can be compared with the new ones and adjust weights
    
    def evaluate(self, obs, actions):                                                   # current network evaluates past actions
        dist = Categorical(logits=self.actor(obs))
        return dist.log_prob(actions), dist.entropy(), self.value(obs)                  # entropy = how spread probabilities are, high entropy = still exploring
                            # generally used to re-score actions collected earlier, so training can compare with new


class PPOTrainer:
    # prepare trainer state
    # create envs and model
    # set up initial obs & logs
    # prepare optimizer
    def __init__(self, make_env, ppo_cfg, seed, log_path):
        self.cfg = ppo_cfg                                                              # save configs in class
        torch.manual_seed(seed)                                                         # init random generators
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)

        self.envs = [make_env() for _ in range(ppo_cfg["n_envs"])]                      # parallell envs running in the same time, training gets varied data per update
        e0 = self.envs[0]                                                               # reads n_agents, obs_dim, n_actions, STAT_KEYS from first created env
        self.N, self.D, self.A = e0.n_agents, e0.obs_dim, e0.n_actions                  # n.o agents, obs size, n.o available actions for e0
        self.stat_keys = e0.STAT_KEYS
        self.model = ActorCritic(self.D, self.A, ppo_cfg["hidden"])                     # called called from above, stays default:hidden
        self.opt = torch.optim.Adam(self.model.parameters(), lr=ppo_cfg["lr"], eps=1e-5)    # Adam optimizer over model's params, optimizer actually turns the weights
                                                # lr = learning rate, how much the weights nudge

        # reseting every copy of the env with a different seed, stacks the result in array: copies x agents x numbers
        self.obs = np.stack([env.reset(seed=seed * 1000 + k) for k, env in enumerate(self.envs)])   # stack glues all observations from all games together in array form
        self.active = np.stack([env.alive_mask for env in self.envs])
                                    # maching table of which agents are alive, the network should not learn from dead agents, failed = took bad actions

        # create CSV log and write the header row, all iterations after this will just append new lines to the file
        # basically acts as the data storage
        self.log_path = log_path
        self.header = ["iter", "env_steps", "time_s", "n_episodes", "entropy",
                       "policy_loss", "value_loss"] + self.stat_keys
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(self.header)

    # 1. Collect --------------------------------
    # play N numbers of episodes and log everything from all of them
    # learning curve data
    def collect(self):
        # build buffer dictionary of empty arrays
        # store observations, actions, log-probabilities, critic's value, rewards, done flags, mask
        T, E, N, D = self.cfg["rollout_len"], len(self.envs), self.N, self.D
                        # steps/env, n.o envs, N-agents, obs size
        buf = dict(obs=np.zeros((T, E, N, D), np.float32), act=np.zeros((T, E, N), np.int64),
                   logp=np.zeros((T, E, N), np.float32), val=np.zeros((T, E, N), np.float32),
                   rew=np.zeros((T, E, N), np.float32), done=np.zeros((T, E, N), np.float32),
                   mask=np.zeros((T, E, N), np.float32)                                   # save obs and mask in buffer
                   )
        finished = []
        for t in range(T):
            buf["obs"][t], buf["mask"][t] = self.obs, self.active
                    # for each step, save what everbody can see & who is alive
            with torch.no_grad():                                                               # no notes registred
                a, lp, v = self.model.act(torch.as_tensor(self.obs.reshape(E * N, D)))
                    # stack all agents in 1 table and ask all of them for their observations
                    # as_tensor converts from numpy to PyTorch format
            a_np = a.numpy().reshape(E, N)
                    # converts back to numpy and reshape into 1 row per env, so each env can get its own's agent's actiions
            buf["act"][t], buf["logp"][t], buf["val"][t] = a_np, lp.numpy().reshape(E, N), v.numpy().reshape(E, N)
            for e, env in enumerate(self.envs):
                obs, rew, agent_done, ep_done, info = env.step(a_np[e])                                  # for all envs, take a step and record
                buf["rew"][t, e], buf["done"][t, e] = rew, agent_done                           # record rewards & done flags for taken step
                if ep_done:
                    finished.append(info["episode_stats"])                                      # record stats if the taken step finished the game
                    obs = env.reset()                                                           # reset entire episode, so another can run immediatly after, trainer never pauses to wait for the whole batch of envs to finish beforen it can start another batch
                self.obs[e], self.active[e] = obs, env.alive_mask                               # update observations and alive mask as games finish and restart
        with torch.no_grad():   # no notes
            last_v = self.model.value(torch.as_tensor(self.obs.reshape(E * N, D))).numpy().reshape(E, N)
                    # save the last forecast of the critic as a current evaluation of the current step
                    # needed for next stage, where last reward is compared to current one
        return buf, last_v, finished

    # 2. Judge(advantage) --------------------------------
    # work out advantage for each action
    # For each step in the past, 3 things are known:
        # 1. Critics value/forecast (V) before the step was executed
        # 2. Actual reward (r) the agent received
        # 3. The new value (V_next) the critic forecasts for next step
            # using gamma reward: delta = r + 0.99 * V_next - V; 0.99 = gamma
            # this tells the agent to not abandon the bad action immediatly, as the "later reward" can still be worth it
    # Say critic forecasts V=5, r = 1 (which makes it a bad action), but the next forecast V_next=4.5.
        # delta = 1 + 0.99*4.5 - 5 = 0.455, which is still > 0, positive
    
    # these things can spike noise, so they are blended with the following steps (using GAE)
    # each step's judgement leans on the next
    def advantages(self, buf, last_v):
        g, lam = self.cfg["gamma"], self.cfg["gae_lambda"]
        T = buf["rew"].shape[0]
        adv = np.zeros_like(buf["rew"])
        last = 0.0
        for t in reversed(range(T)):
            next_v = last_v if t == T - 1 else buf["val"][t + 1]
            nonterminal = 1.0 - buf["done"][t]                                  # becomes 0 if episode ended, avoids bleeding
                                                                                # this way failure is not inherited or passed on from/to other episodes
            delta = buf["rew"][t] + g * next_v * nonterminal - buf["val"][t]
            last = delta + g * lam * nonterminal * last
            adv[t] = last
        return adv, adv + buf["val"]                                            # adv + buf["val"] is used to train the critic

    # 3. Adjust --------------------------------
    # adjust weights based on found advantage
    def update(self, buf, adv, ret, progress):
        c = self.cfg
        for group in self.opt.param_groups:
            group["lr"] = c["lr"] * (1.0 - progress)
                # lr - learning rate is big at the beginning, and shrinks down to nothing at the end
        keep = buf["mask"].reshape(-1) > 0                              # reshape it into a row list and turn it into True/False cases
        # observations, actions, old probabilities, advantages, return targets as [list] will only keep the True values
        # False ones, dead agents, present no usefull experience for the network
        obs = torch.as_tensor(buf["obs"].reshape(-1, self.D)[keep])
        act = torch.as_tensor(buf["act"].reshape(-1)[keep])
        old_lp = torch.as_tensor(buf["logp"].reshape(-1)[keep])
        advt = torch.as_tensor(adv.reshape(-1)[keep])
        rett = torch.as_tensor(ret.reshape(-1)[keep])
        advt = (advt - advt.mean()) / (advt.std() + 1e-8)               # advatange averages 0, size = 1, size of the weight change are similar from batch to batch

        # dont use all data in 1 big step
        # its shuffled and cut into minibatches, ran through 4 times (4 epochs)
        n = len(act)
        mb = max(1, n // c["minibatches"])
        stats = []
        for _ in range(c["epochs"]):
            perm = torch.randperm(n)                                    # samples are shuffled by randperm
            for s in range(0, n, mb):
                idx = perm[s:s + mb]                                    # sample numbers in the current minibatch
                new_lp, ent, val = self.model.evaluate(obs[idx], act[idx])
                ratio = (new_lp - old_lp[idx]).exp()
                            # evaluates new and old probabilities and uses a ratio to learn the better actions based on current observations
                pg = torch.max(-advt[idx] * ratio, -advt[idx] * ratio.clamp(1 - c["clip"], 1 + c["clip"])).mean()
                            # reward network for using probability of good action (positive advantage), but no more than 20%
                vl = 0.5 * ((val - rett[idx]) ** 2).mean()              # minimizing the critic's mistake in forecast
                loss = pg + c["vf_coef"] * vl - c["ent_coef"] * ent.mean()      # total loss number out of policy, critic, entropy, less loss means better decisions
                self.opt.zero_grad()                                            # forgets previous round's advice
                loss.backward()                                                # checks the loss effect if weight is changed in a certain way, checks if it's worth
                nn.utils.clip_grad_norm_(self.model.parameters(), c["max_grad_norm"])   # shrink the advice in case it comes out as too large
                self.opt.step()                                                 # change the weights a little bit towards the direction that shrinks the loss
                stats.append((ent.mean().item(), pg.item(), vl.item()))         # append all to the stats
        return np.mean(stats, axis=0)                                           # log: average entropy & losses


    # 4. Training loop --------------------------------
    # ties the above steps together, so the trainer loops to learn
    def train(self, total_env_steps, verbose=True, watch_key=None):
        steps_per_iter = self.cfg["rollout_len"] * len(self.envs)
        n_iters = max(1, total_env_steps // steps_per_iter)
                        # 1 iteration = N steps/env (based on size) x number of envs
        t0 = time.time()
        for it in range(1, n_iters + 1):            # cycle
            buf, last_v, finished = self.collect()
            adv, ret = self.advantages(buf, last_v)
            ent, pl, vl, = self.update(buf, adv, ret, progress=(it - 1) / n_iters)

            means = ([np.mean([s[k] for s in finished]) for k in self.stat_keys])
                        # for each statistic, collect it from each episide, take the mean out of it into one line list
            row = [it, it * steps_per_iter, round(time.time() - t0, 1), len(finished), ent, pl, vl] + means

            with open(self.log_path, "a", newline="") as f:
                csv.writer(f).writerow(row)
            if verbose and (it % 10 == 0 or it == 1):
                extra = f"  {watch_key}={means[self.stat_keys.index(watch_key)]:.3f}" if watch_key else ""
                print(f"iter {it}/{n_iters} env_steps={it * steps_per_iter} "
                      f"episodes={len(finished)} entropy={ent:.2f}{extra} [{time.time() - t0:0f}s]", flush=True
                      )
        return self.model