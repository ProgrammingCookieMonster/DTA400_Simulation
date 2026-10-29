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
import torch.nn as nn                                                                   # nn: layers for the network
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
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0)                                   # expects 1 value: the final reward
        )

    def value(self, obs):
        return self.critic(obs).squeeze(-1)                                             # runs critic, squeezes output just [batch_size] row

    def act(self, obs, deterministic=False):
                # False deterministic has a random chance to allow the agent to pick choices with a low change so it can still explore paths
                # True one is a greedy evaluation mode, it will only pick the option with the highest trained score
        dist = Categorical(logits=self.actor(obs))
                            # logits = raw score outputs generated by last linear layer before probability activation function is applied
                            # passed in a Caterogical, turns the raw logits into positive valid form e.g [0.65, 0.05, 0.10, .20] adding up to 1.0 = 100% chance distribution
        a = dist.probs.argmax(-1) if deterministic else dist.sample()                   # only runs highest change action if deterministic, has change to explore otherwise
        return a, dist.log_prob(a), self.value(obs)
                            # must return old probabilities too, so they can be compared with the new ones and adjust weights
    
    def evaluate(self, obs, actions):                                                   # recomputes probability
        dist = Categorical(logits=self.actor(obs))
        return dist.log_prob(actions), dist.entropy(), self.value(obs)                  # entropy = how spread probabilities are, high entropy = still exploring
                            # generally used to re-score actions collected earlier, so training can compare with new


class PPOTrainer:
    pass