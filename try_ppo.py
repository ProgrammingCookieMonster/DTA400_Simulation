# small attent to watch the trainer work

import torch
from envs.coin_env import CoinEnv
from ppo import PPOTrainer

torch.set_num_threads(1)                                                                    # use 1 CPU core for network this small

ppo_cfg = dict(lr=3e-4, gamma=0.99, gae_lambda=0.95, clip=0.2, ent_coef=0.01,               # configuration disctionary
               vf_coef=0.5, epochs=4, minibatches=4, max_grad_norm=0.5,
               hidden=128, n_envs=16, rollout_len=128
               )

trainer = PPOTrainer(lambda: CoinEnv("proxy"), ppo_cfg, seed=1, log_path="demo_tests/try_log.csv")     # lambda builds 1 CoinEnv, the trainer then makes 16 copy of it, given configurations
trainer.train(100_000, watch_key="coins")                                                   # plays 100k game steps, will look for the average coins per episode, proxy wants more coins
print("Finished training. Learning Curve saved in try_log.csv")

### Expectation from test file
## Given the data & configuration
# -- expects about 48 calculated iteration, where: entropy shrinks as actions become more sure && coin average raises as proxy is made to teach and encourage coin collection