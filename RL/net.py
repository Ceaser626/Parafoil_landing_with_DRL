import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


class Agent(nn.Module):
    def __init__(self, env):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            self.layer_init(nn.Linear(env.obs_dimension, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 32)),
            nn.Tanh(),
            self.layer_init(nn.Linear(32, 1)),)
        self.actor_mean = nn.Sequential(
            self.layer_init(nn.Linear(env.obs_dimension, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            self.layer_init(nn.Linear(64, 32)),
            nn.Tanh(),
            self.layer_init(nn.Linear(32, env.act_dimension)),)
        log_std = -0.5 * np.ones(env.act_dimension, dtype=np.float32)
        self.actor_logstd = nn.Parameter(torch.as_tensor(log_std))

    @staticmethod
    def layer_init(layer, std=0.1, bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        return torch.squeeze(self.critic(x), -1)

    def get_action_and_value(self, x, action=None, deterministic=False):
        action_mean = self.actor_mean(x)
        action_std = torch.exp(self.actor_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            if deterministic:  # for test
                action = action_mean.detach()
            else:
                action = probs.sample()
        return action, probs.log_prob(action).sum(axis=-1), probs.entropy().sum(axis=-1), self.get_value(x)
