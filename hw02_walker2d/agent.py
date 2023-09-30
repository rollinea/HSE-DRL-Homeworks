import random
import numpy as np
import os
import torch
from torch.distributions import Normal

class Agent:
    def __init__(self):
        self.model = torch.load(__file__[:-8] + "/agent.pkl")
        
    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array(state)).float()
            out = self.model(state)
            action_dim = out.size(-1) // 2
            out = out[None, :]
            mu = out[:, :action_dim]
            sigma = torch.exp(out[:, action_dim:])
            distr = Normal(mu, sigma)
            return torch.tanh(distr.sample().squeeze()).cpu().numpy()

    def reset(self):
        pass


