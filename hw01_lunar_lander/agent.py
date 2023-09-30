import gym
import pickle
import torch

from torch import nn
from torch.nn import functional as F


class DQN_model(nn.Module):
    def __init__(self, state_dim=8, action_dim=4):
        super(DQN_model, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Agent:

    def __init__(self):
        self.model = DQN_model()
        self.model.load_state_dict(torch.load(__file__[:-8] + "/agent.pkl"))
        self.model.eval()

        
    def act(self, state):
        state = torch.tensor(state).float()
        return self.model(torch.tensor(state).float()).max(dim=0)[1].item()

