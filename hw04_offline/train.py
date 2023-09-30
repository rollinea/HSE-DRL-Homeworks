import pybullet_envs
from gym import make
from collections import deque
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.distributions import Normal
import random
import copy
import pickle
from tqdm.auto import tqdm

GAMMA = 0.99
TAU = 0.005
CRITIC_LR = 3e-4
ACTOR_LR = 3e-4
DEVICE = "cpu"
BATCH_SIZE = 256
ITERATIONS = 1000000
SEED = 42
EPS = 0.2
LAMBDA = 2.0


def soft_update(target, source):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_((1 - TAU) * tp.data + TAU * sp.data)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.action_dim = action_dim
        self.state_dim = state_dim

        # Stochastic policy

        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, action_dim * 2))

    def act(self, state):
        out = self.model(state)
        mu = out[:, :self.action_dim]
        sigma = torch.exp(out[:, self.action_dim:])
        distr = Normal(mu, sigma)
        pi_action = distr.rsample()
        logp_pi = distr.log_prob(pi_action).sum(-1)
        pi_action = torch.tanh(pi_action)
        return pi_action, logp_pi

    def compute_proba(self, state, action):
        out = self.model(state)
        mu = out[:, :self.action_dim]
        sigma = torch.exp(out[:, self.action_dim:])
        distr = Normal(mu, sigma)
        return distr.log_prob(action).sum(-1)


class Critic(nn.Module):
    # Calculates Q_function
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 1)
        )
    def forward(self, state, action):
        return self.model(torch.cat([state, action], dim=-1)).view(-1)


class Algo:
    def __init__(self, state_dim, action_dim, data):
        # TODO: You can modify anything here
        self.actor = Actor(state_dim, action_dim).to(DEVICE)
        self.critic_1 = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_2 = Critic(state_dim, action_dim).to(DEVICE)

        self.actor_optim = Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_1_optim = Adam(self.critic_1.parameters(), lr=ACTOR_LR)
        self.critic_2_optim = Adam(self.critic_2.parameters(), lr=ACTOR_LR)

        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic_1 = copy.deepcopy(self.critic_1)
        self.target_critic_2 = copy.deepcopy(self.critic_2)

        self.loss = nn.MSELoss()
        self.replay_buffer = list(data)

    def update(self, steps):
        if len(self.replay_buffer) > BATCH_SIZE * 16:
            # Sample batch
            transitions = [self.replay_buffer[random.randint(0, len(self.replay_buffer) - 1)] for _ in
                           range(BATCH_SIZE)]
            state, action, next_state, reward, done = zip(*transitions)
            state = torch.tensor(np.array(state), device=DEVICE, dtype=torch.float)
            action = torch.tensor(np.array(action), device=DEVICE, dtype=torch.float)
            next_state = torch.tensor(np.array(next_state), device=DEVICE, dtype=torch.float)
            reward = torch.tensor(np.array(reward), device=DEVICE, dtype=torch.float)
            done = torch.tensor(np.array(done), device=DEVICE, dtype=torch.float)

            # Update critic
            self.critic_1_optim.zero_grad()
            self.critic_2_optim.zero_grad()

            q1 = self.critic_1(state, action)
            q2 = self.critic_2(state, action)

            with torch.no_grad():
                next_action, _ = self.actor.act(next_state)
                q_next = torch.minimum(self.target_critic_1(next_state, next_action), self.target_critic_1(next_state, next_action))
                y_true =  reward + GAMMA * (1 - done) * q_next

            critic_1_loss = self.loss(q1, y_true)
            critic_2_loss = self.loss(q2, y_true)
            loss = critic_1_loss + critic_2_loss

            loss.backward()

            self.critic_1_optim.step()
            self.critic_2_optim.step()

            # Update actor
            self.actor_optim.zero_grad()
            with torch.no_grad():
                pi_action, _ = self.actor.act(state) # get action from current policy
                v = torch.minimum(self.critic_1(state, pi_action),
                                  self.critic_2(state, pi_action))
                q = torch.minimum(self.critic_1(state, action),
                                  self.critic_2(state, action))

                adv = q - v
                w = F.softmax(adv / LAMBDA, dim=0)

            pi_logp = self.actor.compute_proba(state, action)
            actor_loss = (-pi_logp * w).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            soft_update(self.target_critic_1, self.critic_1)
            soft_update(self.target_critic_2, self.critic_2)
            soft_update(self.target_actor, self.actor)

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state])).float().to(self.device)
            action, logp = self.actor.act(state)
        return action.cpu().numpy()[0], logp.cpu().numpy()[0]

    def save(self):
        torch.save(self.actor.model, "agent.pkl")


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    with open("./offline_data.pkl", "rb") as f:
        data = pickle.load(f)

    algo = Algo(state_dim=32, action_dim=8, data=data)

    for i in tqdm(range(ITERATIONS)):
        steps = 0
        algo.update(i)

        if (i + 1) % (ITERATIONS//10000) == 0:
            algo.save()

