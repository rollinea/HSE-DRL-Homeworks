import pybullet_envs
# Don't forget to install PyBullet!
from gym import make
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
import random
from torch.distributions import Normal
from tqdm import tqdm
ENV_NAME = "Walker2DBulletEnv-v0"

LAMBDA = 0.97
GAMMA = 0.99

ACTOR_LR = 2e-4 / 2
CRITIC_LR = 1e-4 * 2


CLIP = 0.2
ENTROPY_COEF = 1e-2
BATCHES_PER_UPDATE = 256
BATCH_SIZE = 256

MIN_TRANSITIONS_PER_UPDATE = 2048
MIN_EPISODES_PER_UPDATE = 4

ITERATIONS = 1000
SEED = 42


#Orthogonal initialization of weights and constant initialization of biases
def init_weights(module, scale=np.sqrt(2), val=0.0):
    nn.init.orthogonal_(module.weight, scale)
    nn.init.constant_(module.weight, val)
    return module

class Actor(nn.Module):

    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Advice: use same log_sigma for all states to improve stability
        # You can do this by defining log_sigma as nn.Parameter(torch.zeros(...))

        self.action_dim = action_dim
        self.state_dim = state_dim

        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim * 2))

    def compute_proba(self, state, action):
        # Returns probability of action according to current policy and distribution of actions
        out = self.model(state)
        mu = out[:, :self.action_dim]
        sigma = torch.exp(out[:, self.action_dim:])  # we need positive values
        distr = Normal(mu, sigma)
        return torch.exp(distr.log_prob(action).sum(-1)), distr

    def act(self, state):
        # Returns an action (with tanh), not-transformed action (without tanh) and distribution of non-transformed actions
        # Remember: agent is not deterministic, sample actions from distribution (e.g. Gaussian)
        out = self.model(state)
        mu = out[:, :self.action_dim]
        sigma = torch.exp(out[:, self.action_dim:])  # we need positive values
        distr = Normal(mu, sigma)
        action_sample = distr.sample()
        return torch.tanh(action_sample), action_sample, distr


class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

    def get_value(self, state):
        return self.model(state)


class PPO:
    def __init__(self, state_dim, action_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)
        self.actor_optim = Adam(self.actor.parameters(), ACTOR_LR)
        self.critic_optim = Adam(self.critic.parameters(), CRITIC_LR)


    def update(self, trajectories):
        transitions = [t for traj in trajectories for t in traj]  # Turn a list of trajectories into list of transitions
        state, action, old_prob, target_value, advantage = zip(*transitions)
        state = np.array(state)
        action = np.array(action)
        old_prob = np.array(old_prob)
        target_value = np.array(target_value)
        advantage = np.array(advantage)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        for _ in range(BATCHES_PER_UPDATE):
            idx = np.random.randint(0, len(transitions), BATCH_SIZE)  # Choose random batch
            s = torch.tensor(state[idx]).float()
            a = torch.tensor(action[idx]).float()
            op = torch.tensor(old_prob[idx]).float()  # Probability of the action in state s.t. old policy
            v = torch.tensor(target_value[idx]).float()  # Estimated by lambda-returns
            adv = torch.tensor(advantage[idx]).float()  # Estimated by generalized advantage estimation

            # TODO: Update actor here
            self.actor_optim.zero_grad()
            cur_proba, distr = self.actor.compute_proba(s, a)
            frac = cur_proba / op
            actor_loss = -torch.min(frac * adv, torch.clamp(frac, 1 - CLIP, 1 + CLIP) * adv)

            entropy = distr.entropy().mean()
            actor_loss -= entropy * ENTROPY_COEF

            actor_loss = actor_loss.mean()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5) #clipping grad_norm
            self.actor_optim.step()

            # TODO: Update critic here
            self.critic_optim.zero_grad()
            pred = self.critic.get_value(s)
            critic_loss = F.smooth_l1_loss(pred.ravel(), v)
            #critic_loss = torch.min(critic_loss, ((torch.clamp(pred, pred - CLIP, pred + CLIP) - v)**2).sum())
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optim.step()

    def get_value(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state])).float()
            value = self.critic.get_value(state)
        return value.cpu().item()

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array([state])).float().to(self.device)
            action, pure_action, distr = self.actor.act(state)
            prob = torch.exp(distr.log_prob(pure_action).sum(-1))  # one number
        return action.cpu().numpy()[0], pure_action.cpu().numpy()[0], prob.cpu().item()

    def save(self):
        torch.save(self.actor.model, "agent.pkl")


def evaluate_policy(env, agent, episodes=5):
    returns = []
    for _ in range(episodes):
        done = False
        state = env.reset()
        total_reward = 0.

        while not done:
            state, reward, done, _ = env.step(agent.act(state)[0])
            total_reward += reward
        returns.append(total_reward)
    return returns


def compute_lambda_returns_and_gae(trajectory):
    # trajectory is (state, action without tanh, reward, p, value)
    lambda_returns = []
    gae = []  # generilized advantage estimate
    last_lr = 0.
    last_v = 0.
    for _, _, r, _, v in reversed(trajectory):
        ret = r + GAMMA * (last_v * (1 - LAMBDA) + last_lr * LAMBDA)
        last_lr = ret
        last_v = v
        lambda_returns.append(last_lr)
        gae.append(last_lr - v)

    # Each transition contains state, action, old action probability, value estimation and advantage estimation
    return [(s, a, p, v, adv) for (s, a, _, p, _), v, adv in zip(trajectory, reversed(lambda_returns), reversed(gae))]


def sample_episode(env, agent):
    s = env.reset()
    d = False
    trajectory = []
    while not d:
        a, pa, p = agent.act(s)  # action with tanh, action without tanh, p
        v = agent.get_value(s)  # get value function from critic
        ns, r, d, _ = env.step(a)  # get next state, reward, done or not
        trajectory.append((s, pa, r, p, v))  # state, action without tanh, reward, p, value
        s = ns
    return compute_lambda_returns_and_gae(
        trajectory)  # we don't want to use clean rewards or only value fucntion from critic, let's mix it


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


    env = make(ENV_NAME)
    env.seed(SEED)
    env.action_space.seed(SEED)

    ppo = PPO(state_dim=env.observation_space.shape[0],
              action_dim=env.action_space.shape[0])  # state_dim = 22, action_dim = 6
    state = env.reset()
    episodes_sampled = 0
    steps_sampled = 0

    for i in tqdm(range(ITERATIONS)):
            trajectories = []
            steps_ctn = 0

            while len(trajectories) < MIN_EPISODES_PER_UPDATE or steps_ctn < MIN_TRANSITIONS_PER_UPDATE:
                traj = sample_episode(env, ppo)  # list with transitions
                steps_ctn += len(traj)
                trajectories.append(traj)
            episodes_sampled += len(trajectories)
            steps_sampled += steps_ctn

            ppo.update(trajectories)

            if (i + 1) % (ITERATIONS // 100) == 0:
                rewards = evaluate_policy(env, ppo, 5)
                print(
                    f"Step: {i + 1}, Reward mean: {np.mean(rewards)}, Reward std: {np.std(rewards)}, Episodes: {episodes_sampled}, Steps: {steps_sampled}")
                ppo.save()
