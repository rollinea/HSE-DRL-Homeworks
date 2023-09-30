from gym import make
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from collections import deque, namedtuple
import random
import copy
from tqdm.auto import tqdm

GAMMA = 0.99
INITIAL_STEPS = 1024
TRANSITIONS = 500000
STEPS_PER_UPDATE = 4
STEPS_PER_TARGET_UPDATE = STEPS_PER_UPDATE * 1000
BATCH_SIZE = 128
LEARNING_RATE = 0.001
SEED = 42

class DQN_model(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN_model, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, transition):
        self.memory.append(transition)

    def sample(self, ids):
        samples = [self.memory[i] for i in ids]
        return zip(*samples)

    def __len__(self):
        return len(self.memory)


class DQN:
    def __init__(self, state_dim, action_dim):
        self.steps = 0  # Do not change
        self.replay_buffer = ReplayBuffer(capacity=100000)
        self.Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

        self.model = DQN_model(state_dim, action_dim)  # Torch model
        self.target_model = copy.deepcopy(self.model)

        self.optimizer = Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.loss = nn.MSELoss()



    def consume_transition(self, transition):
        # Add transition to a replay buffer.
        # Hint: use deque with specified maxlen. It will remove old experience automatically.
        self.replay_buffer.push(transition)

    def sample_batch(self):
        # Sample batch from a replay buffer.
        # Hints:
        # 1. Use random.randint
        # 2. Turn your batch into a numpy.array before turning it to a Tensor. It will work faster
        ids = np.random.randint(0, len(self.replay_buffer), BATCH_SIZE)
        state, action, next_state, reward, done = self.replay_buffer.sample(ids)

        state = torch.tensor(np.array(state)).float()
        next_state = torch.tensor(np.array(next_state)).float()
        reward = torch.tensor(np.array(reward)).float()
        action = torch.tensor(np.array(action))
        done = torch.tensor(np.array(done))

        return (state, next_state, reward, action, done)

    def train_step(self, batch):
        # Use batch to update DQN's network.
        state, next_state, reward, action, done = batch
        target_model = self.target_model
        target = torch.zeros(reward.size()[0]).float()

        with torch.no_grad():
            target[~done] = self.model(next_state).max(1)[0][~done]

        target = reward + target * GAMMA
        loss = self.loss(self.model(state).gather(1, action.view(-1, 1)).flatten(), target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        # Update weights of a target Q-network here. You may use copy.deepcopy to do this or 
        # assign a values of network parameters via PyTorch methods.
        self.target_model = copy.deepcopy(dqn.model)

    def act(self, state, target=False):
        # Compute an action. Do not forget to turn state to a Tensor and then turn an action to a numpy array.
        return self.model(torch.tensor(state).float()).max(dim=0)[1].item()

    def update(self, transition):
        # You don't need to change this
        self.consume_transition(transition)
        if self.steps % STEPS_PER_UPDATE == 0:
            batch = self.sample_batch()
            self.train_step(batch)
        if self.steps % STEPS_PER_TARGET_UPDATE == 0:
            self.update_target_network()
        self.steps += 1

    def save(self):
        torch.save(self.model.state_dict(), "agent.pkl")


def evaluate_policy(agent, episodes=5):
    env = make("LunarLander-v2")
    returns = []
    for _ in tqdm(range(episodes)):
        done = False
        state, info = env.reset()
        total_reward = 0.

        while not done:
            state, reward, terminated, truncated, _ = env.step(agent.act(state))
            done = terminated or truncated
            total_reward += reward
        returns.append(total_reward)
    return returns


if __name__ == "__main__":
    env = make("LunarLander-v2")  # создали среду
    dqn = DQN(state_dim=env.observation_space.shape[0], action_dim=env.action_space.n)
    eps = 0.1
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    state, _ = env.reset()  # текущее состояние np.array() из 8-ми элементов

    for _ in range(INITIAL_STEPS):  # просто случайно самплим действия чтобы пополнить буфер, здесь нет никакой политики
        action = env.action_space.sample()  # самплим действие - их 4 штуки, какое-то одно выпадет
        next_state, reward, terminated, truncated, _ = env.step(action)  # получили следующее состояние
        done = terminated or truncated
        dqn.consume_transition((state, action, next_state, reward, done))  # добавили в нашу очередь

        state = next_state if not done else env.reset()[0]

    for i in range(TRANSITIONS):
        # Epsilon-greedy policy
        if random.random() < eps:  # совершаем рандом
            action = env.action_space.sample()
        else:  # совершаем действие основанное на политике (предсказывает наша сетка)
            action = dqn.act(state)

        next_state, reward, terminated,truncated, _ = env.step(action)  # совершаем действие
        done = terminated or truncated
        dqn.update((state, action, next_state, reward, done))

        state = next_state if not done else env.reset()[0]

        if (i + 1) % (TRANSITIONS // 100) == 0:
            rewards = evaluate_policy(dqn, 5)
            dqn.save()

