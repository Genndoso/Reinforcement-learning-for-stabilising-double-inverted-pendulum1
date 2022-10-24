from torch import nn
import torch
import numpy as np
from collections import deque
import random
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from config_DDQN import config
import os
SAVE_DIR = "saved_models"
PLOTS_DIR = "plots"
from torch.utils.tensorboard import SummaryWriter

class Memory:
    def __init__(self, len):
        self.rewards = deque(maxlen=len)
        self.state = deque(maxlen=len)
        self.action = deque(maxlen=len)
        self.is_done = deque(maxlen=len)

    def update(self, state, action, reward, done):
        # if the episode is finished we do not save to new state. Otherwise we have more states per episode than rewards
        # and actions whcih leads to a mismatch when we sample from memory.
        if not done:
            self.state.append(state)
        self.action.append(action)
        self.rewards.append(reward)
        self.is_done.append(done)

    def sample(self, batch_size):
        """
        sample "batch_size" many (state, action, reward, next state, is_done) datapoints.
        """
        n = len(self.is_done)

        idx = random.sample(range(0, n - 1), batch_size)

        return torch.Tensor(self.state)[idx].to(device), torch.LongTensor(self.action)[idx].to(device), \
               torch.Tensor(self.state)[1 + np.array(idx)].to(device), torch.Tensor(self.rewards)[idx].to(device), \
               torch.Tensor(self.is_done)[idx].to(device)

    def reset(self):
        self.rewards.clear()
        self.state.clear()
        self.action.clear()
        self.is_done.clear()


class QNetwork(nn.Module):
    def __init__(self, action_dim, state_dim, hidden_dim):
        super(QNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, action_dim),

        )
    def forward(self, x):
        x = self.net(x)
        return x


class DDQN:
    def __init__(self, env, discount=0.99, lr=2e-4):

        super().__init__()
        self.discount_rate = discount
        self.learning_rate = lr
        self.min_episodes = config['min_episodes']
        self.eps = config['eps']
        self.eps_decay = config['eps_decay']
        self.eps_min = config['eps_min']
        self.update_step = config['update_step']
        self.batch_size = config['batch_size']
        self.update_repeats = config['update_repeats']
        self.max_episodes = config['max_episodes']
        self.seed = config['seed']
        self.max_memory_size = config['max_memory_size']
        self.measure_step = config['measure_step']
        self.measure_repeats =  config['measure_repeats']
        self.hidden_dim =  config['hidden_dim']
        self.horizon = config['horizon']

        self.num_actions = env.action_space.shape[0]
        # self.load_file = kwargs['load_file']
        self.env = env
        torch.manual_seed(self.seed)

        self.primary_q = QNetwork(action_dim=self.num_actions, state_dim=self.env.observation_space.shape[0],
                                  hidden_dim=self.hidden_dim).to(device)
        self.target_q = QNetwork(action_dim=self.num_actions, state_dim=self.env.observation_space.shape[0],
                                 hidden_dim=self.hidden_dim).to(device)
        # self.discretized_actions = [
        #     (((self.env.action_space.high[0] - self.env.action_space.low[0]) * i / (self.num_actions - 1)) - 1)
        #     for i in range(self.num_actions)]

    def save_models(self, file_name=None):
        if file_name is None:
            file_name = f"DDQN-{self.max_episodes}"

            os.makedirs(SAVE_DIR, exist_ok=True)
            path = SAVE_DIR + "/" + file_name
            path = os.ensure_unique_path(path)

            torch.save({
                'primary_q': self.primary_q.state_dict(),
                'target_q': self.target_q.state_dict(),
            }, path)

    def select_action(self, model, state, eps):
        state = torch.Tensor(state).to(device)
        with torch.no_grad():
            values = model(state)

            # select a random action wih probability eps
        if random.random() <= eps:
            action = np.random.randint(0, self.num_actions)
        else:
            action = np.argmax(values.cpu().numpy())

        return action

    def train(self, batch_size, primary, target, optim, memory, discount_rate):
        states, actions, next_states, rewards, is_done = memory.sample(batch_size)

        q_values = primary(states)

        next_q_values = primary(next_states)
        next_q_state_values = target(next_states)

        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = rewards + discount_rate * next_q_value * (1 - is_done)

        loss = (q_value - expected_q_value.detach()).pow(2).mean()

        optim.zero_grad()
        loss.backward()
        optim.step()

    def evaluate(self, eval_repeats):
        """
        Runs a greedy policy with respect to the current Q-Network for "repeats" many episodes. Returns the average
        episode reward.
        """
        self.primary_q.eval()
        scores = []
        for e in range(eval_repeats):
            state, done = self.env.reset()

            perform = 0
            c = 0
            while c <= 300:
                state = torch.Tensor(state).to(device)
                with torch.no_grad():
                    values = self.primary_q(state)
                action = np.argmax(values.cpu().numpy())
                print(action)
                state, reward, done, _ = self.env.step(self.env.action_space[action])
                perform += reward
                c += 1

            scores.append([e, perform])

        scores = np.array(scores)
        self.primary_q.train()
        return scores[:, 1].mean(), scores

    def update_parameters(self, current_model, target_model):
        target_model.load_state_dict(current_model.state_dict())

    def run(self):
        # transfer parameters from Q_1 to Q_2
        self.update_parameters(self.primary_q, self.target_q)

        directory_plots = "DDQN_plots"
        if not os.path.exists(directory_plots):
            os.makedirs(directory_plots)

        writer = SummaryWriter(log_dir=directory_plots)

        # we only train Q_1
        for param in self.target_q.parameters():
            param.requires_grad = False

        optimizer = torch.optim.Adam(self.primary_q.parameters(), lr=self.learning_rate)

        self.performance = []
        memory = Memory(self.max_memory_size)
        for episode in range(self.max_episodes):
            # display the performance
            if episode % self.measure_step == 0:
                mean, _ = self.evaluate(self.measure_repeats)
                self.performance.append([episode, mean])
                print(f"\nEpisode: {episode}\nMean accumulated reward: {mean}\neps: {self.eps}")
                writer.add_scalar('Mean accumulated reward', mean, episode)

            state, done = self.env.reset()
            memory.state.append(state)

            i = 0
            while not done:
                i += 1
                action = self.select_action(self.target_q, state, self.eps)
                state, reward, done, _ = self.env.step(self.env.action_space[action])
                if i > self.horizon:
                    done = True

                    #     # render the environment if render == True
                    # if self.do_render and episode % self.render_step == 0:
                    #       self.env.render()

                    # save state, action, reward sequence
                memory.update(state, action, reward, done)

            if episode >= self.min_episodes and episode % self.update_step == 0:
                for _ in range(self.update_repeats):
                    self.train(self.batch_size, self.primary_q, self.target_q, optimizer, memory, self.discount_rate)

                # transfer new parameter from Q_1 to Q_2
                self.update_parameters(self.primary_q, self.target_q)

            self.eps = max(self.eps * self.eps_decay, self.eps_min)

        return np.array(self.performance)


