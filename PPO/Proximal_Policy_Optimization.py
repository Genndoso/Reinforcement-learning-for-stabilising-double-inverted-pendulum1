from Environment import DoublePendulumEnv, ActionSpaceCartPole, ObservationSpaceCartPole
from torch import nn
import torch
from torch.utils.tensorboard import SummaryWriter
import gym
from torch.distributions import Normal, Categorical
from gym.spaces import Box
import random
from IPython.display import clear_output
from torch.distributions import MultivariateNormal
from .config_PPO import config


class RolloutBuffer:
    """
    Buffer for storing a state parameters
    actions: list of actions.
    states: list of states.
    logprobs: list of log of probability of action taken at current timestep under given policy.
    rewards: list of rewards for actions performed.
    dones: list of done flag for every action taken.
    """
    def __init__(self):
        # Initialize all list.
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.dones = []

    def clear(self):
        # Clear all the list.
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.dones[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim = ObservationSpaceCartPole(), action_dim = ActionSpaceCartPole(), action_std_init = config['action_std'], hidden_size = config['hidden_size']):
        super(ActorCritic, self).__init__()
        self.state_dim = 6
        self.action_dim = 1
        # Instead of directly using variance as input to normal distribution, standard deviation is set as hyperparameter
        # and used to calculate the variance.
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init)
        # Actor NN
        self.hidden_size = hidden_size
        self.actor = nn.Sequential(
            nn.Linear(state_dim, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(self.hidden_size, action_dim),
            nn.Tanh()
        )
        # Critic NN
        self.critic = nn.Sequential(
            nn.Linear(state_dim, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, 1)
        )

    def set_action_std(self, new_action_std):
        """
        Performance of PPO is sensitive to standard deviation. The standard deviation is decaying by 0.05
        every 90000 timestep. This function sets new standard deviation to be used while creating normal distribution.
        """
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std)

    def get_action(self, state):
        """
        Called during sampling phase.
        Passing a State through actor network to get the mean and plot normal distribution based on that mean to sample
        an action.
        """
        action_mean = self.actor(state)  # output of actor network
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)  # Variance of Normal distribution.
        policy = MultivariateNormal(action_mean, cov_mat)  # Generating a policy based on (mean, variance)

        action = policy.sample()  # Action sampeled from policy and to be applied.
        action_logprob = policy.log_prob(action)  # log of prob. of that action given the distribution.

        # Since, we are only interested in the action and its prob., we do not perform SGD on them and can detach the
        # computational graph associated with it.
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        """
        Called during update phase.
        In order to calculate the ratio of prob. of taking an action given a state under new policy, we need to
        pass the old sampled state and action taken old policy and get mean, value and logprob under new policy.
        New policy means updated model weights compared to the weights using which the action and value was approximated
        during sampling phase.
        """
        action_mean = self.actor(state)
      #  action_mean = torch.clamp(action_mean, min = config['action_low'], max = config['action_high'])
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var)
        policy = MultivariateNormal(action_mean, cov_mat)

        if self.action_dim == 1:
            action = action.reshape(-1, self.action_dim)
        action_logprobs = policy.log_prob(action)
        policy_entropy = policy.entropy()
        state_values = self.critic(state) #critic
        # The 'action_logprobs' will be used to calculate the ratio for surrogate loss.
        # The 'state_values' will be used to calculate the MSE for critic loss.
        # The 'policy_entropy' will be used give bonus for exploration in final loss.
        return action_logprobs, state_values, policy_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor = config['lr_actor'], lr_critic = config['lr_critic'], action_std_init = config['action_std']):
        self.action_std = action_std_init

        self.gamma = config['gamma']  # Discount factor
        self.eps_clip = config['eps_clip']  # Clipping range for Clipped Surrogate Function.
        self.K_epochs = config['max_number_of_epoch']  # Total number of epochs.

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, action_std_init)
        # Adam Optimizer to perform optimization given the loss on NN parameters.
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        # Sampling is always done under old policy(aka old weights) under the evaluation for update step is always done
        # under new policy(aka update/new weights). After 800 timestep * Number of Epochs,
        # old_policy parameters set same as new_policy parameters.
        self.policy_old = ActorCritic(state_dim, action_dim, action_std_init)
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Mean Squared Error
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        # Setting new standard deviation.
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")
        # Calculate the new standard deviation corresponding to decay rate.
        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 2)
        # In case, standard deviation decays below threshold, set it to minimum.
        if self.action_std <= min_action_std:
            self.action_std = min_action_std
            print("setting actor output action_std to min_action_std : ", self.action_std)
        else:
            print("setting actor output action_std to : ", self.action_std)
        self.set_action_std(self.action_std)

    def select_action(self, state):
        """
        Selecting action under old policy during sampling phase. At same time, save the state and reward into
        rollout buffer to be used during evaluation.
        """
        with torch.no_grad():
            state = torch.FloatTensor(state)
            action, action_logprob = self.policy_old.get_action(state)

        action = torch.clamp(action, min=config['action_low'], max=config['action_high'])
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)

        return action.detach().numpy().flatten()

    def update(self):
        """
        Note: Adam is Gradient descent algorithm. Therefore, it tries to find global minimum. For surrogate loss,
        gradient ascent is requirred and therefore a negative surrogate loss is used during optimization using
        Adam. Maximization of loss function is equal to minimizing a negative loss function.
        """
        # Using Monte Carlo estimates of return we can calcuate the advantage function.
        rewards = []
        discounted_reward = 0

        # reward for terminal state is zero. Starting from terminal state, add the discounted reward collected
        # till the initial state and store them into rewards list. So, no bootstrapping.

        for reward, done in zip(reversed(self.buffer.rewards), reversed(self.buffer.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # By normalizing the rewards, variance in advantage is reduced further.
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)  # to avoid division by zero.

        # The buffer has multiple list. The list as it is cannot be passed to the NN made using pytorch.
        # Pytorch works with tensors and therefore, a conversion is done.

        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach()

        # Using the sampled date from old policy, calculate loss over multiple trajectories and perform optimization.
        # Data/ state tuple corresponding to 800 time steps is processed 'K_epochs' times before updating old_policy
        # parameters to same as new updated policy parameters.

        for _ in range(self.K_epochs):
            logprobs, state_values, policy_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding surrogate loss and then maximizing it. Since, maximization of surrogate loss corresponds to
            # increase in prob. of action which gives higher reward given the state.
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = torch.min(surr1, surr2)

            # Combined loss of PPO
            value_loss = self.MseLoss(state_values, rewards)
           # print(policy_entropy, policy_loss, value_loss)

            loss = -policy_loss + 0.5 * value_loss #- 0.03 * policy_entropy

            # Optimization step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Finally, old_policy is same as new _policy until next update phase.
        self.policy_old.load_state_dict(self.policy.state_dict())
        # Prepare buffer for next round of sampling by clearing all previous entries.
        self.buffer.clear()

    def save(self, checkpoint_path):
        """
        Save the weights and biases of old policy to be used later for evaluating PPO performance via rendering the
        Environment.
        """
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        """
        Load previously trained model parameters for testing.
        """
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))


def unscaled_action(scaled_action, action_low= config['action_low'], action_high = config['action_high']):
    """
    A tanh() activation function is applied before getting output from actor network. Therefore, the mean is bounded
    to (-1, 1). An unscaling of action is needed to explore
    the action space of control problem.
    """
    return action_low + (0.5 *(scaled_action + 1.0) * (action_high - action_low))