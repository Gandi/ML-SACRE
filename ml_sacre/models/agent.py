from ml_sacre.helpers import (
    state_to_tensor,
    action_to_tensor,
    tensor_to_state)
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Bernoulli
import numpy as np
from math import inf


# ========== Param init ==========
# Policy / Value NN
HIDDEN_SIZE = 512
DROPOUT_PROB = 0.2

# PPO
LR = 0.000001
GAMMA = 0.99
CLIP_RATIO = 0.2
VF_COEF = 0.5
ENTROPY_COEF = 0.1

# PPOAgent training
N_EPISODES = 1000
N_MIN_TIME_STEPS = 50
N_MAX_TIME_STEPS = 50
N_OPTIMIZATION_EPOCHS = 1
COEF_DECAYING_EXPLORATION_NOISE = 0.5
VALIDATION_PATIENCE = 20


class Policy(nn.Module):

    def __init__(self, env):
        super(Policy, self).__init__()

        self.n_resources = len(env.resources)
        state_size = self.n_resources * (3 + env.n_prediction_steps)

        self.fcin = nn.Linear(state_size, HIDDEN_SIZE)
        self.fc1 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)

        self.fcout_reallocating = nn.ModuleList(
            [nn.Linear(HIDDEN_SIZE, 1) for _ in range(self.n_resources)])

        self.dropout1 = nn.Dropout(p=DROPOUT_PROB)
        self.dropout2 = nn.Dropout(p=DROPOUT_PROB)
        self.dropout3 = nn.Dropout(p=DROPOUT_PROB)

    def forward(self, x):
        x = F.relu(self.fcin(x))
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout3(x)

        logits_reallocating = [self.fcout_reallocating[i](x)
                               for i in range(self.n_resources)]
        probs_reallocating = [F.sigmoid(logits_reallocating[i])
                              for i in range(self.n_resources)]

        return probs_reallocating


class Value(nn.Module):

    def __init__(self, env):
        super(Value, self).__init__()

        state_size = len(env.resources) * (3 + env.n_prediction_steps)

        self.fcin = nn.Linear(state_size, HIDDEN_SIZE)
        self.fc1 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fc3 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.fcout = nn.Linear(HIDDEN_SIZE, 1)

        self.dropout1 = nn.Dropout(p=DROPOUT_PROB)
        self.dropout2 = nn.Dropout(p=DROPOUT_PROB)
        self.dropout3 = nn.Dropout(p=DROPOUT_PROB)
        self.dropout4 = nn.Dropout(p=DROPOUT_PROB)

    def forward(self, x):
        x = F.relu(self.fcin(x))
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.dropout3(x)
        x = F.relu(self.fc3(x))
        x = self.dropout4(x)
        state_value = self.fcout(x)
        return state_value


class PPOAgent:

    def __init__(self,
                 env,
                 lr=LR,
                 gamma=GAMMA,
                 clip_ratio=CLIP_RATIO,
                 vf_coef=VF_COEF,
                 entropy_coef=ENTROPY_COEF,
                 optimization_epochs=N_OPTIMIZATION_EPOCHS):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.vf_coef = vf_coef
        self.entropy_coef = entropy_coef
        self.optimization_epochs = optimization_epochs
        self.policy = Policy(self.env)
        self.policy_optimizer = optim.Adam(self.policy.parameters(),
                                           lr=self.lr)
        self.value = Value(self.env)
        self.value_optimizer = optim.Adam(self.value.parameters(),
                                          lr=self.lr)

    def select_action(self,
                      state_tensor,
                      exploration_noise=0):
        probs_reallocating = self.policy(state_tensor)

        state = tensor_to_state(state_tensor,
                                self.env.resources,
                                self.env.n_prediction_steps)

        dist_reallocating = {}
        reallocating = {}
        allocation = {}
        for i, res in enumerate(self.env.resources):
            if exploration_noise > 0:
                reallocating_noise = np.random.normal(0, exploration_noise)
                allocation_noise = np.random.normal(0, exploration_noise)

            dist_reallocating[res.name] = Bernoulli(probs_reallocating[i])

            reallocating[res.name] = dist_reallocating[res.name].sample()
            if exploration_noise > 0:
                reallocating[res.name] = torch.clamp(
                    reallocating[res.name] * (1 + reallocating_noise),
                    min=0,
                    max=1).round()

            if reallocating[res.name]:
                allocation[res.name] = state[res.name]['Predicted'].max() *\
                    res.coef_margin
                if exploration_noise > 0:
                    allocation[res.name] *= (1 + allocation_noise)
                allocation[res.name] = torch.clamp(
                    allocation[res.name],
                    min=0,
                    max=state[res.name]['Requested']).round()
            else:
                allocation[res.name] = reallocating[res.name]

        action = {
            res.name:
            {
                'Reallocating': reallocating[res.name].int().item(),
                'Allocation': allocation[res.name].int().item()
            } for res in self.env.resources
        }

        log_prob = sum(
            dist_reallocating[res.name].log_prob(reallocating[res.name])
            for res in self.env.resources)

        return action, log_prob

    def compute_returns(self, rewards):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        return returns

    def compute_baseline(self, states):
        values = self.value(states)
        return values.squeeze()

    def update_policy(self,
                      states,
                      actions,
                      log_probs,
                      returns):
        states = torch.stack(states).detach()
        actions = torch.stack(actions).detach()
        log_probs = torch.stack(log_probs).detach()

        reallocating = {res.name: actions[:, i * 2]
                        for i, res in enumerate(self.env.resources)}

        for _ in range(self.optimization_epochs):

            new_probs_reallocating = self.policy(states)
            new_dist_reallocating = {}
            new_log_probs_reallocating = {}

            for i, res in enumerate(self.env.resources):
                new_dist_reallocating[res.name] =\
                    Bernoulli(new_probs_reallocating[i])

                new_log_probs_reallocating[res.name] =\
                    new_dist_reallocating[res.name]\
                    .log_prob(reallocating[res.name])

            new_log_probs = sum(new_log_probs_reallocating[res.name]
                                for res in self.env.resources)

            baseline = self.compute_baseline(states)
            advantages = returns - baseline

            ratio = torch.exp(new_log_probs - log_probs)
            obj = ratio * advantages.view(-1, 1)
            obj_clipped = ratio.clamp(1 - self.clip_ratio,
                                      1 + self.clip_ratio)\
                * advantages.view(-1, 1)
            policy_loss = -torch.min(obj, obj_clipped).mean()

            value_loss = F.smooth_l1_loss(baseline, returns)

            entropy =\
                sum(new_dist_reallocating[res.name].entropy().mean()
                    for res in self.env.resources) / len(self.env.resources)

            loss = policy_loss +\
                self.vf_coef * value_loss -\
                self.entropy_coef * entropy

            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            loss.backward()
            self.policy_optimizer.step()
            self.value_optimizer.step()

            return loss, policy_loss, value_loss, entropy


def train(env,
          used_train_data,
          predicted_train_data,
          used_validation_data,
          predicted_validation_data,
          model_policy_path=None,
          plot_path=None):
    print('==== PPO agent training ====')

    agent = PPOAgent(env)

    episode_rewards = []
    losses = []
    policy_losses = []
    value_losses = []
    entropies = []
    validation_episode_rewards = []
    validation_best_episode_rewards = []

    best_validation_reward = -inf
    no_improvement_counter = 0

    for episode in range(N_EPISODES):
        states = []
        actions = []
        rewards = []
        log_probs = []

        state = env.reset()

        exploration_noise = COEF_DECAYING_EXPLORATION_NOISE *\
            max((1 - episode / N_EPISODES), 0)

        n_steps = random.randint(N_MIN_TIME_STEPS, N_MAX_TIME_STEPS)
        i_start = random.randint(0, N_MAX_TIME_STEPS - n_steps)

        for i in range(i_start, i_start + n_steps - 1):
            state_vec = state_to_tensor(state, env.resources)
            states.append(state_vec)

            action, log_prob = agent.select_action(
                state_tensor=state_vec,
                exploration_noise=exploration_noise)
            action_vec = action_to_tensor(action, env.resources)
            actions.append(action_vec)
            log_probs.append(log_prob)

            next_used = {res.name: used_train_data[res.name][i]
                         for res in env.resources}
            next_predicted = {res.name: predicted_train_data[res.name][i]
                              for res in env.resources}
            next_state, reward, _ =\
                env.step(action=action,
                         next_used=next_used,
                         next_predicted=next_predicted)
            rewards.append(reward)

            state = next_state

        returns = agent.compute_returns(rewards)

        loss,\
            policy_loss,\
            value_loss,\
            entropy = agent.update_policy(states, actions, log_probs, returns)

        average_reward = sum(rewards) / n_steps
        episode_rewards.append(average_reward)
        losses.append(loss.item())
        policy_losses.append(policy_loss.item())
        value_losses.append(value_loss.item())
        entropies.append(entropy.item())

        # validation
        if episode % 10 == 0 or episode == N_EPISODES - 1:
            validation_rewards = []

            state = env.reset()

            n_validation_steps =\
                len(used_validation_data[list(used_validation_data.keys())[0]])
            for i in range(n_validation_steps):
                state_vec = state_to_tensor(state, env.resources)

                action, _ = agent.select_action(
                    state_tensor=state_vec,
                    exploration_noise=exploration_noise)
                action_vec = action_to_tensor(action, env.resources)

                next_used = {
                    res.name: used_validation_data[res.name][i]
                    for res in env.resources}
                next_predicted = {
                    res.name: predicted_validation_data[res.name][i]
                    for res in env.resources}
                next_state, reward, _ =\
                    env.step(action=action,
                             next_used=next_used,
                             next_predicted=next_predicted)
                validation_rewards.append(reward)

                state = next_state

            average_validation_reward =\
                sum(validation_rewards) / n_validation_steps

            validation_episode_rewards.append(average_validation_reward)

            print(f'Episode {episode}, ' +
                  f'Average training reward: {average_reward} ' +
                  f'Average validation reward: {average_validation_reward}')

            if average_validation_reward > best_validation_reward:
                best_validation_reward = average_validation_reward
                no_improvement_counter = 0

                # saving best model so far
                if model_policy_path:
                    torch.save(agent.policy.state_dict(), model_policy_path)
            else:
                no_improvement_counter += 1

            if no_improvement_counter >= VALIDATION_PATIENCE:
                print('Early stop')
                break

        else:
            validation_episode_rewards.append(validation_episode_rewards[-1]
                                              if validation_episode_rewards
                                              else np.nan)

        validation_best_episode_rewards.append(best_validation_reward)

    env.close()

    if plot_path:
        fig = plt.figure(plt.figure(figsize=[5.5, 4], dpi=80))
        plt.xlabel('Training episode')
        plt.ylabel('Average reward')
        plt.title('PPO agent training')
        plt.plot(episode_rewards,
                 linestyle='solid',
                 label='Training')
        plt.plot(validation_episode_rewards,
                 linestyle='solid',
                 label='Validation')
        plt.plot(validation_best_episode_rewards,
                 linestyle='solid',
                 label='Best validation')
        plt.legend()
        plt.savefig(plot_path,
                    bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure(plt.figure(figsize=[5.5, 4], dpi=80))
        plt.xlabel('Training episode')
        plt.ylabel('Total loss')
        plt.title('PPO agent training')
        plt.plot(losses,
                 label='Total loss')
        plt.savefig(plot_path.replace('reward', 'total_loss'),
                    bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure(plt.figure(figsize=[5.5, 4], dpi=80))
        plt.xlabel('Training episode')
        plt.ylabel('Policy loss')
        plt.title('PPO agent training')
        plt.plot(policy_losses,
                 label='Policy loss')
        plt.savefig(plot_path.replace('reward', 'policy_loss'),
                    bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure(plt.figure(figsize=[5.5, 4], dpi=80))
        plt.xlabel('Training episode')
        plt.ylabel('Value loss')
        plt.title('PPO agent training')
        plt.plot(value_losses,
                 label='Value loss')
        plt.savefig(plot_path.replace('reward', 'value_loss'),
                    bbox_inches='tight')
        plt.close(fig)

        fig = plt.figure(plt.figure(figsize=[5.5, 4], dpi=80))
        plt.xlabel('Training episode')
        plt.ylabel('Entropy')
        plt.title('PPO agent training')
        plt.plot(entropies,
                 label='Entropy')
        plt.savefig(plot_path.replace('reward', 'entropy'),
                    bbox_inches='tight')
        plt.close(fig)

    return agent
