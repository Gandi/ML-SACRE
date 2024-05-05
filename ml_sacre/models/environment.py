from ml_sacre.helpers import normalize
import gymnasium as gym
from gymnasium.spaces import Dict, Discrete, Tuple


class Environment(gym.Env):

    def __init__(self,
                 resources,
                 n_prediction_steps):
        super(Environment, self).__init__()

        self.resources = resources
        self.n_prediction_steps = n_prediction_steps

        self.requested = {res.name: res.requested_value
                          for res in self.resources}
        self.used = {res.name: res.requested_value
                     for res in self.resources}
        self.predicted = {res.name: [res.requested_value
                                     for _ in range(self.n_prediction_steps)]
                          for res in self.resources}
        self.allocated = {res.name: res.requested_value
                          for res in self.resources}

        self.reallocating = {res.name: 0 for res in self.resources}

        self.state_space = Dict({
            res.name: Dict({
                'Requested': Discrete(res.requested_value + 1),
                'Used': Discrete(res.requested_value + 1),
                'Allocated': Discrete(res.requested_value + 1),
                'Predicted': Tuple([Discrete(res.requested_value + 1)
                                    for _ in range(self.n_prediction_steps)])
            }) for res in self.resources
        })

        self.action_space = Dict({
            res.name: Dict({
                'Reallocating': Discrete(2),
                'Allocation': Discrete(res.requested_value + 1)
            }) for res in self.resources
        })

        self.min_reward = sum(res.reward_underallocation +
                              res.reward_reallocation
                              for res in self.resources)
        self.max_reward = sum(res.reward_optimal_allocation
                              for res in self.resources)

    def make_state(self):
        state = {
            res.name: {
                'Requested': self.requested[res.name],
                'Used': self.used[res.name],
                'Allocated': self.allocated[res.name],
                'Predicted': self.predicted[res.name]
            } for res in self.resources
        }

        return state

    def step(self,
             action,
             next_used,
             next_predicted):
        assert self.action_space.contains(action), 'Invalid action'

        self.used = next_used
        self.predicted = next_predicted

        for res in self.resources:
            self.reallocating[res.name] = action[res.name]['Reallocating']
            if self.reallocating[res.name]:
                self.allocated[res.name] = action[res.name]['Allocation']

        reward = self.calculate_reward()

        state = self.make_state()

        return state, reward, {}

    def reset(self):
        self.used = {res.name: res.requested_value
                     for res in self.resources}
        self.predicted = {res.name: [res.requested_value
                                     for _ in range(self.n_prediction_steps)]
                          for res in self.resources}
        self.allocated = {res.name: res.requested_value
                          for res in self.resources}

        self.reallocating = {res.name: 0 for res in self.resources}

        state = self.make_state()

        return state

    def calculate_reward(self):
        reward = {res.name: 0 for res in self.resources}

        for res in self.resources:
            used_weighted = self.used[res.name] * res.coef_margin

            if self.allocated[res.name] <= self.used[res.name]:
                reward[res.name] += res.reward_underallocation

            elif self.allocated[res.name] <= used_weighted:
                reward[res.name] +=\
                    (self.allocated[res.name] - self.used[res.name]) /\
                    (used_weighted - self.used[res.name]) *\
                    (res.reward_optimal_allocation -
                        res.reward_underallocation) +\
                    res.reward_underallocation

            else:  # self.allocated[res.name] <= self.requested[res.name]
                reward[res.name] +=\
                    (self.allocated[res.name] - used_weighted) /\
                    (self.requested[res.name] - used_weighted) *\
                    (res.reward_requested_allocation -
                        res.reward_optimal_allocation) +\
                    res.reward_optimal_allocation

            if self.reallocating[res.name]:
                reward[res.name] += res.reward_reallocation

        reward = sum(reward[res.name] for res in self.resources)
        return normalize(reward, self.min_reward, self.max_reward)

    def render(self, mode='human'):
        state = self.make_state()
        print(state)
