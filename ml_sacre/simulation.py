from ml_sacre.helpers import (
    load_dataframe,
    save_dataframe,
    df_to_state_tensor,
    load_yaml,
    save_yaml,
    print_result,
    get_class_name)
import numpy as np
from abc import ABC, abstractmethod
import os
import pandas as pd


class Simulation(ABC):
    df_test = None
    df_reward = None

    def __init__(self, env, input_dir, output_dir):
        self.env = env
        self.input_dir = input_dir
        self.output_dir = output_dir

        self.sim_type = get_class_name(self)
        self.sim_name = self.sim_type
        for res in self.env.resources:
            self.sim_name += f'_{res.coef_margin}'

        self.col_sim = {res.name:
                        f'{res.name} Allocated {self.sim_name} ({res.unit})'
                        for res in self.env.resources}
        self.col_sim_realloc = {res.name:
                                f'{res.name} Reallocating {self.sim_name}'
                                for res in self.env.resources}
        self.col_reward = f'Reward {self.sim_name}'

        self.init_paths()

    def init_paths(self):
        self.df_train_predicted_path = os.path.join(
            self.output_dir, 'dataframes/df_train_predicted.csv')
        self.df_validation_predicted_path = os.path.join(
            self.output_dir, 'dataframes/df_validation_predicted.csv')
        self.df_test_predicted_path = os.path.join(
            self.output_dir, 'dataframes/df_test_predicted.csv')

        self.df_allocated_path = os.path.join(
            self.output_dir, 'dataframes/df_allocated.csv')

        self.df_reward_path = os.path.join(
            self.output_dir, 'dataframes/df_reward.csv')

        self.result_path = os.path.join(
            self.output_dir, 'result.yaml')

    @abstractmethod
    def select_action(self, state_tensor):
        pass

    def simulate(self):
        print(f'>>>> Simulating {self.sim_name} <<<<')

        if not os.path.exists(self.df_allocated_path):
            self.df_test = load_dataframe(self.df_test_predicted_path)
        else:
            self.df_test = load_dataframe(self.df_allocated_path)

        df = self.df_test
        n_steps = df.shape[0]

        if not os.path.exists(self.df_reward_path):
            self.df_reward = pd.DataFrame(
                {self.col_reward: [np.nan] * n_steps})
        else:
            self.df_reward = load_dataframe(self.df_reward_path)

        for res in self.env.resources:
            df[self.col_sim[res.name]] = np.nan
            df.loc[0, self.col_sim[res.name]] = res.requested_value

        total_reward = 0
        self.df_reward[self.col_reward] = np.nan
        self.env.reset()

        for i in range(1, n_steps):
            state = df_to_state_tensor(df,
                                       i - 1,
                                       self.env.resources,
                                       self.col_sim)

            action = self.select_action(state)[0]

            for res in self.env.resources:
                df.loc[i, self.col_sim[res.name]] =\
                    action[res.name]['Allocation']\
                    if action[res.name]['Reallocating']\
                    else df.loc[i - 1][self.col_sim[res.name]]

            next_used = {res.name: df.loc[i][res.col_used]
                         for res in self.env.resources}
            next_predicted = {res.name: df.loc[i][res.col_predicted]
                              for res in self.env.resources}

            _, reward, _ = self.env.step(
                action=action,
                next_used=next_used,
                next_predicted=next_predicted)

            total_reward += reward
            self.df_reward.loc[i, self.col_reward] = reward

        for res in self.env.resources:
            df[self.col_sim[res.name]] =\
                df[self.col_sim[res.name]].astype(int)
            df[self.col_sim_realloc[res.name]] =\
                (df[self.col_sim[res.name]] != df[self.col_sim[res.name]]
                 .shift(1).fillna(df[self.col_sim[res.name]].iloc[0]))\
                .astype(int)

        result = self.calculate_result()

        print('====== Test ======')
        print(f'Time steps: {n_steps}')
        print('Average reward per time step: ' +
              f'{total_reward / n_steps}')
        print('Result:')
        print_result(result)

        try:
            results = load_yaml(self.result_path)
        except FileNotFoundError:
            results = {}
        results.update(result)
        save_yaml(results, self.result_path)

        save_dataframe(df, self.df_allocated_path)

        save_dataframe(self.df_reward, self.df_reward_path)

        return df, self.df_reward, result

    def calculate_result(self):
        df = self.df_test
        n_steps = df.shape[0]

        result = {}
        for res in self.env.resources:
            result[res.name] = {'Efficiency': {},
                                'Stability': {},
                                'QoS': {}}

            # average % of unused freed
            result[res.name]['Efficiency']['Average % of unused freed'] = int(
                ((df[res.col_requested] - df[self.col_sim[res.name]]) /
                 (df[res.col_requested] - df[res.col_used]) * 100)
                .mean().round().astype(int))

            # % of time steps reallocating
            result[res.name]['Stability']['Reallocation probability %'] = int(
                df[self.col_sim_realloc[res.name]].sum() / n_steps * 100)

            # % of time steps underallocated
            result[res.name]['QoS']['Underallocation probability %'] = int(
                (df[res.col_used] > df[self.col_sim[res.name]]).sum() /
                n_steps * 100)
            # average % of used underallocated
            result[res.name]['QoS']['Average % of used underallocated'] =\
                int(((df[self.col_sim[res.name]] - df[res.col_used])
                     .clip(upper=0) / df[res.col_used] * 100)
                    .mean().round().astype(int))

        result = {self.sim_name: result}

        return result
