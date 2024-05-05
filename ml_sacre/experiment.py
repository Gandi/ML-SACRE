from ml_sacre.models.prediction import Prediction
from ml_sacre.ml_sacre_scaler import ML_SACRE
from ml_sacre.benchmarks.requested_scaler import\
    RequestedScaler
from ml_sacre.benchmarks.predicted_scaler import\
    PredictedScaler
from ml_sacre.helpers import (
    delete_dir_if_exists,
    make_dir_if_not_exists,
    load_dataframe,
    save_dataframe,
    average_dicts_with_confidence,
    load_yaml,
    save_yaml,
    print_result)
import os
import pandas as pd
import matplotlib.pyplot as plt
import time


class Experiment:
    sims = []

    def __init__(self, config):
        self.config = config

        self.init_paths()

        self.base_env = self.config.envs[0]

    def init_paths(self):
        output_dir = self.config.output_dir

        self.output_dirs = [os.path.join(
            output_dir, f'exp_{i + 1}')
            for i in range(self.config.n_trials)]
        self.df_allocated_paths = [os.path.join(
            output_dir, 'dataframes/df_allocated.csv')
            for output_dir in self.output_dirs]
        self.df_reward_paths = [os.path.join(
            output_dir, 'dataframes/df_reward.csv')
            for output_dir in self.output_dirs]
        self.result_paths = [os.path.join(
            output_dir, 'result.yaml')
            for output_dir in self.output_dirs]

        self.agg_dir = os.path.join(
            output_dir, 'exp_aggregated')
        self.df_agg_allocated_path = os.path.join(
            self.agg_dir, 'dataframes/df_allocated.csv')
        self.df_agg_reward_path = os.path.join(
            self.agg_dir, 'dataframes/df_reward.csv')

    def init_dirs(self):
        output_dir = self.config.output_dir

        delete_dir_if_exists(output_dir)
        make_dir_if_not_exists(output_dir)
        save_yaml(self.config.conf, os.path.join(output_dir, 'conf.yaml'))

        make_dir_if_not_exists(os.path.join(self.agg_dir, 'plots/'))

    def run(self, load_existing_agents=False):
        if not load_existing_agents:
            self.init_dirs()

        et = time.time()
        for i in range(self.config.n_trials):
            t = time.time()

            print('----')
            print(f'==== Trial {i + 1} / ' +
                  f'{self.config.n_trials} ====')
            print('----')

            if not load_existing_agents:
                pt = time.time()
                Prediction(self.base_env,
                           self.config.input_dir,
                           self.output_dirs[i])
                pt = time.time() - pt
                print(f'TS time: {pt}s')

            trial_sims = [RequestedScaler(self.base_env,
                                          self.config.input_dir,
                                          self.output_dirs[i]),
                          PredictedScaler(self.base_env,
                                          self.config.input_dir,
                                          self.output_dirs[i])]
            trial_sims.extend([ML_SACRE(env,
                                        self.config.input_dir,
                                        self.output_dirs[i],
                                        load_existing_agents)
                               for env in self.config.envs])
            self.sims.append(trial_sims)

            for sim in trial_sims:
                sim.simulate()

            t = time.time() - t
            print(f'Trial time: {t}s')
        et = time.time() - et
        print(f'Experiment time: {et}s')

    def aggregate(self):
        if not self.sims:
            pass

        print('----')
        print('==== Average experiment results (90% confidence interval) ====')
        print('----')
        self.aggregate_allocations()
        self.aggregate_rewards()
        self.aggregate_results()
        self.plot_agg_allocation()
        self.plot_agg_rewards()
        self.plot_agg_freed()
        self.plot_agg_reallocation()
        self.plot_agg_underallocation()
        self.plot_agg_underallocation_amount()

    def load_allocated_dfs(self):
        df_list = []
        for df_path in self.df_allocated_paths:
            df_list.append(load_dataframe(df_path))
        return df_list

    def load_reward_dfs(self):
        df_list = []
        for df_path in self.df_reward_paths:
            df_list.append(load_dataframe(df_path))
        return df_list

    def load_results(self):
        result_list = []
        for result_path in self.result_paths:
            result_list.append(load_yaml(result_path))
        return result_list

    def aggregate_allocations(self):
        df_list = self.load_allocated_dfs()
        # excluding Predicted and Timestamp columns for averaging
        df_timestamp = df_list[0][['Timestamp']].copy()
        df_list = [df[df.columns.drop(df.filter(like=' Predicted '))]
                   for df in df_list]
        df_list = [df.drop(columns=['Timestamp'])
                   for df in df_list]
        # re-including Timestamp column
        df = pd.concat([df_timestamp, sum(df_list) / len(df_list)], axis=1)
        save_dataframe(
            df, os.path.join(self.agg_dir, 'dataframes/df_allocated.csv'))

    def aggregate_rewards(self):
        df_list = self.load_reward_dfs()
        df = sum(df_list) / len(df_list)
        save_dataframe(
            df, os.path.join(self.agg_dir, 'dataframes/df_reward.csv'))

    def aggregate_results(self):
        result_list = self.load_results()
        result = average_dicts_with_confidence(*result_list)
        print_result(result)
        save_yaml(result, os.path.join(self.agg_dir, 'result_90%_conf.yaml'))

    def plot_agg_rewards(self):
        df = load_dataframe(self.df_agg_reward_path)

        fig = plt.figure(plt.figure(figsize=[5.5, 4], dpi=80))
        plt.xlabel('Time step')
        plt.ylabel('Cumulative reward' +
                   '\nper time step')
        plt.title('Reward benchmark')

        for i, sim in enumerate(self.sims[0]):
            col = df.filter(like='Reward').filter(
                like=sim.sim_name).columns[0]
            plt.plot((df[col]
                      .cumsum() / df.shape[0]).to_list(),
                     label=sim.sim_name,
                     alpha=0.6)

        plt.legend()
        plt.savefig(os.path.join(self.agg_dir, 'plots/reward.png'),
                    bbox_inches='tight')
        plt.close(fig)

    def plot_agg_freed(self):
        df = load_dataframe(self.df_agg_allocated_path)

        for res in self.base_env.resources:
            fig = plt.figure(plt.figure(figsize=[5.5, 4], dpi=80))
            plt.xlabel('Time step')
            plt.ylabel(f'Cumulative % of used {res.name} freed' +
                       '\nper time step')
            plt.title(f'{res.name} Efficiency benchmark')

            col_requested = df.filter(like=f'{res.name}')\
                .filter(like='Requested').columns[0]
            col_used = df.filter(like=f'{res.name}')\
                .filter(like='Used').columns[0]
            for i, sim in enumerate(self.sims[0]):
                col = df.filter(like=f'{res.name}')\
                    .filter(like='Allocated')\
                    .filter(like=sim.sim_name).columns[0]
                plt.plot(

                    (((df[col_requested] - df[col]) /
                      (df[col_requested] - df[col_used]) * 100)

                     .cumsum() / df.shape[0]).to_list(),
                    label=sim.sim_name,
                    alpha=0.6)

            plt.legend()
            plt.savefig(os.path.join(self.agg_dir,
                                     f'plots/freed_{res.name}.png'),
                        bbox_inches='tight')
            plt.close(fig)

    def plot_agg_reallocation(self):
        df = load_dataframe(self.df_agg_allocated_path)

        for res in self.base_env.resources:
            fig = plt.figure(plt.figure(figsize=[5.5, 4], dpi=80))
            plt.xlabel('Time step')
            plt.ylabel(f'Cumulative {res.name} reallocation probability %' +
                       '\nper time step')
            plt.title(f'{res.name} Stability benchmark')

            for i, sim in enumerate(self.sims[0]):
                col = df.filter(like=f'{res.name}')\
                    .filter(like='Reallocating')\
                    .filter(like=sim.sim_name).columns[0]
                plt.plot(

                    ((df[col] * 100)

                     .cumsum() / df.shape[0]).to_list(),
                    label=sim.sim_name,
                    alpha=0.6)

            plt.legend()
            plt.savefig(os.path.join(self.agg_dir,
                                     f'plots/reallocation_{res.name}.png'),
                        bbox_inches='tight')
            plt.close(fig)

    def plot_agg_underallocation(self):
        df = load_dataframe(self.df_agg_allocated_path)

        for res in self.base_env.resources:
            fig = plt.figure(plt.figure(figsize=[5.5, 4], dpi=80))
            plt.xlabel('Time step')
            plt.ylabel(f'Cumulative {res.name} underallocation probability %' +
                       '\nper time step')
            plt.title(f'{res.name} QoS benchmark')

            col_used = df.filter(like=f'{res.name}')\
                .filter(like='Used').columns[0]
            for i, sim in enumerate(self.sims[0]):
                col = df.filter(like=f'{res.name}')\
                    .filter(like='Allocated')\
                    .filter(like=sim.sim_name).columns[0]
                plt.plot(

                    (((df[col_used] > df[col]) * 100)

                     .cumsum() / df.shape[0]).to_list(),
                    label=sim.sim_name,
                    alpha=0.6)

            plt.legend()
            plt.savefig(os.path.join(self.agg_dir,
                                     f'plots/underallocation_{res.name}.png'),
                        bbox_inches='tight')
            plt.close(fig)

    def plot_agg_underallocation_amount(self):
        df = load_dataframe(self.df_agg_allocated_path)

        for res in self.base_env.resources:
            fig = plt.figure(plt.figure(figsize=[5.5, 4], dpi=80))
            plt.xlabel('Time step')
            plt.ylabel(f'Cumulative {res.name} % of used underallocated' +
                       '\nper time step')
            plt.title(f'{res.name} QoS benchmark')

            col_used = df.filter(like=f'{res.name}')\
                .filter(like='Used').columns[0]
            for i, sim in enumerate(self.sims[0]):
                col = df.filter(like=f'{res.name}')\
                    .filter(like='Allocated')\
                    .filter(like=sim.sim_name).columns[0]
                plt.plot(

                    (((df[col] - df[col_used]).clip(upper=0) /
                      df[col_used] * 100)

                     .cumsum() / df.shape[0]).to_list(),
                    label=sim.sim_name,
                    alpha=0.6)

            plt.legend()
            plt.savefig(os.path.join(
                self.agg_dir,
                f'plots/underallocation_amount_{res.name}.png'),
                bbox_inches='tight')
            plt.close(fig)

    def plot_agg_allocation(self):
        df = load_dataframe(self.df_agg_allocated_path)

        for res in self.base_env.resources:
            fig = plt.figure(plt.figure(figsize=[5.5, 4], dpi=80))
            plt.xlabel('Time step')
            plt.ylabel(f'{res.name} allocation ({res.unit})' +
                       '\nin the last 100 time steps')
            plt.title(f'{res.name} scaling')

            col_used = df.filter(like=f'{res.name}')\
                .filter(like='Used').columns[0]
            plt.plot(df[-100:][col_used],
                     label=f'{res.name} Used',
                     color='lightgray',
                     linestyle='solid',
                     linewidth='3')
            for i, sim in enumerate(self.sims[0]):
                col = df.filter(like=f'{res.name}')\
                    .filter(like='Allocated')\
                    .filter(like=sim.sim_name).columns[0]
                plt.plot(df[-100:][col],
                         label=sim.sim_name,
                         alpha=0.6)

            plt.legend(loc='upper left')
            plt.savefig(os.path.join(self.agg_dir,
                                     f'plots/allocation_{res.name}.png'),
                        bbox_inches='tight')
            plt.close(fig)
