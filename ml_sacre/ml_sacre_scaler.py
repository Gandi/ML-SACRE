from ml_sacre.helpers import (
    load_dataframe,
    load_model,
    make_dir_if_not_exists)
from ml_sacre.models.agent import (
    train as train_agent,
    Policy,
    PPOAgent)
from ml_sacre.simulation import Simulation
import os


class ML_SACRE(Simulation):
    model_agent = None

    def __init__(self,
                 env,
                 input_dir,
                 output_dir,
                 load_existing_agent=False):
        super().__init__(env=env,
                         input_dir=input_dir,
                         output_dir=output_dir)

        self.load_existing_agent = load_existing_agent

    def init_paths(self):
        super().init_paths()

        make_dir_if_not_exists(
            os.path.join(self.output_dir, 'agent/models/'))
        make_dir_if_not_exists(
            os.path.join(self.output_dir, 'agent/plots/'))

        self.model_policy_path = os.path.join(
            self.output_dir, 'agent/models/model_policy' +
            f'{self.sim_name.replace(self.sim_type, "")}.pth')

        self.plot_agent_path = os.path.join(
            self.output_dir, 'agent/plots/agent_training_reward' +
            f'{self.sim_name.replace(self.sim_type, "")}.png')

    def train_agent(self):
        used_train_data = {
            res.name: self.df_train[res.col_used].to_list()
            for res in self.env.resources}
        predicted_train_data = {
            res.name: self.df_train[res.col_predicted].to_list()
            for res in self.env.resources}
        used_validation_data = {
            res.name: self.df_validation[res.col_used].to_list()
            for res in self.env.resources}
        predicted_validation_data = {
            res.name: self.df_validation[res.col_predicted].to_list()
            for res in self.env.resources}

        self.model_agent = train_agent(
            env=self.env,
            used_train_data=used_train_data,
            predicted_train_data=predicted_train_data,
            used_validation_data=used_validation_data,
            predicted_validation_data=predicted_validation_data,
            model_policy_path=self.model_policy_path,
            plot_path=self.plot_agent_path)

    def load_agent(self):
        model_policy = load_model(ModelClass=Policy,
                                  filepath=self.model_policy_path,
                                  model_params=self.env)
        self.model_agent = PPOAgent(self.env)
        self.model_agent.policy = model_policy

    def select_action(self, state_tensor):
        return self.model_agent.select_action(state_tensor)

    def simulate(self):
        self.df_train = load_dataframe(
            self.df_train_predicted_path)
        self.df_validation = load_dataframe(
            self.df_validation_predicted_path)

        if not self.load_existing_agent:
            self.train_agent()
        else:
            self.load_agent()

        super().simulate()
