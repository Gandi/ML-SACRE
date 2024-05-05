from ml_sacre.models.resource import Resource
from ml_sacre.models.environment import Environment
from ml_sacre.helpers import load_yaml
import os


class Config:

    def __init__(self, filepath):
        self.conf = load_yaml(filepath)

        self.experiment_name = self.conf['experiment_name']

        self.input_dir = self.conf['input_dirpath']
        self.output_dir = os.path.join(
            self.conf['output_dirpath'], self.experiment_name)

        self.n_trials = self.conf['n_trials']

        self.envs = []
        n_prediction_steps = self.conf['n_prediction_steps']
        env_conf = self.conf['environments']
        for coef_margin in env_conf['coef_margins']:
            resources = []
            for res_name in coef_margin:
                res_conf = env_conf['resources'][res_name]
                res = Resource(name=res_name,
                               unit=res_conf['unit'],
                               requested_value=res_conf['requested_value'],
                               coef_margin=coef_margin[res_name],
                               reward_reallocation=res_conf
                               ['reward_reallocation'],
                               reward_underallocation=res_conf
                               ['reward_underallocation'],
                               reward_requested_allocation=res_conf
                               ['reward_requested_allocation'],
                               reward_optimal_allocation=res_conf
                               ['reward_optimal_allocation'])
                resources.append(res)

            env = Environment(resources=resources,
                              n_prediction_steps=n_prediction_steps)
            self.envs.append(env)
