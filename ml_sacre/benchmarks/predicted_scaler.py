from ml_sacre.helpers import tensor_to_state
from ml_sacre.simulation import Simulation


class PredictedScaler(Simulation):

    def __init__(self, env, input_dir, output_dir):
        super().__init__(env=env,
                         input_dir=input_dir,
                         output_dir=output_dir)

    def select_action(self, state_tensor):
        state = tensor_to_state(state_tensor,
                                self.env.resources,
                                self.env.n_prediction_steps)
        action = {
            res.name:
                {
                    'Reallocating': 1,
                    'Allocation': max(min(int(state[res.name]['Predicted'][0]),
                                          res.requested_value),
                                      0)
                } for res in self.env.resources
        }
        return [action]
