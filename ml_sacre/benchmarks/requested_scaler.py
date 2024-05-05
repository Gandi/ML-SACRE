from ml_sacre.simulation import Simulation


class RequestedScaler(Simulation):

    def __init__(self, env, input_dir, output_dir):
        super().__init__(env=env,
                         input_dir=input_dir,
                         output_dir=output_dir)

    def select_action(self, state_tensor):
        action = {
            res.name:
                {
                    'Reallocating': 0,
                    'Allocation': res.requested_value
                } for res in self.env.resources
        }
        return [action]
