from ml_sacre.config import Config
from ml_sacre.experiment import Experiment


CONFPATH = '/input/dir/conf.yaml'


if __name__ == '__main__':

    config = Config(CONFPATH)

    experiment = Experiment(config)
    experiment.run()
    experiment.aggregate()

    print('----')
    print('Done')
    print(f'Output dir: {config.output_dir}/')
