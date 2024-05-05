# ML-SACRE

This project is the code support for the paper *ML-SACRE: Machine Learning-based Stable Auto-scaling of Cloud Resources with Efficiency*. It contains:
- The code for the presented method, i.e. the ML-SACRE algorithm and its underlying machine learning models.
- The code to run the presented simulation experiments from the paper and generate their results.

## Setup
### Requirements

- [Python](https://www.python.org/downloads/) >= 3.9
- [Poetry](https://python-poetry.org/). It can be installed using:
```
pip install poetry
```

### Installation

Before installing the project, make sure the [requirements](#requirements) are satisfied.
Then, from the project root directory, run the following command:
```
poetry install
```
Optional (but recommended):
```
poetry shell
```
This will activate a virtual environment to run project scripts in.

## Running
### Input

An example cofiguration can be found in [example_configuration.py](example_configuration.py). It is the configuration used for the simulations in the paper. The `input_dirpath` and `output_dirpath` fields need to be filled with real directory paths. The other parameters in the configuration can also be customized.

Training parameters for the time series and PPO agent models can be found (and modified) in the model files themselves: [models/time_series.py](ml_sacre/models/time_series.py) and [models/agent.py](ml_sacre/models/agent.py) respectively. The current parameters are those used in the paper.

The input directory specified in the configuration should contain 3 *.csv* files: `df_train.csv`, `df_validation.csv` and `df_test.csv`. The dataframe column names should contain a `Timestamp` column, as well as *Requested* and *Used* resource columns matching each resource item in the configuration file, along the following pattern: `[Resource name] Requested ([Resource unit])` and `[Resource name] Used ([Resource unit])`. The separator used is tab (`\t`). For example:
```
Timestamp	CPU Requested (%)	RAM Requested (GB)	CPU Used (%)	RAM Used (GB)
2023-12-04 00:50:00	100	17	8	5
2023-12-04 01:00:00	100	17	2	5
2023-12-04 01:10:00	100	17	6	5
2023-12-04 01:20:00	100	17	2	5
...
```

### Running an experiment (simulation batch)

To run a set of simulations, prepare the [inputs](#input) as described above, and update the configuration file path in the main file [main.py](ml_sacre/main.py). Then run the main script:
```
python main.py
```

### Outputs

The output directory as set in the configuration file will contain all results and corresponding plots for each experiment trial (set of simulations), as well as aggregated results and plots.