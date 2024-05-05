import pandas as pd
import torch
import ast
import numpy as np
import os
import yaml
from collections import OrderedDict
import shutil
from scipy.stats import t


def separate_paths(filepath):
    dirpath = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    return dirpath, filename


def make_dir_if_not_exists(filepath):
    dirpath = os.path.dirname(filepath)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def delete_file_if_exists(filepath):
    if os.path.exists(filepath):
        os.remove(filepath)


def delete_dir_if_exists(dirpath):
    if os.path.exists(dirpath):
        shutil.rmtree(dirpath)


def load_yaml(filepath):
    with open(filepath, 'r') as f:
        content_dict = yaml.load(f, yaml.SafeLoader)
        return content_dict


def save_yaml(content_dict, filepath):
    make_dir_if_not_exists(filepath)
    with open(filepath, 'w') as f:
        yaml.dump(content_dict, f,
                  default_flow_style=False,
                  Dumper=yaml.SafeDumper)


def save_dataframe(df, filepath, sep='\t'):
    make_dir_if_not_exists(filepath)
    df.to_csv(filepath, sep=sep, index=False)


def load_dataframe(filepath, sep='\t', header=0):
    df = pd.read_csv(filepath,
                     sep=sep,
                     header=header)
    # loading lists properly
    for col in df.columns:
        try:
            df[col] = df[col].apply(lambda x: ast.literal_eval(x))
        except ValueError:
            pass
        except SyntaxError:
            pass
    return df


def split_train_test_df(df, test_coef, validation_coef):
    n = df.shape[0]
    train_size = int(n * (1 - test_coef))
    df_train = df.head(train_size)
    df_test = df.tail(n - train_size)

    n = df_train.shape[0]
    train_size = int(n * (1 - validation_coef))
    df_train = df_train.head(train_size)
    df_validation = df_train.tail(n - train_size)

    return df_train, df_validation, df_test


def save_model(model, filepath):
    if not filepath.endswith('.pth'):
        filepath += '.pth'
    make_dir_if_not_exists(filepath)
    torch.save(model.state_dict(), filepath)


def load_model(ModelClass, filepath, model_params=None):
    model = ModelClass(model_params)
    state_dict = torch.load(filepath)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def state_to_tensor(state, resources):
    state_vec = []
    for res in resources:
        state_vec.extend([state[res.name]['Requested'],
                          state[res.name]['Used'],
                          state[res.name]['Allocated'],
                          *state[res.name]['Predicted']])
    state_vec = np.array(state_vec)
    return torch.FloatTensor(state_vec)


def tensor_to_state(state_tensor, resources, n_prediction_steps):
    state = {}
    i = 0
    for res in resources:
        state[res.name] = {
            'Requested': state_tensor[i],
            'Used': state_tensor[i + 1],
            'Allocated': state_tensor[i + 2],
            'Predicted': state_tensor[i + 3:i + 3 + n_prediction_steps]}
        i += 3 + n_prediction_steps
    return state


def action_to_tensor(action, resources):
    action_vec = []
    for res in resources:
        action_vec.extend([action[res.name]['Reallocating'],
                           action[res.name]['Allocation']])
    action_vec = np.array(action_vec)
    return torch.FloatTensor(action_vec)


def df_to_state(df, i_row, resources, col_allocated):
    df_row = df.loc[i_row]
    state = {
        res.name: {
            'Requested': df_row[res.col_requested],
            'Used': df_row[res.col_used],
            'Allocated': df_row[col_allocated[res.name]],
            'Predicted': df_row[res.col_predicted]
        } for res in resources
    }
    return state


def df_to_state_tensor(df, i_row, resources, col_allocated):
    state = df_to_state(df, i_row, resources, col_allocated)
    return state_to_tensor(state, resources)


def average_dicts(*dict_list):
    if not dict_list:
        return {}

    avg_dict = OrderedDict.fromkeys(dict_list[0].keys())

    for key in avg_dict.keys():
        if all(isinstance(d[key], dict) for d in dict_list):
            avg_dict[key] = average_dicts(*[d[key] for d in dict_list])
        else:
            avg_dict[key] = round(
                sum(d[key] for d in dict_list) / len(dict_list))

    return dict(avg_dict)


def average_dicts_with_confidence(*dict_list):
    if not dict_list:
        return {}

    avg_dict = OrderedDict.fromkeys(dict_list[0].keys())

    for key in avg_dict.keys():
        if all(isinstance(d[key], dict) for d in dict_list):
            avg_dict[key] = average_dicts_with_confidence(
                *[d[key] for d in dict_list])
        else:
            values = [d[key] for d in dict_list]
            mean = np.mean(values)
            n = len(values)
            std_err = np.std(values, ddof=1) / np.sqrt(n)
            t_score = t.ppf(0.95, df=n - 1)  # 90% confidence interval
            margin_of_error = t_score * std_err
            avg_dict[key] = f'{round(mean)} +- {round(margin_of_error)} '
    return dict(avg_dict)


def print_result(result):
    for sim_name in result:
        print(f'>> {sim_name} <<')
        for res_name in result[sim_name]:
            print(f'{res_name}')
            for criterion in result[sim_name][res_name]:
                print(f'\t{criterion}')
                for metric in result[sim_name][res_name][criterion]:
                    print(f'\t\t{metric}: ' +
                          f'{result[sim_name][res_name][criterion][metric]}')


def normalize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)


def get_class_name(obj):
    return obj.__class__.__name__
