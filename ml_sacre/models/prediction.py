from ml_sacre.helpers import (
    load_dataframe,
    load_model,
    save_dataframe,
    make_dir_if_not_exists)
from ml_sacre.models.time_series import (
    train as train_ts,
    predict_df as predict_ts,
    plot as plot_ts,
    TimeSeriesModel)
import pandas as pd
import os


class Prediction():
    model_ts = None

    def __init__(self,
                 env,
                 input_dir,
                 output_dir,
                 load_existing_model=False):
        self.env = env
        self.input_dir = input_dir
        self.output_dir = output_dir

        self.init_paths()

        if not load_existing_model:
            self.df_train = load_dataframe(
                self.df_train_path)
            self.df_validation = load_dataframe(
                self.df_validation_path)
            self.df_test = load_dataframe(
                self.df_test_path)

            self.train_ts()
        else:
            self.df_train = load_dataframe(
                self.df_train_predicted_path)
            self.df_validation = load_dataframe(
                self.df_validation_predicted_path)
            self.df_test = load_dataframe(
                self.df_test_predicted_path)

            self.load_ts()

        self.predict_ts()

    def init_paths(self):
        self.df_train_path = os.path.join(
            self.input_dir, 'df_train.csv')
        self.df_validation_path = os.path.join(
            self.input_dir, 'df_validation.csv')
        self.df_test_path = os.path.join(
            self.input_dir, 'df_test.csv')

        make_dir_if_not_exists(
            os.path.join(self.output_dir, 'dataframes/'))
        make_dir_if_not_exists(
            os.path.join(self.output_dir, 'ts/models/'))
        make_dir_if_not_exists(
            os.path.join(self.output_dir, 'ts/plots/'))

        self.df_train_predicted_path = os.path.join(
            self.output_dir, 'dataframes/df_train_predicted.csv')
        self.df_validation_predicted_path = os.path.join(
            self.output_dir, 'dataframes/df_validation_predicted.csv')
        self.df_test_predicted_path = os.path.join(
            self.output_dir, 'dataframes/df_test_predicted.csv')

        self.model_ts_path = os.path.join(
            self.output_dir, 'ts/models/model_ts.pth')

        self.plot_ts_paths = [os.path.join(
            self.output_dir, f'ts/plots/ts_{res.name}.png')
            for res in self.env.resources]

    def train_ts(self):
        requested_resources = [res.requested_value
                               for res in self.env.resources]
        target_columns = [res.col_used
                          for res in self.env.resources]

        self.model_ts = train_ts(
            df_train=self.df_train,
            df_validation=self.df_validation,
            target_columns=target_columns,
            n_prediction_steps=self.env.n_prediction_steps,
            output_path=self.model_ts_path)

        plot_ts(
            df_train=pd.concat([self.df_train, self.df_validation],
                               ignore_index=True),
            df_test=self.df_test,
            model=self.model_ts,
            target_columns=target_columns,
            n_prediction_steps=self.env.n_prediction_steps,
            y_max=requested_resources,
            title='Time series prediction',
            x_label='Time step',
            output_paths=self.plot_ts_paths)

    def load_ts(self):
        target_columns = [res.col_used
                          for res in self.env.resources]
        self.model_ts = load_model(ModelClass=TimeSeriesModel,
                                   filepath=self.model_ts_path,
                                   model_params=target_columns)

    def predict_ts(self):
        requested_resources = [res.requested_value
                               for res in self.env.resources]
        target_columns = [res.col_used
                          for res in self.env.resources]
        predicted_columns = [res.col_predicted
                             for res in self.env.resources]

        self.df_train = predict_ts(
            model=self.model_ts,
            df=self.df_train,
            target_columns=target_columns,
            predicted_columns=predicted_columns,
            n_prediction_steps=self.env.n_prediction_steps,
            fill_values=requested_resources)
        self.df_validation = predict_ts(
            model=self.model_ts,
            df=self.df_validation,
            target_columns=target_columns,
            predicted_columns=predicted_columns,
            n_prediction_steps=self.env.n_prediction_steps,
            fill_values=requested_resources)
        self.df_test = predict_ts(
            model=self.model_ts,
            df=self.df_test,
            target_columns=target_columns,
            predicted_columns=predicted_columns,
            n_prediction_steps=self.env.n_prediction_steps,
            fill_values=requested_resources)

        save_dataframe(self.df_train, self.df_train_predicted_path)
        save_dataframe(self.df_validation, self.df_validation_predicted_path)
        save_dataframe(self.df_test, self.df_test_predicted_path)
