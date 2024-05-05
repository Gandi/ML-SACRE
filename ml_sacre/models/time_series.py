from ml_sacre.helpers import save_model
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt
from math import inf


# ========== Param init ==========
# NN
HIDDEN_SIZE = 256
DROPOUT_PROB = 0.2

# Training
LOOKBACK = 10
N_EPOCHS = 100
BATCH_SIZE = 8
VALIDATION_PATIENCE = 2


def create_dataset(df, target_columns, n_prediction_steps):
    dataset = df[target_columns].astype('float32').to_numpy()

    X, y = [], []
    for i in range(LOOKBACK, len(dataset) - n_prediction_steps + 1):
        features = dataset[i - LOOKBACK:i]
        targets = dataset[i + n_prediction_steps - LOOKBACK:
                          i + n_prediction_steps]
        X.append(torch.tensor(features))
        y.append(torch.tensor(targets))

    return torch.stack(X), torch.stack(y)


def create_dataset_Xonly(df, target_columns):
    dataset = df[target_columns].astype('float32').to_numpy()

    X = []
    for i in range(LOOKBACK, len(dataset)):
        features = dataset[i - LOOKBACK:i]
        X.append(torch.tensor(features))

    return torch.stack(X)


class TimeSeriesModel(nn.Module):

    def __init__(self, target_columns):
        super().__init__()
        self.size = len(target_columns)
        self.lstm1 = nn.ModuleList(
            [nn.LSTM(input_size=1,
                     hidden_size=HIDDEN_SIZE,
                     num_layers=1,
                     batch_first=True) for _ in range(self.size)])
        self.lstm2 = nn.ModuleList(
            [nn.LSTM(input_size=HIDDEN_SIZE,
                     hidden_size=HIDDEN_SIZE,
                     num_layers=1,
                     batch_first=True) for _ in range(self.size)])
        self.dropout1 = nn.ModuleList([nn.Dropout(DROPOUT_PROB)
                                       for _ in range(self.size)])
        self.dropout2 = nn.ModuleList([nn.Dropout(DROPOUT_PROB)
                                       for _ in range(self.size)])
        self.linear = nn.ModuleList(
            [nn.Linear(HIDDEN_SIZE, 1) for _ in range(self.size)])

    def forward(self, x):
        s = x.shape
        x_list = []
        for i in range(self.size):
            x_i = x[:, :, i].reshape((s[0], s[1], 1))
            x_i, _ = self.lstm1[i](x_i)
            x_i = self.dropout1[i](x_i)
            x_i, _ = self.lstm2[i](x_i)
            x_i = self.dropout2[i](x_i)
            x_i = self.linear[i](x_i)
            x_list.append(x_i)
        x = torch.cat(x_list, dim=2)
        return x


def train(df_train,
          df_validation,
          target_columns,
          n_prediction_steps,
          output_path=None):
    print('==== Time series training ====')

    X_train, y_train =\
        create_dataset(df=df_train,
                       target_columns=target_columns,
                       n_prediction_steps=n_prediction_steps)
    X_validation, y_validation =\
        create_dataset(df=df_validation,
                       target_columns=target_columns,
                       n_prediction_steps=n_prediction_steps)

    model = TimeSeriesModel(target_columns)
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    loader = data.DataLoader(data.TensorDataset(X_train, y_train),
                             shuffle=True,
                             batch_size=BATCH_SIZE)

    previous_validation_rmse = inf
    no_improvement_counter = 0
    for epoch in range(N_EPOCHS):
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validation and early stopping
        if epoch % 10 != 0 and not epoch == N_EPOCHS - 1:
            continue
        model.eval()
        with torch.no_grad():
            y_pred = model(X_train)
            train_rmse = np.sqrt(loss_fn(y_pred, y_train))
            y_pred = model(X_validation)
            validation_rmse = np.sqrt(loss_fn(y_pred, y_validation))
        print(f'Epoch {epoch}: ' +
              f'train RMSE {train_rmse}, validation RMSE {validation_rmse}')

        improvement_pct = (previous_validation_rmse - validation_rmse) /\
            previous_validation_rmse * 100

        if improvement_pct <= 10:
            no_improvement_counter += 1
        else:
            previous_validation_rmse = validation_rmse
            no_improvement_counter = 0

            if output_path:
                save_model(model, output_path)

        if no_improvement_counter >= VALIDATION_PATIENCE:
            print('Early stop')
            break

    return model


def predict(model,
            df,
            target_columns,
            n_prediction_steps):
    X = create_dataset_Xonly(df=df,
                             target_columns=target_columns)
    return model(X)[:, -n_prediction_steps:, :].detach().numpy()


def predict_df(model,
               df,
               target_columns,
               predicted_columns,
               n_prediction_steps,
               fill_values):
    predictions = predict(model=model,
                          df=df,
                          target_columns=target_columns,
                          n_prediction_steps=n_prediction_steps)
    df = df.copy()

    # shifting for lookback
    for i in range(len(target_columns)):
        shift = np.ones(n_prediction_steps) * fill_values[i]
        shift = np.array([shift for i in range(LOOKBACK)])
        prediction = np.concatenate([shift, predictions[:, :, i]])
        df[predicted_columns[i]] = [np.ceil(row).astype(int).tolist()
                                    for row in prediction]

    return df


def plot(df_train,
         df_test,
         model,
         target_columns,
         n_prediction_steps,
         y_max,
         title,
         x_label,
         output_paths=None):
    n_train = df_train.shape[0]
    n = n_train + df_test.shape[0]
    X = np.concatenate([df_train[target_columns].astype('float32').to_numpy(),
                        df_test[target_columns].astype('float32').to_numpy()])

    with torch.no_grad():
        predictions_train = predict(model=model,
                                    df=df_train,
                                    target_columns=target_columns,
                                    n_prediction_steps=n_prediction_steps)
        predictions_test = predict(model=model,
                                   df=df_test,
                                   target_columns=target_columns,
                                   n_prediction_steps=n_prediction_steps)

    for i, target_column in enumerate(target_columns):
        # shift train predictions for plotting
        train_plot = np.ones(n) * np.nan
        train_plot[LOOKBACK:n_train] =\
            predictions_train[:, -n_prediction_steps, i]
        # shift test predictions for plotting
        test_plot = np.ones(n) * np.nan
        test_plot[n_train + LOOKBACK:n] =\
            predictions_test[:, -n_prediction_steps, i]

        fig = plt.figure(plt.figure(figsize=[5.5, 4], dpi=80))
        plt.xlabel(x_label)
        plt.ylabel(target_column)
        plt.title(title)
        plt.ylim((0, y_max[i]))
        plt.plot(X[:, i],
                 linestyle='solid',
                 label='Real data')
        plt.plot(train_plot,
                 linestyle='solid',
                 label='Prediction on training\nand validation data')
        plt.plot(test_plot,
                 linestyle='solid',
                 label='Prediction on test data')
        plt.legend()

        if output_paths:
            plt.savefig(output_paths[i], bbox_inches='tight')

        plt.close(fig)
