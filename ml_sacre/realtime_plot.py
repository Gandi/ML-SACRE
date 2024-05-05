from ml_sacre.helpers import load_dataframe
import dash
from dash import dcc, html
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
import plotly.express as px
from collections import deque


timetamp_column = 'Timestamp'
requested_column_cpu = 'CPU Requested (%)'
used_column_cpu = 'CPU Used (%)'
predicted_column_cpu = 'CPU Predicted (%)'
allocated_column_cpu = 'CPU Allocated ML_SACRE_2_2 (%)'
requested_column_ram = 'RAM Requested (GB)'
used_column_ram = 'RAM Used (GB)'
predicted_column_ram = 'RAM Predicted (GB)'
allocated_column_ram = 'RAM Allocated ML_SACRE_2_2 (GB)'


def realtime_plot(df_path):
    df = load_dataframe(df_path)

    # using only one predicted step
    df[predicted_column_cpu] = df[predicted_column_cpu].apply(
        lambda x: x[0])
    df[predicted_column_ram] = df[predicted_column_ram].apply(
        lambda x: x[0])

    # shifting predicted
    df[predicted_column_cpu] = df[predicted_column_cpu].shift(1)
    df[predicted_column_ram] = df[predicted_column_ram].shift(1)

    timestamp = df[timetamp_column].to_list()
    cpu_requested = df[requested_column_cpu].to_list()
    cpu_used = df[used_column_cpu].to_list()
    cpu_predicted = df[predicted_column_cpu].to_list()
    cpu_allocated = df[allocated_column_cpu].to_list()
    ram_requested = df[requested_column_ram].to_list()
    ram_used = df[used_column_ram].to_list()
    ram_predicted = df[predicted_column_ram].to_list()
    ram_allocated = df[allocated_column_ram].to_list()

    app = dash.Dash(__name__)

    app.layout = html.Div(
        [
            dcc.Graph(id='live-update-cpu-graph'),
            dcc.Graph(id='live-update-ram-graph'),
            dcc.Interval(id='interval-component', interval=500),
        ]
    )

    X_timestamp = deque(maxlen=20)
    Y_cpu_requested = deque(maxlen=20)
    Y_cpu_used = deque(maxlen=20)
    Y_cpu_predicted = deque(maxlen=20)
    Y_cpu_allocated = deque(maxlen=20)
    Y_ram_requested = deque(maxlen=20)
    Y_ram_used = deque(maxlen=20)
    Y_ram_predicted = deque(maxlen=20)
    Y_ram_allocated = deque(maxlen=20)

    X_timestamp.append(timestamp.pop(0))
    Y_cpu_requested.append(cpu_requested.pop(0))
    Y_cpu_used.append(cpu_used.pop(0))
    Y_cpu_predicted.append(cpu_predicted.pop(0))
    Y_cpu_allocated.append(cpu_allocated.pop(0))
    Y_ram_requested.append(ram_requested.pop(0))
    Y_ram_used.append(ram_used.pop(0))
    Y_ram_predicted.append(ram_predicted.pop(0))
    Y_ram_allocated.append(ram_allocated.pop(0))

    def update_data():
        if not timestamp:
            # deque empty
            raise PreventUpdate
        X_timestamp.append(timestamp.pop(0))
        Y_cpu_requested.append(cpu_requested.pop(0))
        Y_cpu_used.append(cpu_used.pop(0))
        Y_cpu_predicted.append(cpu_predicted.pop(0))
        Y_cpu_allocated.append(cpu_allocated.pop(0))
        Y_ram_requested.append(ram_requested.pop(0))
        Y_ram_used.append(ram_used.pop(0))
        Y_ram_predicted.append(ram_predicted.pop(0))
        Y_ram_allocated.append(ram_allocated.pop(0))
        return X_timestamp,\
            Y_cpu_requested,\
            Y_cpu_used,\
            Y_cpu_predicted,\
            Y_cpu_allocated,\
            Y_ram_requested,\
            Y_ram_used,\
            Y_ram_predicted,\
            Y_ram_allocated

    @app.callback(
        [dash.dependencies.Output('live-update-cpu-graph', 'figure'),
         dash.dependencies.Output('live-update-ram-graph', 'figure')],
        [dash.dependencies.Input('interval-component', 'n_intervals')],
    )
    def update_graph(n):
        X_timestamp,\
            Y_cpu_requested,\
            Y_cpu_used,\
            Y_cpu_predicted,\
            Y_cpu_allocated,\
            Y_ram_requested,\
            Y_ram_used,\
            Y_ram_predicted,\
            Y_ram_allocated = update_data()
        colors = px.colors.diverging.Spectral
        layout_cpu = go.Layout(title='CPU allocation',
                               xaxis=dict(title='Time'),
                               yaxis=dict(title='CPU (%)',
                                          range=[0,
                                                 None]))
        layout_ram = go.Layout(title='RAM allocation',
                               xaxis=dict(title='Time'),
                               yaxis=dict(title='RAM (GB)',
                                          range=[0,
                                                 None]))
        data_cpu_requested = go.Scatter(x=list(X_timestamp),
                                        y=list(Y_cpu_requested),
                                        mode='lines',
                                        name='Requested',
                                        line=dict(color=colors[2]))
        data_cpu_used = go.Scatter(x=list(X_timestamp),
                                   y=list(Y_cpu_used),
                                   mode='lines',
                                   name='Used',
                                   line=dict(color=colors[-2]))
        data_cpu_predicted = go.Scatter(x=list(X_timestamp),
                                        y=list(Y_cpu_predicted),
                                        mode='lines',
                                        name='Predicted',
                                        line=dict(color=colors[-3],
                                                  dash='dot'))
        data_cpu_allocated = go.Scatter(x=list(X_timestamp),
                                        y=list(Y_cpu_allocated),
                                        mode='lines',
                                        name='Allocated',
                                        line=dict(color=colors[1],
                                                  dash='dash'))
        data_ram_requested = go.Scatter(x=list(X_timestamp),
                                        y=list(Y_ram_requested),
                                        mode='lines',
                                        name='Requested',
                                        line=dict(color=colors[2]))
        data_ram_used = go.Scatter(x=list(X_timestamp),
                                   y=list(Y_ram_used),
                                   mode='lines',
                                   name='Used',
                                   line=dict(color=colors[-2]))
        data_ram_predicted = go.Scatter(x=list(X_timestamp),
                                        y=list(Y_ram_predicted),
                                        mode='lines',
                                        name='Predicted',
                                        line=dict(color=colors[-3],
                                                  dash='dot'))
        data_ram_allocated = go.Scatter(x=list(X_timestamp),
                                        y=list(Y_ram_allocated),
                                        mode='lines',
                                        name='Allocated',
                                        line=dict(color=colors[1],
                                                  dash='dash'))
        return {'layout': layout_cpu,
                'data': [data_cpu_requested,
                         data_cpu_allocated,
                         data_cpu_used,
                         data_cpu_predicted]},\
            {'layout': layout_ram,
             'data': [data_ram_requested,
                      data_ram_allocated,
                      data_ram_used,
                      data_ram_predicted]}

    return app.run_server(debug=True)


if __name__ == '__main__':

    df_path = '/df/path/df_allocated.csv'
    realtime_plot(df_path)
