import time

from sklearn.metrics import mean_absolute_error

import xgb_utils
from utils.data_preprocessing.generator import *

target_variable = "temp"
target_sequence_length = 24
import plotly.graph_objects as go
import pickle


def xgboost_out(DATA):
    hyperparameters = {
        "in_length": 1,
        "step_size": 12,
        "n_estimators": 20,
        "max_depth": 6,
        "subsample": 0.5,
        "min_child_weight": 1,
        "selected_features": [target_variable]
    }

    temppred = xgb_utils.load_data(DATA)
    test_data = temppred

    test_indices = xgb_utils.get_indices_entire_sequence(
        data=test_data,
        window_size=hyperparameters["in_length"] + target_sequence_length,
        step_size=24
    )
    x_test, y_test = xgb_utils.get_xgboost_x_y(
        indices=test_indices,
        data=test_data[hyperparameters["selected_features"]].to_numpy(),
        target_sequence_length=target_sequence_length,
        input_seq_len=hyperparameters["in_length"]
    )
    with open('models/xgb_model.pkl', 'rb') as f:
        trained_model = pickle.load(f)
    start = time.time()
    test_forecasts = trained_model.predict(x_test)
    z = (time.time() - start)
    test_mae = mean_absolute_error(y_test, test_forecasts)

    plot_df = pd.DataFrame({"Forecasts": test_forecasts.flatten(), "Targets": y_test.flatten()},
                           index=range(len(y_test.flatten())))

    plt.plot(plot_df.index, plot_df["Forecasts"].rolling(3).mean(), label="Forecasts")
    plt.plot(plot_df.index, plot_df["Targets"].rolling(3).mean(), label="Targets")
    plot_df["anomaly_xgb"] = plot_df.Targets - plot_df.Forecasts

    plot_df.loc[abs(plot_df["anomaly_xgb"]) >= 0.9, "anomaly_xgb"] = 1
    plot_df.loc[abs(plot_df["anomaly_xgb"]) <= 0.9, "anomaly_xgb"] = 0

    fig = go.Figure()

    a = plot_df.loc[plot_df['anomaly_xgb'] == 1, ['Targets']]

    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Targets"].rolling(3).mean(),
                             mode='lines',
                             name='Временной ряд'))
    fig.add_trace(go.Scatter(x=a.index, y=a['Targets'].rolling(3).mean(),
                             mode='markers',
                             name='Аномалия'))
    fig.update_layout(showlegend=True)

    return [fig, str(test_mae), str(DATA.temp.max()), str(DATA.temp.min()), str(DATA.temp.mean()),
            str(len(a) / (TEST_SIZE) * 100), str(z)]
