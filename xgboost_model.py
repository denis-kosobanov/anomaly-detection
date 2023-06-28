import time

import xgboost as xgb
from sklearn.metrics import mean_absolute_error, roc_auc_score, recall_score, f1_score
from sklearn.multioutput import MultiOutputRegressor

import xgb_utils
from utils.data_preprocessing.generator import *

target_variable = "temp"
target_sequence_length = 24
import plotly.graph_objects as go
import pickle


def xgboost_learn(DATA, train_size):
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

    training_data = temppred[:len(temppred) - TEST_SIZE]

    test_data = temppred[len(temppred) - TEST_SIZE:]

    param_grid = {
        'n_estimators': [100, 500],
        'max_depth': [3, 5, 10],
        'learning_rate': [0.01, 0.1]
    }
    lags_grid = [48, 72]


    training_indices = xgb_utils.get_indices_entire_sequence(
        data=training_data,
        window_size=hyperparameters["in_length"] + target_sequence_length,
        step_size=hyperparameters["step_size"]
    )

    # Получение (X,Y) пар тренировочных данных
    x_train, y_train = xgb_utils.get_xgboost_x_y(
        indices=training_indices,
        data=training_data[hyperparameters["selected_features"]].to_numpy(),
        target_sequence_length=target_sequence_length,
        input_seq_len=hyperparameters["in_length"]
    )

    test_indices = xgb_utils.get_indices_entire_sequence(
        data=test_data,
        window_size=hyperparameters["in_length"] + target_sequence_length,
        step_size=24
    )

    # Получение (X,Y) пар тестовых данных
    x_test, y_test = xgb_utils.get_xgboost_x_y(
        indices=test_indices,
        data=test_data[hyperparameters["selected_features"]].to_numpy(),
        target_sequence_length=target_sequence_length,
        input_seq_len=hyperparameters["in_length"]
    )

    model = xgb.XGBRegressor(
        n_estimators=hyperparameters["n_estimators"],
        max_depth=hyperparameters["max_depth"],
        subsample=hyperparameters["subsample"],
        min_child_weight=hyperparameters["min_child_weight"],
        objective="reg:squarederror",
        tree_method="hist"
    )
    start = time.time()
    trained_model = MultiOutputRegressor(model).fit(x_train, y_train)
    z = (time.time() - start)
    with open('models/xgb_model.pkl', 'wb') as f:
        pickle.dump(trained_model, f)
    test_forecasts = trained_model.predict(x_test)
    test_mae = mean_absolute_error(y_test, test_forecasts)
    print(test_mae)
    print("Mean test data value: {}".format(np.mean(y_test)))

    plot_df = pd.DataFrame({"Forecasts": test_forecasts.flatten(), "Targets": y_test.flatten()},
                           index=range(len(y_test.flatten())))

    plt.plot(plot_df.index, plot_df["Forecasts"].rolling(3).mean(), label="Forecasts")
    plt.plot(plot_df.index, plot_df["Targets"].rolling(3).mean(), label="Targets")
    plot_df["anomaly_xgb"] = plot_df.Targets - plot_df.Forecasts
    print(plot_df["anomaly_xgb"])
    plot_df.loc[abs(plot_df["anomaly_xgb"]) >= 0.8, "anomaly_xgb"] = 1
    plot_df.loc[abs(plot_df["anomaly_xgb"]) <= 0.8, "anomaly_xgb"] = 0

    fig = go.Figure()

    a = plot_df.loc[plot_df['anomaly_xgb'] == 1, ['Targets']]
    print(a)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Targets"].rolling(3).mean(),
                             mode='lines',
                             name='Временной ряд'))
    fig.add_trace(go.Scatter(x=a.index, y=a['Targets'].rolling(3).mean(),
                             mode='markers',
                             name='Аномалия'))
    fig.update_layout(showlegend=True)

    from sklearn.metrics import accuracy_score
    acc_output = []

    if GEN_ANOMALY == True:
        acc_output.append(accuracy_score(plot_df['anomaly_xgb'], test_data["anomaly"][:4080]))
        acc_output.append(roc_auc_score(plot_df['anomaly_xgb'], test_data["anomaly"][:4080]))
        acc_output.append(recall_score(plot_df['anomaly_xgb'], test_data["anomaly"][:4080]))
        acc_output.append(f1_score(plot_df['anomaly_xgb'], test_data["anomaly"][:4080]))

    return [fig, z, acc_output]
