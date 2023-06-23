import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, recall_score, f1_score

plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 1.5
import plotly.graph_objects as go

from catboost import CatBoostRegressor
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import backtesting_forecaster
import time

from utils.data_preprocessing.generator import *


def catboost_learn(DATA, train_size):
    df = DATA
    df["timestamp"] = df["date"] + " " + df["time"]

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df.drop(columns=['date', 'time'], axis=1)

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df.plot(x='timestamp', y='temp')

    df['hours'] = df['timestamp'].dt.hour
    df['daylight'] = ((df['hours'] >= 7) & (df['hours'] <= 22)).astype(int)

    df['DayOfTheWeek'] = df['timestamp'].dt.dayofweek
    df['WeekDay'] = (df['DayOfTheWeek'] < 5).astype(int)

    df['time_epoch'] = (df['timestamp'].astype(np.int64) / 100000000000).astype(np.int64)

    df['categories'] = df['WeekDay'] * 2 + df['daylight']
    """
        добавляем категориальные признаки
    """

    exog_variables = [column for column in df.columns
                      if column.startswith(('hours', 'DayOfTheWeek'))]

    data_train = df.loc[:train_size]
    data_val = df.loc[8097:len(df) - TEST_SIZE]
    data_test = df.loc[len(df) - TEST_SIZE + 1:]

    """
        обучаем модель
    """

    # impotance = forecaster.get_feature_importance()
    start = time.time()
    forecaster = ForecasterAutoreg(
        regressor=CatBoostRegressor(random_state=123, silent=True),
        lags=24
    )
    z = (time.time() - start)

    param_grid = {
        'n_estimators': [100, 500],
        'max_depth': [3, 5, 10],
        'learning_rate': [0.01, 0.1]
    }

    # Lags used as predictors
    lags_grid = [72, [1, 2, 3, 23, 24, 25, 71, 72, 73]]

    metric, predictions = backtesting_forecaster(
        forecaster=forecaster,
        y=df['temp'],
        exog=df[exog_variables],
        initial_train_size=len(df.loc[:len(df) - TEST_SIZE]),
        fixed_train_size=False,
        steps=36,
        refit=False,
        metric='mean_squared_error',
        verbose=False
    )

    print(f"Backtest error: {metric}")

    data_test["anomaly_catboost"] = data_test.temp - predictions.pred
    print(data_test["anomaly_catboost"])
    data_test.loc[abs(data_test["anomaly_catboost"]) >= 0.9, "anomaly_catboost"] \
        = 1
    data_test.loc[abs(data_test["anomaly_catboost"]) <= 0.9, "anomaly_catboost"] \
        = 0

    fig = go.Figure()

    a = data_test.loc[data_test['anomaly_catboost'] == 1, ['temp']]
    print(a)
    fig.add_trace(go.Scatter(x=data_test.index, y=data_test.temp.rolling(3).mean(),
                             mode='lines',
                             name='Временной ряд'))
    fig.add_trace(go.Scatter(x=a.index, y=a.temp.rolling(3).mean(),
                             mode='markers',
                             name='Аномалия'))
    fig.update_layout(showlegend=True)
    from sklearn.metrics import accuracy_score

    acc_output = []
    if GEN_ANOMALY == True:
        acc_output.append(accuracy_score(data_test.anomaly, data_test.anomaly_catboost))
        acc_output.append(roc_auc_score(data_test.anomaly, data_test.anomaly_catboost))
        acc_output.append(recall_score(data_test.anomaly, data_test.anomaly_catboost))
        acc_output.append(f1_score(data_test.anomaly, data_test.anomaly_catboost))

    return [fig, z, acc_output]
