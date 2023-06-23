import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 1.5
import plotly.graph_objects as go
from catboost import CatBoostRegressor
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import backtesting_forecaster
import time
from utils.data_preprocessing.generator import *


def catboost_out(DATA):
    df = DATA
    df["timestamp"] = df["date"] + " " + df["time"]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.drop(columns=['date', 'time'], axis=1)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.plot(x='timestamp', y='temp')
    """
        категориальные признаки
    """
    df['hours'] = df['timestamp'].dt.hour
    df['daylight'] = ((df['hours'] >= 7) & (df['hours'] <= 22)).astype(int)

    df['DayOfTheWeek'] = df['timestamp'].dt.dayofweek
    df['WeekDay'] = (df['DayOfTheWeek'] < 5).astype(int)
    df['time_epoch'] = (df['timestamp'].astype(np.int64) / 100000000000).astype(np.int64)
    df['categories'] = df['WeekDay'] * 2 + df['daylight']
    exog_variables = [column for column in df.columns
                      if column.startswith(('hours', 'DayOfTheWeek'))]

    data_test = df.loc[len(df) - TEST_SIZE + 1:]

    # impotance = forecaster.get_feature_importance()
    start = time.time()
    forecaster = ForecasterAutoreg(
        regressor=CatBoostRegressor(random_state=123, silent=True),
        lags=24
    )
    z = (time.time() - start)

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
    """
        ошибка на тенировочной выборке mse
    """

    print(f"Backtest error: {metric}")

    data_test["anomaly_catboost"] = data_test.temp - predictions.pred
    mean = abs(data_test["anomaly_catboost"].mean())
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

    return [fig, str(mean), str(df.temp.max()), str(df.temp.min()), str(df.temp.mean()), str(len(a) / (len(df))),
            str(z)]
