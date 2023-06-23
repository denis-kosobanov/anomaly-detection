import pickle
import time

import plotly.graph_objects as go

from utils.data_preprocessing.generator import *


def prophet_out(DATA):
    df = DATA

    df["timestamp"] = df["date"] + " " + df["time"]

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[['timestamp', 'temp', 'anomaly']]

    df = df.rename(columns={'timestamp': 'ds',
                            'temp': 'y'})

    """
        создаем выборку
    """
    test = df.iloc[0:len(df)]
    """
        загружаем предобученную модель
    """
    my_model = pickle.load(open('models/prophet_model.sav', 'rb'))

    start = time.time()
    forecast = my_model.predict(test)
    z = (time.time() - start)

    """
        создаем график
    """

    test["anomaly_prophet"] = forecast['yhat'].values - test["y"].values
    mean = test["anomaly_prophet"].mean()

    test.loc[abs(test["anomaly_prophet"]) >= 0.95, "anomaly_prophet"] = 1
    test.loc[abs(test["anomaly_prophet"]) <= 0.95, "anomaly_prophet"] = 0

    a = test.loc[test['anomaly_prophet'] == 1, ['ds', 'y']]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test['ds'], y=test['y'],
                             mode='lines',
                             name='Исходный временной ряд'))
    fig.add_trace(go.Scatter(x=a['ds'], y=a['y'],
                             mode='markers',
                             name='Аномалия'))
    fig.update_layout(showlegend=True)

    return [fig, str(abs(mean)), str(df.y.max()), str(df.y.min()), str(df.y.mean()), str(len(a) / (len(df))), str(z)]
