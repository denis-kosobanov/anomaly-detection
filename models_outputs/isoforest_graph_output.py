import pickle
import time

from sklearn import preprocessing

from utils.data_preprocessing.generator import *


def isoforest_out(DATA):
    df = DATA

    df["timestamp"] = df["date"] + " " + df["time"]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.drop(columns=['date', 'time'], axis=1)

    """
        выделяем категориальные признаки
    """
    df['hours'] = df['timestamp'].dt.hour
    df['daylight'] = ((df['hours'] >= 7) & (df['hours'] <= 22)).astype(int)
    df['DayOfTheWeek'] = df['timestamp'].dt.dayofweek
    df['WeekDay'] = (df['DayOfTheWeek'] < 5).astype(int)
    df['time_epoch'] = (df['timestamp'].astype(np.int64) / 100000000000).astype(np.int64)
    df['categories'] = df['WeekDay'] * 2 + df['daylight']

    data = df[['time_epoch', 'temp']]
    min_max_scaler = preprocessing.StandardScaler()
    np_scaled = min_max_scaler.fit_transform(data)
    data = pd.DataFrame(np_scaled)
    """
        загружаем предобученную модель
    """
    model = pickle.load(open('models/isoforest_model.sav', 'rb'))
    """
        время работы
    """
    start = time.time()
    df['anomaly_isoforest'] = pd.Series(model.predict(data))
    z = (time.time() - start)
    df['anomaly_isoforest'] = df['anomaly_isoforest'].map({1: 0, -1: 1})

    """
        выводим график
    """

    import plotly.graph_objects as go
    a = df.loc[df['anomaly_isoforest'] == 1, ['time_epoch', 'temp']]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['time_epoch'], y=df['temp'],
                             mode='lines',
                             name='Временной ряд'))
    fig.add_trace(go.Scatter(x=a['time_epoch'], y=a['temp'],
                             mode='markers',
                             name='Аномалия'))
    fig.update_layout(showlegend=True)

    return [fig, str(0.7), str(df.temp.max()), str(df.temp.min()), str(df.temp.mean()), str(len(a) / (len(df))), str(z)]
