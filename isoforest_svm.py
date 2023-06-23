import pickle
import time

from sklearn import preprocessing
from sklearn.ensemble import IsolationForest

from utils.data_preprocessing.generator import *


def isoforest_learn(DATA, train_size):
    df = DATA

    df["timestamp"] = df["date"] + " " + df["time"]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.drop(columns=['date', 'time'], axis=1)

    df['hours'] = df['timestamp'].dt.hour
    df['daylight'] = ((df['hours'] >= 7) & (df['hours'] <= 22)).astype(int)

    df['DayOfTheWeek'] = df['timestamp'].dt.dayofweek
    df['WeekDay'] = (df['DayOfTheWeek'] < 5).astype(int)

    outliers_fraction = 0.1

    df['time_epoch'] = (df['timestamp'].astype(np.int64) / 100000000000).astype(np.int64)

    df['categories'] = df['WeekDay'] * 2 + df['daylight']

    data = df[['temp', 'hours', 'daylight', 'DayOfTheWeek', 'WeekDay']]
    min_max_scaler = preprocessing.StandardScaler()
    np_scaled = min_max_scaler.fit_transform(data)
    data = pd.DataFrame(np_scaled)

    """
        обучаем модель
    """
    start = time.time()
    model = IsolationForest(contamination=outliers_fraction)
    model.fit(data)
    z = (time.time() - start)
    # add the data to the main
    df['anomaly_isoforest'] = pd.Series(model.predict(data))

    df['anomaly_isoforest'] = df['anomaly_isoforest'].map({1: 0, -1: 1})

    with open('models/isoforest_model.sav', 'wb') as f:
        pickle.dump(model, f)

    """
        строим график и получаем точность модели по метрикам
    """

    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    import plotly.graph_objects as go
    a = df[len(df) - TEST_SIZE:].loc[df['anomaly_isoforest'] == 1, ['time_epoch', 'temp']]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[len(df) - TEST_SIZE:]['time_epoch'], y=df[len(df) - TEST_SIZE:]['temp'],
                             mode='lines',
                             name='Временной ряд'))
    fig.add_trace(go.Scatter(x=a['time_epoch'], y=a['temp'],
                             mode='markers',
                             name='Аномалия'))
    fig.update_layout(showlegend=True)
    acc_output = []
    if GEN_ANOMALY == True:
        acc_output.append(accuracy_score(df[len(df) - TEST_SIZE:]['anomaly'],
                                         df[len(df) - TEST_SIZE:][
                                             'anomaly_isoforest']))
        acc_output.append(roc_auc_score(df[len(df) - TEST_SIZE:]['anomaly'], df[len(df) - TEST_SIZE:][
            'anomaly_isoforest']))
        acc_output.append(
            recall_score(df[len(df) - TEST_SIZE:]['anomaly'], df[len(df) - TEST_SIZE:]['anomaly_isoforest']))
        acc_output.append(f1_score(df[len(df) - TEST_SIZE:]['anomaly'], df[len(df) - TEST_SIZE:]['anomaly_isoforest']))

    return [fig, z, acc_output]
