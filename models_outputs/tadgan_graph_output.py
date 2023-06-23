import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.models import Sequential, Model
from utils.data_preprocessing.generator import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import time
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras


def get_reconstruction_segment(model, values, start, end):
    """
    Автокодировщик model получает на вход сигнал values и
    возвращает реконструированные (декодированные) значения
    для заданного сегмента [start, end].
    Поскольку автокодировщик требует постоянное число сэмплов на входе, то для
    последнего набора данных берется сегмент [end-WINWOW_SIZE, end].
    """
    num = int((end - start) / WINDOW_SIZE)
    data = []
    left, right = 0, WINDOW_SIZE

    for i in range(num):
        result = model.predict(np.array(values[left:right]).reshape(1, -1))
        data = np.r_[data, result[0]]
        left += WINDOW_SIZE
        right += WINDOW_SIZE

    if left < end:
        result = model.predict(np.array(values[end - WINDOW_SIZE:end]).reshape(1, -1)).reshape(-1, 1)
        data = np.r_[data, result[-end + left:].squeeze()]

    return np.array(data)


def check_anomaly_pointwise_abs(ys, ys_hat, threeshold):
    """
    Поточечное сравнение, модуль расстояния
    """
    result = []
    for y1, y2 in zip(ys, ys_hat):
        if np.abs(y1 - y2) > threeshold:
            result.append(1)
        else:
            result.append(0)

    return np.array(result)


LATENT_VECTOR_SIZE = 10
WINDOW_SIZE = 100


def TadGan_out(DATA):
    df = DATA

    df["timestamp"] = df["date"] + " " + df["time"]

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df = df[["temp", "anomaly"]]

    xs_test = df.index.values[0: len(df)]
    ys_test = df.temp.values[0: len(df)]
    an_test = df.anomaly.values[0: len(df)]

    ys_test = MinMaxScaler((-1, 1)).fit_transform(ys_test.reshape(-1, 1)).squeeze()

    ae_model = keras.models.load_model('models/tadgan_model')
    start = time.time()
    predictions_ = get_reconstruction_segment(ae_model, ys_test, 0, len(ys_test))
    z = (time.time() - start)
    labels_ = check_anomaly_pointwise_abs(ys_test, predictions_, 0.4)
    l_, r_ = 0, len(ys_test)

    plt.figure(figsize=(20, 10))
    plt.plot(xs_test[l_:r_], ys_test[l_:r_], label='значения ряда')
    plt.plot(xs_test[l_:r_], predictions_[l_:r_], c='g', label="предсказанные значения")
    plt.scatter(xs_test[l_:r_], an_test[l_:r_] - 5, c='r', label="размеченные аномалии")
    plt.scatter(xs_test[l_:r_], np.array(labels_[l_:r_]) - 3, c='b', label="найденные аномалии")

    dict = {'time': xs_test[l_:r_], 'temp': ys_test[l_:r_], 'anomaly': np.array(labels_[l_:r_])}
    anomalies = pd.DataFrame(dict)
    anomaly = anomalies.loc[anomalies['anomaly'] == 1, ['time', 'temp']]

    print(anomaly)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs_test[l_:r_], y=ys_test[l_:r_],
                             mode='lines',
                             name='Исходный временной ряд'))
    fig.add_trace(go.Scatter(x=anomaly.time, y=anomaly.temp,
                             mode='markers',
                             name='Аномалия'))
    fig.update_layout(showlegend=True)

    return [fig, "", str(df.temp.max()), str(df.temp.min()), str(df.temp.mean()), str(len(anomaly) / (TEST_SIZE) * 100),
            str(z)]
