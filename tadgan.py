import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.models import Sequential, Model
from utils.data_preprocessing.generator import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from tqdm import tqdm
import time
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler

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


def init_model():
    adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.0002, beta_1=0.5)

    E = Sequential()
    E.add(L.InputLayer(WINDOW_SIZE))
    E.add(L.Dense(WINDOW_SIZE * 2, activation='relu'))
    E.add(L.Dropout(0.3))
    E.add(L.Dense(WINDOW_SIZE, activation='relu'))
    E.add(L.Dropout(0.3))
    E.add(L.Dense(LATENT_VECTOR_SIZE))

    G = Sequential()
    G.add(L.Dense(128, input_dim=LATENT_VECTOR_SIZE))
    G.add(L.LeakyReLU(0.2))
    G.add(L.Dropout(0.3))
    G.add(L.Dense(256))
    G.add(L.LeakyReLU(0.2))
    G.add(L.Dropout(0.3))
    G.add(L.Dense(512))
    G.add(L.LeakyReLU(0.2))
    G.add(L.Dropout(0.3))
    G.add(L.Dense(WINDOW_SIZE, activation='tanh'))
    G.compile(loss='binary_crossentropy', optimizer=adam)

    Cx = Sequential()
    Cx.add(L.Dense(1024, input_dim=WINDOW_SIZE))
    Cx.add(L.LeakyReLU(0.2))
    Cx.add(L.Dropout(0.3))
    Cx.add(L.Dense(512, input_dim=WINDOW_SIZE))
    Cx.add(L.LeakyReLU(0.2))
    Cx.add(L.Dropout(0.3))
    Cx.add(L.Dense(256))
    Cx.add(L.LeakyReLU(0.2))
    Cx.add(L.Dropout(0.3))
    Cx.add(L.Dense(1, activation='sigmoid'))
    Cx.compile(loss='binary_crossentropy', optimizer=adam)

    Cz = Sequential()
    Cz.add(L.Dense(1024, input_dim=LATENT_VECTOR_SIZE))
    Cz.add(L.LeakyReLU(0.2))
    Cz.add(L.Dropout(0.3))
    Cz.add(L.Dense(512))
    Cz.add(L.LeakyReLU(0.2))
    Cz.add(L.Dropout(0.3))
    Cz.add(L.Dense(256))
    Cz.add(L.LeakyReLU(0.2))
    Cz.add(L.Dropout(0.3))
    Cz.add(L.Dense(1, activation='sigmoid'))
    Cz.compile(loss='binary_crossentropy', optimizer=adam)

    ae_input = L.Input(WINDOW_SIZE)
    ae_code = E(ae_input)
    ae_reconstruction = G(ae_code)
    ae_model = Model(inputs=ae_input, outputs=ae_reconstruction)
    ae_model.compile(loss='mse', optimizer=adam)

    cx_gan_input = L.Input(LATENT_VECTOR_SIZE)
    cx_gan_code = G(cx_gan_input)
    cx_gan_output = Cx(cx_gan_code)
    cx_gan_model = Model(inputs=cx_gan_input, outputs=cx_gan_output)
    cx_gan_model.compile(loss='binary_crossentropy', optimizer=adam)

    cz_gan_input = L.Input(WINDOW_SIZE)
    cz_gan_code = E(cz_gan_input)
    cz_gan_output = Cz(cz_gan_code)
    cz_gan_model = Model(inputs=cz_gan_input, outputs=cz_gan_output)
    cz_gan_model.compile(loss='binary_crossentropy', optimizer=adam)

    return E, G, Cx, Cz, ae_model, cx_gan_model, cz_gan_model


def train_model(X_train, epochs=1, batch_size=128):
    batchCount = int(X_train.shape[0] / batch_size)

    for epoch in range(1, epochs + 1):
        print("-" * 10, "Epoch: {}, batchCount {}".format(epoch, batchCount), "-" * 10)

        for _ in range(batchCount):

            # обучение дискриминатора Cx
            Cx.trainable = True
            idxs = np.random.choice(len(X_train) - WINDOW_SIZE, size=batch_size, replace=False)
            fake = G.predict(np.random.normal(0, 1, size=(batch_size, LATENT_VECTOR_SIZE)))
            X = []
            for i in idxs:
                X.append(X_train[i:i + WINDOW_SIZE])
            X = np.r_[X, fake]
            labels = np.r_[np.ones(shape=batch_size) * 0.95, np.zeros(shape=batch_size)]
            cx_loss = Cx.train_on_batch(X, labels)

            # обучение генератора cx_gan_model обманывать дискриминатор Cx
            Cx.trainable = False
            labels = np.ones(shape=batch_size)
            X = np.random.normal(0, 1, size=(batch_size, LATENT_VECTOR_SIZE))
            cx_g_loss = cx_gan_model.train_on_batch(X, labels)

            # обучение дискриминатора Cz
            Cz.trainable = True
            idxs = np.random.choice(len(X_train) - WINDOW_SIZE, size=batch_size, replace=False)
            fake = np.array([X_train[i:i + WINDOW_SIZE] for i in idxs])
            fake = E.predict(fake)
            X = np.random.normal(0, 1, size=(batch_size, LATENT_VECTOR_SIZE))
            X = np.r_[X, fake]
            labels = np.r_[np.ones(shape=batch_size) * 0.95, np.zeros(shape=batch_size)]
            cz_loss = Cz.train_on_batch(X, labels)

            # обучение генератора cx_gan_model обманывать дискриминатор Cz
            Cz.trainable = False
            idxs = np.random.choice(len(X_train) - WINDOW_SIZE, size=batch_size, replace=False)
            X = []
            for i in idxs:
                X.append(X_train[i:i + WINDOW_SIZE])
            X = np.array(X)
            labels = np.ones(shape=batch_size)
            cz_g_loss = cz_gan_model.train_on_batch(X, labels)

            # обучение автокодировщика AE
            idxs = np.random.choice(len(X_train) - WINDOW_SIZE, size=batch_size, replace=False)
            X = []
            for i in idxs:
                X.append(X_train[i:i + WINDOW_SIZE])
            E.trainable = True
            G.trainable = True
            X = np.array(X)
            ae_loss = ae_model.train_on_batch(X, X)

        # оценка ошибок и периодическое сохранение картинок
        aeLoss.append(ae_loss)
        cxLoss.append(cx_loss)
        czLoss.append(cz_loss)
        cx_g_Loss.append(cx_g_loss)
        cz_g_Loss.append(cz_g_loss)

        if epoch % 4 == 0:
            print("Эпоха {}".format(epoch))
            print("ae_loss {}, cx_loss {}, cz_loss {}, cx_g_loss {}, cz_g_loss {}".format(ae_loss,
                                                                                          cx_loss,
                                                                                          cz_loss,
                                                                                          cx_g_loss,
                                                                                          cz_g_loss))
            ae_model.save_weights("./ae_model-weights-epoch-{}.h5".format(epoch))


aeLoss = []
cxLoss = []
czLoss = []
cx_g_Loss = []
cz_g_Loss = []

E, G, Cx, Cz, ae_model, cx_gan_model, cz_gan_model = init_model()
def TadGan_learn(DATA, epochs, train_size):

    df = DATA

    df["timestamp"] = df["date"] + " " + df["time"]

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df = df[["temp", "anomaly"]]

    ys_train = df.temp.values[:len(df)-TEST_SIZE]
    xs_test = df.index.values[len(df)-TEST_SIZE:]
    ys_test = df.temp.values[len(df)-TEST_SIZE:]
    an_test = df.anomaly.values[len(df)-TEST_SIZE:]


    ys_train = MinMaxScaler((-1,1)).fit_transform(ys_train.reshape(-1,1)).squeeze()
    ys_test = MinMaxScaler((-1,1)).fit_transform(ys_test.reshape(-1,1)).squeeze()

    start = time.time()
    train_model(ys_train, epochs, 128)
    z = (time.time() - start)
    ae_model.save('models/tadgan_model')

    predictions_ = get_reconstruction_segment(ae_model, ys_test, 0, len(ys_test))
    labels_ = check_anomaly_pointwise_abs(ys_test, predictions_, 0.4)
    l_, r_ = 0, len(ys_test)

    plt.figure(figsize=(20,10))
    plt.plot(xs_test[l_:r_], ys_test[l_:r_], label='значения ряда')
    plt.plot(xs_test[l_:r_], predictions_[l_:r_], c='g', label="предсказанные значения")
    plt.scatter(xs_test[l_:r_], an_test[l_:r_]-5, c='r', label="размеченные аномалии")
    plt.scatter(xs_test[l_:r_], np.array(labels_[l_:r_])-3, c='b', label="найденные аномалии")

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
    print(an_test[:len(labels_)])
    print(labels_)
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    acc_output = []
    if GEN_ANOMALY == True:
        acc_output.append(accuracy_score(an_test, labels_))
        acc_output.append(roc_auc_score(an_test, labels_))
        acc_output.append(recall_score(an_test, labels_))
        acc_output.append(f1_score(an_test, labels_))
    return [fig, z, acc_output]
