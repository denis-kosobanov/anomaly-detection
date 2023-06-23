import plotly.graph_objects as go
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score

from utils.data_preprocessing.generator import *


def rnn_learn(DATA, epoch, train_size):
    df = DATA

    df["timestamp"] = df["date"] + " " + df["time"]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.drop(columns=['date', 'time'], axis=1)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.plot(x='timestamp', y='temp')

    """
        извлекаем пареметры для обучения
    """

    df['hours'] = df['timestamp'].dt.hour
    df['daylight'] = ((df['hours'] >= 7) & (df['hours'] <= 22)).astype(int)
    df['DayOfTheWeek'] = df['timestamp'].dt.dayofweek
    df['WeekDay'] = (df['DayOfTheWeek'] < 5).astype(int)

    df['time_epoch'] = (df['timestamp'].astype(np.int64) / 100000000000).astype(np.int64)
    df['categories'] = df['WeekDay'] * 2 + df['daylight']

    data_n = df[['temp', 'hours', 'daylight', 'DayOfTheWeek', 'WeekDay']]
    min_max_scaler = preprocessing.StandardScaler()
    np_scaled = min_max_scaler.fit_transform(data_n)
    data_n = pd.DataFrame(np_scaled)

    prediction_time = 1
    testdatasize = 4050
    unroll_length = 50
    testdatacut = testdatasize + unroll_length + 1

    x_train = data_n[0:train_size].values
    y_train = data_n[prediction_time:train_size][0].values

    """
        тестовая выборка
    """
    x_test = data_n[0 - testdatacut:-prediction_time].values
    y_test = data_n[prediction_time - testdatacut:][0].values

    def unroll(data, sequence_length=24):
        result = []
        for index in range(len(data) - sequence_length):
            result.append(data[index: index + sequence_length])
        return np.asarray(result)

    # adapt the datasets for the sequence data shape
    x_train = unroll(x_train, unroll_length)
    x_test = unroll(x_test, unroll_length)
    y_train = y_train[-x_train.shape[0]:]
    y_test = y_test[-x_test.shape[0]:]

    from keras.layers import Dense, Activation, Dropout
    from keras.layers import LSTM
    from keras.models import Sequential
    import time

    model = Sequential()

    model.add(LSTM(units=32, input_dim=x_train.shape[-1], return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        units=1))
    model.add(Activation('linear'))

    model.compile(loss='mse', optimizer='rmsprop')

    start = time.time()
    model.fit(
        x_train,
        y_train,
        batch_size=3028,
        epochs=epoch,  # 30
        validation_split=0.1)
    z = (time.time() - start)
    model.save('models/rnn_model')
    loaded_model = model
    diff = []
    ratio = []
    p = loaded_model.predict(x_test)

    for u in range(len(y_test)):
        pr = p[u][0]
        ratio.append((y_test[u] / pr) - 1)
        diff.append(abs(y_test[u] - pr))

    diff = pd.Series(diff)
    number_of_outliers = int(0.05 * len(diff))
    threshold = diff.nlargest(number_of_outliers).min()

    test = (diff >= threshold).astype(int)

    complement = pd.Series(0, index=np.arange(len(data_n) - testdatasize))
    df['anomaly_rnn'] = complement._append(test, ignore_index='True')

    a = df.loc[df['anomaly_rnn'] == 1, ['time_epoch', 'temp']]

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
        acc_output.append(accuracy_score(df[0 - testdatacut:-prediction_time]['anomaly'],
                                         df[0 - testdatacut:-prediction_time]['anomaly_rnn']))
        acc_output.append(roc_auc_score(df[0 - testdatacut:-prediction_time]['anomaly'],
                                        df[0 - testdatacut:-prediction_time]['anomaly_rnn']))
        acc_output.append(recall_score(df[0 - testdatacut:-prediction_time]['anomaly'],
                                       df[0 - testdatacut:-prediction_time]['anomaly_rnn']))
        acc_output.append(f1_score(df[0 - testdatacut:-prediction_time]['anomaly'],
                                   df[0 - testdatacut:-prediction_time]['anomaly_rnn']))

    return [fig, z, acc_output]
