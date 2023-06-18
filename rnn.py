"""
Anomaly Detection using Brutlag algorithm
-----------------------------------------
This file contains the implementation brutlag algorithm and it can be used to detect anomalies
in time series data.
Dataset
-------
The dataset contains India's monthly average temperature (°C) recorded for a period of 2000-2018.
"""


from sklearn import preprocessing
from utils.data_preprocessing.generator import *
import plotly.graph_objects as go


def rnn_learn(DATA):
    df = DATA
    # ensemble = pd.read_csv(r"outputs/ensemble_out_316.csv", sep=',')
    # df["anomaly"] = 0
    # df["anomaly"].iloc[-4100:, ] = ensemble["target"].iloc[-4100:, ]
    # print(df["anomaly"].value_counts())
    # if (GEN_ANOMALY == True):
    #     df = generate_anomaly_data(df, WINDOW_COUNT, WINDOW_SIZE_LIST)
    # print(df["anomaly"].value_counts())
    df["timestamp"] = df["date"] + " " + df["time"]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.drop(columns = ['date', 'time'], axis = 1)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.plot(x='timestamp', y='temp')

    df['hours'] = df['timestamp'].dt.hour
    df['daylight'] = ((df['hours'] >= 7) & (df['hours'] <= 22)).astype(int)
    df['DayOfTheWeek'] = df['timestamp'].dt.dayofweek
    df['WeekDay'] = (df['DayOfTheWeek'] < 5).astype(int)
    outliers_fraction = 0.01
    df['time_epoch'] = (df['timestamp'].astype(np.int64)/100000000000).astype(np.int64)
    df['categories'] = df['WeekDay']*2 + df['daylight']


    data_n = df[['temp', 'hours', 'daylight', 'DayOfTheWeek', 'WeekDay']]
    min_max_scaler = preprocessing.StandardScaler()
    np_scaled = min_max_scaler.fit_transform(data_n)
    data_n = pd.DataFrame(np_scaled)




    # important parameters and train/test size
    prediction_time = 1
    testdatasize = 4050
    unroll_length = 50
    testdatacut = testdatasize + unroll_length  + 1
    print(data_n)
    #train data
    x_train = data_n[0:-prediction_time-testdatacut].values
    y_train = data_n[prediction_time:-testdatacut  ][0].values

    # test data
    x_test = data_n[0-testdatacut:-prediction_time].values
    y_test = data_n[prediction_time-testdatacut:  ][0].values


    def unroll(data,sequence_length=24):
        result = []
        for index in range(len(data) - sequence_length):
            result.append(data[index: index + sequence_length])
        return np.asarray(result)

    # adapt the datasets for the sequence data shape
    x_train = unroll(x_train,unroll_length)
    x_test  = unroll(x_test,unroll_length)
    y_train = y_train[-x_train.shape[0]:]
    y_test  = y_test[-x_test.shape[0]:]
    print(x_train)
    # see the shape
    print("x_train", x_train.shape)
    print("y_train", y_train.shape)
    print("x_test", x_test.shape)
    print("y_test", y_test.shape)
    import tensorflow as tf
    from keras.layers import Dense, Activation, Dropout
    from keras.layers import LSTM
    from keras.models import Sequential
    import time #helper libraries
    from keras.models import model_from_json
    import sys


    model = Sequential()

    model.add(LSTM(units = 32, input_dim=x_train.shape[-1],return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units = 100,return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        units=1))
    model.add(Activation('linear'))

    start = time.time()
    model.compile(loss='mse', optimizer='rmsprop')
    z = (time.time() - start)


    model.fit(
        x_train,
        y_train,
        batch_size=3028,
        epochs=30, #30
        validation_split=0.1)

    model.save('models/rnn_model')
    loaded_model = model
    diff=[]
    ratio=[]
    p = loaded_model.predict(x_test)

    for u in range(len(y_test)):
        pr = p[u][0]
        ratio.append((y_test[u]/pr)-1)
        diff.append(abs(y_test[u] - pr))


    print(diff)
    diff = pd.Series(diff)
    number_of_outliers = int(0.01*len(diff))
    threshold = diff.nlargest(number_of_outliers).min()

    test = (diff >= threshold).astype(int)
    print(test)
    complement = pd.Series(0, index=np.arange(len(data_n)-testdatasize))
    df['anomaly_rnn'] = complement._append(test, ignore_index='True')
    # df['anomaly_rnn'] = test
    print(df['anomaly_rnn'].value_counts())

    a = df.loc[df['anomaly_rnn'] == 1, ['timestamp', 'temp']]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[len(df)-TEST_SIZE: ]['timestamp'], y=df[len(df)-TEST_SIZE: ]['temp'],
                        mode='lines',
                        name='Временной ряд'))
    fig.add_trace(go.Scatter(x=a['timestamp'], y=a['temp'],
                        mode='markers',
                        name='Аномалия'))
    fig.update_layout(showlegend=True)

    return [fig, z, prediction_time]



from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# print(df['anomaly'].value_counts(()))
# if GEN_ANOMALY == True:
#     print(accuracy_score(df[0 - testdatacut:-prediction_time]['anomaly'],
#                         df[0 - testdatacut:-prediction_time]['anomaly_rnn']))
#     print(roc_auc_score(df[0-testdatacut:-prediction_time]['anomaly'], df[0-testdatacut:-prediction_time]['anomaly_rnn']))
#     print(recall_score(df[0-testdatacut:-prediction_time]['anomaly'], df[0-testdatacut:-prediction_time]['anomaly_rnn']))
#     print(f1_score(df[0-testdatacut:-prediction_time]['anomaly'], df[0-testdatacut:-prediction_time]['anomaly_rnn']))


# print(roc_auc_score(df_out[prediction_time-testdatacut:  ]['target'], df[prediction_time-testdatacut:  ]['anomaly_rnn']))
# print(recall_score(df_out[prediction_time-testdatacut:  ]['target'], df[prediction_time-testdatacut:  ]['anomaly_rnn']))
# print(f1_score(df_out[prediction_time-testdatacut:  ]['target'], df[prediction_time-testdatacut:  ]['anomaly_rnn']))