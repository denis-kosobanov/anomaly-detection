from sklearn import preprocessing
import plotly.graph_objects as go
from tensorflow import keras
from utils.data_preprocessing.generator import *
def rnn_out(DATA):
    df = DATA
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

    # important parameters and train/test size
    prediction_time = 1
    testdatasize = 4050
    unroll_length = 50
    testdatacut = testdatasize + unroll_length + 1
    print(data_n)

    # test data
    x_test = data_n[0 - testdatacut:-prediction_time].values
    y_test = data_n[prediction_time - testdatacut:][0].values

    def unroll(data,sequence_length=24):
        result = []
        for index in range(len(data) - sequence_length):
            result.append(data[index: index + sequence_length])
        return np.asarray(result)

    # adapt the datasets for the sequence data shape
    x_test  = unroll(x_test,unroll_length)
    y_test  = y_test[-x_test.shape[0]:]

    print("x_test", x_test.shape)
    print("y_test", y_test.shape)

    # model.save('16_model')
    loaded_model = keras.models.load_model('models/rnn_model')
    #loaded_model = model
    diff=[]
    ratio=[]
    p = loaded_model.predict(x_test)
    # predictions = lstm.predict_sequences_multiple(loaded_model, x_test, 50, 50)
    for u in range(len(y_test)):
        pr = p[u][0]
        ratio.append((y_test[u]/pr)-1)
        diff.append(abs(y_test[u] - pr))

    print(diff)
    diff = pd.Series(diff)
    number_of_outliers = int(0.05*len(diff))
    threshold = diff.nlargest(number_of_outliers).min()

    test = (diff >= threshold).astype(int)
    print(test)
    complement = pd.Series(0, index=np.arange(len(data_n) - testdatasize))
    df['anomaly_rnn'] = complement._append(test, ignore_index='True')

    print(df['anomaly_rnn'].value_counts())

    a = df.loc[df['anomaly_rnn'] == 1, ['timestamp', 'temp']]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[len(df) - TEST_SIZE:]['timestamp'], y=df[len(df) - TEST_SIZE:]['temp'],
                             mode='lines',
                             name='Временной ряд'))
    fig.add_trace(go.Scatter(x=a['timestamp'], y=a['temp'],
                             mode='markers',
                             name='Аномалия'))
    fig.update_layout(showlegend=True)

    return fig

