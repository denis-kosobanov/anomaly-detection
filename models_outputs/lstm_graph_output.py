import plotly.graph_objects as go
from utils.data_preprocessing.generator import *
import tensorflow as tf
import time
import seaborn as sns
import numpy as np
from tensorflow import keras
def lstm_out(DATA):

    df = DATA

    df["timestamp"] = df["date"] + " " + df["time"]

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df.drop(columns = ['date', 'time'], axis = 1)
    print(df.columns)
    print(len(df))

    test =  df


    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler = scaler.fit(test[['temp']])

    test['temp'] = scaler.transform(test[['temp']])

    def create_dataset(X, y, time_steps=1):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            v = X.iloc[i:(i + time_steps)].values
            Xs.append(v)
            ys.append(y.iloc[i + time_steps])
        return np.array(Xs), np.array(ys)

    time_steps = 30

    X_test, y_test = create_dataset(test[['temp']], test.temp, time_steps)

    model = keras.models.load_model('models/lstm_model')

    model.evaluate(X_test, y_test)
    start = time.time()
    X_test_pred = model.predict(X_test)
    z = (time.time() - start)
    test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)

    sns.distplot(test_mae_loss, bins=50, kde=True)

    THRESHOLD = 0.7

    test_score_df = pd.DataFrame(test[time_steps:])
    test_score_df['loss'] = test_mae_loss
    test_score_df['threshold'] = THRESHOLD
    test_score_df['anomaly_lstm'] = test_score_df.loss > test_score_df.threshold
    test_score_df['temp'] = test[time_steps:].temp


    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test[time_steps:].timestamp, y=test_score_df.loss,
                        mode='lines',
                        name='Test Loss'))
    fig.add_trace(go.Scatter(x=test[time_steps:].timestamp, y=test_score_df.threshold,
                        mode='lines',
                        name='Threshold'))
    fig.update_layout(showlegend=True)
    #fig.show()

    anomalies = test_score_df[test_score_df.anomaly_lstm == True]

    print(test[time_steps:].temp)
    print(anomalies.temp)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test[time_steps:].timestamp, y=test[time_steps:].temp,
                        mode='lines',
                        name='Временной ряд'))
    fig.add_trace(go.Scatter(x=anomalies.timestamp, y=anomalies.temp,
                        mode='markers',
                        name='Аномалия'))
    fig.update_layout(showlegend=True)
    #fig.show()

    #header = ['anomaly_lstm']
    #test_score_df.to_csv('outputs/output_lstm.csv', columns = header)


    print(df['anomaly'].value_counts(()))


    return [fig, str(abs(test_mae_loss.mean())), str(df.temp.max()), str(df.temp.min()), str(df.temp.mean()), str(len(anomalies)/(len(df))), str(z)]