import pickle
import time

import plotly.graph_objects as go
from prophet import Prophet

from utils.data_preprocessing.generator import *


def prophet_learn(DATA, train_size):
    df = DATA

    df["timestamp"] = df["date"] + " " + df["time"]

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[['timestamp', 'temp', 'anomaly']]

    df = df.rename(columns={'timestamp': 'ds',
                            'temp': 'y'})

    train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
    print(train.shape, test.shape)

    start = time.time()
    my_model = Prophet()
    my_model.fit(df)
    z = (time.time() - start)
    with open('models/prophet_model.sav', 'wb') as f:
        pickle.dump(my_model, f)

    forecast = my_model.predict(test)
    print(forecast)
    print(test)
    test["anomaly_prophet"] = forecast['yhat'].values - test["y"].values
    print(test["anomaly_prophet"].value_counts())
    test.loc[abs(test["anomaly_prophet"]) >= 0.9, "anomaly_prophet"] = 1
    test.loc[abs(test["anomaly_prophet"]) <= 0.9, "anomaly_prophet"] = 0

    a = test.loc[test['anomaly_prophet'] == 1, ['ds', 'y']]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test['ds'], y=test['y'],
                             mode='lines',
                             name='Исходный временной ряд'))
    fig.add_trace(go.Scatter(x=a['ds'], y=a['y'],
                             mode='markers',
                             name='Аномалия'))
    fig.update_layout(showlegend=True)
    acc_output = []
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    if GEN_ANOMALY == True:
        acc_output.append(accuracy_score(test['anomaly'], test['anomaly_prophet']))
        acc_output.append(roc_auc_score(test['anomaly'], test['anomaly_prophet']))
        acc_output.append(recall_score(test['anomaly'], test['anomaly_prophet']))
        acc_output.append(f1_score(test['anomaly'], test['anomaly_prophet']))

    return [fig, z, acc_output]
