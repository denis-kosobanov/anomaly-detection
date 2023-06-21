import pandas as pd

import numpy as np
import matplotlib
import seaborn
import matplotlib.dates as md
from matplotlib import pyplot as plt

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.covariance import EllipticEnvelope
#from pyemma import msm # not available on Kaggle Kernel
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from utils.data_preprocessing.generator import *
import pickle
import time
def isoforest_out(DATA):

    df = DATA

    df["timestamp"] = df["date"] + " " + df["time"]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.drop(columns = ['date', 'time'], axis = 1)


    df['hours'] = df['timestamp'].dt.hour
    df['daylight'] = ((df['hours'] >= 7) & (df['hours'] <= 22)).astype(int)
    # the day of the week (Monday=0, Sunday=6) and if it's a week end day or week day.
    df['DayOfTheWeek'] = df['timestamp'].dt.dayofweek
    df['WeekDay'] = (df['DayOfTheWeek'] < 5).astype(int)
    # An estimation of anomly population of the dataset (necessary for several algorithm)
    outliers_fraction = 0.1
    # time with int to plot easily
    df['time_epoch'] = (df['timestamp'].astype(np.int64)/100000000000).astype(np.int64)
    # creation of 4 distinct categories that seem useful (week end/day week & night/day)
    df['categories'] = df['WeekDay']*2 + df['daylight']


    data = df[['temp', 'hours', 'daylight', 'DayOfTheWeek', 'WeekDay']]
    min_max_scaler = preprocessing.StandardScaler()
    np_scaled = min_max_scaler.fit_transform(data)
    data = pd.DataFrame(np_scaled)
    # reduce to 2 importants features
    pca = PCA(n_components=2)
    data = pca.fit_transform(data)
    # standardize these 2 new features
    min_max_scaler = preprocessing.StandardScaler()
    np_scaled = min_max_scaler.fit_transform(data)
    data = pd.DataFrame(np_scaled)
    model = pickle.load(open('models/isoforest_model.sav', 'rb'))
    # train isolation forest
    start = time.time()
    model.fit(data)
    z = (time.time() - start)
    # add the data to the main
    df['anomaly_isoforest'] = pd.Series(model.predict(data))
    print(df['anomaly_isoforest'])
    df['anomaly_isoforest'] = df['anomaly_isoforest'].map( {1: 0, -1: 1} )
    print(df['anomaly_isoforest'].value_counts())

    model =  OneClassSVM(nu=0.95 * outliers_fraction)
    data = pd.DataFrame(np_scaled)
    model.fit(data)


    df['anomaly_svm'] = pd.Series(model.predict(data))
    df['anomaly_svm'] = df['anomaly_svm'].map( {1: 0, -1: 1} )
    print(df['anomaly_svm'].value_counts())
    #df.loc[df['anomaly_svm'] == 1, 'anomaly'] = 1


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


    return [fig, str(0.7), str(df.temp.max()), str(df.temp.min()), str(df.temp.mean()), str(len(a)/(len(df))), str(z)]