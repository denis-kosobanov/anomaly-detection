import numpy as np
import tensorflow as tf
import seaborn as sns
from matplotlib.pylab import rcParams
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import openpyxl

sns.set(style='whitegrid', palette='muted')
rcParams['figure.figsize'] = 14, 8
np.random.seed(1)
tf.random.set_seed(1)
from generator import *
pd.options.mode.chained_assignment = None

df = DATA
ensemble = pd.read_csv(r"outputs/ensemble_out_316.csv", sep=',')
df["anomaly"] = 0
df["anomaly"].iloc[-4100:, ] = ensemble["target"].iloc[-4100:, ]
if (GEN_ANOMALY == True):
    #df['anomaly'] = 0
    df = generate_anomaly_data(df, WINDOW_COUNT, WINDOW_SIZE_LIST)
df["timestamp"] = df["date"] + " " + df["time"]

df['timestamp'] = pd.to_datetime(df['timestamp'])

df.drop(columns = ['date', 'time'], axis = 1)
print(df.columns)
print(len(df))
train_size = len(df)-TEST_SIZE-30
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
print(train.shape, test.shape)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler = scaler.fit(train[['temp']])

train['temp'] = scaler.transform(train[['temp']])
test['temp'] = scaler.transform(test[['temp']])

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 30

X_train, y_train = create_dataset(train[['temp']], train.temp, time_steps)
X_test, y_test = create_dataset(test[['temp']], test.temp, time_steps)

print(X_train.shape)
timesteps = X_train.shape[1]
num_features = X_train.shape[2]
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, RepeatVector, TimeDistributed

model = Sequential([
    LSTM(128, input_shape=(timesteps, num_features)),
    Dropout(0.2),
    RepeatVector(timesteps),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    TimeDistributed(Dense(num_features))
])

model.compile(loss='mae', optimizer='adam')
model.summary()

es = tf.keras.callbacks.EarlyStopping(restore_best_weights=True, patience=5)
history = model.fit(
    X_train, y_train,
    epochs=1,#100
    batch_size=32,
    validation_split=0.1,
    callbacks = [es],
    shuffle=False
)

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()

model.evaluate(X_test, y_test)

X_train_pred = model.predict(X_train)

train_mae_loss = pd.DataFrame(np.mean(np.abs(X_train_pred - X_train), axis=1), columns=['Error'])

sns.distplot(train_mae_loss, bins=50, kde=True)

X_test_pred = model.predict(X_test)

test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)

sns.distplot(test_mae_loss, bins=50, kde=True)

THRESHOLD = 0.50

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
fig.show()

anomalies = test_score_df[test_score_df.anomaly_lstm == True]

fig = go.Figure()
fig.add_trace(go.Scatter(x=test[time_steps:].timestamp, y=test[time_steps:].temp,
                    mode='lines',
                    name='Временной ряд'))
fig.add_trace(go.Scatter(x=anomalies.timestamp, y=anomalies.temp,
                    mode='markers',
                    name='Аномалия'))
fig.update_layout(showlegend=True)
fig.show()

header = ['anomaly_lstm']
test_score_df.to_csv('outputs/output_lstm.csv', columns = header)




from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
print(df['anomaly'].value_counts(()))

from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

print(accuracy_score(df['anomaly'][-4100:, ], test_score_df['anomaly_lstm']))
print(roc_auc_score(df['anomaly'][-4100:, ], test_score_df['anomaly_lstm']))
print(recall_score(df['anomaly'][-4100:, ], test_score_df['anomaly_lstm']))
print(f1_score(df['anomaly'][-4100:, ], test_score_df['anomaly_lstm']))
