
import warnings
warnings.filterwarnings('ignore')

from plotly import graph_objs as go
import time


from utils.data_preprocessing.generator import *





class HoltWinters:
    """
    # series - исходный временной ряд
    # slen - длина сезона
    # alpha, beta, gamma - коэффициенты модели Хольта-Винтерса
    # n_preds - горизонт предсказаний
    # scaling_factor - задаёт ширину доверительного интервала по Брутлагу (обычно принимает значения от 2 до 3)

    """

    def __init__(self, series, slen, alpha, beta, gamma, n_preds, scaling_factor=2.5):
        self.series = series
        self.slen = slen
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_preds = n_preds
        self.scaling_factor = scaling_factor

    def initial_trend(self):
        sum = 0.0
        for i in range(self.slen):
            sum += float(self.series[i + self.slen] - self.series[i]) / self.slen
        return sum / self.slen

    def initial_seasonal_components(self):
        seasonals = {}
        season_averages = []
        n_seasons = int(len(self.series) / self.slen)
        # вычисляем сезонные средние
        for j in range(n_seasons):
            season_averages.append(sum(self.series[self.slen * j:self.slen * j + self.slen]) / float(self.slen))
        # вычисляем начальные значения
        for i in range(self.slen):
            sum_of_vals_over_avg = 0.0
            for j in range(n_seasons):
                sum_of_vals_over_avg += self.series[self.slen * j + i] - season_averages[j]
            seasonals[i] = sum_of_vals_over_avg / n_seasons
        return seasonals

    def triple_exponential_smoothing(self):
        self.result = []
        self.Smooth = []
        self.Season = []
        self.Trend = []
        self.PredictedDeviation = []
        self.UpperBond = []
        self.LowerBond = []

        seasonals = self.initial_seasonal_components()

        for i in range(len(self.series) + self.n_preds):
            if i == 0:  # инициализируем значения компонент
                smooth = self.series[0]
                trend = self.initial_trend()
                self.result.append(self.series[0])
                self.Smooth.append(smooth)
                self.Trend.append(trend)
                self.Season.append(seasonals[i % self.slen])

                self.PredictedDeviation.append(0)

                self.UpperBond.append(self.result[0] +
                                      self.scaling_factor *
                                      self.PredictedDeviation[0])

                self.LowerBond.append(self.result[0] -
                                      self.scaling_factor *
                                      self.PredictedDeviation[0])

                continue
            if i >= len(self.series):  # прогнозируем
                m = i - len(self.series) + 1
                self.result.append((smooth + m * trend) + seasonals[i % self.slen])

                # во время прогноза с каждым шагом увеличиваем неопределенность
                self.PredictedDeviation.append(self.PredictedDeviation[-1] * 1.01)

            else:
                val = self.series[i]
                last_smooth, smooth = smooth, self.alpha * (val - seasonals[i % self.slen]) + (1 - self.alpha) * (
                            smooth + trend)
                trend = self.beta * (smooth - last_smooth) + (1 - self.beta) * trend
                seasonals[i % self.slen] = self.gamma * (val - smooth) + (1 - self.gamma) * seasonals[i % self.slen]
                self.result.append(smooth + trend + seasonals[i % self.slen])

                # Отклонение рассчитывается в соответствии с алгоритмом Брутлага
                self.PredictedDeviation.append(self.gamma * np.abs(self.series[i] - self.result[i])
                                               + (1 - self.gamma) * self.PredictedDeviation[-1])

            self.UpperBond.append(self.result[-1] +
                                  self.scaling_factor *
                                  self.PredictedDeviation[-1])

            self.LowerBond.append(self.result[-1] -
                                  self.scaling_factor *
                                  self.PredictedDeviation[-1])

            self.Smooth.append(smooth)
            self.Trend.append(trend)
            self.Season.append(seasonals[i % self.slen])



# Минимизируем функцию потерь с ограничениями на параметры

def timeseriesCVscore(x, data):
    # вектор ошибок
    error = []

    # кусок исходного временного ряда
    # граница установлена так, чтобы на изначальном train-отрезке было не менее пяти полных сезонов
    timeseries_slice = data[:24 * 7 * 5]

    alpha, beta, gamma = x

    while len(timeseries_slice) <= len(data):
        # если выходим за рамки временного ряда - заканчиваем
        try:
            # строим модель с текущими параметрами, предсказываем на n_preds шагов вперёд
            model = HoltWinters(timeseries_slice, slen=24 * 7, alpha=alpha, beta=beta, gamma=gamma, n_preds=100)
            model.triple_exponential_smoothing()

            # считаем ошибку и добавляем в вектор ошибок
            er = data[len(model.result) - 1] - model.result[-1]
            error.append(er)

            # увеличиваем срез для следующего обучения
            timeseries_slice = data[:(len(model.result))]
        except IndexError:
            break

    # Считаем средний квадрат ошибки по вектору ошибок
    mean_squared_error = np.mean(np.array(error) ** 2)
    return mean_squared_error

def holt_winters_learn(DATA, train_size):
    df = DATA
    df["timestamp"] = df["date"] + " " + df["time"]

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.drop(columns=['date', 'time'], axis=1)
    data = df.temp
    alpha_final, beta_final, gamma_final= [0.0020669123607705564, 2.9321831921447217e-05, 0.18868791512613298]
    # Передаем оптимальные значения модели,
    start = time.time()
    model = HoltWinters(data[:-128], slen = 24*7, alpha = alpha_final, beta =
    beta_final, gamma = gamma_final, n_preds = 128, scaling_factor =0.56)
    model.triple_exponential_smoothing()
    z = (time.time() - start)


    Anomalies = np.array([np.NaN]*len(data))
    Anomalies[data.values<model.LowerBond] = data.values[data.values<model.LowerBond]

    df['d'] = np.where(Anomalies > 0, 1, 0)
    print(df['d'].value_counts())
    a = df[len(df)-TEST_SIZE: ].loc[df['d'] == 1, ['timestamp', 'temp']]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[len(df) - TEST_SIZE:]['timestamp'],
                             y=df[len(df) - TEST_SIZE:]['temp'],
                             mode='lines',
                             name='Временной ряд'))
    fig.add_trace(go.Scatter(x=a['timestamp'], y=a['temp'],
                             mode='markers',
                             name='Аномалия'))
    fig.update_layout(showlegend=True)



    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import accuracy_score
    acc_output = []
    if GEN_ANOMALY == True:
        acc_output.append(accuracy_score(df[len(df) - TEST_SIZE:]['anomaly'], df[len(
            df) - TEST_SIZE:]['d']))
        acc_output.append(roc_auc_score(df[len(df)-TEST_SIZE: ]['anomaly'], df[len(
            df)-TEST_SIZE: ]['d']))
        acc_output.append(recall_score(df[len(df)-TEST_SIZE: ]['anomaly'], df[len(
            df)-TEST_SIZE: ]['d']))
        acc_output.append(f1_score(df[len(df)-TEST_SIZE: ]['anomaly'], df[len(
            df)-TEST_SIZE: ]['d']))
    return [fig, z, acc_output]
