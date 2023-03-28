import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo


def create_temperature_anomalies_plot(df: pd.DataFrame) -> None:
    # Создаем график
    fig = go.Figure()

    # Копируем исходный dataframe
    copy_df = df.copy()

    # Создаем новый столбец 'datetime'
    copy_df['datetime'] = pd.to_datetime(copy_df['date'].astype(str) + ' ' + copy_df['time'].astype(str))

    # Добавляем основной график температуры
    fig.add_trace(go.Scatter(x=copy_df['datetime'], y=copy_df['temp'], mode='lines+markers', name='Температура'))

    # Добавляем маркеры для аномалий
    anomalies = copy_df[copy_df['anomaly'] == 1.0]
    fig.add_trace(
        go.Scatter(x=anomalies['datetime'], y=anomalies['temp'], mode='markers', marker=dict(color='red', size=10),
                   name='Аномалия'))

    # Настраиваем заголовки и оси графика
    fig.update_layout(title='Температура и аномалии', xaxis_title='Date', yaxis_title='Температура')

    # Выводим график
    pyo.plot(fig)


def create_new_rows_plot(df: pd.DataFrame) -> None:
    # Создаем график
    fig = go.Figure()

    # Копируем исходный dataframe
    copy_df = df.copy()

    # Создаем новый столбец 'datetime'
    copy_df['datetime'] = pd.to_datetime(copy_df['date'].astype(str) + ' ' + copy_df['time'].astype(str))

    # Добавляем основной график температуры
    fig.add_trace(go.Scatter(x=copy_df['datetime'], y=copy_df['temp'], mode='lines+markers', name='Температура'))

    # Добавляем маркеры для новых строк
    new_rows = copy_df[copy_df['new'] == 1.0]
    fig.add_trace(
        go.Scatter(x=new_rows['datetime'], y=new_rows['temp'], mode='markers', marker=dict(color='green', size=15),
                   name='Новое значение'))

    # Настраиваем заголовки и оси графика
    fig.update_layout(title='Температура и новые значения', xaxis_title='Date', yaxis_title='Температура')

    # Выводим график
    pyo.plot(fig)
