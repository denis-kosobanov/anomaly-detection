import math
import random
import statistics as statistics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.data_preprocessing.settings import *


def generate_anomaly_data(data: pd.DataFrame) -> pd.DataFrame:
    for _ in range(WINDOW_COUNT):
        # Рандомный выбор размера окна
        random_window_size = random.choice(WINDOW_SIZE_LIST)

        # Выбор окна в данных
        window = pd.DataFrame(_random_select_window(data, random_window_size))

        # Получение данных столбца 'temp' в виде списка
        window_temp = [float(i) for i in window['temp'].to_list()]

        # Генерация аномалий
        window_temp_anomaly = generate_anomaly_window(
            window_temp,
            slice_count=SLICE_COUNT,
            temp_range=TEMP_RANGE,
            k_min=K_MIN,
            k_max=K_MAX
        )

        index_list = list(window.index.values)

        # Присваивание новых значений и пометка как аномалия
        for index, new_temp in zip(index_list, window_temp_anomaly):
            data.loc[index, ['temp', 'anomaly']] = round(new_temp, 2), 1.0

    return data


def generate_anomaly_window(data: list, slice_count=1, k_min=-3, k_max=3, temp_range=5) -> list[float]:
    df = pd.DataFrame(data)
    df.plot(title="Исходный массив")

    k = 0
    while k == 0:
        k = random.randint(k_min, k_max)

    divide_data = _divide_array(data, slice_count)

    for data_part in divide_data:
        lim = temp_range
        mn = statistics.mean(data_part)
        x = np.linspace(0, math.pi * k, len(data_part))
        y = np.sin(x)

        df = pd.DataFrame(y)
        df.plot(title=f"График массива y\nk = {k}")

        for i in range(len(data_part)):
            p = random.randint(-2, 2)
            noise = random.random()
            fx = mn + y[i] * lim + noise * p
            data_part[i] = fx

    concatenated_data = [item for sublist in divide_data for item in sublist]

    df = pd.DataFrame(concatenated_data)
    df.plot(title="Результат алгоритма")

    # plt.show()

    return concatenated_data


def _random_select_window(data: pd.DataFrame, window_size=70) -> pd.DataFrame:
    """
        Выбирает случайное окно указанного размера в датафрейме.
        Если в выбранном окне есть строка в столбце 'anomaly' со значением 1.0, выбирает заново
        Args:
            data (pd.DataFrame): Датафрейм, в котором должно быть выбрано случайное окно.
            window_size (int, optional): Размер окна, которое должно быть выбрано. По умолчанию равно 70.

        Returns:
            pd.DataFrame: Датафрейм выбранного окна
    """
    while True:
        start = random.randint(0, len(data) - window_size)
        window = data[start: start + window_size]
        if not any(window["anomaly"]):
            return data[start: start + window_size]


def _divide_array(array: list, N: int) -> list:
    length = len(array)
    step = length // N
    remainder = length % N
    index = 0
    split = []

    for i in range(N):
        part = step + (i < remainder)
        split.append(array[index: index + part])
        index += part

    return split
