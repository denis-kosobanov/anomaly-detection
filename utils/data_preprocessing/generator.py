import math
import random
import statistics as statistics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from settings import *


def generate_anomaly_data(data: pd.DataFrame, window_count: int, window_size_list: list) -> pd.DataFrame:
    for _ in range(window_count):
        # Рандомный выбор размера окна
        random_window_size = random.choice(window_size_list)

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


def modify_data_chunks(data, chunk_size, alphas, noise_scales, mode='multiply'):
    num_chunks = len(data) // chunk_size
    new_data_chunks = []

    for i in range(num_chunks):
        chunk = data[i * chunk_size: (i + 1) * chunk_size]

        # Apply modification and add random noise based on chosen mode
        if mode == 'multiply':
            modified_chunk = alphas[i] * chunk + noise_scales[i] * np.random.randn(chunk_size)
        elif mode == 'divide':
            modified_chunk = chunk / alphas[i] + noise_scales[i] * np.random.randn(chunk_size)
        else:
            raise ValueError("Invalid mode. Choose 'multiply' or 'divide'.")

        new_data_chunks.append(modified_chunk)

    # Include the remaining data if it is not a multiple of chunk_size
    if len(data) % chunk_size != 0:
        remaining_chunk = data[num_chunks * chunk_size:]
        if mode == 'multiply':
            modified_remaining_chunk = alphas[-1] * remaining_chunk + noise_scales[-1] * np.random.randn(
                len(remaining_chunk))
        elif mode == 'divide':
            modified_remaining_chunk = remaining_chunk / alphas[-1] + noise_scales[-1] * np.random.randn(
                len(remaining_chunk))
        new_data_chunks.append(modified_remaining_chunk)

    new_data = np.concatenate(new_data_chunks)
    return new_data


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
