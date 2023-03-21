import numpy as np
import pandas as pd


def interpolate_time_series(x_data, y_data, degree):
    # Получаем индексы пропущенных значений (np.nan)
    nan_idxs = np.isnan(y_data)

    # Выполняем полиномиальную интерполяцию с заданным интерполяционным полиномом
    coefficients = np.polyfit(x_data[~nan_idxs], y_data[~nan_idxs], degree)

    # Вычисляем интерполированные значения y для пропущенных значений x
    interpolated_y_values = np.polyval(coefficients, x_data[nan_idxs])

    # Заменяем пропущенные значения интерполированными значениями
    y_data[nan_idxs] = interpolated_y_values

    return y_data


def selected_data(path: str, start_date: str, end_date: str) -> pd.DataFrame:
    # Открываем файл на чтение
    with open(path, "r") as file:
        # Считываем содержимое файла
        data = file.readlines()

    # Преобразуем каждую строку в список значений
    data = [line.strip().split() for line in data]

    # Создаем DataFrame с заданными названиями столбцов
    df = pd.DataFrame(data, columns=["date", "time", "temp"])

    # Добавляем колонку 'anomaly' и заполняем нулями
    df['anomaly'] = float(0.0)

    # Преобразование колонки 'date' в формат datetime
    df['date'] = pd.to_datetime(df['date'])

    # Выбираем данные в указанном диапазоне
    data = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

    return data


def check_gaps(df: pd.DataFrame):
    # Преобразование даты в тип данных datetime и сортировка
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)

    # Вычисление разницы между датами
    df['date_diff'] = df['date'].diff().dt.days

    # Удаление первой строки, т.к. нам не нужен date_diff для первой даты
    df = df.iloc[1:]

    # Проверка на пропуски с разницей в один день и один месяц
    day_gaps = df[df['date_diff'] != 1]
    month_gaps = df[df['date_diff'] != 30]
    # print("Пропуски с разницей в один день:")
    # print(day_gaps[df["date_diff"] != 0.0])
    # print("Пропуски с разницей в один месяц:")
    # print(month_gaps[df["date_diff"] != 0.0])
    # month_gaps.to_csv(r'month_gaps.csv', index=False)
