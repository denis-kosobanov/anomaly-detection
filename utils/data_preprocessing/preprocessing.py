import pandas as pd


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
    #df['anomaly'] = float(0.0)

    # Преобразование колонки 'date' в формат datetime
    df['date'] = pd.to_datetime(df['date'])

    # Выбираем данные в указанном диапазоне
    data = df[(df['date'] >= start_date) & (df['date'] <= end_date)]

    return data


def reindex_and_interpolate_temp(df: pd.DataFrame) -> pd.DataFrame:
    # Объединение столбцов 'date' и 'time' и преобразование в datetime
    df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str))
    df = df.drop(columns=['date', 'time'])

    # Разница между соседними записями
    df['diff'] = df['datetime'].diff().dt.total_seconds() / 60

    # Удаление записей с разницей меньше 10 минут
    df = df[df['diff'] >= 10].reset_index(drop=True)

    # Создание промежуточных записей
    new_rows = []
    for i in range(len(df) - 1):
        row1 = df.iloc[i]
        row2 = df.iloc[i + 1]
        diff = row2['diff'] - 10
        num_new_rows = int(diff // 10)
        for j in range(num_new_rows):
            new_row = row1.copy()
            new_row['datetime'] += pd.to_timedelta(10 * (j + 1), unit='m')
            new_row['temp'] = None
            new_rows.append(new_row)

    # Добавление промежуточных записей в исходный DataFrame
    df = df._append(new_rows, ignore_index=True).sort_values(by='datetime').reset_index(drop=True)

    # Приведение столбца 'temp' к типу float
    df['temp'] = df['temp'].astype(float)

    # Интерполяция пропущенных значений 'temp'
    df['temp'] = df['temp'].interpolate()

    # Разделение столбца 'datetime
    df['date'] = df['datetime'].dt.date
    df['time'] = df['datetime'].dt.time
    df['time'] = df['time'].astype(str)
    df['date'] = df['date'].astype(str)

    # Удаление столбца 'diff' и 'datetime'
    df = df.drop(columns=['diff', 'datetime'])

    # Столбцы в нужном порядке
    df = df[['date', 'time', 'temp', 'anomaly']]

    return df
