from typing import Tuple

import pandas as pd

from utils.data_preprocessing.generator import *


def get_xgboost_x_y(
        indices: list,
        data: np.array,
        target_sequence_length,
        input_seq_len: int
) -> Tuple[np.array, np.array]:



    for i, idx in enumerate(indices):


        data_instance = data[idx[0]:idx[1]]

        x = data_instance[0:input_seq_len]

        assert len(x) == input_seq_len

        y = data_instance[input_seq_len:input_seq_len + target_sequence_length]

        if i == 0:

            all_y = y.reshape(1, -1)

            all_x = x.reshape(1, -1)

        else:

            all_y = np.concatenate((all_y, y.reshape(1, -1)), axis=0)

            all_x = np.concatenate((all_x, x.reshape(1, -1)), axis=0)

    return all_x, all_y


def load_data(DATA):

    df = DATA
    df["timestamp"] = df["date"] + " " + df["time"]

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df.drop(columns=['date', 'time'], axis=1)

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # df = generate_anomaly_data(df, WINDOW_COUNT , WINDOW_SIZE_LIST)
    df.plot(x='timestamp', y='temp')

    df['hours'] = df['timestamp'].dt.hour
    df['daylight'] = ((df['hours'] >= 7) & (df['hours'] <= 22)).astype(int)

    df['DayOfTheWeek'] = df['timestamp'].dt.dayofweek
    df['WeekDay'] = (df['DayOfTheWeek'] < 5).astype(int)

    outliers_fraction = 0.05

    df['time_epoch'] = (df['timestamp'].astype(np.int64) / 100000000000).astype(np.int64)

    target_variable = "temp"

    timestamp_col = "timestamp"



    df[target_variable] = pd.to_numeric(df["temp"])



    df.index = pd.to_datetime(df[timestamp_col])


    df = df[[target_variable, "anomaly"]]


    df.sort_values(by=timestamp_col, ascending=True, inplace=True)

    return df


def get_indices_entire_sequence(
        data: pd.DataFrame,
        window_size: int,
        step_size: int
) -> list:


    stop_position = len(data) - 1


    subseq_first_idx = 0

    subseq_last_idx = window_size

    indices = []

    while subseq_last_idx <= stop_position:
        indices.append((subseq_first_idx, subseq_last_idx))

        subseq_first_idx += step_size

        subseq_last_idx += step_size

    return indices
