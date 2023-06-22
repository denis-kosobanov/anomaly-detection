import numpy as np
from typing import Tuple
import pandas as pd
from utils.data_preprocessing.generator import *

def get_xgboost_x_y(
        indices: list,
        data: np.array,
        target_sequence_length,
        input_seq_len: int
) -> Tuple[np.array, np.array]:
    """
    Args:
        indices: List of index positions at which data should be sliced
        data: A univariate time series
        target_sequence_length: The forecasting horizon, m
        input_seq_len: The length of the model input, n
    Output:
        all_x: np.array of shape (number of instances, input seq len)
        all_y: np.array of shape (number of instances, target seq len)
    """
    print("Preparing data..")

    # Loop over list of training indices
    for i, idx in enumerate(indices):

        # Slice data into instance of length input length + target length
        data_instance = data[idx[0]:idx[1]]

        x = data_instance[0:input_seq_len]

        assert len(x) == input_seq_len

        y = data_instance[input_seq_len:input_seq_len + target_sequence_length]

        # Create all_y and all_x objects in first loop iteration
        if i == 0:

            all_y = y.reshape(1, -1)

            all_x = x.reshape(1, -1)

        else:

            all_y = np.concatenate((all_y, y.reshape(1, -1)), axis=0)

            all_x = np.concatenate((all_x, x.reshape(1, -1)), axis=0)

    print("Finished preparing data!")

    return all_x, all_y


def load_data(DATA):
    # Read data
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

    # Convert separator from "," to "." and make numeric

    df[target_variable] = pd.to_numeric(df["temp"])

    # Convert HourDK to proper date time and make it index

    df.index = pd.to_datetime(df[timestamp_col])

    # Discard all cols except DKK prices
    df = df[[target_variable, "anomaly"]]

    # Order by ascending time stamp
    df.sort_values(by=timestamp_col, ascending=True, inplace=True)

    return df


def get_indices_entire_sequence(
        data: pd.DataFrame,
        window_size: int,
        step_size: int
) -> list:
    """
    Produce all the start and end index positions that is needed to produce
    the sub-sequences.
    Returns a list of tuples. Each tuple is (start_idx, end_idx) of a sub-
    sequence. These tuples should be used to slice the dataset into sub-
    sequences. These sub-sequences should then be passed into a function
    that slices them into input and target sequences.

    Args:
        data (pd.DataFrame): Partitioned data set, e.g. training data
        window_size (int): The desired length of each sub-sequence. Should be
                           (input_sequence_length + target_sequence_length)
                           E.g. if you want the model to consider the past 100
                           time steps in order to predict the future 50
                           time steps, window_size = 100+50 = 150
        step_size (int): Size of each step as the data sequence is traversed
                         by the moving window.
                         If 1, the first sub-sequence will be [0:window_size],
                         and the next will be [1:window_size].
    Return:
        indices: a list of tuples
    """

    stop_position = len(data) - 1  # 1- because of 0 indexing

    # Start the first sub-sequence at index position 0
    subseq_first_idx = 0

    subseq_last_idx = window_size

    indices = []

    while subseq_last_idx <= stop_position:
        indices.append((subseq_first_idx, subseq_last_idx))

        subseq_first_idx += step_size

        subseq_last_idx += step_size

    return indices