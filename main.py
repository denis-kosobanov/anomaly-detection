import math
import random
import pandas as pd
import matplotlib.pyplot as plt
import statistics as statistics
import numpy as np

from data_preprocessing import *
from generator import *
from settings import *
from graph import *

if __name__ == "__main__":
    df = selected_data("data.txt", '2022-01-01', '2022-04-30')
    df = reindex_and_interpolate_temp(df)
    create_new_rows_plot(df)
    df = df.drop(columns=['new'])
    df = generate_anomaly_data(df, WINDOW_COUNT, WINDOW_SIZE_LIST)
    create_temperature_anomalies_plot(df)
    print(df)
