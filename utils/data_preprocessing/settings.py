"""
    Описание глобальных переменных:
        WINDOW_SIZE_LIST: список возможных размеров окон. 6 записей ~ 1 час, 60 записей ~ 10 часов и т.д.;
        WINDOW_COUNT: кол-во окон для генерации аномалий;
        SLICE_COUNT: кол-во аномалий в окне;
        TEMP_RANGE: радиус для генерации шума в данных;
        K_MIN: минимальное значение для генерации функции шаблона.
        K_MAX: максимальное значение для генерации функции шаблона;
"""
import pandas as pd

GEN_ANOMALY = True
ENSEMLE = False
if ENSEMLE == True:
    ensemble = pd.read_csv(r"ensemble.csv", sep=';')
DATA = pd.read_csv(r"C:/Users/dadon/PycharmProjects/pythonProject20/first_half 210.csv", sep=';')

TEST_SIZE = 4100
WINDOW_SIZE_LIST = [60, 70, 80]
WINDOW_COUNT = 3
SLICE_COUNT = 5
TEMP_RANGE = 1
K_MIN = -3
K_MAX = 3
