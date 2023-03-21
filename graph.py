import pandas as pd
import matplotlib.pyplot as plt


def anomaly_graph(df: pd.DataFrame) -> None:
    # Separation of data into normal and abnormal
    normal_data = df[df['anomaly'] == 0]
    anomaly_data = df[df['anomaly'] == 1]

    # Настройка размера графика
    plt.figure(figsize=(15, 5))

    # Drawing a graph
    plt.plot(normal_data.index, normal_data['temp'], 'bo-', label='Normal')
    plt.plot(anomaly_data.index, anomaly_data['temp'], 'ro-', label='Anomaly')

    # Display axis labels and title
    plt.xlabel('Index')
    plt.ylabel('Temperature')
    plt.title('Temperature vs Index')
    plt.legend()

    # Graph display
    plt.show()


def plot_temp_anomaly(df):
    a = df.loc[df['anomaly'] == 0, 'temp']
    b = df.loc[df['anomaly'] == 1, 'temp']

    fig, axs = plt.subplots()
    axs.hist([a, b], bins=32, stacked=True, color=['blue', 'red'])
    plt.legend()
    axs.set_xlabel("Температура")
    axs.set_ylabel("Кол-во записей")
    plt.show()
