import os
from functools import partial

import plotly
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QIcon
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import QScrollArea, QTextEdit, QLabel, QLineEdit, QGridLayout, QGroupBox, QHBoxLayout
from statsmodels.tsa.stattools import adfuller

from catboost_model import *

from holt_winters import *

from isoforest_svm import *
from lstm import *
from models_outputs.catboost_graph_output import catboost_out

from models_outputs.isoforest_graph_output import isoforest_out
from models_outputs.lstm_graph_output import lstm_out
from models_outputs.prophet_graph_output import prophet_out
from models_outputs.rnn_graph_output import rnn_out
from models_outputs.tadgan_graph_output import TadGan_out
from models_outputs.xgboost_output import xgboost_out
from prophet_model import *
from rnn import *
from tadgan import *
from utils.data_preprocessing.preprocessing import reindex_and_interpolate_temp
from utils.valedate import get_text_line_edit
from xgboost_model import *


class Ui_MainWindow(object):
    def __init__(self):
        self.data = None
        self.exec_model_rb = ["XGBoost", "SARIMA", "CatBoost", "Holt-Winters", "Isolation Forest"]

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowModality(QtCore.Qt.NonModal)
        MainWindow.resize(1374, 591)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.main_layout = QtWidgets.QHBoxLayout()
        self.main_layout.setObjectName("main_layout")
        self.main_par_layout = QtWidgets.QVBoxLayout()
        self.main_par_layout.setObjectName("main_par_layout")
        self.open_file_layout = QtWidgets.QGridLayout()
        self.open_file_layout.setObjectName("open_file_layout")
        self.open_button = QtWidgets.QPushButton(self.centralwidget)
        self.open_button.setObjectName("open_button")
        self.open_button.clicked.connect(self.load_file)
        self.open_file_layout.addWidget(self.open_button, 1, 0, 1, 1)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.open_file_layout.addWidget(self.label, 0, 0, 1, 1)
        self.filename_label = QtWidgets.QLabel(self.centralwidget)
        self.filename_label.setObjectName("filename_label")
        self.open_file_layout.addWidget(self.filename_label, 1, 1, 1, 1)
        self.main_par_layout.addLayout(self.open_file_layout)
        self.preproc_layout = QtWidgets.QVBoxLayout()
        self.preproc_layout.setObjectName("preproc_layout")
        self.preproc_label = QtWidgets.QLabel(self.centralwidget)
        self.preproc_label.setAlignment(QtCore.Qt.AlignCenter)
        self.preproc_label.setObjectName("preproc_label")
        self.preproc_layout.addWidget(self.preproc_label)
        self.count_records_layout = QHBoxLayout()
        self.count_records_layout.setObjectName(u"count_records_layout")
        self.count_records_label = QLabel(self.centralwidget)
        self.count_records_label.setObjectName(u"count_records_label")
        self.count_records_layout.addWidget(self.count_records_label)
        self.count_records = QLabel(self.centralwidget)
        self.count_records.setObjectName(u"count_records")
        self.count_records_layout.addWidget(self.count_records)
        self.preproc_layout.addLayout(self.count_records_layout)
        self.preproc_button = QtWidgets.QPushButton(self.centralwidget)
        self.preproc_button.setObjectName("preproc_button")
        self.preproc_button.clicked.connect(self.on_preproc_button)
        self.preproc_layout.addWidget(self.preproc_button)
        self.main_par_layout.addLayout(self.preproc_layout)
        self.generate_layout = QtWidgets.QGridLayout()
        self.generate_layout.setObjectName("generate_layout")
        self.generate_cb = QtWidgets.QCheckBox(self.centralwidget)
        self.generate_cb.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.generate_cb.setObjectName("generate_cb")
        self.generate_cb.stateChanged.connect(self.generate_cb_toggle)
        self.generate_layout.addWidget(self.generate_cb, 0, 0, 1, 2)
        self.generate_gb = QtWidgets.QGroupBox(self.centralwidget)
        self.generate_gb.setObjectName("generate_gb")
        self.generate_gb.setCheckable(False)
        self.generate_gb.setEnabled(False)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.generate_gb)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.generate_par_layout = QtWidgets.QGridLayout()
        self.generate_par_layout.setObjectName("generate_par_layout")
        self.slice_lineEdit = QtWidgets.QLineEdit(self.generate_gb)
        self.slice_lineEdit.setObjectName("slice_lineEdit")
        self.slice_lineEdit.setText(str(SLICE_COUNT))
        self.generate_par_layout.addWidget(self.slice_lineEdit, 1, 1, 1, 1)
        self.range_label = QtWidgets.QLabel(self.generate_gb)
        self.range_label.setObjectName("range_label")
        self.generate_par_layout.addWidget(self.range_label, 2, 0, 1, 1)
        self.windows_label = QtWidgets.QLabel(self.generate_gb)
        self.windows_label.setObjectName("windows_label")
        self.generate_par_layout.addWidget(self.windows_label, 0, 0, 1, 1)
        self.windows_lineEdit = QtWidgets.QLineEdit(self.generate_gb)
        self.windows_lineEdit.setObjectName("windows_lineEdit")
        self.windows_lineEdit.setText(str(WINDOW_COUNT))
        self.generate_par_layout.addWidget(self.windows_lineEdit, 0, 1, 1, 1)
        self.range_lineEdit = QtWidgets.QLineEdit(self.generate_gb)
        self.range_lineEdit.setObjectName("range_lineEdit")
        self.range_lineEdit.setText(str(TEMP_RANGE))
        self.generate_par_layout.addWidget(self.range_lineEdit, 2, 1, 1, 1)
        self.slice_label = QtWidgets.QLabel(self.generate_gb)
        self.slice_label.setObjectName("slice_label")
        self.generate_par_layout.addWidget(self.slice_label, 1, 0, 1, 1)
        self.horizontalLayout_2.addLayout(self.generate_par_layout)
        self.generate_par_layout2 = QtWidgets.QGridLayout()
        self.generate_par_layout2.setObjectName("generate_par_layout2")
        self.min_lineEdit = QtWidgets.QLineEdit(self.generate_gb)
        self.min_lineEdit.setObjectName("min_lineEdit")
        self.min_lineEdit.setText(str(K_MIN))
        self.generate_par_layout2.addWidget(self.min_lineEdit, 2, 1, 1, 1)
        self.max_label = QtWidgets.QLabel(self.generate_gb)
        self.max_label.setObjectName("max_label")
        self.generate_par_layout2.addWidget(self.max_label, 1, 0, 1, 1)
        self.max_lineEdit = QtWidgets.QLineEdit(self.generate_gb)
        self.max_lineEdit.setObjectName("max_lineEdit")
        self.max_lineEdit.setText(str(K_MAX))
        self.generate_par_layout2.addWidget(self.max_lineEdit, 1, 1, 1, 1)
        self.min_label = QtWidgets.QLabel(self.generate_gb)
        self.min_label.setObjectName("min_label")
        self.generate_par_layout2.addWidget(self.min_label, 2, 0, 1, 1)
        self.syns_lable = QtWidgets.QLabel(self.generate_gb)
        self.syns_lable.setObjectName("syns_lable")
        self.generate_par_layout2.addWidget(self.syns_lable, 0, 0, 1, 2)
        self.horizontalLayout_2.addLayout(self.generate_par_layout2)
        self.generate_layout.addWidget(self.generate_gb, 1, 0, 1, 2)
        self.main_par_layout.addLayout(self.generate_layout)
        self.generate_button = QtWidgets.QPushButton(self.centralwidget)
        self.generate_button.clicked.connect(self.generate_anomaly)
        self.generate_button.setObjectName("generate_button")
        self.generate_button.setEnabled(False)
        self.main_par_layout.addWidget(self.generate_button)
        self.models_layout = QtWidgets.QVBoxLayout()
        self.models_layout.setObjectName("models_layout")
        self.models_label = QtWidgets.QLabel(self.centralwidget)
        self.models_label.setAlignment(QtCore.Qt.AlignCenter)
        self.models_label.setObjectName("models_label")
        self.models_layout.addWidget(self.models_label)
        self.model_rb_layout = QtWidgets.QGridLayout()
        self.model_rb_layout.setObjectName("model_rb_layout")
        self.model_rb_2 = QtWidgets.QRadioButton(self.centralwidget)
        self.model_rb_2.setObjectName("model_rb_2")
        self.model_rb_layout.addWidget(self.model_rb_2, 1, 0, 1, 1)
        self.model_rb_5 = QtWidgets.QRadioButton(self.centralwidget)
        self.model_rb_5.setObjectName("model_rb_5")
        self.model_rb_layout.addWidget(self.model_rb_5, 0, 2, 1, 1)
        self.model_rb_4 = QtWidgets.QRadioButton(self.centralwidget)
        self.model_rb_4.setObjectName("model_rb_4")
        self.model_rb_layout.addWidget(self.model_rb_4, 1, 1, 1, 1)
        self.model_rb_6 = QtWidgets.QRadioButton(self.centralwidget)
        self.model_rb_6.setObjectName("model_rb_6")
        self.model_rb_layout.addWidget(self.model_rb_6, 1, 2, 1, 1)
        self.model_rb_1 = QtWidgets.QRadioButton(self.centralwidget)
        self.model_rb_1.setObjectName("model_rb_1")
        self.model_rb_layout.addWidget(self.model_rb_1, 0, 0, 1, 1)
        self.model_rb_3 = QtWidgets.QRadioButton(self.centralwidget)
        self.model_rb_3.setObjectName("model_rb_3")
        self.model_rb_layout.addWidget(self.model_rb_3, 0, 1, 1, 1)
        self.model_rb_7 = QtWidgets.QRadioButton(self.centralwidget)
        self.model_rb_7.setObjectName("model_rb_7")
        self.model_rb_layout.addWidget(self.model_rb_7, 0, 3, 1, 1)
        self.model_rb_8 = QtWidgets.QRadioButton(self.centralwidget)
        self.model_rb_8.setObjectName("model_rb_8")
        self.model_rb_layout.addWidget(self.model_rb_8, 1, 3, 1, 1)
        self.models_layout.addLayout(self.model_rb_layout)
        self.main_par_layout.addLayout(self.models_layout)
        self.learn_par_layout = QtWidgets.QVBoxLayout()
        self.learn_par_layout.setObjectName("learn_par_layout")

        # Групп бокс параметры обучения
        self.learn_par_gb = QGroupBox(self.centralwidget)
        self.learn_par_gb.setObjectName(u"learn_par_gb")
        self.horizontalLayout = QHBoxLayout(self.learn_par_gb)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.learn_par_gb_layout = QGridLayout()
        self.learn_par_gb_layout.setObjectName(u"learn_par_gb_layout")
        self.epoch_lineEdit = QLineEdit(self.learn_par_gb)
        self.epoch_lineEdit.setObjectName(u"epoch_lineEdit")
        self.learn_par_gb_layout.addWidget(self.epoch_lineEdit, 0, 1, 1, 1)
        self.epoch_label = QLabel(self.learn_par_gb)
        self.epoch_label.setObjectName(u"epoch_label")
        self.learn_par_gb_layout.addWidget(self.epoch_label, 0, 0, 1, 1)
        self.train_sample_label = QLabel(self.learn_par_gb)
        self.train_sample_label.setObjectName(u"train_sample_label")
        self.learn_par_gb_layout.addWidget(self.train_sample_label, 1, 0, 1, 1)
        self.train_sample_lineEdit = QLineEdit(self.learn_par_gb)
        self.train_sample_lineEdit.setObjectName(u"train_sample_lineEdit")
        self.learn_par_gb_layout.addWidget(self.train_sample_lineEdit, 1, 1, 1, 1)
        self.horizontalLayout.addLayout(self.learn_par_gb_layout)
        self.learn_par_layout.addWidget(self.learn_par_gb)

        self.learn_button = QtWidgets.QPushButton(self.centralwidget)
        self.learn_button.clicked.connect(self.learn)
        self.learn_button.setObjectName("learn_button")
        self.learn_par_layout.addWidget(self.learn_button)
        self.main_par_layout.addLayout(self.learn_par_layout)

        self.main_layout.addLayout(self.main_par_layout)
        self.graph_layout = QtWidgets.QVBoxLayout()
        self.graph_layout.setObjectName("graph_layout")
        self.result_layout = QtWidgets.QVBoxLayout()
        self.result_layout.setObjectName("result_layout")
        self.log_layout = QtWidgets.QVBoxLayout()
        self.log_layout.setObjectName("log_layout")
        self.plot_widget = QWebEngineView()
        self.plot_widget.setHtml("")
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.log_text_edit)
        self.log_layout.addWidget(self.scroll_area)
        self.graph_layout.addWidget(self.plot_widget)

        self.start_button = QtWidgets.QPushButton(self.centralwidget)
        self.start_button.setObjectName("start_button")
        self.start_button.clicked.connect(self.veltest)

        self.graph_layout.addWidget(self.start_button)
        self.main_layout.addLayout(self.graph_layout)
        self.main_layout.addLayout(self.log_layout)
        self.gridLayout.addLayout(self.main_layout, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.set_toggle_models_rb()
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Anomaly Detection"))
        MainWindow.setWindowIcon(QIcon('resources/icon.png'))
        self.open_button.setText(_translate("MainWindow", "Открыть"))
        self.label.setText(_translate("MainWindow", "Выбрать файл с данными"))
        self.filename_label.setText(_translate("MainWindow", "Название файла"))
        self.preproc_label.setText(_translate("MainWindow", "Предобработка"))
        self.count_records_label.setText(_translate("MainWindow", "Количество записей:"))
        self.preproc_button.setText(_translate("MainWindow", "Предобработать"))
        self.generate_cb.setText(_translate("MainWindow", "Синтетические аномалии"))
        self.generate_gb.setTitle(_translate("MainWindow", "Параметры генерации"))
        self.range_label.setText(_translate("MainWindow", "Радиус:"))
        self.windows_label.setText(_translate("MainWindow", "Окна:"))
        self.slice_label.setText(_translate("MainWindow", "Срезы:"))
        self.max_label.setText(_translate("MainWindow", "Макс:"))
        self.min_label.setText(_translate("MainWindow", "Мин:"))
        self.syns_lable.setText(_translate("MainWindow", "Параметры синусойды"))
        self.generate_button.setText(_translate("MainWindow", "Сгенерировать"))
        self.models_label.setText(_translate("MainWindow", "Модель"))
        self.model_rb_2.setText(_translate("MainWindow", "LSTM"))
        self.model_rb_5.setText(_translate("MainWindow", "XGBoost"))
        self.model_rb_4.setText(_translate("MainWindow", "SARIMA"))
        self.model_rb_6.setText(_translate("MainWindow", "CatBoost"))
        self.model_rb_1.setText(_translate("MainWindow", "RNN"))
        self.model_rb_3.setText(_translate("MainWindow", "TadGAN"))
        self.model_rb_7.setText(_translate("MainWindow", "Holt-Winters"))
        self.model_rb_8.setText(_translate("MainWindow", "Isolation Forest"))
        self.learn_par_gb.setTitle(_translate("MainWindow", "Параметры обучения"))
        self.learn_button.setText(_translate("MainWindow", "Обучение"))
        self.start_button.setText(_translate("MainWindow", "Запуск"))
        self.epoch_label.setText(_translate("MainWindow", "Эпохи"))
        self.train_sample_label.setText(_translate("MainWindow", "Обучающая выборка"))

    def generate_cb_toggle(self, state):
        if state == 2:
            self.generate_gb.setEnabled(True)
            self.generate_button.setEnabled(True)
        else:
            self.generate_gb.setEnabled(False)
            self.generate_button.setEnabled(False)

    def set_toggle_models_rb(self):
        for i in range(self.model_rb_layout.count()):
            widget = self.model_rb_layout.itemAt(i).widget()
            if widget:
                widget.toggled.connect(partial(self.on_model_rb_toggled, widget.text()))

    def on_model_rb_toggled(self, text):
        if text in self.exec_model_rb:
            self.epoch_label.setVisible(False)
            self.epoch_lineEdit.setVisible(False)
        else:
            self.epoch_label.setVisible(True)
            self.epoch_lineEdit.setVisible(True)

    # Обновить строку счетчика записей
    def update_data_counter(self):
        self.count_records.setText(str(len(self.data)))

    # Предобработкать загруженные данные
    def on_preproc_button(self):
        self.data = reindex_and_interpolate_temp(self.data)
        self.log_text_edit.append(f"Данные предобработаны.\nКол-во строк изменилось на {len(self.data)}")
        self.update_data_counter()

    # Обновить глобальные параметры генератора
    def update_generate_settings(self):
        lines_edit = [self.windows_lineEdit, self.slice_lineEdit, self.range_lineEdit, self.max_lineEdit,
                      self.min_lineEdit]
        global WINDOW_COUNT, SLICE_COUNT, TEMP_RANGE, K_MAX, K_MIN
        for le in lines_edit:
            if get_text_line_edit(le) is None:
                return False
        WINDOW_COUNT = get_text_line_edit(self.windows_lineEdit)
        SLICE_COUNT = get_text_line_edit(self.slice_lineEdit)
        TEMP_RANGE = get_text_line_edit(self.range_lineEdit)
        K_MAX = get_text_line_edit(self.max_lineEdit)
        K_MIN = get_text_line_edit(self.min_lineEdit)
        return True

    # Сгенерировать аномалии на загруженных данных
    def generate_anomaly(self):
        self.data["timestamp"] = self.data["date"] + " " + self.data["time"]
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])

        if not self.update_generate_settings():
            return
        self.data = generate_anomaly_data(self.data)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.data['timestamp'], y=self.data['temp'],
                                 mode='lines',
                                 name='Временной ряд'))
        html = '<html><body>'
        html += plotly.offline.plot(fig, output_type='div', include_plotlyjs='cdn')
        html += '</body></html>'
        self.log_text_edit.append("Сгенерированы синтетические аномалии " + str(self.data["anomaly"].value_counts()))
        self.log_text_edit.append('---' * 15)
        self.plot_widget.setHtml(html)

    # Обучение модели
    def learn(self):
        mod = ""
        if self.model_rb_1.isChecked() == True:
            mod = "RNN"
            fig = rnn_learn(self.data, int(self.epoch_lineEdit.text()), int(self.train_sample_lineEdit.text()))
        if self.model_rb_2.isChecked() == True:
            mod = "LSTM"
            fig = lstm_learn(self.data, int(self.epoch_lineEdit.text()), int(self.train_sample_lineEdit.text()))
        if self.model_rb_8.isChecked() == True:
            mod = "IsolationForest"
            fig = isoforest_learn(self.data, int(self.train_sample_lineEdit.text()))
        elif self.model_rb_4.isChecked() == True:
            mod = "SARIMA"
            fig = prophet_learn(self.data, int(self.train_sample_lineEdit.text()))
        elif self.model_rb_3.isChecked() == True:
            mod = "tadgan"
            fig = TadGan_learn(self.data, self.train_sample_lineEdit.text(), int(self.train_sample_lineEdit.text()))
        elif self.model_rb_5.isChecked() == True:
            mod = "xgboost"
            fig = xgboost_learn(self.data, self.train_sample_lineEdit.text())
        elif self.model_rb_6.isChecked() == True:
            mod = "catboost"
            fig = catboost_learn(self.data, self.train_sample_lineEdit.text())
        elif self.model_rb_7.isChecked() == True:
            mod = "holt_winters"
            fig = holt_winters_learn(self.data, self.train_sample_lineEdit.text())
        html = '<html><body>'
        html += plotly.offline.plot(fig[0], output_type='div', include_plotlyjs='cdn')
        html += '</body></html>'
        self.plot_widget.setHtml(html)

        self.log_text_edit.append("Модель " + mod + " переобучена c параметрами:")
        self.log_text_edit.append("Эпох: " + self.epoch_lineEdit.text())
        self.log_text_edit.append("Размер обучающей выборки: " + self.train_sample_lineEdit.text())
        self.log_text_edit.append("время обучения: " + str(fig[1]))
        self.log_text_edit.append("точность по roc: " + str(fig[2][1]))
        self.log_text_edit.append("точность по pr: " + str(fig[2][2]))
        self.log_text_edit.append("точность по f1: " + str(fig[2][3]))
        self.log_text_edit.append('---' * 15)

    # Запуск модели
    def veltest(self):
        mod = ""
        if self.model_rb_1.isChecked() == True:
            mod = "rnn"
            fig = rnn_out(self.data)
        elif self.model_rb_2.isChecked() == True:
            mod = "lstm"
            fig = lstm_out(self.data)
        elif self.model_rb_8.isChecked() == True:
            mod = "isoforest"
            fig = isoforest_out(self.data)
        elif self.model_rb_4.isChecked() == True:
            mod = "prophet"
            fig = prophet_out(self.data)
        elif self.model_rb_3.isChecked() == True:
            mod = "tadgan"
            fig = TadGan_out(self.data)
        elif self.model_rb_5.isChecked() == True:
            mod = "xgboost"
            fig = xgboost_out(self.data)
        elif self.model_rb_6.isChecked() == True:
            mod = "catboost"
            fig = catboost_out(self.data)


        elif self.model_rb_7.isChecked() == True:
            mod = "holt_winters"
            fig = holt_winters_learn(self.data, None)


        html = '<html><body>'
        html += plotly.offline.plot(fig[0], output_type='div', include_plotlyjs='cdn')
        html += '</body></html>'
        self.plot_widget.setHtml(html)
        self.log_text_edit.append("Запуск модели")
        if mod != "tadgan" or mod != "holt_winters":
            self.log_text_edit.append("Средняя ошибка предсказания: " + fig[1])

        if mod != "holt_winters":
            self.log_text_edit.append("Время работы: " + fig[6])
            self.log_text_edit.append("Максимальная температура: " + fig[2])
            self.log_text_edit.append("Минимальная температура: " + fig[3])
            self.log_text_edit.append("Средняя температура: " + fig[4])
            self.log_text_edit.append("Процент аномалий в ряде: " + fig[5] + "%")



    def load_file(self):
        _translate = QtCore.QCoreApplication.translate
        fname = QtWidgets.QFileDialog.getOpenFileName()
        self.data = pd.read_csv(fname[0], sep=';')
        self.update_data_counter()
        self.filename_label.setText(_translate("MainWindow", os.path.basename(fname[0])))
        self.data["timestamp"] = self.data["date"] + " " + self.data["time"]
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        if (fname[0].split('/')[len(fname[0].split('/')) - 1] != "412_1"):
            ensemble = pd.read_csv(r"outputs/ensemble_out_412.csv", sep=',')
        if (fname[0].split('/')[len(fname[0].split('/')) - 1] != "412_2"):
            ensemble = pd.read_csv(r"outputs/ensemble_out_412_second.csv", sep=',')
        if (fname[0].split('/')[len(fname[0].split('/')) - 1] != "210"):
            ensemble = pd.read_csv(r"outputs/ensemble_out_210.csv", sep=',')
        if (fname[0].split('/')[len(fname[0].split('/')) - 1] != "210_2"):
            ensemble = pd.read_csv(r"outputs/ensemble_out_210_second.csv", sep=',')
        if (fname[0].split('/')[len(fname[0].split('/')) - 1] != "420"):
            ensemble = pd.read_csv(r"outputs/ensemble_out_420.csv", sep=',')
        if (fname[0].split('/')[len(fname[0].split('/')) - 1] != "420_2"):
            ensemble = pd.read_csv(r"outputs/ensemble_out_420_second.csv", sep=',')
        if (fname[0].split('/')[len(fname[0].split('/')) - 1] != "316"):
            ensemble = pd.read_csv(r"outputs/ensemble_out_316.csv", sep=',')
        if (fname[0].split('/')[len(fname[0].split('/')) - 1] != "316_2"):
            ensemble = pd.read_csv(r"outputs/ensemble_out_316_second.csv", sep=',')
        ensemble = pd.read_csv(r"outputs/ensemble_out_412.csv", sep=',')
        self.data["anomaly"] = 0
        self.data["anomaly"].iloc[-4100:, ] = ensemble["target"].iloc[-4100:, ]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.data['timestamp'], y=self.data['temp'],
                                 mode='lines',
                                 name='Временной ряд'))

        self.plot_widget.setHtml(fig.to_html(include_plotlyjs='cdn'))
        self.log_text_edit.append("открыт файл " + fname[0])
        self.test_stationarity()
        html = '<html><body>'
        html += plotly.offline.plot(fig, output_type='div', include_plotlyjs='cdn')
        html += '</body></html>'
        self.plot_widget.setHtml(html)

        return fname

    def test_stationarity(self):
        dftest = adfuller(self.data['temp'], autolag='AIC')
        dfoutput = pd.Series(dftest[0:4],
                             index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value (%s)' % key] = value

        critical_value = dftest[4]['5%']
        test_statistic = dftest[0]
        alpha = 1e-3
        pvalue = dftest[1]
        if pvalue and alpha and test_statistic and critical_value:
            self.log_text_edit.append("Ряд стационарен")
            self.log_text_edit.append('---' * 15)
        else:
            self.log_text_edit.append("Ряд не стационарен")
            self.log_text_edit.append('---' * 15)
