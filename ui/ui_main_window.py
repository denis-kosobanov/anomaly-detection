from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import QScrollArea, QTextEdit

import pandas as pd
import plotly
from rnn import *
from rnn_graph_output import rnn_out
from utils.data_preprocessing.generator import *
import plotly.graph_objects as go

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowModality(QtCore.Qt.NonModal)
        MainWindow.resize(1374, 591)

        self.data = None

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
        self.count_records_label = QtWidgets.QLabel(self.centralwidget)
        self.count_records_label.setObjectName("count_records_label")
        self.preproc_layout.addWidget(self.count_records_label)
        self.preproc_button = QtWidgets.QPushButton(self.centralwidget)
        self.preproc_button.setObjectName("preproc_button")
        self.preproc_layout.addWidget(self.preproc_button)
        self.main_par_layout.addLayout(self.preproc_layout)
        self.generate_layout = QtWidgets.QGridLayout()
        self.generate_layout.setObjectName("generate_layout")
        self.generate_cb = QtWidgets.QCheckBox(self.centralwidget)
        self.generate_cb.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.generate_cb.setObjectName("generate_cb")
        self.generate_layout.addWidget(self.generate_cb, 0, 0, 1, 2)
        self.generate_gb = QtWidgets.QGroupBox(self.centralwidget)
        self.generate_gb.setObjectName("generate_gb")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.generate_gb)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.generate_par_layout = QtWidgets.QGridLayout()
        self.generate_par_layout.setObjectName("generate_par_layout")
        self.slice_lineEdit = QtWidgets.QLineEdit(self.generate_gb)
        self.slice_lineEdit.setObjectName("slice_lineEdit")
        self.generate_par_layout.addWidget(self.slice_lineEdit, 1, 1, 1, 1)
        self.range_label = QtWidgets.QLabel(self.generate_gb)
        self.range_label.setObjectName("range_label")
        self.generate_par_layout.addWidget(self.range_label, 2, 0, 1, 1)
        self.windows_label = QtWidgets.QLabel(self.generate_gb)
        self.windows_label.setObjectName("windows_label")
        self.generate_par_layout.addWidget(self.windows_label, 0, 0, 1, 1)
        self.windows_lineEdit = QtWidgets.QLineEdit(self.generate_gb)
        self.windows_lineEdit.setObjectName("windows_lineEdit")
        self.generate_par_layout.addWidget(self.windows_lineEdit, 0, 1, 1, 1)
        self.range_lineEdit = QtWidgets.QLineEdit(self.generate_gb)
        self.range_lineEdit.setObjectName("range_lineEdit")
        self.generate_par_layout.addWidget(self.range_lineEdit, 2, 1, 1, 1)
        self.slice_label = QtWidgets.QLabel(self.generate_gb)
        self.slice_label.setObjectName("slice_label")
        self.generate_par_layout.addWidget(self.slice_label, 1, 0, 1, 1)
        self.horizontalLayout_2.addLayout(self.generate_par_layout)
        self.generate_par_layout2 = QtWidgets.QGridLayout()
        self.generate_par_layout2.setObjectName("generate_par_layout2")
        self.min_lineEdit = QtWidgets.QLineEdit(self.generate_gb)
        self.min_lineEdit.setObjectName("min_lineEdit")
        self.generate_par_layout2.addWidget(self.min_lineEdit, 2, 1, 1, 1)
        self.max_label = QtWidgets.QLabel(self.generate_gb)
        self.max_label.setObjectName("max_label")
        self.generate_par_layout2.addWidget(self.max_label, 1, 0, 1, 1)
        self.max_lineEdit = QtWidgets.QLineEdit(self.generate_gb)
        self.max_lineEdit.setObjectName("max_lineEdit")
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
        self.learn_par_gb = QtWidgets.QGroupBox(self.centralwidget)
        self.learn_par_gb.setObjectName("learn_par_gb")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.learn_par_gb)
        self.gridLayout_2.setObjectName("gridLayout_2")
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

        plot_widget = QWebEngineView()
        plot_widget.setHtml("")

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

        self.graph_layout.addWidget(plot_widget)

        self.start_button = QtWidgets.QPushButton(self.centralwidget)
        self.start_button.setObjectName("start_button")

        self.graph_layout.addWidget(self.start_button)
        self.main_layout.addLayout(self.graph_layout)
        self.main_layout.addLayout(self.log_layout)
        self.gridLayout.addLayout(self.main_layout, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Anomaly Detection"))
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



    def generate_anomaly(self):
        self.data["anomaly"] = 0
        #self.data["anomaly"].iloc[-4100:, ] = ensemble["target"].iloc[-4100:, ]
        if (GEN_ANOMALY == True):
            self.data = generate_anomaly_data(self.data, WINDOW_COUNT, WINDOW_SIZE_LIST)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.data['timestamp'], y=self.data['temp'],
                                 mode='lines',
                                 name='Временной ряд'))
        html = '<html><body>'
        html += plotly.offline.plot(fig, output_type='div', include_plotlyjs='cdn')
        html += '</body></html>'
        self.log_text_edit.append("сгенерированы синтетические аномалии " + str(self.data["anomaly"].value_counts()))
        self.plot_widget.setHtml(html)

    def learn(self):
        mod = ""
        if self.model_rb_1.isChecked() == True:
            mod = "RNN"
            fig = rnn_learn(self.data)
        html = '<html><body>'
        html += plotly.offline.plot(fig[0], output_type='div', include_plotlyjs='cdn')
        html += '</body></html>'
        self.plot_widget.setHtml(html)
        self.log_text_edit.append("модель " + mod + " переобучена")
        self.log_text_edit.append("время обучения " + str(fig[1]))
        self.log_text_edit.append("точность " + str(fig[2]))

    def veltest(self):
        if self.model_rb_1.isChecked() == True:
            fig = rnn_out(self.data)
        # elif self.model_rb_6.isChecked() == True:
        #     fig = catboost_out(self.data)
        # elif self.model_rb_7.isChecked() == True:
        #     fig = isoforest_out(self.data)
        # we create html code of the figure
        html = '<html><body>'
        html += plotly.offline.plot(fig, output_type='div', include_plotlyjs='cdn')
        html += '</body></html>'
        self.plot_widget.setHtml(html)

    def load_file(self):
        _translate = QtCore.QCoreApplication.translate
        fname = QtWidgets.QFileDialog.getOpenFileName()
        self.data = pd.read_csv(fname[0], sep=';')
        self.filename_label.setText(_translate("MainWindow", fname[0]))
        self.data["timestamp"] = self.data["date"] + " " + self.data["time"]
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.data['timestamp'], y=self.data['temp'],
                                 mode='lines',
                                 name='Временной ряд'))
        html = '<html><body>'
        html += plotly.offline.plot(fig, output_type='div', include_plotlyjs='cdn')
        html += '</body></html>'
        self.plot_widget.setHtml(html)
        self.log_text_edit.append("отррыт файл "+ fname[0])
        return fname
