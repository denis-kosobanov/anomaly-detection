import sys

from PyQt5 import QtWidgets
from PyQt5.QtGui import QFont

from ui.ui_main_window import Ui_MainWindow

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    font = QFont()
    font.setPointSize(10)
    app.setFont(font)
    MainWindow.show()
    sys.exit(app.exec_())
