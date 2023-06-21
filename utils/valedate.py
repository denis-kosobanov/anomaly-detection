from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QToolTip


def check_type_line_edit(line_edit, check_type=int):
    try:
        if check_type(line_edit.text()):
            return True
    except ValueError:
        QToolTip.showText(line_edit.mapToGlobal(line_edit.rect().topRight()),
                          f"Ожидалось значение типа {check_type}")
        # Скрытие подсказки через 3 секунды
        QTimer.singleShot(3000, QToolTip.hideText)
        return False


def check_line_edit(line_edit):
    if not line_edit.text():
        QToolTip.showText(line_edit.mapToGlobal(line_edit.rect().topRight()), "Поле ввода пустое!")
        # Скрытие подсказки через 3 секунды
        QTimer.singleShot(3000, QToolTip.hideText)
        return False
    return True


def get_text_line_edit(line_edit, text_type=int):
    if check_line_edit(line_edit) and check_type_line_edit(line_edit, text_type):
        return text_type(line_edit.text())
    return None
