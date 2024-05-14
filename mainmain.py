from PyQt6.QtWidgets import QApplication

from GoogleCloudStorage import GoogleCloudStorage
from Interface.MainWindow import MainWindow

app = QApplication([])
default_font = app.font()
default_font.setPointSize(16)
app.setFont(default_font)
main_window = MainWindow()
main_window.showMaximized()
main_window.cloud_manager = GoogleCloudStorage()
app.exec()
