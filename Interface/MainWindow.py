from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget
from Interface.CalibrationTab import CalibrationTab
from Interface.SettingsTab import SettingsTab


class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("Eye Tracker")
        screen_size = self.screen().size()
        self.setFixedSize(screen_size)

        # widget = CalibrationTab()
        # self.setCentralWidget(widget)

        self.initial_position = self.pos()
        self.setWindowTitle("My App")

        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.TabPosition.North)
        tabs.setDocumentMode(True)


        tabs.addTab(SettingsTab(), "Settings")
        tabs.addTab(CalibrationTab(), "Calibration")
        for tab_name in ["Gaze Tracking", "Cheating Detection"]:
            tabs.addTab(CalibrationTab(), tab_name)

        self.setCentralWidget(tabs)

    def moveEvent(self, event):
        if event.pos() != self.initial_position:
            self.move(self.initial_position)




app = QApplication([])
main_window = MainWindow()
main_window.showMaximized()
app.exec()