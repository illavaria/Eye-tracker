from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget
from Interface.CalibrationTab import CalibrationTab
from Interface.GazeDirectionTab import GazeDirectionTab
from Interface.SettingsTab import SettingsTab
from main import Camera


class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()

        self.setWindowTitle("Eye Tracker")
        screen_size = self.screen().size()
        self.setFixedSize(screen_size)
        self.initial_position = self.pos()

        self.camera = Camera(0, '', '')
        self.tab_using_camera = 0

        self.tabs_list = [
            SettingsTab(self, "Settings"),
            CalibrationTab(self, "Calibration"),
            GazeDirectionTab(self,"Gaze Direction")
        ]

        self.tabs = QTabWidget()
        self.tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.tabs.currentChanged.connect(self.tab_changed)
        self.tabs.setDocumentMode(True)

        for tab in self.tabs_list:
            self.tabs.addTab(tab, tab.tab_name)

        self.setCentralWidget(self.tabs)




    def moveEvent(self, event):
        if event.pos() != self.initial_position:
            self.move(self.initial_position)

    def tab_changed(self, index: int) -> None:
        if self.camera.is_capturing():
            self.camera.stop_capture()
            if self.tab_using_camera == 3 or self.tab_using_camera == 4:
                self.tabs_list[self.tab_using_camera].timer.stop()

        self.tabs_list[index].tab_selected()
        self.tab_using_camera = index




app = QApplication([])
main_window = MainWindow()
main_window.showMaximized()
app.exec()