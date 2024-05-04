from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QHBoxLayout, QComboBox, QLabel, QPushButton, QBoxLayout, QVBoxLayout

from Interface.AbstractTab import AbstractTab


class SettingsTab(AbstractTab):
    def __init__(self, parent_class, tab_name):
        super().__init__(parent_class, tab_name)
        self.camera = parent_class.camera
        self.calibration_label = QLabel(self)
        self.calibration_label.setText('Calibration status: not taken')
        self.layout = QVBoxLayout(self)
        self.camera_list_combobox = QComboBox(self)
        self.camera_list_combobox.currentIndexChanged.connect(self.onSelectedCameraChanged)
        self.onReloadCameraPortsPressed()

        self.reload_camera_button = QPushButton("Reload camera list")
        self.reload_camera_button.clicked.connect(self.onReloadCameraPortsPressed)

        camera_layout = QHBoxLayout()
        camera_layout.addWidget(QLabel("Available cameras:"))
        camera_layout.addWidget(self.camera_list_combobox)
        camera_layout.addWidget(self.reload_camera_button)


        self.layout.addLayout(camera_layout)
        self.layout.addWidget(self.calibration_label)
        self.setLayout(self.layout)

    def onSelectedCameraChanged(self):
        self.parent_class.camera.camera_number = self.camera_list_combobox.currentIndex()

    def onReloadCameraPortsPressed(self):
        available_ports = self.parent_class.camera.list_ports()
        camera_names = self.parent_class.camera.get_camera_names(available_ports)
        self.camera_list_combobox.clear()
        self.camera_list_combobox.addItems(camera_names)

    def tab_selected(self):
        if self.parent_class.calibration_taken:
            self.calibration_label.setText('Calibration status: taken')
        self.layout.update()