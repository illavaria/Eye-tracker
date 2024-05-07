from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QHBoxLayout, QComboBox, QLabel, QPushButton, QBoxLayout, QVBoxLayout, QMessageBox

from GoogleCloudStorage import GoogleCloudStorage
from Interface.AbstractTab import AbstractTab


class SettingsTab(AbstractTab):
    def __init__(self, parent_class, tab_name):
        super().__init__(parent_class, tab_name)
        self.camera = parent_class.camera
        self.eyes_recognizer = self.parent_class.eyes_recognizer
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

        self.cloud_manager = parent_class.cloud_manager
        self.versions_combobox = QComboBox(self)
        self.versions = self.cloud_manager.get_versions()
        self.versions_combobox.addItems(GoogleCloudStorage.get_versions_names(self.versions))
        self.load_version_button = QPushButton("Load version")
        self.load_version_button.clicked.connect(self.onLoadVersionPressed)

        cloud_layout = QHBoxLayout()
        cloud_layout.addWidget(QLabel("Model's versions:"))
        cloud_layout.addWidget(self.versions_combobox)
        cloud_layout.addWidget(self.load_version_button)

        self.layout.addLayout(camera_layout)
        self.layout.addLayout(cloud_layout)
        self.layout.addWidget(self.calibration_label)
        self.setLayout(self.layout)

    def onSelectedCameraChanged(self):
        self.parent_class.camera.camera_number = self.camera_list_combobox.currentIndex()

    def onReloadCameraPortsPressed(self):
        available_ports = self.parent_class.camera.list_ports()
        camera_names = self.parent_class.camera.get_camera_names(available_ports)
        self.camera_list_combobox.clear()
        self.camera_list_combobox.addItems(camera_names)

    def onLoadVersionPressed(self):
        chosen_version = self.versions[self.versions_combobox.currentIndex()]
        version_id = chosen_version['id']
        print(version_id)
        if self.cloud_manager.download_files_from_folder(version_id):
            QMessageBox.information(self, "Download finished", f"Downloaded model version {chosen_version['name']} successfully.")
            self.eyes_recognizer.load_state("left.pth", "right.pth")
            print(self.parent_class.eyes_recognizer.is_loaded)
    #        change state label if the model wasn't loaded before


    def tab_selected(self):
        if self.parent_class.calibration_taken:
            self.calibration_label.setText('Calibration status: taken')
        self.layout.update()