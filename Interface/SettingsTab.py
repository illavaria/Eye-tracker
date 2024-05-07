from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QHBoxLayout, QComboBox, QLabel, QPushButton, QBoxLayout, QVBoxLayout, QMessageBox

from GoogleCloudStorage import GoogleCloudStorage
from Interface.AbstractTab import AbstractTab
from ModelFileManager import ModelFileManager


class SettingsTab(AbstractTab):
    def __init__(self, parent_class, tab_name):
        super().__init__(parent_class, tab_name)
        self.camera = parent_class.camera
        self.eyes_recognizer = self.parent_class.eyes_recognizer
        self.calibration_label = QLabel(self)
        self.calibration_label.setText('Calibration status: not taken')
        self.calibration_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.model_loaded_label = QLabel(self)
        self.model_loaded_label.setFixedHeight(35)
        self.model_loaded_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        if self.eyes_recognizer.is_loaded:
            self.model_loaded_label.setText(f'Model is loaded, version: {self.parent_class.model_version}')
        else:
            self.model_loaded_label.setText(f'Model is not loaded')
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
        self.download_versions_combobox = QComboBox(self)
        self.download_versions = self.cloud_manager.get_versions()
        self.download_versions_combobox.addItems(GoogleCloudStorage.get_versions_names(self.download_versions))
        self.download_version_button = QPushButton("Download version")
        self.download_version_button.clicked.connect(self.onDownloadVersionPressed)

        cloud_layout = QHBoxLayout()
        cloud_layout.addWidget(QLabel("Model's versions available for download:"))
        cloud_layout.addWidget(self.download_versions_combobox)
        cloud_layout.addWidget(self.download_version_button)


        self.loaded_versions_combobox = QComboBox(self)
        self.loaded_versions = ModelFileManager.get_loaded_versions()
        self.loaded_versions_combobox.addItems(self.loaded_versions)
        self.loaded_version_button = QPushButton("Load version")
        self.loaded_version_button.clicked.connect(self.onLoadVersionPressed)

        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Downloaded model's versions:"))
        model_layout.addWidget(self.loaded_versions_combobox)
        model_layout.addWidget(self.loaded_version_button)

        self.layout.addLayout(camera_layout)
        self.layout.addLayout(cloud_layout)
        self.layout.addLayout(model_layout)
        self.layout.addWidget(self.model_loaded_label)
        self.layout.addWidget(self.calibration_label)
        self.setLayout(self.layout)

    def onSelectedCameraChanged(self):
        self.parent_class.camera.camera_number = self.camera_list_combobox.currentIndex()

    def onReloadCameraPortsPressed(self):
        available_ports = self.parent_class.camera.list_ports()
        camera_names = self.parent_class.camera.get_camera_names(available_ports)
        self.camera_list_combobox.clear()
        self.camera_list_combobox.addItems(camera_names)

    def onDownloadVersionPressed(self):
        chosen_version = self.download_versions[self.download_versions_combobox.currentIndex()]
        version_id = chosen_version['id']
        version_name = chosen_version['name']
        print(version_id)
        if self.cloud_manager.download_files_from_folder(version_id, version_name):
            QMessageBox.information(self, "Download finished", f"Downloaded model version {chosen_version['name']} successfully.")
        self.loaded_versions = ModelFileManager.get_loaded_versions()
        self.loaded_versions_combobox.clear()
        self.loaded_versions_combobox.addItems(self.loaded_versions)

    def onLoadVersionPressed(self):
        if len(self.loaded_versions) == 0:
            QMessageBox.warning(self, 'No downloaded model', "There aren't any downloaded models on the device. "
                                                             "Please download a model then try to load it.")
            return
        chosen_version = self.loaded_versions[self.loaded_versions_combobox.currentIndex()]
        self.eyes_recognizer.load_state(chosen_version + '_left.pth', chosen_version + '_right.pth')
        self.parent_class.model_version = chosen_version
        QMessageBox.information(self, "Loaded succeeded", f"Model version {chosen_version} was loaded successfully.")
        self.model_loaded_label.setText(f'Model is loaded, version: {self.parent_class.model_version}')

    def tab_selected(self):
        if self.parent_class.calibration_taken:
            self.calibration_label.setText('Calibration status: taken')
        self.layout.update()